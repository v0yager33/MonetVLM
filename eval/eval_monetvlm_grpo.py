import argparse
import json
import os
import re
import sys
import time
import math
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.distributed as dist
from PIL import Image
from tqdm import tqdm
from transformers.cache_utils import DynamicCache

# 导入你本地的模型定义
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vlm_model import SparkVLM, SparkVLMConfig
from dataset import smart_resize
from inference import load_model

BOXED_PATTERN = re.compile(r"\\boxed\{([^}]*)\}")

def extract_boxed_answer(text: str) -> Optional[str]:
    if not text: return None
    matches = BOXED_PATTERN.findall(text)
    if matches:
        return matches[-1].strip().upper()
    return None

def load_test_dataset(test_path: str, limit: int = 0) -> List[Dict[str, Any]]:
    records = []
    if not os.path.exists(test_path): return []
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            records.append(json.loads(line.strip()))
            if 0 < limit <= len(records): break
    return records

def sample_next_token(logits: torch.Tensor, temperature: float = 0.0, top_p: float = 0.9) -> torch.Tensor:
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)
    scaled_logits = logits / temperature
    sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    remove_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
    sorted_logits[remove_mask] = float("-inf")
    probs = F.softmax(sorted_logits, dim=-1)
    sampled_index = torch.multinomial(probs, num_samples=1)
    return torch.gather(sorted_indices, -1, sampled_index)

@torch.no_grad()
def generate_batch_parallel(
    model: torch.nn.Module,
    processor: Any,
    tokenizer: Any,
    batch_samples: List[Dict[str, str]],
    max_new_tokens: int = 512,
    temperature: float = 0.6,
    top_p: float = 0.95,
    device: str = "cuda",
) -> List[str]:
    batch_size = len(batch_samples)
    if batch_size == 0: return []
    
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_token_id
    
    all_prefill_logits, all_prefill_pasts, prompt_lengths = [], [], []
    
    # === Prefill: 使用你原有的 image_grid_thw 逻辑 ===
    for sample in batch_samples:
        image = Image.open(sample["image_path"]).convert("RGB")
        image = smart_resize(image, patch_size=14, compress_grid=2)
        width, height = image.size
        # 对齐你原代码的 grid 计算
        grid_t, grid_h, grid_w = 1, height // 28, width // 28
        image_pad_num = grid_t * grid_h * grid_w
        image_grid_thw = torch.tensor([[grid_t, grid_h, grid_w]], dtype=torch.long, device=device)
        image_sizes = torch.tensor([[height, width]], dtype=torch.long, device=device)
        
        pixel_values = processor.image_processor(images=image, do_resize=False, return_tensors="pt")["pixel_values"].to(device=device, dtype=torch.bfloat16)
        
        messages = [{"role": "system", "content": "You are a helpful multimodal assistant."}, {"role": "user", "content": sample["query"]}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        text = text.replace("<image>", "<|image_pad|>" * image_pad_num)
        input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
        prompt_lengths.append(input_ids.shape[1])
        
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_sizes=image_sizes,
                image_grid_thw=image_grid_thw, # 传入原代码要求的参数
                use_cache=True,
            )
            all_prefill_logits.append(outputs.logits[:, -1, :])
            all_prefill_pasts.append(outputs.past_key_values)
    
    # === 合并 KV Cache (适配 past.layers[idx].keys 结构) ===
    max_prompt_len = max(prompt_lengths)
    num_layers = len(all_prefill_pasts[0])
    batch_past = DynamicCache()
    
    for layer_idx in range(num_layers):
        layer_keys, layer_values = [], []
        for p_idx in range(batch_size):
            # 严格按照你原代码的 keys/values 访问方式
            pk = all_prefill_pasts[p_idx].layers[layer_idx].keys
            pv = all_prefill_pasts[p_idx].layers[layer_idx].values
            pad_len = max_prompt_len - pk.shape[2]
            if pad_len > 0:
                pk = F.pad(pk, (0, 0, pad_len, 0), value=0.0)
                pv = F.pad(pv, (0, 0, pad_len, 0), value=0.0)
            layer_keys.append(pk)
            layer_values.append(pv)
        batch_past.update(torch.cat(layer_keys, dim=0), torch.cat(layer_values, dim=0), layer_idx)

    # === Decode 阶段 ===
    batch_attention_mask = torch.zeros(batch_size, max_prompt_len, dtype=torch.long, device=device)
    for idx, plen in enumerate(prompt_lengths): batch_attention_mask[idx, max_prompt_len - plen:] = 1
        
    next_logits = torch.cat(all_prefill_logits, dim=0)
    all_generated = [[] for _ in range(batch_size)]
    finished = [False] * batch_size
    
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(max_new_tokens):
            next_token_ids = sample_next_token(next_logits, temperature, top_p)
            active = False
            for i in range(batch_size):
                if not finished[i]:
                    if next_token_ids[i].item() == eos_token_id: finished[i] = True
                    else:
                        all_generated[i].append(next_token_ids[i:i+1])
                        active = True
            if not active or all(finished): break
            
            # Pad 已经结束的序列
            for i in range(batch_size):
                if finished[i]: next_token_ids[i] = pad_token_id

            batch_attention_mask = torch.cat([batch_attention_mask, torch.ones(batch_size, 1, dtype=torch.long, device=device)], dim=1)
            outputs = model(input_ids=next_token_ids, attention_mask=batch_attention_mask, past_key_values=batch_past, use_cache=True)
            next_logits, batch_past = outputs.logits[:, -1, :], outputs.past_key_values

    return [tokenizer.decode(torch.cat(gen).view(-1), skip_special_tokens=True) if gen else "" for gen in all_generated]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_rounds", type=int, default=1)
    parser.add_argument("--output_path", type=str, default="eval_results.jsonl")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank, world_size = dist.get_rank(), dist.get_world_size()
        torch.cuda.set_device(rank)
    else: rank, world_size = 0, 1

    if rank == 0:
        if os.path.exists(args.output_path): os.remove(args.output_path)
    
    model, processor, tokenizer = load_model(args.model_dir)
    model.eval()
    test_data = load_test_dataset(args.test_path, limit=args.limit)
    
    chunk_size = (len(test_data) + world_size - 1) // world_size
    local_data = test_data[rank * chunk_size : (rank + 1) * chunk_size]
    
    all_round_accs = []
    start_time = time.time()

    for r in range(args.num_rounds):
        correct_in_round = 0
        total_in_round = 0
        
        pbar = tqdm(range(0, len(local_data), args.batch_size), desc=f"Rank {rank} Round {r+1}", disable=rank!=0)
        for i in pbar:
            batch = local_data[i : i + args.batch_size]
            batch_samples = [s for s in batch if os.path.exists(s["image_path"])]
            if not batch_samples: continue
            
            resps = generate_batch_parallel(model, processor, tokenizer, batch_samples, args.max_new_tokens, args.temperature, args.top_p)
            
            for resp, sample in zip(resps, batch_samples):
                gt = sample["answer"].upper().strip()
                pred = extract_boxed_answer(resp)
                is_correct = (pred == gt) if pred else False
                
                total_in_round += 1
                if is_correct: correct_in_round += 1
                
                record = {
                    "round": r + 1, "image_path": sample["image_path"],
                    "gt": gt, "pred": pred or "N/A",
                    "is_correct": is_correct, "response": resp
                }
                
                with open(args.output_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
            if rank == 0: pbar.set_postfix({"acc": f"{100*correct_in_round/max(1, total_in_round):.2f}%"})

        if world_size > 1:
            stats = torch.tensor([correct_in_round, total_in_round], device="cuda")
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            correct_in_round, total_in_round = stats.tolist()

        if rank == 0:
            round_acc = (correct_in_round / total_in_round * 100) if total_in_round > 0 else 0
            all_round_accs.append(round_acc)
            print(f"\n[Round {r+1} Complete] Global Acc: {round_acc:.2f}%")

    if rank == 0:
        total_duration = time.time() - start_time
        summary_path = args.output_path.replace(".jsonl", "_summary.json")
        mean_acc = sum(all_round_accs) / len(all_round_accs) if all_round_accs else 0
        summary_data = {
            "config": vars(args),
            "summary": {
                "mean_accuracy": round(mean_acc, 2),
                "per_round_accs": [round(x, 2) for x in all_round_accs],
                "total_samples": len(test_data),
                "elapsed_time": round(total_duration, 2)
            }
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=4, ensure_ascii=False)
        print(f"\nSummary saved to {summary_path}")

    if dist.is_initialized(): dist.destroy_process_group()

if __name__ == "__main__":
    main()