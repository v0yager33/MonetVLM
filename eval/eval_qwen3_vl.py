import argparse
import json
import math
import os
import re
import time
from collections import defaultdict
from PIL import Image 
from tqdm import tqdm
from vllm import LLM, SamplingParams

# 严格匹配 \boxed{...}
BOXED_PATTERN = re.compile(r"\\boxed\{([^}]*)\}")

def extract_boxed_answer(text):
    """严格提取 boxed 内部内容并转大写"""
    if not text: return None
    matches = BOXED_PATTERN.findall(text)
    if matches:
        return matches[-1].strip().upper()
    return None

def load_test_dataset(test_path, limit=0):
    records = []
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"找不到测试集文件: {test_path}")
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))
                if 0 < limit <= len(records): break
    return records

def build_vllm_inputs(test_data, tokenizer):
    """预处理多模态输入"""
    inputs, metadata = [], []
    print(f"正在预处理 {len(test_data)} 个样本...")
    for sample in test_data:
        image_path = sample.get("image_path", "")
        if not os.path.exists(image_path): continue
        
        image = Image.open(image_path).convert("RGB")
        query = sample.get("query", "").replace("<image>", "").strip()
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "image", "image": image}, 
                {"type": "text", "text": query}
            ]}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs.append({"prompt": prompt, "multi_modal_data": {"image": image}})
        metadata.append({
            "gt": sample.get("answer", "").upper().strip(),
            "genre": sample.get("genre", "Unknown"),
            "image_path": image_path
        })
    return inputs, metadata

def run_evaluation_round(llm, sampling_params, vllm_inputs, batch_metadata, output_path, round_idx):
    correct_count = 0
    format_count = 0
    
    outputs = llm.generate(vllm_inputs, sampling_params=sampling_params)

    for output, meta in zip(outputs, batch_metadata):
        res_text = output.outputs[0].text
        pred = extract_boxed_answer(res_text)
        is_correct = (pred == meta["gt"]) if pred else False
        
        if pred: format_count += 1
        if is_correct: correct_count += 1
        
        record = {
            "round": round_idx + 1,
            "gt": meta["gt"],
            "pred": pred or "N/A",
            "is_correct": is_correct,
            "response": res_text
        }
        
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
    acc = (correct_count / len(vllm_inputs) * 100) if vllm_inputs else 0
    fmt_rate = (format_count / len(vllm_inputs) * 100) if vllm_inputs else 0
    return acc, fmt_rate

def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL WikiArt Multi-round Eval")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--num_rounds", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.6)
    # 补全缺失参数
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--output_path", type=str, default="eval_results.jsonl")
    parser.add_argument("--max_model_len", type=int, default=16384)
    parser.add_argument("--gpu_util", type=float, default=0.85)
    args = parser.parse_args()

    if os.path.exists(args.output_path): os.remove(args.output_path)
    summary_path = args.output_path.replace(".jsonl", "_summary.json")

    # 1. 初始化 vLLM
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_util,
        mm_processor_kwargs={"max_pixels": 512 * 512}
    )
    tokenizer = llm.get_tokenizer()
    
    # 2. 预处理数据
    test_data = load_test_dataset(args.test_path)
    vllm_inputs, batch_metadata = build_vllm_inputs(test_data, tokenizer)
    
    # 使用完整的采样参数
    sampling_params = SamplingParams(
        temperature=args.temperature, 
        top_p=args.top_p, 
        max_tokens=args.max_new_tokens
    )
    
    all_round_accs = []
    start_time = time.time()

    # 3. 推理循环
    for i in range(args.num_rounds):
        acc, fmt = run_evaluation_round(llm, sampling_params, vllm_inputs, batch_metadata, args.output_path, i)
        all_round_accs.append(acc)
        print(f"Round {i+1}/{args.num_rounds} | Acc: {acc:.2f}% | Format: {fmt:.2f}%")

    # 4. 统计
    total_duration = time.time() - start_time
    mean_acc = sum(all_round_accs) / len(all_round_accs)
    
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

    print(f"\n任务完成！平均准确率: {mean_acc:.2f}%")
    print(f"统计文件已生成: {summary_path}")

if __name__ == "__main__":
    main()