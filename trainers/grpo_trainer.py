"""
SparkVLM 自定义 GRPO Trainer - 纯 PyTorch 实现 DeepSeek-R1 的组内相对策略优化

核心算法:
1. 对每个 Prompt 生成 G 个回复 (自定义 generate，支持多模态输入)
2. 用奖励函数计算每个回复的奖励 R_i
3. 组内归一化: A_i = (R_i - mean(R_group)) / std(R_group)
4. 计算当前策略和参考策略的 log_prob
5. Clipped Surrogate Loss (类似 PPO) + KL 散度惩罚
"""

import os
import re
import copy
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
from PIL import Image
from dataset import smart_resize


class GRPOTrainer:
    """
    DeepSeek-R1 论文中的 Group Relative Policy Optimization。
    """

    def __init__(
        self,
        model,
        tokenizer,
        reward_func,
        train_dataset,
        processor=None,
        output_dir="save/vlm_grpo",
        num_generations=4,
        max_completion_length=256,
        learning_rate=5e-7,
        num_epochs=1,
        mini_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        clip_epsilon=0.2,
        kl_coeff=0.01,
        bf16=True,
        logging_steps=1,
        eval_dataset=None,
        eval_steps=1000,
        eval_on_start=True,
        eval_batch_size=256,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.reward_func = reward_func
        self.processor = processor
        self.train_dataset = train_dataset
        self.output_dir = output_dir
        self.num_generations = num_generations
        self.max_completion_length = max_completion_length
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.mini_batch_size = mini_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.warmup_ratio = warmup_ratio
        self.max_grad_norm = max_grad_norm
        self.clip_epsilon = clip_epsilon
        self.kl_coeff = kl_coeff
        self.bf16 = bf16
        self.logging_steps = logging_steps
        self.eval_dataset = eval_dataset
        self.eval_steps = eval_steps
        self.eval_on_start = eval_on_start
        self.eval_batch_size = eval_batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs(output_dir, exist_ok=True)

        # 参考模型：冻结的 SFT 模型副本，用于计算 KL 散度
        self.ref_model = copy.deepcopy(model)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

        # 启用 gradient checkpointing 减少训练时中间激活的显存占用
        if hasattr(self.model, 'llm') and hasattr(self.model.llm, 'gradient_checkpointing_enable'):
            self.model.llm.gradient_checkpointing_enable()
            print("  [Memory] Enabled gradient checkpointing for LLM")

    def _load_image(self, image_path):
        """按需加载单张图片，返回 pixel_values, image_sizes, image_grid_thw。"""
        if not image_path or not os.path.exists(image_path):
            return None, None, None
        try:
            image = Image.open(image_path).convert("RGB")
            image = smart_resize(image, patch_size=14, compress_grid=2)
        except (OSError, IOError):
            return None, None, None
        width, height = image.size
        grid_t, grid_h, grid_w = 1, height // 28, width // 28
        pixel_values = self.processor.image_processor(
            images=image, do_resize=False, return_tensors="pt"
        )["pixel_values"].to(dtype=torch.bfloat16, device=self.device)
        image_sizes = torch.tensor([[height, width]], dtype=torch.long, device=self.device)
        image_grid_thw = torch.tensor([[grid_t, grid_h, grid_w]], dtype=torch.long, device=self.device)
        return pixel_values, image_sizes, image_grid_thw

    def _encode_prompt(self, prompt_messages):
        """将 prompt messages 编码为 input_ids。"""
        text = self.tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
        return input_ids.to(self.device)

    def _model_forward(self, model, **kwargs):
        """统一的 model forward 封装，自动处理 bf16 autocast。"""
        if self.bf16:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                return model(**kwargs)
        return model(**kwargs)

    # =========================================================================
    # 串行生成（保留备用）：每次生成 1 个回复，循环 G 次
    # 优点：显存占用低；缺点：速度慢（G 倍）
    # 如需使用，在 _grpo_step 中将 _generate_batch 调用替换为循环调用 _generate_one
    # =========================================================================
    @torch.no_grad()
    def _generate_one(self, model, prompt_ids, pixel_values=None, image_sizes=None, image_grid_thw=None):
        """
        串行版本：使用 KV Cache 的自回归生成，每次生成 1 个回复。
        返回: (completion_ids [1, comp_len], full_ids [1, prompt_len + comp_len])
        """
        eos_token_id = self.tokenizer.eos_token_id
        generated_ids = []

        # Prefill
        fwd = {"input_ids": prompt_ids, "use_cache": True}
        if pixel_values is not None:
            fwd["pixel_values"] = pixel_values
            fwd["image_sizes"] = image_sizes
            fwd["image_grid_thw"] = image_grid_thw
        outputs = self._model_forward(model, **fwd)
        next_logits = outputs.logits[:, -1, :]
        past_kv = outputs.past_key_values

        for _ in range(self.max_completion_length):
            next_token_id = self._sample_top_p(next_logits)
            if next_token_id.item() == eos_token_id:
                break
            generated_ids.append(next_token_id)
            outputs = self._model_forward(model, input_ids=next_token_id, past_key_values=past_kv, use_cache=True)
            next_logits = outputs.logits[:, -1, :]
            past_kv = outputs.past_key_values

        if generated_ids:
            completion_ids = torch.cat(generated_ids, dim=1)
        else:
            completion_ids = torch.zeros((1, 0), dtype=torch.long, device=self.device)
        full_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        return completion_ids, full_ids

    # =========================================================================
    # 并行生成（默认使用）：一次 Prefill 共享，Decode 阶段 batch_size=G 并行
    # 优点：速度快（约 G 倍加速）；缺点：显存占用高
    # =========================================================================
    @torch.no_grad()
    def _generate_batch(self, model, prompt_ids, num_generations, pixel_values=None, image_sizes=None, image_grid_thw=None):
        """
        G 个 rollout 共享一次 Prefill，Decode 阶段 batch 并行生成。

        返回:
            completions_ids: list of [1, comp_len_i] tensors, 长度为 G
            completions_full_ids: list of [1, prompt_len + comp_len_i] tensors, 长度为 G
            completions_text: list of str, 长度为 G
        """
        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else eos_token_id
        prompt_length = prompt_ids.shape[1]

        # === Prefill: 只做一次 ===
        fwd = {"input_ids": prompt_ids, "use_cache": True}
        if pixel_values is not None:
            fwd["pixel_values"] = pixel_values
            fwd["image_sizes"] = image_sizes
            fwd["image_grid_thw"] = image_grid_thw
        outputs = self._model_forward(model, **fwd)
        prefill_logits = outputs.logits[:, -1, :]  # [1, vocab]
        prefill_past = outputs.past_key_values

        # 在 batch 维度 repeat G 次
        batch_logits = prefill_logits.repeat(num_generations, 1)  # [G, vocab]
        # DynamicCache 的 batch repeat：使用内置方法在 batch 维度复制 KV cache
        prefill_past.batch_repeat_interleave(num_generations)
        batch_past = prefill_past

        # === Decode: batch_size=G 并行 ===
        all_generated = [[] for _ in range(num_generations)]
        finished = [False] * num_generations
        next_logits = batch_logits

        for _ in range(self.max_completion_length):
            # 对每个序列独立采样
            next_token_ids = self._sample_top_p(next_logits)  # [G, 1]

            # 检查 EOS
            for idx in range(num_generations):
                if finished[idx]:
                    continue
                if next_token_ids[idx].item() == eos_token_id:
                    finished[idx] = True
                else:
                    all_generated[idx].append(next_token_ids[idx:idx+1])

            if all(finished):
                break

            # 已结束的序列用 pad 填充，保持 batch 对齐
            for idx in range(num_generations):
                if finished[idx]:
                    next_token_ids[idx] = pad_token_id

            # Batch decode
            outputs = self._model_forward(model, input_ids=next_token_ids, past_key_values=batch_past, use_cache=True)
            next_logits = outputs.logits[:, -1, :]
            batch_past = outputs.past_key_values

        # 组装结果
        completions_ids = []
        completions_full_ids = []
        completions_text = []
        for idx in range(num_generations):
            if all_generated[idx]:
                comp_ids = torch.cat(all_generated[idx], dim=1)
            else:
                comp_ids = torch.zeros((1, 0), dtype=torch.long, device=self.device)
            full_ids = torch.cat([prompt_ids, comp_ids], dim=1)
            text = self.tokenizer.decode(comp_ids[0], skip_special_tokens=True)
            completions_ids.append(comp_ids)
            completions_full_ids.append(full_ids)
            completions_text.append(text)

        return completions_ids, completions_full_ids, completions_text

    def _sample_top_p(self, logits, temperature=0.7, top_p=0.9):
        """Top-p (nucleus) 采样，支持 batch。"""
        scaled = logits / temperature
        sorted_logits, sorted_indices = torch.sort(scaled, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
        sorted_logits[mask] = float("-inf")
        probs = F.softmax(sorted_logits, dim=-1)
        sampled_index = torch.multinomial(probs, num_samples=1)
        return sorted_indices.gather(1, sampled_index)

    def _precompute_vision_features(self, model, pixel_values, image_sizes):
        """预计算视觉特征（ViT + C-Abstractor + MLP），只编码一次图片。"""
        with torch.no_grad():
            B = pixel_values.shape[0]
            all_features = []
            for i in range(B):
                h, w = image_sizes[i]
                valid_pv = pixel_values[i:i+1, :, :h, :w]
                image_embeds, H_p, W_p = model.extract_dynamic_vision_features(valid_pv)
                new_h, new_w = H_p // model.compress_grid, W_p // model.compress_grid
                image_embeds = image_embeds.view(1, H_p, W_p, -1)
                image_embeds = image_embeds.view(1, new_h, model.compress_grid, new_w, model.compress_grid, -1)
                image_embeds = image_embeds.permute(0, 1, 3, 2, 4, 5).contiguous()
                image_embeds = image_embeds.view(1, new_h * new_w, -1)
                feats = model.linear2(F.silu(model.linear1(image_embeds)))
                all_features.append(feats.squeeze(0))
            return torch.cat(all_features, dim=0)  # [num_tokens, llm_dim]

    def _compute_log_probs(self, model, full_ids, prompt_length, pixel_values=None, image_sizes=None, image_grid_thw=None, vision_features=None):
        """计算 completion 部分每个 token 的 log probability（单条）。"""
        fwd = {"input_ids": full_ids}
        if vision_features is not None:
            fwd["vision_features"] = vision_features
        elif pixel_values is not None:
            fwd["pixel_values"] = pixel_values
            fwd["image_sizes"] = image_sizes
            fwd["image_grid_thw"] = image_grid_thw
        outputs = self._model_forward(model, **fwd)
        logits = outputs.logits
        completion_logits = logits[:, prompt_length - 1:-1, :]
        completion_targets = full_ids[:, prompt_length:]
        log_probs = F.log_softmax(completion_logits, dim=-1)
        token_log_probs = log_probs.gather(2, completion_targets.unsqueeze(-1)).squeeze(-1)
        return token_log_probs.squeeze(0)

    def _compute_log_probs_batch(self, model, completions_full_ids, prompt_length,
                                  pixel_values=None, image_sizes=None, image_grid_thw=None,
                                  vision_features=None):
        """
        批量计算 G 个回复的 log probability（一次 forward 完成）。
        支持传入预计算的 vision_features 避免重复 ViT 编码。

        Args:
            completions_full_ids: list of [1, seq_len_i] tensors（G 个回复，长度可能不同）
            prompt_length: prompt 的 token 长度
            vision_features: 预计算的视觉特征 [num_tokens, llm_dim]（优先使用）
            pixel_values, image_sizes, image_grid_thw: 原始视觉输入（仅在无 vision_features 时使用）

        Returns:
            token_log_probs: [G, max_comp_len] 的 per-token log prob
            completion_mask: [G, max_comp_len] 的 mask，标记有效 token
        """
        num_generations = len(completions_full_ids)
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id

        # 将 G 个不同长度的 full_ids 右 padding 到同一长度
        max_seq_len = max(ids.shape[1] for ids in completions_full_ids)
        padded_ids_list = []
        attention_masks = []
        for ids in completions_full_ids:
            pad_len = max_seq_len - ids.shape[1]
            padded = F.pad(ids, (0, pad_len), value=pad_token_id)
            mask = F.pad(torch.ones(1, ids.shape[1], dtype=torch.long, device=self.device), (0, pad_len), value=0)
            padded_ids_list.append(padded)
            attention_masks.append(mask)

        batch_ids = torch.cat(padded_ids_list, dim=0)  # [G, max_seq_len]
        batch_attention_mask = torch.cat(attention_masks, dim=0)  # [G, max_seq_len]

        fwd = {"input_ids": batch_ids, "attention_mask": batch_attention_mask}
        if vision_features is not None:
            # 将预计算的视觉特征 repeat G 次（只是 embedding 级别的 repeat，远小于 ViT 重新编码）
            fwd["vision_features"] = vision_features.repeat(num_generations, 1)
        elif pixel_values is not None:
            fwd["pixel_values"] = pixel_values.repeat(num_generations, 1, 1, 1)
            fwd["image_sizes"] = image_sizes.repeat(num_generations, 1)
            fwd["image_grid_thw"] = image_grid_thw.repeat(num_generations, 1)

        outputs = self._model_forward(model, **fwd)
        logits = outputs.logits  # [G, max_seq_len, vocab]

        # 提取 completion 部分的 log probs
        completion_logits = logits[:, prompt_length - 1:-1, :]  # [G, max_comp_len, vocab]
        completion_targets = batch_ids[:, prompt_length:]  # [G, max_comp_len]
        completion_mask = batch_attention_mask[:, prompt_length:]  # [G, max_comp_len]

        log_probs = F.log_softmax(completion_logits, dim=-1)
        token_log_probs = log_probs.gather(2, completion_targets.unsqueeze(-1)).squeeze(-1)  # [G, max_comp_len]

        return token_log_probs, completion_mask

    def _compute_rewards(self, completions_text, target=None):
        """调用奖励函数，返回每个 completion 的最终奖励。"""
        return self.reward_func(completions_text, target=target)

    def _grpo_step(self, prompt_messages, target=None, pixel_values=None, image_sizes=None, image_grid_thw=None):
        """
        执行一次 GRPO 更新步骤（显存优化版）:
        1. 并行生成 G 个回复（共享 Prefill，Decode batch 并行）
        2. 生成后立即清理 KV Cache 释放显存
        3. 预计算视觉特征（ViT 只编码一次）
        4. current_model 批量计算 log_probs（1 次 forward）
        5. ref_model 逐条计算 log_probs，避免 G 倍峰值显存
        6. 计算 clipped surrogate loss + KL penalty
        """
        prompt_ids = self._encode_prompt(prompt_messages)
        prompt_length = prompt_ids.shape[1]

        # Step 1: 并行生成 G 个回复
        self.model.eval()
        completions_ids, completions_full_ids, completions_text = self._generate_batch(
            self.model, prompt_ids, self.num_generations,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            image_grid_thw=image_grid_thw,
        )
        self.model.train()

        # Step 2: 生成完毕后立即清理显存（KV Cache 已不再需要）
        torch.cuda.empty_cache()

        # 过滤空回复
        valid_indices = [i for i in range(self.num_generations) if completions_ids[i].shape[1] > 0]
        if not valid_indices:
            return torch.tensor(0.0, device=self.device, requires_grad=True), 0.0, 0.0

        valid_full_ids = [completions_full_ids[i] for i in valid_indices]

        # Step 3: 计算奖励并进行组内归一化
        raw_rewards = self._compute_rewards(completions_text, target=target)
        rewards_tensor = torch.tensor(raw_rewards, dtype=torch.float32, device=self.device)
        reward_mean = rewards_tensor.mean()
        reward_std = rewards_tensor.std()
        if reward_std < 1e-8:
            advantages = torch.zeros_like(rewards_tensor)
        else:
            advantages = (rewards_tensor - reward_mean) / (reward_std + 1e-8)
        valid_advantages = advantages[valid_indices]

        # Step 4: 预计算视觉特征（ViT 只编码一次，后续 current + ref 共享）
        vision_feats = None
        if pixel_values is not None:
            vision_feats = self._precompute_vision_features(self.model, pixel_values, image_sizes)

        # Step 5: current_model 批量计算 log_probs（1 次 forward，需要梯度）
        current_token_log_probs, completion_mask = self._compute_log_probs_batch(
            self.model, valid_full_ids, prompt_length,
            vision_features=vision_feats,
        )

        # Step 6: ref_model 逐条计算 log_probs（避免 G 倍峰值显存，无梯度）
        # 预计算 ref_model 的视觉特征（ref_model 有自己的 linear1/linear2 权重）
        ref_vision_feats = None
        if pixel_values is not None:
            ref_vision_feats = self._precompute_vision_features(self.ref_model, pixel_values, image_sizes)

        with torch.no_grad():
            ref_log_probs_list = []
            for full_ids in valid_full_ids:
                token_lp = self._compute_log_probs(
                    self.ref_model, full_ids, prompt_length,
                    vision_features=ref_vision_feats,
                )
                ref_log_probs_list.append(token_lp)

            # 将逐条结果 pad 到同一长度，与 current 对齐
            max_comp_len = completion_mask.shape[1]
            padded_ref_list = []
            for lp in ref_log_probs_list:
                pad_len = max_comp_len - lp.shape[0]
                padded_ref_list.append(F.pad(lp, (0, pad_len), value=0.0))
            ref_token_log_probs = torch.stack(padded_ref_list, dim=0)  # [G_valid, max_comp_len]

        # Step 7: 计算 clipped surrogate loss + KL penalty
        log_ratio = current_token_log_probs - ref_token_log_probs  # [G_valid, max_comp_len]
        ratio = torch.exp(log_ratio)
        clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)

        advantages_expanded = valid_advantages.unsqueeze(1)  # [G_valid, 1]
        token_surrogate = -torch.min(ratio * advantages_expanded, clipped_ratio * advantages_expanded)
        token_kl = self.kl_coeff * (ratio - log_ratio - 1.0)

        masked_loss = (token_surrogate + token_kl) * completion_mask.float()
        total_valid_tokens = completion_mask.float().sum()
        if total_valid_tokens > 0:
            total_loss = masked_loss.sum() / total_valid_tokens
        else:
            total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        return total_loss, reward_mean.item(), reward_std.item()

    # =========================================================================
    # 多 Prompt 并行生成：M 个 prompt 各生成 G 个回复，总 batch = M*G
    # =========================================================================
    @torch.no_grad()
    def _generate_batch_multi_prompt(self, model, prompt_ids_list, num_generations,
                                      vision_features_list=None):
        """
        多 Prompt 并行 Rollout：M 个 prompt 各生成 G 个回复，Decode 阶段 batch_size = M*G。

        Prefill 阶段逐个处理（各自独立编码），然后将 M*G 个序列的 KV Cache
        拼接成一个大 batch 做并行 Decode。

        Args:
            model: 用于生成的模型
            prompt_ids_list: list of [1, prompt_len_i] tensors，M 个 prompt
            num_generations: 每个 prompt 生成的回复数 G
            vision_features_list: list of vision_features tensors（或 None），M 个

        Returns:
            per_prompt_results: list of M dicts, each containing:
                - completions_ids: list of G [1, comp_len] tensors
                - completions_full_ids: list of G [1, full_len] tensors
                - completions_text: list of G strings
                - prompt_length: int
        """
        from transformers.cache_utils import DynamicCache

        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else eos_token_id
        num_prompts = len(prompt_ids_list)
        total_seqs = num_prompts * num_generations

        # === Prefill: 逐个 prompt 做 Prefill，收集 KV Cache 和首 token logits ===
        all_prefill_logits = []
        all_prefill_pasts = []
        prompt_lengths = []

        for prompt_idx in range(num_prompts):
            prompt_ids = prompt_ids_list[prompt_idx]
            prompt_lengths.append(prompt_ids.shape[1])

            fwd = {"input_ids": prompt_ids, "use_cache": True}
            if vision_features_list is not None and vision_features_list[prompt_idx] is not None:
                fwd["vision_features"] = vision_features_list[prompt_idx]
            outputs = self._model_forward(model, **fwd)
            prefill_logits = outputs.logits[:, -1, :]  # [1, vocab]
            prefill_past = outputs.past_key_values

            # repeat G 次
            batch_logits = prefill_logits.repeat(num_generations, 1)  # [G, vocab]
            prefill_past.batch_repeat_interleave(num_generations)

            all_prefill_logits.append(batch_logits)
            all_prefill_pasts.append(prefill_past)

        # === 合并 M 个 prompt 的 KV Cache 为一个大 batch (M*G) ===
        max_prompt_len = max(prompt_lengths)
        num_layers = len(all_prefill_pasts[0])
        merged_past_keys = []
        merged_past_values = []

        for layer_idx in range(num_layers):
            layer_keys = []
            layer_values = []
            for prompt_idx in range(num_prompts):
                past = all_prefill_pasts[prompt_idx]
                key = past.layers[layer_idx].keys      # [G, num_heads, seq_len_i, head_dim]
                value = past.layers[layer_idx].values  # [G, num_heads, seq_len_i, head_dim]
                seq_len = key.shape[2]
                pad_len = max_prompt_len - seq_len
                if pad_len > 0:
                    key = F.pad(key, (0, 0, pad_len, 0), value=0.0)    # 左 padding
                    value = F.pad(value, (0, 0, pad_len, 0), value=0.0)
                layer_keys.append(key)
                layer_values.append(value)
            merged_past_keys.append(torch.cat(layer_keys, dim=0))
            merged_past_values.append(torch.cat(layer_values, dim=0))

        batch_past = DynamicCache()
        for layer_idx in range(num_layers):
            batch_past.update(merged_past_keys[layer_idx], merged_past_values[layer_idx], layer_idx)

        del all_prefill_pasts, merged_past_keys, merged_past_values

        # 合并首 token logits
        batch_logits = torch.cat(all_prefill_logits, dim=0)  # [M*G, vocab]

        # 构建 attention_mask：左 padding 的位置为 0，其余为 1
        batch_attention_mask = torch.zeros(total_seqs, max_prompt_len, dtype=torch.long, device=self.device)
        for prompt_idx in range(num_prompts):
            seq_len = prompt_lengths[prompt_idx]
            pad_len = max_prompt_len - seq_len
            start = prompt_idx * num_generations
            end = start + num_generations
            batch_attention_mask[start:end, pad_len:] = 1

        # === Decode: batch_size = M*G 并行 ===
        all_generated = [[] for _ in range(total_seqs)]
        finished = [False] * total_seqs
        next_logits = batch_logits

        for _ in range(self.max_completion_length):
            next_token_ids = self._sample_top_p(next_logits)  # [M*G, 1]

            for idx in range(total_seqs):
                if finished[idx]:
                    continue
                if next_token_ids[idx].item() == eos_token_id:
                    finished[idx] = True
                else:
                    all_generated[idx].append(next_token_ids[idx:idx+1])

            if all(finished):
                break

            for idx in range(total_seqs):
                if finished[idx]:
                    next_token_ids[idx] = pad_token_id

            new_col = torch.ones(total_seqs, 1, dtype=torch.long, device=self.device)
            batch_attention_mask = torch.cat([batch_attention_mask, new_col], dim=1)

            outputs = self._model_forward(
                model, input_ids=next_token_ids,
                attention_mask=batch_attention_mask,
                past_key_values=batch_past, use_cache=True,
            )
            next_logits = outputs.logits[:, -1, :]
            batch_past = outputs.past_key_values

        # === 组装结果：按 prompt 分组 ===
        per_prompt_results = []
        for prompt_idx in range(num_prompts):
            prompt_ids = prompt_ids_list[prompt_idx]
            completions_ids = []
            completions_full_ids = []
            completions_text = []
            for g in range(num_generations):
                seq_idx = prompt_idx * num_generations + g
                if all_generated[seq_idx]:
                    comp_ids = torch.cat(all_generated[seq_idx], dim=1)
                else:
                    comp_ids = torch.zeros((1, 0), dtype=torch.long, device=self.device)
                full_ids = torch.cat([prompt_ids, comp_ids], dim=1)
                text = self.tokenizer.decode(comp_ids[0], skip_special_tokens=True)
                completions_ids.append(comp_ids)
                completions_full_ids.append(full_ids)
                completions_text.append(text)
            per_prompt_results.append({
                "completions_ids": completions_ids,
                "completions_full_ids": completions_full_ids,
                "completions_text": completions_text,
                "prompt_length": prompt_lengths[prompt_idx],
            })

        return per_prompt_results

    def _grpo_step_batch(self, mini_batch_samples):
        """
        批量 GRPO 步骤：一次处理 mini_batch 内所有 prompt，M*G 个序列并行生成。

        Args:
            mini_batch_samples: list of sample dicts (M 个)

        Returns:
            total_loss: 整个 mini_batch 的平均 loss（带梯度）
            avg_reward: 平均奖励
            valid_count: 有效 prompt 数量
        """
        # Step 1: 预处理所有 prompt
        prompt_ids_list = []
        targets = []
        vision_features_list = []
        pixel_values_list = []
        image_sizes_list = []
        image_grid_thw_list = []

        for sample in mini_batch_samples:
            prompt_messages = sample["prompt"]
            image_path = sample.get("image_path", "")
            pixel_values, image_sizes, image_grid_thw = self._load_image(image_path)
            if image_path and pixel_values is None:
                continue

            prompt_ids = self._encode_prompt(prompt_messages)
            prompt_ids_list.append(prompt_ids)
            targets.append(sample.get("target"))

            if pixel_values is not None:
                vision_feats = self._precompute_vision_features(self.model, pixel_values, image_sizes)
                vision_features_list.append(vision_feats)
                pixel_values_list.append(pixel_values)
                image_sizes_list.append(image_sizes)
                image_grid_thw_list.append(image_grid_thw)
            else:
                vision_features_list.append(None)
                pixel_values_list.append(None)
                image_sizes_list.append(None)
                image_grid_thw_list.append(None)

        num_valid = len(prompt_ids_list)
        if num_valid == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True), 0.0, 0

        # Step 2: M*G 并行生成（所有 prompt 的 G 个回复在 Decode 阶段并行）
        self.model.eval()
        per_prompt_results = self._generate_batch_multi_prompt(
            self.model, prompt_ids_list, self.num_generations,
            vision_features_list=vision_features_list,
        )
        self.model.train()
        torch.cuda.empty_cache()

        # Step 3: 逐 prompt 计算 GRPO loss
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=False)
        total_reward_sum = 0.0
        actual_valid = 0

        for prompt_idx in range(num_valid):
            result = per_prompt_results[prompt_idx]
            completions_ids = result["completions_ids"]
            completions_full_ids = result["completions_full_ids"]
            completions_text = result["completions_text"]
            prompt_length = result["prompt_length"]
            target = targets[prompt_idx]

            valid_gen_indices = [i for i in range(self.num_generations) if completions_ids[i].shape[1] > 0]
            if not valid_gen_indices:
                continue

            valid_full_ids = [completions_full_ids[i] for i in valid_gen_indices]

            raw_rewards = self._compute_rewards(completions_text, target=target)
            rewards_tensor = torch.tensor(raw_rewards, dtype=torch.float32, device=self.device)
            reward_mean = rewards_tensor.mean()
            reward_std = rewards_tensor.std()
            if reward_std < 1e-8:
                advantages = torch.zeros_like(rewards_tensor)
            else:
                advantages = (rewards_tensor - reward_mean) / (reward_std + 1e-8)
            valid_advantages = advantages[valid_gen_indices]

            vision_feats = vision_features_list[prompt_idx]

            current_token_log_probs, completion_mask = self._compute_log_probs_batch(
                self.model, valid_full_ids, prompt_length,
                vision_features=vision_feats,
            )

            ref_vision_feats = None
            if pixel_values_list[prompt_idx] is not None:
                ref_vision_feats = self._precompute_vision_features(
                    self.ref_model, pixel_values_list[prompt_idx], image_sizes_list[prompt_idx]
                )

            with torch.no_grad():
                ref_log_probs_list = []
                for full_ids in valid_full_ids:
                    token_lp = self._compute_log_probs(
                        self.ref_model, full_ids, prompt_length,
                        vision_features=ref_vision_feats,
                    )
                    ref_log_probs_list.append(token_lp)

                max_comp_len = completion_mask.shape[1]
                padded_ref_list = []
                for lp in ref_log_probs_list:
                    pad_len = max_comp_len - lp.shape[0]
                    padded_ref_list.append(F.pad(lp, (0, pad_len), value=0.0))
                ref_token_log_probs = torch.stack(padded_ref_list, dim=0)

            log_ratio = current_token_log_probs - ref_token_log_probs
            ratio = torch.exp(log_ratio)
            clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)

            advantages_expanded = valid_advantages.unsqueeze(1)
            token_surrogate = -torch.min(ratio * advantages_expanded, clipped_ratio * advantages_expanded)
            token_kl = self.kl_coeff * (ratio - log_ratio - 1.0)

            masked_loss = (token_surrogate + token_kl) * completion_mask.float()
            total_valid_tokens = completion_mask.float().sum()
            if total_valid_tokens > 0:
                prompt_loss = masked_loss.sum() / total_valid_tokens
            else:
                prompt_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

            total_loss = total_loss + prompt_loss
            total_reward_sum += reward_mean.item()
            actual_valid += 1

        if actual_valid == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True), 0.0, 0

        avg_reward = total_reward_sum / actual_valid
        return total_loss, avg_reward, actual_valid

    def _build_optimizer(self):
        return AdamW([p for p in self.model.parameters() if p.requires_grad], lr=self.learning_rate, weight_decay=0.01)

    def _build_scheduler(self, optimizer, total_steps):
        warmup_steps = int(total_steps * self.warmup_ratio)
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def save_checkpoint(self, step=None):
        if step is not None:
            save_path = os.path.join(self.output_dir, f"checkpoint-step-{step}")
        else:
            save_path = self.output_dir
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"  [Checkpoint] Saved to {save_path}")

    def train(self):
        """
        主训练循环
        mini_batch_size 个 prompt 的 loss 累加后一次 backward，
        每 gradient_accumulation_steps 个 mini-batch 做一次 optimizer.step()。
        """
        self.model.to(self.device)
        self.ref_model.to(self.device)

        total_prompts = len(self.train_dataset)
        effective_batch_size = self.mini_batch_size * self.gradient_accumulation_steps
        total_steps = (total_prompts * self.num_epochs) // effective_batch_size
        optimizer = self._build_optimizer()
        scheduler = self._build_scheduler(optimizer, max(total_steps, 1))

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  GRPO Training Config:")
        print(f"    Trainable params: {trainable_params:,}")
        print(f"    Group size (G): {self.num_generations}")
        print(f"    Mini-batch size: {self.mini_batch_size}")
        print(f"    Gradient accumulation steps: {self.gradient_accumulation_steps}")
        print(f"    Effective batch size: {effective_batch_size}")
        print(f"    Max completion length: {self.max_completion_length}")
        print(f"    Clip epsilon: {self.clip_epsilon}")
        print(f"    KL coefficient: {self.kl_coeff}")
        print(f"    Total prompts: {total_prompts}")
        print(f"    Total optimization steps: {total_steps}")

        global_step = 0
        accumulated_loss = 0.0
        accumulated_reward = 0.0
        mini_batch_count = 0
        start_time = time.time()

        if self.eval_dataset and self.eval_on_start:
            print("=== Baseline evaluation before training ===")
            self.evaluate()

        for epoch in range(self.num_epochs):
            num_mini_batches = (total_prompts + self.mini_batch_size - 1) // self.mini_batch_size
            progress_bar = tqdm(
                total=total_steps,
                initial=global_step,
                desc=f"Epoch {epoch + 1}/{self.num_epochs}", unit="step",
            )
            for mb_idx in range(num_mini_batches):
                # 取出当前 mini-batch 的样本
                start_idx = mb_idx * self.mini_batch_size
                end_idx = min(start_idx + self.mini_batch_size, total_prompts)
                mini_batch_samples = self.train_dataset[start_idx:end_idx]

                # 对 mini-batch 内所有 prompt 并行执行 GRPO（M*G 个序列同时生成）
                try:
                    mb_loss, mb_avg_reward, valid_count = self._grpo_step_batch(mini_batch_samples)
                except torch.cuda.OutOfMemoryError:
                    print(f"\n  [OOM] CUDA out of memory at mini-batch {mb_idx}, skipping...")
                    torch.cuda.empty_cache()
                    optimizer.zero_grad()
                    continue

                if valid_count == 0:
                    continue

                # 对整个 mini-batch 的累积 loss 做一次 backward
                scaling_factor = self.gradient_accumulation_steps * valid_count
                scaled_loss = mb_loss / scaling_factor
                try:
                    scaled_loss.backward()
                except torch.cuda.OutOfMemoryError:
                    print(f"\n  [OOM] CUDA out of memory during backward at mini-batch {mb_idx}, skipping...")
                    torch.cuda.empty_cache()
                    optimizer.zero_grad()
                    continue

                accumulated_loss += mb_loss.item() / valid_count
                accumulated_reward += mb_avg_reward
                mini_batch_count += 1

                # 每个 mini-batch 都更新进度条信息（不推进进度）
                accum_count = ((mini_batch_count - 1) % self.gradient_accumulation_steps) + 1
                progress_bar.set_postfix(
                    loss=f"{accumulated_loss / accum_count:.4f}",
                    reward=f"{accumulated_reward / accum_count:.3f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                    accum=f"{accum_count}/{self.gradient_accumulation_steps}",
                )

                # 每 gradient_accumulation_steps 个 mini-batch 做一次优化步骤
                if mini_batch_count % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    progress_bar.update(1)

                    accumulated_loss = 0.0
                    accumulated_reward = 0.0

                    if self.eval_dataset and global_step % self.eval_steps == 0:
                        self.evaluate()
                        
                    # ckpt 保存是 eval 频率 1/10
                    if self.eval_dataset and global_step % (10 * self.eval_steps) == 0:
                        self.save_checkpoint(step=global_step)

            progress_bar.close()

            # epoch 结束时处理剩余未更新的梯度
            if mini_batch_count % self.gradient_accumulation_steps != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

        self.save_checkpoint()
        total_time = time.time() - start_time
        print(f"\n  GRPO Training complete! Total time: {total_time:.1f}s")

    @torch.no_grad()
    def evaluate(self):
        """在验证集上进行 batch 贪婪解码评测，计算准确率。"""
        if not self.eval_dataset:
            return
        self.model.eval()
        boxed_pattern = re.compile(r"\\boxed\{([^}]*)\}")
        correct_count = 0
        total_count = 0
        batch_size = self.eval_batch_size

        print(f"\n  [Eval] Running validation (batch_size={batch_size})...")
        num_batches = (len(self.eval_dataset) + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(num_batches), desc="Evaluating", unit="batch"):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(self.eval_dataset))
            batch_samples = self.eval_dataset[start:end]

            prompt_ids_list = []
            vision_features_list = []
            targets = []
            valid_indices = []

            for idx, sample in enumerate(batch_samples):
                prompt_messages = sample["prompt"]
                target = sample.get("target", "")
                image_path = sample.get("image_path", "")

                pixel_values, image_sizes, image_grid_thw = self._load_image(image_path)
                if image_path and pixel_values is None:
                    continue

                prompt_ids = self._encode_prompt(prompt_messages)
                prompt_ids_list.append(prompt_ids)
                targets.append(target)
                valid_indices.append(idx)

                if pixel_values is not None:
                    vision_feats = self._precompute_vision_features(self.model, pixel_values, image_sizes)
                    vision_features_list.append(vision_feats)
                else:
                    vision_features_list.append(None)

            if not prompt_ids_list:
                continue

            completions_text = self._greedy_generate_batch(
                prompt_ids_list, vision_features_list,
            )

            for text, target in zip(completions_text, targets):
                match = boxed_pattern.search(text)
                predicted_answer = match.group(1).strip() if match else ""
                if predicted_answer == target.strip():
                    correct_count += 1
                total_count += 1

        accuracy = correct_count / total_count if total_count > 0 else 0.0
        print(f"  [Eval] Accuracy: {correct_count}/{total_count} = {accuracy:.4f}")
        self.model.train()
        return accuracy

    @torch.no_grad()
    def _greedy_generate_batch(self, prompt_ids_list, vision_features_list):
        """
        Batch 贪婪解码：多个 prompt 共享 Decode 阶段并行生成。

        Args:
            prompt_ids_list: list of [1, prompt_len_i] tensors
            vision_features_list: list of vision_features (or None)

        Returns:
            completions_text: list of str, 每个 prompt 的生成文本
        """
        from transformers.cache_utils import DynamicCache

        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else eos_token_id
        num_prompts = len(prompt_ids_list)

        # Prefill: 逐个 prompt 编码
        all_prefill_logits = []
        all_prefill_pasts = []
        prompt_lengths = []

        for prompt_idx in range(num_prompts):
            prompt_ids = prompt_ids_list[prompt_idx]
            prompt_lengths.append(prompt_ids.shape[1])

            fwd = {"input_ids": prompt_ids, "use_cache": True}
            if vision_features_list[prompt_idx] is not None:
                fwd["vision_features"] = vision_features_list[prompt_idx]
            outputs = self._model_forward(self.model, **fwd)
            all_prefill_logits.append(outputs.logits[:, -1, :])
            all_prefill_pasts.append(outputs.past_key_values)

        # 合并 KV Cache
        max_prompt_len = max(prompt_lengths)
        num_layers = len(all_prefill_pasts[0])
        merged_past_keys = []
        merged_past_values = []

        for layer_idx in range(num_layers):
            layer_keys = []
            layer_values = []
            for prompt_idx in range(num_prompts):
                past = all_prefill_pasts[prompt_idx]
                key = past.layers[layer_idx].keys
                value = past.layers[layer_idx].values
                seq_len = key.shape[2]
                pad_len = max_prompt_len - seq_len
                if pad_len > 0:
                    key = F.pad(key, (0, 0, pad_len, 0), value=0.0)
                    value = F.pad(value, (0, 0, pad_len, 0), value=0.0)
                layer_keys.append(key)
                layer_values.append(value)
            merged_past_keys.append(torch.cat(layer_keys, dim=0))
            merged_past_values.append(torch.cat(layer_values, dim=0))

        batch_past = DynamicCache()
        for layer_idx in range(num_layers):
            batch_past.update(merged_past_keys[layer_idx], merged_past_values[layer_idx], layer_idx)

        del all_prefill_pasts, merged_past_keys, merged_past_values

        batch_logits = torch.cat(all_prefill_logits, dim=0)

        # Attention mask
        batch_attention_mask = torch.zeros(num_prompts, max_prompt_len, dtype=torch.long, device=self.device)
        for prompt_idx in range(num_prompts):
            pad_len = max_prompt_len - prompt_lengths[prompt_idx]
            batch_attention_mask[prompt_idx, pad_len:] = 1

        # Decode: batch 并行贪婪解码
        all_generated = [[] for _ in range(num_prompts)]
        finished = [False] * num_prompts
        next_logits = batch_logits

        for _ in range(self.max_completion_length):
            next_token_ids = next_logits.argmax(dim=-1, keepdim=True)

            for idx in range(num_prompts):
                if finished[idx]:
                    continue
                if next_token_ids[idx].item() == eos_token_id:
                    finished[idx] = True
                else:
                    all_generated[idx].append(next_token_ids[idx:idx+1])

            if all(finished):
                break

            for idx in range(num_prompts):
                if finished[idx]:
                    next_token_ids[idx] = pad_token_id

            new_col = torch.ones(num_prompts, 1, dtype=torch.long, device=self.device)
            batch_attention_mask = torch.cat([batch_attention_mask, new_col], dim=1)

            outputs = self._model_forward(
                self.model, input_ids=next_token_ids,
                attention_mask=batch_attention_mask,
                past_key_values=batch_past, use_cache=True,
            )
            next_logits = outputs.logits[:, -1, :]
            batch_past = outputs.past_key_values

        # 组装结果
        completions_text = []
        for idx in range(num_prompts):
            if all_generated[idx]:
                comp_ids = torch.cat(all_generated[idx], dim=1)
                text = self.tokenizer.decode(comp_ids[0], skip_special_tokens=True)
            else:
                text = ""
            completions_text.append(text)

        return completions_text
