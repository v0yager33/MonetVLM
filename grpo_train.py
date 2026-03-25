"""
Phase 3: GRPO (Group Relative Policy Optimization) 强化学习对齐训练
使用自定义纯 PyTorch GRPO Trainer，实现 DeepSeek-R1 的组内相对策略优化。
"""

import argparse
import json
import os
import torch
from transformers import AutoProcessor, AutoTokenizer
from vlm_model import SparkVLM, SparkVLMConfig
from trainers.grpo_trainer import GRPOTrainer
from reward_functions import compute_reward


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 3: GRPO Alignment Training (DeepSeek-R1 Style)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the SFT model dir (Phase 2 output)")
    parser.add_argument("--dataset_path", type=str, default=None, help="Path to GRPO dataset (jsonl)")
    parser.add_argument("--output_dir", type=str, default="save/vlm_grpo")
    parser.add_argument("--num_generations", type=int, default=4, help="Group size G")
    parser.add_argument("--max_completion_length", type=int, default=2048)
    parser.add_argument("--learning_rate", type=float, default=5e-7)
    parser.add_argument("--clip_epsilon", type=float, default=0.2)
    parser.add_argument("--kl_coeff", type=float, default=0.01)
    parser.add_argument("--mini_batch_size", type=int, default=1, help="Number of prompts per mini-batch")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--val_dataset_path", type=str, default=None, help="Path to validation dataset (jsonl)")
    parser.add_argument("--eval_steps", type=int, default=1000, help="Run validation every N optimization steps")
    parser.add_argument("--no_eval_on_start", action="store_true", help="Disable baseline evaluation before training")
    parser.add_argument("--eval_batch_size", type=int, default=4, help="Batch size for validation inference")
    return parser.parse_args()

def load_grpo_dataset(dataset_path):
    """加载 GRPO 数据集 (JSONL 格式)。"""
    dataset = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line.strip()))
    return dataset

def preprocess_grpo_dataset(raw_dataset):
    """懒加载模式：只存元数据，图片在训练时由 GRPOTrainer 按需加载。"""
    processed = []
    for record in raw_dataset:
        sample = {
            "prompt": [
                {"role": "system", "content": "You are a helpful multimodal assistant."},
                {"role": "user", "content": record.get("query", "")},
            ],
            "target": record.get("answer", ""),
            "image_path": record.get("image_path", ""),
        }
        processed.append(sample)
    return processed


def main():
    args = parse_args()

    print(f"Loading SFT model from {args.model_path}...")
    config = SparkVLMConfig.from_pretrained(args.model_path)
    model = SparkVLM.from_pretrained(args.model_path, config=config, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    processor = AutoProcessor.from_pretrained(config.vision_model_path)

    # GRPO 阶段全量微调：解冻 ViT（SFT 阶段冻结的）
    for param in model.vision_model.parameters():
        param.requires_grad = True
    print(f"  Unfroze vision model for full fine-tuning")

    # 加载数据集
    if args.dataset_path:
        print(f"Loading GRPO dataset from {args.dataset_path}...")
        raw_dataset = load_grpo_dataset(args.dataset_path)
    else:
        print("No dataset provided, please specify --dataset_path")
        return

    print(f"Preprocessing {len(raw_dataset)} train samples...")
    train_dataset = preprocess_grpo_dataset(raw_dataset)
    print(f"  Preprocessed {len(train_dataset)} train samples")

    # 加载验证集（可选）
    eval_dataset = None
    if args.val_dataset_path:
        print(f"Loading validation dataset from {args.val_dataset_path}...")
        raw_val = load_grpo_dataset(args.val_dataset_path)
        print(f"Preprocessing {len(raw_val)} val samples...")
        eval_dataset = preprocess_grpo_dataset(raw_val)
        print(f"  Preprocessed {len(eval_dataset)} val samples")

    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_func=compute_reward,
        train_dataset=train_dataset,
        processor=processor,
        output_dir=args.output_dir,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        learning_rate=args.learning_rate,
        mini_batch_size=args.mini_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_epochs=args.num_epochs,
        clip_epsilon=args.clip_epsilon,
        kl_coeff=args.kl_coeff,
        bf16=True,
        eval_dataset=eval_dataset,
        eval_steps=args.eval_steps,
        eval_on_start=not args.no_eval_on_start,
        eval_batch_size=args.eval_batch_size,
    )

    print(f"Starting GRPO Training (Group Size G={args.num_generations})...")
    trainer.train()


if __name__ == "__main__":
    main()
