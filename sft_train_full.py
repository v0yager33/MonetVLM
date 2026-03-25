import argparse
import os
import torch
from transformers import AutoProcessor
from vlm_model import SparkVLM, SparkVLMConfig
from dataset import VLMDataset, VLMDataCollator
from trainers.sft_trainer import SFTTrainer

def parse_args():
    parser = argparse.ArgumentParser(description="SFT SparkVLM (Full Model)")
    parser.add_argument("--pretrained_model_path", type=str, required=True, help="Path to the phase 1 output dir")
    parser.add_argument("--jsonl_path", type=str, nargs='+', required=True, help="Path(s) to the SFT dataset file(s)")
    parser.add_argument("--output_dir", type=str, default="save/vlm_sft_full")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Default LR for all modules")
    parser.add_argument("--vit_lr", type=float, default=None, help="LR for ViT, overrides learning_rate")
    parser.add_argument("--adapter_lr", type=float, default=None, help="LR for adapter (linear1+linear2), overrides learning_rate")
    parser.add_argument("--llm_lr", type=float, default=None, help="LR for LLM, overrides learning_rate")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--val_jsonl_path", type=str, nargs='+', default=None, help="Path(s) to the validation dataset file(s), each evaluated separately")
    parser.add_argument("--eval_steps", type=int, default=1000, help="Run eval every N optimization steps")
    return parser.parse_args()

def main():
    args = parse_args()

    print(f"Loading pretrained config and weights from {args.pretrained_model_path}...")
    config = SparkVLMConfig.from_pretrained(args.pretrained_model_path)
    model = SparkVLM.from_pretrained(args.pretrained_model_path, config=config, torch_dtype=torch.bfloat16)

    # 解冻全部参数（ViT + Adapter + LLM）进行全参微调
    print("Unfreezing all parameters for full model SFT...")
    for param in model.parameters():
        param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters())
    print(f"All {trainable_params:,} parameters are trainable.")

    print("Loading Processor & Dataset...")
    processor = AutoProcessor.from_pretrained(config.vision_model_path)
    tokenizer = model.tokenizer

    train_dataset = VLMDataset(args.jsonl_path, processor, tokenizer, config)
    data_collator = VLMDataCollator(tokenizer)

    eval_datasets = {}
    if args.val_jsonl_path:
        for val_path in args.val_jsonl_path:
            eval_name = os.path.splitext(os.path.basename(val_path))[0]
            eval_datasets[eval_name] = VLMDataset(val_path, processor, tokenizer, config)

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        data_collator=data_collator,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        vit_lr=args.vit_lr,
        adapter_lr=args.adapter_lr,
        llm_lr=args.llm_lr,
        num_epochs=args.num_epochs,
        per_device_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        bf16=True,
        logging_steps=5,
        eval_datasets=eval_datasets,
        eval_steps=args.eval_steps,
    )

    print("Starting SFT Training (Full Model) ...")
    trainer.train()

if __name__ == "__main__":
    main()
