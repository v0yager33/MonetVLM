import argparse
import torch
from transformers import AutoProcessor, AutoTokenizer, Trainer, TrainingArguments
from vlm_model import SparkVLM, SparkVLMConfig
from dataset import VLMDataset, VLMDataCollator

def parse_args():
    parser = argparse.ArgumentParser(description="Phase 1: Pretrain SparkVLM Adapter")
    parser.add_argument("--vision_model_path", type=str, default="/chatgpt_nas/dukaixuan.dkx/models/siglip2-so400m-patch14-384")
    parser.add_argument("--llm_model_path", type=str, default="/chatgpt_nas/dukaixuan.dkx/models/Qwen3-1.7B")
    parser.add_argument("--jsonl_path", type=str, nargs='+', required=True, help="Path(s) to the pretrain dataset file(s)")
    parser.add_argument("--output_dir", type=str, default="save/vlm_pretrain_adapter")
    return parser.parse_args()

def main():
    args = parse_args()
    
    config = SparkVLMConfig(
        vision_model_path=args.vision_model_path, 
        llm_model_path=args.llm_model_path,
        freeze_vision_model=True
    )
    
    print("Initialize VLM Model (bfloat16)...")
    model = SparkVLM(config)
    
    # 阶段一：冻结 LLM 和 Vision，仅训练 Adapter
    print("Freezing LLM parameters. Only Adapter will be trained.")
    for param in model.llm.parameters():
        param.requires_grad = False
    for param in model.vision_model.parameters():
        param.requires_grad = False
        
    print("Loading Processor & Dataset...")
    processor = AutoProcessor.from_pretrained(config.vision_model_path)
    tokenizer = model.tokenizer
    
    train_dataset = VLMDataset(args.jsonl_path, processor, tokenizer, config)
    data_collator = VLMDataCollator(tokenizer)
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        do_train=True,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=1e-3, # Adapter 预训练学习率通常较高
        num_train_epochs=1,
        save_strategy="epoch",
        bf16=True, 
        logging_steps=10,
        dataloader_num_workers=2,
        remove_unused_columns=False, 
        report_to="tensorboard",
        logging_dir=f"{args.output_dir}/runs"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    print("Starting Phase 1 Training (Adapter Only)...")
    trainer.train()
    
    print(f"Training complete! Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    
if __name__ == "__main__":
    main()
