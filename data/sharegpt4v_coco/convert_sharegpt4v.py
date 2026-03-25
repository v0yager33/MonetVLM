"""
将 ShareGPT4V-COCO 数据集转换为 SparkVLM VLMDataset 所需的 conversations 格式。
原始格式: {"id": "...", "image": "coco/train2017/xxx.jpg", "caption": "..."}
目标格式: {"image_path": "/abs/path/to/img.jpg", "conversations": [{"from": "human", "value": "...<image>"}, {"from": "assistant", "value": "..."}]}
"""

import json
import os
import random

SOURCE_JSONL = "/chatgpt_nas/dukaixuan.dkx/datasets/ShareGPT4V-COCO/ShareGPT4V-COCO.jsonl"
IMAGE_ROOT = "/chatgpt_nas/dukaixuan.dkx/datasets/ShareGPT4V-COCO"
OUTPUT_DIR = "/chatgpt_nas/dukaixuan.dkx/reproduceVLM/data"

# 多样化的图像描述 query 模板，避免模型只学会一种提问方式
QUERY_TEMPLATES = [
    "Describe this image in detail.\n<image>",
    "What do you see in this image?\n<image>",
    "Please provide a detailed description of the image.\n<image>",
    "Can you describe what's happening in this picture?\n<image>",
    "Tell me about this image.\n<image>",
    "What is shown in this photograph?\n<image>",
    "Explain the contents of this image.\n<image>",
    "Give a comprehensive description of this image.\n<image>",
    "What can you observe in this picture?\n<image>",
    "Describe the scene depicted in this image.\n<image>",
    "What's going on in this image?\n<image>",
    "Look at this image and describe it.\n<image>",
    "Please describe what you see.\n<image>",
    "Provide a thorough description of this photo.\n<image>",
    "Analyze and describe this image.\n<image>",
]


def convert():
    random.seed(42)

    converted = []
    skipped = 0

    with open(SOURCE_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line.strip())

            image_rel_path = item.get("image", "")
            image_abs_path = os.path.join(IMAGE_ROOT, image_rel_path)

            if not os.path.exists(image_abs_path):
                skipped += 1
                continue

            caption = item.get("caption", "").strip()
            if not caption:
                skipped += 1
                continue

            query = random.choice(QUERY_TEMPLATES)

            converted.append({
                "image_path": image_abs_path,
                "conversations": [
                    {"from": "human", "value": query},
                    {"from": "assistant", "value": caption},
                ],
            })

    # 打乱顺序
    random.shuffle(converted)

    # 按 9:1 划分训练集和验证集
    split_idx = int(len(converted) * 0.9)
    train_data = converted[:split_idx]
    val_data = converted[split_idx:]

    train_path = os.path.join(OUTPUT_DIR, "sharegpt4v_coco_train.jsonl")
    val_path = os.path.join(OUTPUT_DIR, "sharegpt4v_coco_val.jsonl")

    with open(train_path, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Conversion complete!")
    print(f"  Total valid samples: {len(converted)}")
    print(f"  Skipped (missing image or caption): {skipped}")
    print(f"  Train set: {len(train_data)} -> {train_path}")
    print(f"  Val set:   {len(val_data)} -> {val_path}")

    # 打印一条样例
    if train_data:
        print(f"\nSample:")
        print(json.dumps(train_data[0], indent=2, ensure_ascii=False)[:500])


if __name__ == "__main__":
    convert()
