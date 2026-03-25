"""
将 SFT 数据集的 conversations 字段按 image_path 合并到 GRPO 数据集中。
合并后的 GRPO 数据集同时包含 SFT 和 GRPO 所需的字段，无需维护独立的 SFT 数据集。

注意：GRPO 数据集中的图片可能是 SFT 数据集的子集，只合并匹配到的。
"""

import json
import sys
import os

SFT_PATH = "wikiart_artist_sft.jsonl"
GRPO_TRAIN_PATH = "wikiart_artist_grpo_train.jsonl"
GRPO_TEST_PATH = "wikiart_artist_grpo_test.jsonl"


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))
    return records


def save_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_sft_lookup(sft_records):
    """按 image_path 建立 SFT conversations 的查找表。"""
    lookup = {}
    for record in sft_records:
        image_path = record.get("image_path", "")
        if image_path:
            lookup[image_path] = record.get("conversations", [])
    return lookup


def merge_sft_into_grpo(grpo_records, sft_lookup):
    """将 SFT 的 conversations 合并到 GRPO 记录中。"""
    matched_count = 0
    for record in grpo_records:
        image_path = record.get("image_path", "")
        if image_path in sft_lookup:
            record["conversations"] = sft_lookup[image_path]
            matched_count += 1
    return grpo_records, matched_count


def main():
    print(f"Loading SFT dataset from {SFT_PATH}...")
    sft_records = load_jsonl(SFT_PATH)
    print(f"  Loaded {len(sft_records)} SFT records")

    sft_lookup = build_sft_lookup(sft_records)
    print(f"  Built lookup table with {len(sft_lookup)} unique image_paths")

    for grpo_path in [GRPO_TRAIN_PATH, GRPO_TEST_PATH]:
        print(f"\nProcessing {grpo_path}...")
        grpo_records = load_jsonl(grpo_path)
        print(f"  Loaded {len(grpo_records)} GRPO records")

        merged_records, matched_count = merge_sft_into_grpo(grpo_records, sft_lookup)
        print(f"  Matched {matched_count}/{len(grpo_records)} records with SFT conversations")

        save_jsonl(merged_records, grpo_path)
        print(f"  Saved merged dataset to {grpo_path}")

    print("\nDone! GRPO datasets now contain SFT conversations field.")


if __name__ == "__main__":
    main()
