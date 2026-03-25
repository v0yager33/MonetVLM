"""
基于 wikiart_artist_sft.jsonl 构造 GRPO 选择题数据集。

每条记录生成一个 6 选 1 的选择题：
- 正确答案为该图片的真实 genre
- 5 个干扰项从其余 genre 中随机抽取
- 选项顺序随机打乱，正确答案位置随机
- analysis 字段来自 SFT 数据中的 assistant 回复

输入：wikiart_artist_sft.jsonl（含 image_path + conversations）
输出：wikiart_artist_grpo.jsonl

用法：
    python generate_grpo.py
    python generate_grpo.py --num_options 4   # 4 选 1
    python generate_grpo.py --seed 42         # 固定随机种子
"""

import argparse
import json
import os
import random

# ─── 配置 ───────────────────────────────────────────────────────────────────────

INPUT_SFT_JSONL = "/chatgpt_nas/dukaixuan.dkx/reproduceVLM/data/wikiart_artist/wikiart_artist_sft.jsonl"
INPUT_META_JSONL = "/chatgpt_nas/dukaixuan.dkx/reproduceVLM/data/wikiart_artist/wikiart_artist.jsonl"
OUTPUT_TRAIN_JSONL = "/chatgpt_nas/dukaixuan.dkx/reproduceVLM/data/wikiart_artist/wikiart_artist_grpo_train.jsonl"
OUTPUT_TEST_JSONL = "/chatgpt_nas/dukaixuan.dkx/reproduceVLM/data/wikiart_artist/wikiart_artist_grpo_test.jsonl"
TEST_SIZE = 200

# 数据集中实际存在的 16 种 genre
ALL_GENRES = [
    "Abstract Expressionism",
    "Analytical Cubism",
    "Art Nouveau",
    "Baroque",
    "Cubism",
    "Expressionism",
    "Fauvism",
    "Impressionism",
    "Naive Art Primitivism",
    "Northern Renaissance",
    "Pointillism",
    "Post Impressionism",
    "Realism",
    "Romanticism",
    "Symbolism",
    "Synthetic Cubism",
]

OPTION_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

FIXED_QUESTION_PREFIX = (
    "Please analyze the painting shown in the image. "
    "First, briefly describe the content; "
    "then, describe its painting style. "
    "Finally, select the correct art genre from the options below. "
    "You must put your final answer (the option letter) inside \\boxed{}, e.g., \\boxed{A}.\n<image>"
)


def load_genre_mapping(meta_jsonl_path: str) -> dict[str, str]:
    """从元数据文件加载 image_path -> genre 的映射。"""
    mapping = {}
    with open(meta_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line.strip())
            mapping[record["image_path"]] = record["genre"]
    return mapping


def load_sft_records(sft_jsonl_path: str) -> list[dict]:
    """加载 SFT 数据，提取 image_path 和 assistant 的 analysis 文本。"""
    records = []
    with open(sft_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line.strip())
            image_path = record["image_path"]
            analysis = ""
            for turn in record.get("conversations", []):
                if turn["from"] == "assistant":
                    analysis = turn["value"]
                    break
            records.append({
                "image_path": image_path,
                "analysis": analysis,
            })
    return records


def generate_options(correct_genre: str, num_options: int, rng: random.Random) -> tuple[list[str], int]:
    """
    生成选项列表和正确答案的索引。

    返回:
        options: 打乱后的选项列表
        correct_index: 正确答案在列表中的索引（0-based）
    """
    wrong_genres = [g for g in ALL_GENRES if g != correct_genre]
    num_distractors = num_options - 1
    distractors = rng.sample(wrong_genres, min(num_distractors, len(wrong_genres)))

    options = [correct_genre] + distractors
    rng.shuffle(options)
    correct_index = options.index(correct_genre)

    return options, correct_index


def format_query(options: list[str]) -> str:
    """将固定问题和选项拼接成完整的 query 字符串。"""
    options_text = "\n".join(
        f"{OPTION_LABELS[i]}. {option}"
        for i, option in enumerate(options)
    )
    return f"{FIXED_QUESTION_PREFIX}\n\nOptions:\n{options_text}"


def main():
    parser = argparse.ArgumentParser(description="构造 GRPO 选择题数据集")
    parser.add_argument(
        "--num_options", type=int, default=6,
        help="每题选项数量（默认 6）",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子（默认 42）",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # 加载 genre 映射
    print(f"Loading genre mapping from: {INPUT_META_JSONL}")
    genre_mapping = load_genre_mapping(INPUT_META_JSONL)
    print(f"  Loaded {len(genre_mapping)} genre mappings")

    # 加载 SFT 记录
    print(f"Loading SFT records from: {INPUT_SFT_JSONL}")
    sft_records = load_sft_records(INPUT_SFT_JSONL)
    print(f"  Loaded {len(sft_records)} SFT records")

    # 生成 GRPO 数据
    all_grpo_records = []
    skipped_count = 0

    for record in sft_records:
        image_path = record.get("image_path", "")

        genre = genre_mapping.get(image_path)
        if genre is None:
            print(f"  [SKIP] No genre found for: {image_path}")
            skipped_count += 1
            continue

        options, correct_index = generate_options(genre, args.num_options, rng)
        query = format_query(options)
        answer_label = OPTION_LABELS[correct_index]

        grpo_record = dict(record)
        grpo_record["query"] = query
        grpo_record["answer"] = answer_label
        grpo_record["genre"] = genre

        all_grpo_records.append(grpo_record)

    # 打乱数据顺序
    rng.shuffle(all_grpo_records)

    # 划分测试集（最后 TEST_SIZE 条）和训练集
    test_records = all_grpo_records[:TEST_SIZE]
    train_records = all_grpo_records[TEST_SIZE:]

    # 写入训练集
    with open(OUTPUT_TRAIN_JSONL, "w", encoding="utf-8") as f_out:
        for record in train_records:
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

    # 写入测试集
    with open(OUTPUT_TEST_JSONL, "w", encoding="utf-8") as f_out:
        for record in test_records:
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nDone!")
    print(f"  Train set: {OUTPUT_TRAIN_JSONL} ({len(train_records)} records)")
    print(f"  Test set:  {OUTPUT_TEST_JSONL} ({len(test_records)} records)")
    print(f"  Skipped:   {skipped_count} records")


if __name__ == "__main__":
    main()
