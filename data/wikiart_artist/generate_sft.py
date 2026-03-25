"""
调用豆包 VLM API，基于 wikiart_artist.jsonl 中的图片和流派信息，
生成艺术风格分析的 SFT 数据集。

输出格式兼容 VLMDataset 的 conversations 格式：
{"image_path": "...", "conversations": [{"from": "human", "value": "...\n<image>"}, {"from": "assistant", "value": "..."}]}

支持：
- 断点续传：已处理的记录会跳过
- 错误重试：单条失败最多重试 3 次
- 进度显示：每处理 10 条打印一次进度

用法：
    pip install openai
    python generate_sft.py
"""

import argparse
import base64
import io
import json
import os
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image

from openai import OpenAI
from tqdm import tqdm

# ─── 配置 ───────────────────────────────────────────────────────────────────────
API_KEY = ""
BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
MODEL = "doubao-seed-2-0-mini-260215"

INPUT_JSONL = "/chatgpt_nas/dukaixuan.dkx/reproduceVLM/data/wikiart_artist/wikiart_artist.jsonl"
OUTPUT_JSONL = "/chatgpt_nas/dukaixuan.dkx/reproduceVLM/data/wikiart_artist/wikiart_artist_sft.jsonl"

MAX_RETRIES = 10
RETRY_DELAY = 3  # 秒
REQUEST_INTERVAL = 0.5  # 请求间隔，避免限流
MAX_IMAGE_SIZE_MB = 3  # 超过此大小的图片会自动压缩

# 用户侧的固定提问（包含 <image> 占位符，供 VLMDataset 替换为 image_pad）
FIXED_QUESTION = (
    "Please analyze the painting shown in the image. "
    "First, describe the content of the painting; "
    "then, describe its painting style; "
    "finally, state which art genre this work is categorized into.\n<image>"
)

SYSTEM_PROMPT = """# Role

You are a Senior Art Critic and Visual Data Specialist. Your expertise lies in analyzing the visual elements of a painting and articulating its stylistic characteristics with professional precision.

# Task

I will provide you with an image and its official "Genre" label. Your goal is to generate a high-quality analysis based on your actual visual observation of the artwork.

# Constraints

1. **NO ARTIST MENTION**: You are strictly prohibited from mentioning the artist's name (e.g., "Dali"), their biography, or their nationality. Focus solely on the visual attributes of the painting.
2. **Visual-Driven**: Your descriptions must be grounded in the visual evidence of the provided image—describe the composition, lighting, brushwork, and color usage accurately.
3. **Style Alignment**: Ensure your stylistic analysis logically leads to the provided "Genre" conclusion.
4. **Output Only the Analysis Text**: Do NOT output JSON or any formatting markers. Only output the plain analysis text consisting of three parts:
   (1) A detailed description of the visual content.
   (2) An analysis of the artistic style.
   (3) The final genre conclusion based on the provided label.

# Demonstration

For a painting labeled "Post-Impressionism", an example output would be:

This painting depicts a serene coastal landscape during midday. The focal point features a series of sun-drenched, earth-toned buildings situated along a shoreline. The shadows cast by the structures are sharp and deep, contrasting with the calm, turquoise waters and the rugged, barren rock formations in the background. The entire scene evokes a sense of stillness and vast spatial isolation.

In terms of painting style, the work demonstrates a rigorous focus on structural form. Rather than capturing the fleeting effects of light, the artist uses firm, deliberate brushstrokes to define the volume and geometry of the buildings. The color palette is applied with subjective intensity, utilizing broad planes of contrasting colors to reinforce the internal order and structural solidity of the composition.

Based on these visual characteristics and the emphasis on structure over transient impressions, this work is categorized into the **Post Impressionism** genre."""


# ─── 工具函数 ─────────────────────────────────────────────────────────────────────

def encode_image_to_base64(image_path: str) -> str:
    """将本地图片编码为 base64 data URI，超大图片自动压缩。"""
    file_size_mb = os.path.getsize(image_path) / (1024 * 1024)

    if file_size_mb <= MAX_IMAGE_SIZE_MB:
        extension = os.path.splitext(image_path)[1].lower()
        mime_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".bmp": "image/bmp",
            ".webp": "image/webp",
        }
        mime_type = mime_map.get(extension, "image/jpeg")
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"

    # 超大图片：缩小并压缩为 JPEG
    image = Image.open(image_path).convert("RGB")
    max_dimension = 2048
    image.thumbnail((max_dimension, max_dimension), Image.LANCZOS)

    buffer = io.BytesIO()
    quality = 85
    image.save(buffer, format="JPEG", quality=quality)

    while buffer.tell() > MAX_IMAGE_SIZE_MB * 1024 * 1024 and quality > 20:
        buffer = io.BytesIO()
        quality -= 10
        image.save(buffer, format="JPEG", quality=quality)

    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def load_processed_image_paths(output_path: str) -> set:
    """加载已处理的图片路径集合，用于断点续传。"""
    processed = set()
    if not os.path.exists(output_path):
        return processed
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line.strip())
                processed.add(record.get("image_path", ""))
            except json.JSONDecodeError:
                continue
    return processed


def call_vlm_api(client: OpenAI, image_path: str, genre: str) -> str:
    """调用豆包 VLM API 获取画作分析文本。"""
    image_data_uri = encode_image_to_base64(image_path)
    user_text = f'Please analyze this painting. Its official genre label is: "{genre}".'

    response = client.responses.create(
        model=MODEL,
        input=[
            {
                "role": "system",
                "content": [
                    {"type": "input_text", "text": SYSTEM_PROMPT},
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": image_data_uri,
                    },
                    {
                        "type": "input_text",
                        "text": user_text,
                    },
                ],
            },
        ],
    )

    return response.output_text


def process_single_record(client: OpenAI, record: dict) -> dict | None:
    """处理单条记录，返回 conversations 格式的 dict，失败返回 None。"""
    image_path = record["image_path"]
    genre = record["genre"]

    if not os.path.exists(image_path):
        print(f"  [SKIP] Image not found: {image_path}")
        return None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            output_text = call_vlm_api(client, image_path, genre)
            if not output_text or not output_text.strip():
                raise ValueError("Empty response from API")

            return {
                "image_path": image_path,
                "conversations": [
                    {"from": "human", "value": FIXED_QUESTION},
                    {"from": "assistant", "value": output_text.strip()},
                ],
            }
        except Exception as error:
            print(f"  [ERROR] Attempt {attempt}/{MAX_RETRIES} for {os.path.basename(image_path)}: {error}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)
            else:
                traceback.print_exc()
                return None


# ─── 主流程 ───────────────────────────────────────────────────────────────────────

# 线程安全的写入锁和计数器
write_lock = threading.Lock()
counter_lock = threading.Lock()


def worker_task(client: OpenAI, record: dict, f_out, counters: dict, progress_bar: tqdm):
    """单个并发 worker 的任务：处理一条记录并写入结果。"""
    result = process_single_record(client, record)

    with counter_lock:
        if result is not None:
            with write_lock:
                f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                f_out.flush()
            counters["success"] += 1
        else:
            counters["fail"] += 1

        progress_bar.set_postfix(success=counters["success"], fail=counters["fail"])
        progress_bar.update(1)

    return result is not None


def main():
    parser = argparse.ArgumentParser(description="调用豆包 VLM API 生成 wikiart SFT 数据集")
    parser.add_argument(
        "--limit", type=int, default=0,
        help="最多处理多少条新记录，0 表示全部（默认全部）",
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help="并发线程数（默认 8）",
    )
    args = parser.parse_args()

    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

    # 加载输入数据
    with open(INPUT_JSONL, "r", encoding="utf-8") as f:
        all_records = [json.loads(line.strip()) for line in f if line.strip()]
    print(f"Total records in input: {len(all_records)}")

    # 断点续传：跳过已处理的
    processed_paths = load_processed_image_paths(OUTPUT_JSONL)
    print(f"Already processed: {len(processed_paths)}")

    # 过滤出待处理的记录
    pending_records = [r for r in all_records if r["image_path"] not in processed_paths]
    if args.limit > 0:
        pending_records = pending_records[:args.limit]

    print(f"To process: {len(pending_records)} | Workers: {args.workers}")

    if not pending_records:
        print("Nothing to process, exiting.")
        return

    counters = {
        "success": len(processed_paths),
        "fail": 0,
    }

    with open(OUTPUT_JSONL, "a", encoding="utf-8") as f_out:
        with tqdm(total=len(pending_records), desc="Generating SFT", unit="img") as progress_bar:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(worker_task, client, record, f_out, counters, progress_bar): record
                    for record in pending_records
                }

                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as error:
                        record = futures[future]
                        tqdm.write(f"  [FATAL] Unhandled error for {record['image_path']}: {error}")

    print(f"\nDone! Output: {OUTPUT_JSONL}")
    print(f"Total success: {counters['success']}, Total failed: {counters['fail']}")

if __name__ == "__main__":
    main()
