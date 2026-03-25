#!/bin/bash
# ============================================================
# MonetVLM 评测脚本 - GRPO Trainer 风格
#   支持多卡并发、流式小 batch 处理、显存优化
# ============================================================
set -e

# 配置参数
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
echo "Using GPUs: ${CUDA_VISIBLE_DEVICES} (${NUM_GPUS} GPUs)"

WORK_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${WORK_DIR}"
echo "Working directory: ${WORK_DIR}"

TEST_PATH="data/wikiart_artist/wikiart_artist_grpo_test.jsonl"
MAX_NEW_TOKENS=512
BATCH_SIZE=${BATCH_SIZE:-64}  # GRPO 风格：高并发 batch size
OUTPUT_DIR="eval/results"
mkdir -p "${OUTPUT_DIR}"

TEMPERATURE=${TEMPERATURE:-0.6}
TOP_P=${TOP_P:-0.95}
NUM_ROUNDS=${NUM_ROUNDS:-4}

echo "=============================================="
echo "  MonetVLM Evaluation Config"
echo "=============================================="
echo "  Test set:      ${TEST_PATH}"
echo "  GPUs:          ${NUM_GPUS}"
echo "  Batch size:    ${BATCH_SIZE} per GPU"
echo "  Temperature:   ${TEMPERATURE}"
echo "  Top-p:         ${TOP_P}"
echo "  Num rounds:    ${NUM_ROUNDS}"
echo "  Max tokens:    ${MAX_NEW_TOKENS}"
echo "=============================================="

MONETVLM_MODELS=(
    "save/monet_grpo/checkpoint-1-step-800"
    "save/vlm_pretrain_adapter"
    "save/vlm_sft_full"
    "save/vlm_sft_freeze_vit"
    "save/monet_sft"
)

for MODEL_DIR in "${MONETVLM_MODELS[@]}"; do
    if [ ! -d "${MODEL_DIR}" ]; then
        echo "[SKIP] MonetVLM model not found: ${MODEL_DIR}"
        continue
    fi

    MODEL_NAME=$(basename "${MODEL_DIR}")
    OUTPUT_FILE="${OUTPUT_DIR}/${MODEL_NAME}_grpo_results.jsonl"

    echo ""
    echo "=============================================="
    echo "  Evaluating MonetVLM: ${MODEL_NAME}"
    echo "=============================================="

    if [ "${NUM_GPUS}" -eq 1 ]; then
        # 单卡模式
        python eval/eval_monetvlm_grpo.py \
            --model_dir "${MODEL_DIR}" \
            --test_path "${TEST_PATH}" \
            --batch_size ${BATCH_SIZE} \
            --max_new_tokens ${MAX_NEW_TOKENS} \
            --temperature ${TEMPERATURE} \
            --top_p ${TOP_P} \
            --num_rounds ${NUM_ROUNDS} \
            --output_path "${OUTPUT_FILE}"
    else
        echo "多卡错误"
    fi
done

echo ""
echo "=============================================="
echo "  MonetVLM evaluations complete!"
echo "  Results saved to: ${OUTPUT_DIR}/"
echo "=============================================="
