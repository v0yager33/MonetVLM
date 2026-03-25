#!/bin/bash
# ============================================================
# Qwen3-VL 评测脚本
#   在 WikiArt 艺术风格分类测试集上评测 Qwen3-VL 系列模型
# ============================================================
set -e

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
echo "Using GPU: ${CUDA_VISIBLE_DEVICES}"

WORK_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${WORK_DIR}"
echo "Working directory: ${WORK_DIR}"

TEST_PATH="data/wikiart_artist/wikiart_artist_grpo_test.jsonl"
MAX_NEW_TOKENS=512
OUTPUT_DIR="eval/results"
mkdir -p "${OUTPUT_DIR}"

TEMPERATURE=${TEMPERATURE:-0.6}
TOP_P=${TOP_P:-0.95}
NUM_ROUNDS=${NUM_ROUNDS:-16}

echo "=============================================="
echo "  Qwen3-VL Evaluation Config"
echo "=============================================="
echo "  Test set:      ${TEST_PATH}"
echo "  Temperature:   ${TEMPERATURE}"
echo "  Top-p:         ${TOP_P}"
echo "  Num rounds:    ${NUM_ROUNDS}"
echo "  Max tokens:    ${MAX_NEW_TOKENS}"
echo "=============================================="

MODELS_BASE="/chatgpt_nas/dukaixuan.dkx/models"
QWEN3_VL_MODELS=(
    "Qwen3-VL-2B-Instruct"
    "Qwen3-VL-4B-Instruct"
    "Qwen3-VL-8B-Instruct"
)

for MODEL_NAME in "${QWEN3_VL_MODELS[@]}"; do
    MODEL_PATH="${MODELS_BASE}/${MODEL_NAME}"
    if [ ! -d "${MODEL_PATH}" ]; then
        echo "[SKIP] Qwen3-VL model not found: ${MODEL_PATH}"
        continue
    fi

    OUTPUT_FILE="${OUTPUT_DIR}/${MODEL_NAME}_results.jsonl"

    echo ""
    echo "=============================================="
    echo "  Evaluating Qwen3-VL: ${MODEL_NAME}"
    echo "=============================================="

    python eval/eval_qwen3_vl.py \
        --model_path "${MODEL_PATH}" \
        --test_path "${TEST_PATH}" \
        --max_new_tokens ${MAX_NEW_TOKENS} \
        --temperature ${TEMPERATURE} \
        --top_p ${TOP_P} \
        --num_rounds ${NUM_ROUNDS} \
        --output_path "${OUTPUT_FILE}"
done

echo ""
echo "=============================================="
echo "  Qwen3-VL evaluations complete!"
echo "  Results saved to: ${OUTPUT_DIR}/"
echo "=============================================="
