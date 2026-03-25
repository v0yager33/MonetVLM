#!/bin/bash
# ============================================================
# MonetVLM 训练流水线
#   Phase 1: 使用 SFT 数据集做 SFT 训练 (1 epoch)
#   Phase 2: 使用 GRPO 数据集做 GRPO 强化学习训练 (1 epoch)
# ============================================================

set -e

# -------------------- GPU 配置 --------------------
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
echo "Using GPU: ${CUDA_VISIBLE_DEVICES}"

# -------------------- 工作目录 --------------------
WORK_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${WORK_DIR}"
echo "Working directory: ${WORK_DIR}"

# -------------------- 路径配置 --------------------
PRETRAINED_MODEL="save/vlm_sft_full"
TRAIN_DATASET="data/wikiart_artist/wikiart_artist_grpo_train.jsonl"
VAL_DATASET="data/wikiart_artist/wikiart_artist_grpo_test.jsonl"

SFT_OUTPUT="save/monet_sft"
GRPO_OUTPUT="save/monet_grpo"

# -------------------- Phase 1: SFT (1 epoch) --------------------
echo "=============================================="
echo "  Phase 1: SFT on wikiart_artist dataset"
echo "=============================================="

python sft_train_full.py \
    --pretrained_model_path ${PRETRAINED_MODEL} \
    --jsonl_path ${TRAIN_DATASET} \
    --val_jsonl_path ${VAL_DATASET} \
    --eval_steps 50 \
    --output_dir ${SFT_OUTPUT} \
    --learning_rate 1e-5 \
    --vit_lr 2e-6 \
    --adapter_lr 5e-5 \
    --llm_lr 1e-5 \
    --num_epochs 2 \
    --batch_size 4 \
    --gradient_accumulation_steps 4

echo ""
echo "  Phase 1 (SFT) complete! Output: ${SFT_OUTPUT}"
echo ""

# -------------------- Phase 2: GRPO (1 epoch) --------------------
echo "=============================================="
echo "  Phase 2: GRPO on wikiart_artist dataset"
echo "=============================================="

python grpo_train.py \
    --model_path ${SFT_OUTPUT} \
    --dataset_path ${TRAIN_DATASET} \
    --val_dataset_path ${VAL_DATASET} \
    --output_dir ${GRPO_OUTPUT} \
    --num_generations 8 \
    --max_completion_length 512 \
    --learning_rate 1e-6 \
    --clip_epsilon 0.2 \
    --kl_coeff 0.01 \
    --mini_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --eval_steps 40 \
    --eval_batch_size 256

echo ""
echo "=============================================="
echo "  MonetVLM Training Complete!"
echo "  SFT  output: ${SFT_OUTPUT}"
echo "  GRPO output: ${GRPO_OUTPUT}"
echo "=============================================="

# python grpo_train.py \
#     --model_path save/monet_sft \
#     --dataset_path data/wikiart_artist/wikiart_artist_grpo_train.jsonl \
#     --val_dataset_path data/wikiart_artist/wikiart_artist_grpo_test.jsonl \
#     --output_dir save/monet_grpo \
#     --num_generations 8 \
#     --max_completion_length 512 \
#     --learning_rate 1e-6 \
#     --clip_epsilon 0.2 \
#     --kl_coeff 0.01 \
#     --mini_batch_size 16 \
#     --gradient_accumulation_steps 1 \
#     --eval_steps 40 \
#     --eval_batch_size 256