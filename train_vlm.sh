#!/bin/bash
# ============================================================
# SparkVLM 完整训练流水线
#   Phase 0: Adapter 预训练（冻结 ViT + LLM，只训练 Adapter）
#   Phase 1: SFT Stage 1 - 冻结 ViT，训练 Adapter + LLM
#   Phase 2: SFT Stage 2 - 解冻全部参数全参微调
# ============================================================

set -e  # 任何命令失败立即退出

# -------------------- GPU 配置 --------------------
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
echo "Using GPU: ${CUDA_VISIBLE_DEVICES}"

# -------------------- 工作目录 --------------------
WORK_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${WORK_DIR}"
echo "Working directory: ${WORK_DIR}"

# -------------------- 通用配置 --------------------
PRETRAIN_DATA="data/sharegpt4v_coco_train.jsonl"
SFT_TRAIN_DATA="data/sharegpt4v_coco_sft_train.jsonl"
SFT_VAL_DATA="data/sharegpt4v_coco_sft_val.jsonl"
EVAL_STEPS=1000

BATCH_SIZE=2
GRAD_ACCUM=4
NUM_EPOCHS=1

# -------------------- Phase 0: Adapter 预训练 --------------------
echo "=============================================="
echo "  Phase 0: Adapter Pretrain (ViT + LLM Frozen)"
echo "=============================================="

PHASE0_OUTPUT="save/vlm_pretrain_adapter"

python train_proj.py \
    --jsonl_path ${PRETRAIN_DATA} \
    --output_dir ${PHASE0_OUTPUT}

echo ""
echo "  Phase 0 complete! Output: ${PHASE0_OUTPUT}"
echo ""

# -------------------- Phase 1: SFT 冻结 ViT --------------------
echo "=============================================="
echo "  Phase 1: SFT with ViT Frozen"
echo "=============================================="

PHASE1_INPUT="${PHASE0_OUTPUT}"
PHASE1_OUTPUT="save/vlm_sft_freeze_vit"
PHASE1_LR=1e-5
PHASE1_ADAPTER_LR=1e-4
PHASE1_LLM_LR=1e-5

python sft_train_freeze_vit.py \
    --pretrained_model_path ${PHASE1_INPUT} \
    --jsonl_path ${SFT_TRAIN_DATA} \
    --val_jsonl_path ${SFT_VAL_DATA} \
    --eval_steps ${EVAL_STEPS} \
    --output_dir ${PHASE1_OUTPUT} \
    --learning_rate ${PHASE1_LR} \
    --adapter_lr ${PHASE1_ADAPTER_LR} \
    --llm_lr ${PHASE1_LLM_LR} \
    --num_epochs ${NUM_EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM}

echo ""
echo "  Phase 1 complete! Output: ${PHASE1_OUTPUT}"
echo ""

# -------------------- Phase 2: SFT 全参微调 --------------------
echo "=============================================="
echo "  Phase 2: SFT Full Model (ViT Unfrozen)"
echo "=============================================="

PHASE2_INPUT="${PHASE1_OUTPUT}"
PHASE2_OUTPUT="save/vlm_sft_full"
PHASE2_LR=1e-5
PHASE2_VIT_LR=2e-6
PHASE2_ADAPTER_LR=5e-5
PHASE2_LLM_LR=1e-5

python sft_train_full.py \
    --pretrained_model_path ${PHASE2_INPUT} \
    --jsonl_path ${SFT_TRAIN_DATA} \
    --val_jsonl_path ${SFT_VAL_DATA} \
    --eval_steps ${EVAL_STEPS} \
    --output_dir ${PHASE2_OUTPUT} \
    --learning_rate ${PHASE2_LR} \
    --vit_lr ${PHASE2_VIT_LR} \
    --adapter_lr ${PHASE2_ADAPTER_LR} \
    --llm_lr ${PHASE2_LLM_LR} \
    --num_epochs ${NUM_EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM}

echo ""
echo "=============================================="
echo "  All training stages complete!"
echo "  Phase 0 (Adapter Pretrain): ${PHASE0_OUTPUT}"
echo "  Phase 1 (SFT ViT Frozen):   ${PHASE1_OUTPUT}"
echo "  Phase 2 (SFT Full Model):   ${PHASE2_OUTPUT}"
echo "=============================================="
