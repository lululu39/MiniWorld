#!/usr/bin/env bash
# Verified common per-GPU batches for serial Transformer/RTransformer/TaS B.
# See experiments/rtransformer_checkpoint_memory_20260930.md for the corrected profile.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
: "${CUDA_VISIBLE_DEVICES:?Select eight authorized H100 GPUs explicitly}"
IFS=',' read -r -a SELECTED_GPUS <<< "$CUDA_VISIBLE_DEVICES"
if [[ ${#SELECTED_GPUS[@]} -ne 8 || ${NNODES:-1} -ne 1 || ${NPROC_PER_NODE:-8} -ne 8 ]]; then
  echo "This batch/LR profile is for one node with eight GPUs." >&2
  exit 2
fi
export NNODES=1 NPROC_PER_NODE=8 NODE_RANK=0
export MODEL=B BACKBONE="${BACKBONE:-transformer}"
case "$BACKBONE" in transformer|rtransformer|tas) ;; *) echo "Unknown backbone: $BACKBONE" >&2; exit 2 ;; esac
export TRANSFORMER_EXECUTION=serial DF_CHUNK_SIZE=4
export NUM_MEMORY_TOKENS=256 MEMORY_WINDOW_FRAMES=0
export STAGE1_LATENT_FRAMES=8 STAGE2_LATENT_FRAMES=16 STAGE3_LATENT_FRAMES=32 STAGE4_LATENT_FRAMES=64
export STAGE1_BATCH_SIZE=32 STAGE2_BATCH_SIZE=16 STAGE3_BATCH_SIZE=8 STAGE4_BATCH_SIZE=4
export STAGE1_LR=2e-4 STAGE2_LR=4e-5 STAGE3_LR=5.656854249e-5 STAGE4_LR=4e-5
# Preserve video exposure: epochs for stages1/2; 240,000 videos each for3/4.
export STAGE1_EPOCHS=100 STAGE2_EPOCHS=50 STAGE3_MAX_TRAIN_STEPS=3750 STAGE4_MAX_TRAIN_STEPS=7500
export SEED="${SEED:-42}"
export RUN_NAME="${RUN_NAME:-re10k_B_${BACKBONE}_serial_c4_sharedbatch_s${SEED}_$(date -u +%Y%m%dT%H%M%SZ)}"
export OUTPUT_DIR="${OUTPUT_DIR:-/mnt/localssd/experiments/yibo/miniworld/${RUN_NAME}}"
export DATA_ROOT="${DATA_ROOT:-/mnt/localssd/dataset/re10k/training_256}"
export POSE_DIR="${POSE_DIR:-/mnt/localssd/dataset/re10k/training_poses}"
export FILTER_CACHE_DIR="${FILTER_CACHE_DIR:-/mnt/localssd/dataset/re10k/filter_cache}"
export EVAL_DATA_ROOT="${EVAL_DATA_ROOT:-/mnt/localssd/dataset/re10k/test_256}"
export EVAL_POSE_DIR="${EVAL_POSE_DIR:-/mnt/localssd/dataset/re10k/test_poses}"
export EVAL_FILTER_CACHE_DIR="${EVAL_FILTER_CACHE_DIR:-$FILTER_CACHE_DIR}"
export EVAL_EVERY="${EVAL_EVERY:-1000}" EVAL_NUM_VIDEOS=8 EVAL_SEED=42
export EVAL_SAMPLING_STEPS=100 EVAL_CFG_SCALE=2.0 EVAL_ARDIFF_STEP=5
export VAE_CKPT="${VAE_CKPT:-/mnt/localssd/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth}"
export MINIWORLD_WANDB_ENTITY=LVSM-Experiment MINIWORLD_WANDB_PROJECT=miniworld
exec bash scripts/train_re10k.sh "$@"
