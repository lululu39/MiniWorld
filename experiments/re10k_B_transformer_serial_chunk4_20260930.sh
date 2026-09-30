#!/usr/bin/env bash
# User-selected serial Transformer, chunk4, RE10K 8/16/32/64 curriculum.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NNODES=1 NODE_RANK=0 NPROC_PER_NODE=8 MASTER_ADDR=127.0.0.1 MASTER_PORT=29461
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
export MODEL=B BACKBONE=transformer SEED=42
export TRANSFORMER_EXECUTION=serial DF_CHUNK_SIZE=4 STAGE1_LATENT_FRAMES=8
export RUN_NAME=re10k_B_transformer_serial_c4_s42_8xh100_20260930T012919Z
export OUTPUT_DIR=/mnt/localssd/experiments/yibo/miniworld/$RUN_NAME
export DATA_ROOT=/mnt/localssd/dataset/re10k/training_256
export POSE_DIR=/mnt/localssd/dataset/re10k/training_poses
export FILTER_CACHE_DIR=/mnt/localssd/dataset/re10k/filter_cache
export EVAL_DATA_ROOT=/mnt/localssd/dataset/re10k/test_256
export EVAL_POSE_DIR=/mnt/localssd/dataset/re10k/test_poses
export EVAL_FILTER_CACHE_DIR="$FILTER_CACHE_DIR"
export EVAL_EVERY=1000 EVAL_NUM_VIDEOS=8 EVAL_LATENT_FRAMES=0 EVAL_SEED=42
export EVAL_SAMPLING_STEPS=100 EVAL_CFG_SCALE=2.0 EVAL_ARDIFF_STEP=5
export MINIWORLD_WANDB_ENTITY=LVSM-Experiment MINIWORLD_WANDB_PROJECT=miniworld
export VAE_CKPT=/mnt/localssd/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth
exec bash scripts/train_re10k.sh
