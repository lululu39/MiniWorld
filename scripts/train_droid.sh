#!/usr/bin/env bash
# Train action-conditioned MiniWorld on DROID in four curriculum stages.
# Default model is 1B. Set MODEL=0.5B or MODEL=3B to switch model scale.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

BACKBONE="${BACKBONE:-transformer}"
MODEL="${MODEL:-1B}" # choices: B, L, 0.5B, 1B, 3B
DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to the LeRobot DROID root}"
VAE_CKPT="${VAE_CKPT:?Set VAE_CKPT to Wan2.2_VAE.pth}"
if [[ "${NNODES:-${ARNOLD_WORKER_NUM:-1}}" -gt 1 && -z "${RUN_NAME:-}" ]]; then
  echo "Set the same RUN_NAME on every node for a multi-node curriculum." >&2
  exit 2
fi
RUN_NAME="${RUN_NAME:-droid_${BACKBONE}_${MODEL}_$(date -u +%Y%m%dT%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/outputs/${RUN_NAME}}"
WANDB_PROJECT="${MINIWORLD_WANDB_PROJECT:-miniworld}"
WANDB_ENTITY="${MINIWORLD_WANDB_ENTITY:-LVSM-Experiment}"
EVAL_EVERY="${EVAL_EVERY:-0}"
ACTION_CAMERA_VIEWS="${ACTION_CAMERA_VIEWS:-exterior_image_1_left}"
ACTION_KEYS="${ACTION_KEYS:-cartesian_position,gripper_position}"

STAGE1_LATENT_FRAMES="${STAGE1_LATENT_FRAMES:-6}"
STAGE2_LATENT_FRAMES="${STAGE2_LATENT_FRAMES:-16}"
STAGE3_LATENT_FRAMES="${STAGE3_LATENT_FRAMES:-32}"
STAGE4_LATENT_FRAMES="${STAGE4_LATENT_FRAMES:-64}"
STAGE1_BATCH_SIZE="${STAGE1_BATCH_SIZE:-8}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-4}"
STAGE3_BATCH_SIZE="${STAGE3_BATCH_SIZE:-1}"
STAGE4_BATCH_SIZE="${STAGE4_BATCH_SIZE:-1}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-100}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-50}"
STAGE3_MAX_TRAIN_STEPS="${STAGE3_MAX_TRAIN_STEPS:-30000}"
STAGE4_MAX_TRAIN_STEPS="${STAGE4_MAX_TRAIN_STEPS:-30000}"

NNODES="${NNODES:-${ARNOLD_WORKER_NUM:-1}}"
NODE_RANK="${NODE_RANK:-${ARNOLD_ID:-0}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${ARNOLD_WORKER_GPU:-8}}"
MASTER_ADDR="${MASTER_ADDR:-${ARNOLD_WORKER_0_HOST:-127.0.0.1}}"
MASTER_PORT="${MASTER_PORT:-12471}"
TORCHRUN=(uv run --no-sync torchrun --nnodes="${NNODES}" --node_rank="${NODE_RANK}" --nproc_per_node="${NPROC_PER_NODE}" --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}")

COMMON_ARGS=(
  -m miniworld.train
  --dataset droid
  --data_root "${DATA_ROOT}"
  --action_camera_views "${ACTION_CAMERA_VIEWS}"
  --action_keys "${ACTION_KEYS}"
  --wm_model "${MODEL}"
  --backbone "${BACKBONE}"
  --num_memory_tokens "${NUM_MEMORY_TOKENS:-256}"
  --memory_window_frames "${MEMORY_WINDOW_FRAMES:-4}"
  --seed "${SEED:-42}"
  --vae_checkpoint "${VAE_CKPT}"
  --resize_h 240
  --resize_w 320
  --df_chunk_size 2
  --num_workers 8
  --prefetch_factor 2
  --mixed_precision bf16
  --use_muon
  --wandb_project "${WANDB_PROJECT}"
  --wandb_entity "${WANDB_ENTITY}"
  --wandb_group "${RUN_NAME}"
  --eval_every "${EVAL_EVERY}"
  --eval_num_videos "${EVAL_NUM_VIDEOS:-8}"
  --eval_latent_frames "${EVAL_LATENT_FRAMES:-0}"
  --eval_seed "${EVAL_SEED:-42}"
  --eval_sampling_steps "${EVAL_SAMPLING_STEPS:-100}"
  --eval_cfg_scale "${EVAL_CFG_SCALE:-2.0}"
  --eval_ardiff_step "${EVAL_ARDIFF_STEP:-5}"
)

# Held-out paths are explicit; evaluation stays disabled until EVAL_EVERY > 0.
if [[ -n "${EVAL_DATA_ROOT:-}" ]]; then
  COMMON_ARGS+=(--eval_data_root "${EVAL_DATA_ROOT}")
fi
if [[ -n "${EVAL_POSE_DIR:-}" ]]; then
  COMMON_ARGS+=(--eval_pose_dir "${EVAL_POSE_DIR}")
fi
if [[ -n "${EVAL_FILTER_CACHE_DIR:-}" ]]; then
  COMMON_ARGS+=(--eval_filter_cache_dir "${EVAL_FILTER_CACHE_DIR}")
fi

# Optional model/experiment flags, forwarded to each curriculum stage.
COMMON_ARGS+=("$@")

echo "Stage 1/4: latent_frames=${STAGE1_LATENT_FRAMES}, batch=${STAGE1_BATCH_SIZE}, epochs=${STAGE1_EPOCHS}"
"${TORCHRUN[@]}" "${COMMON_ARGS[@]}" \
  --latent_frames "${STAGE1_LATENT_FRAMES}" \
  --batch_size "${STAGE1_BATCH_SIZE}" \
  --max_epochs "${STAGE1_EPOCHS}" \
  --lr 1e-4 \
  --wandb_name "${RUN_NAME}_stage1_lf${STAGE1_LATENT_FRAMES}" \
  --output_dir "${OUTPUT_DIR}/stage1_lf${STAGE1_LATENT_FRAMES}"

STAGE1_CKPT="${OUTPUT_DIR}/stage1_lf${STAGE1_LATENT_FRAMES}/last.pt"

echo "Stage 2/4: latent_frames=${STAGE2_LATENT_FRAMES}, batch=${STAGE2_BATCH_SIZE}, epochs=${STAGE2_EPOCHS}"
"${TORCHRUN[@]}" "${COMMON_ARGS[@]}" \
  --latent_frames "${STAGE2_LATENT_FRAMES}" \
  --batch_size "${STAGE2_BATCH_SIZE}" \
  --max_epochs "${STAGE2_EPOCHS}" \
  --lr 2e-5 \
  --load_pretrained "${STAGE1_CKPT}" \
  --wandb_name "${RUN_NAME}_stage2_lf${STAGE2_LATENT_FRAMES}" \
  --output_dir "${OUTPUT_DIR}/stage2_lf${STAGE2_LATENT_FRAMES}"

STAGE2_CKPT="${OUTPUT_DIR}/stage2_lf${STAGE2_LATENT_FRAMES}/last.pt"

echo "Stage 3/4: latent_frames=${STAGE3_LATENT_FRAMES}, batch=${STAGE3_BATCH_SIZE}, max_train_steps=${STAGE3_MAX_TRAIN_STEPS}"
"${TORCHRUN[@]}" "${COMMON_ARGS[@]}" \
  --latent_frames "${STAGE3_LATENT_FRAMES}" \
  --batch_size "${STAGE3_BATCH_SIZE}" \
  --max_train_steps "${STAGE3_MAX_TRAIN_STEPS}" \
  --lr 2e-5 \
  --load_pretrained "${STAGE2_CKPT}" \
  --wandb_name "${RUN_NAME}_stage3_lf${STAGE3_LATENT_FRAMES}" \
  --output_dir "${OUTPUT_DIR}/stage3_lf${STAGE3_LATENT_FRAMES}"

STAGE3_CKPT="${OUTPUT_DIR}/stage3_lf${STAGE3_LATENT_FRAMES}/last.pt"

echo "Stage 4/4: latent_frames=${STAGE4_LATENT_FRAMES}, batch=${STAGE4_BATCH_SIZE}, max_train_steps=${STAGE4_MAX_TRAIN_STEPS}"
"${TORCHRUN[@]}" "${COMMON_ARGS[@]}" \
  --latent_frames "${STAGE4_LATENT_FRAMES}" \
  --batch_size "${STAGE4_BATCH_SIZE}" \
  --max_train_steps "${STAGE4_MAX_TRAIN_STEPS}" \
  --lr 2e-5 \
  --load_pretrained "${STAGE3_CKPT}" \
  --wandb_name "${RUN_NAME}_stage4_lf${STAGE4_LATENT_FRAMES}" \
  --output_dir "${OUTPUT_DIR}/stage4_lf${STAGE4_LATENT_FRAMES}"

echo "Done: ${OUTPUT_DIR}/stage4_lf${STAGE4_LATENT_FRAMES}/last.pt"

