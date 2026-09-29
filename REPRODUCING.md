# Reproducing the video backbone port

## Environment

From this repository root:

```bash
uv sync --locked
uv run --no-sync python -m miniworld.train --help
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 uv run --no-sync python -m pytest -q tests/test_backbones.py
# First select an idle or explicitly authorized GPU; 0 is an example.
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=2 \
  uv run --no-sync python scripts/check_backbones.py --compile
```

The lock targets Linux x86_64 and Python 3.11. Tested with Python 3.11.15,
uv 0.11.19, Torch 2.8.0+cu128, torchvision 0.23.0, Triton 3.4.0,
FlashAttention 2.8.3 and an H100. The official FlashAttention wheel is pinned
by URL to match Python/Torch/CUDA/C++ ABI; no local CUDA extension build is needed.
Use `uv run --no-sync` after syncing to avoid reinstalling a wheel whose filename
and distribution metadata use different version strings. `requirements.txt`
is the original upstream installation list; `uv.lock` is the reproducible path.

Known upstream packaging issue: decord 0.6.0's public `py3-none` wheel contains
an internal `cp36-cp36m` WHEEL tag. `uv pip check` reports that one platform
mismatch. We retain the upstream artifact without rewriting its metadata;
imports and an actual generated-MP4 decode are checked separately. All other
installed dependency requirements pass. CPU checks cover small latent models;
production training still requires CUDA, datasets and the Wan2.2 VAE checkpoint.

## Model choices

`--wm_model` still chooses B/L/0.5B/1B/3B. `--backbone` independently selects
`transformer` (default), `rtransformer` or `tas`. Existing Transformer parameter
names and checkpoint layout are preserved. No neighboring repo is needed at runtime.

| Backbone | Video attention / state |
| --- | --- |
| Transformer | Existing block-causal DiT, bidirectional within latent chunks; per-layer history KV |
| RTransformer | Same DiT; chunks traverse all layers sequentially. Older history uses final-layer KV; the immediately preceding chunk blends own/final-layer KV with a learned per-head sigmoid initialized at 0.5 |
| TaS | Same DiT plus serial slot reads; bounded preceding raw-video history plus the current bidirectional chunk; fixed learned token banks updated after each layer sweep |

TaS defaults: 256 slots of backbone width, one shared bank, 4 latent frames of
raw history, slot identities, sigmoid EMA initialized at 0.1, final-layer write
source, and assigned writing. Video defaults are deliberately not the LLM's
4096/8192 token settings. All five features can be independently disabled via
`--no-slot_embed`, `--no-gated_ema`, `--no-write_from_last`,
`--no-state_sharing`, `--no-assigned_write`. Memory capacity and raw-history
length use `--num_memory_tokens` and `--memory_window_frames` (0 means no raw
history). Memory/history is per sample and resets for each training clip.

```text
z = x + AdaLN_attention_gate * video_attention(modulated_norm(x))
h = memory_query_norm(z)
y = z + AdaLN_attention_gate * memory_read(h, old_bank)
out = y + AdaLN_mlp_gate * video_mlp(modulated_norm(y))

# After all layers, using their old banks:
source = h_last_layer                  # or each owner's h when WFL is disabled
u = assigned_write(norm(old_bank) + slot_identity, source)
candidate = u + memory_mlp(norm(u))
g = sigmoid(linear(concat(rms(old_bank), rms(u))))
new_bank = (1-g) * old_bank + g * candidate
```

Identity enters read keys and write queries only; memory attention has no RoPE.
Assigned writing softmaxes each input token over slots, then normalizes each
slot's received mass over input tokens. Softmax, mass normalization and EMA
arithmetic use FP32. Persistent banks follow projected token dtype (BF16 during
CUDA autocast; FP32 in CPU tests). Disabling EMA follows LVSM's additive control:
`temporary = old + u; new = temporary + memory_mlp(norm(temporary))`.
This is not the separate LLM fixed-additive-0.05 experiment.

Adaptations for video: retain MiniWorld's 3D RoPE, actions/rays, AdaLN and flow
loss. Memory reads use the existing AdaLN attention gate so DiT's zero-residual
initialization remains intact. The raw-video history window is counted in whole
latent frames before the current chunk, not a token-causal text SWA mask.
All earlier training states remain attached to autograd. The implementation is
a PyTorch/SDPA reference port; it does not port specialized Triton assigned-write
kernels or claim memory/throughput parity with the source implementations.

Streaming returns functional `VideoState` snapshots: tentative denoising can
simulate successive in-flight chunks, but the caller discards its candidate
state. Only completed chunks recomputed at t=0 are committed. CFG branches
have separate banks and KV. Cache eviction retains the bank, removes old raw
KV, and shifts temporal RoPE while preserving the sink. RTransformer retains
raw KV up to the sampler's cache budget; it is not fixed-size recurrent memory.
TaS reads a bounded raw window; its training implementation retains per-layer
raw KV tensors for the clip, while inference uses the sampler's rolling budget.

## Source audit

- MiniWorld baseline: `e484206`, `miniworld/miniworld.py`, `denoiser.py`,
  training/sampling entry points and DROID/RE10K launchers.
- LLM `yibo-dev`: `4a399dc9cb97d9a67354af576af79f8f57e086b2`.
  Read `AGENTS.md`, `docs/tas.md`, `flame/custom_models/transformer/`,
  `recurrent_transformer_gated_attention_aggregate_kv_v254_reusekv/`,
  `tas/modeling_tas.py`, `tas/rtransformer_kv.py`, `tas/lact_writer.py`,
  and `recurrent_lact_ref_v4/modeling_recurrent_lact.py`.
- LVSM's branch is actually `origin/yibo_dev`, not `yibo-dev`:
  `f34753d69aeb0dba495aef738a38d2ad6f43eeff`. Read its `AGENTS.md`,
  `CLAUDE.md`, `docs/lact_memory_tokens.md`, `memory_block_tokens.py`,
  `transformer_diffkvcache_mha.py`, and `memory_block_v2_diffusion_ref.py`.
  LVSM calls TaS `lact_memory_tokens`; these are token banks, not LaCT matrices.

LaCT reference updates three per-head SwiGLU fast-weight matrices from projected
K/V with token learning rates, optional momentum and Newton–Schulz/Muon,
FP32 master updates and row-norm restoration. TaS instead learns slot read/write
operations solely through the outer loss. LaCT was audited for comparison;
this port adds the requested RTransformer and TaS, not a fourth LaCT backbone.

## Launching real experiments

Supply existing datasets and VAE weights; no downloads or real-data experiments
are performed by the synthetic check. Commit and push the intended code first.
Set the public W&B entity to the organization you intend to use (it is not
inherited from LLM or LVSM), then verify authentication without printing keys:

```bash
export WANDB_BASE_URL=https://api.wandb.ai
export WANDB_ENTITY=YOUR_ENTITY
export WANDB_PROJECT=miniworld
# uv run --no-sync wandb login --host https://api.wandb.ai

CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 BACKBONE=tas MODEL=B \
NUM_MEMORY_TOKENS=256 MEMORY_WINDOW_FRAMES=4 SEED=42 \
DATA_ROOT=/path/to/re10k/training_256 POSE_DIR=/path/to/re10k/training_poses \
VAE_CKPT=/path/to/Wan2.2_VAE.pth OUTPUT_DIR=/path/to/experiments/re10k_tas_B \
STAGE1_BATCH_SIZE=1 STAGE2_BATCH_SIZE=1 STAGE3_BATCH_SIZE=1 STAGE4_BATCH_SIZE=1 \
bash scripts/train_re10k.sh
```

Use `BACKBONE=rtransformer` or `transformer` for controls. DROID uses
`scripts/train_droid.sh`, the LeRobot data root and no pose directory. Launchers
use uv and forward additional CLI flags to every curriculum stage. Model sizes
are not parameter-matched: the new memory parameters are additional capacity.
Keep resolution, data split, frame/chunk lengths, conditioning, batch, seed and
optimizer fixed for meaningful comparisons.

The trainer records a config snapshot, Git revision and lock SHA256; checkpoints
include full arguments plus backbone metadata. Sampling restores the backbone
configuration from the checkpoint. Older metadata-free backbone selections
default to Transformer. Resume rejects conflicting recorded backbone settings.
The normal sampling scripts also use uv. Synthetic checks never create W&B runs.

## Validation

The focused suite checks action/pose paths, future-chunk isolation, full vs
chunkwise cached equivalence, early-token and memory-writer gradients,
checkpoint recomputation, independent feature switches, assigned-write math,
non-mutating tentative calls and the real CFG streaming sampler with context
prefill, partial context, rolling eviction and persistent sink.

`outputs/backbone_smoke.json` contains measured synthetic GPU results, including
an eager and fullgraph-Inductor forward/backward comparison and an AdamW update
for each backbone. The smoke models are 2 layers, width64, 2 heads, 6 latent
frames of 2x2 tokens, batch1, with 8 TaS slots. Tests deliberately enable
nonzero output/AdaLN weights so zero-init cannot hide causal/gradient errors.
This establishes executability, not dataset convergence, large-model capacity,
real RGB generation quality or distributed-training performance.

Validated on 2026-09-29: **23 tests passed**; both CLI entry points, Python
byte compilation and shell syntax passed. The actual diffusion-forcing loss
and long latent rollout paths run for all three backbones. MP4 decode returned
four 32x32 RGB frames with expected intensities. `uv sync --locked` succeeds;
`uv pip check` retains only the documented decord wheel-tag warning.

| Synthetic GPU model | Flow loss | Eager backward | Inductor fullgraph backward | AdamW update |
| --- | ---: | --- | --- | --- |
| Transformer | 1.606227 | Pass | Pass | Pass |
| RTransformer | 2.262112 | Pass | Pass | Pass |
| TaS | 2.345978 | Pass | Pass | Pass |

These randomly initialized checks do not compare model quality. No dataset/VAE
training, W&B baseline experiment or distributed run was launched.
