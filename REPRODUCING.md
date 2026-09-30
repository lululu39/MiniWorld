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
retains the legacy pip path with evaluation dependencies; `uv.lock` is the reproducible path.

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
| Transformer | Serial block-causal DiT with per-layer history KV; `--transformer_execution parallel` retains the legacy full-clip reference |
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
This repository defaults explicitly to public `LVSM-Experiment/miniworld`.
Verify public authentication without printing keys:

```bash
export WANDB_BASE_URL=https://api.wandb.ai
export WANDB_ENTITY=LVSM-Experiment
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
The normal sampling scripts also use uv. Synthetic checks never create online W&B runs.

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

## PSNR / SSIM / LPIPS evaluation

`miniworld.metrics.VideoMetrics` evaluates matching **RGB `[T,3,H,W]` floating
point tensors in `[-1,1]`**. It computes all three metrics in FP32, even when
called inside BF16 autocast. Frames are processed in small batches to bound
LPIPS memory. Prediction and target are clamped to `[-1,1]`; PSNR and SSIM use
`[0,1]`, and LPIPS uses its native `[-1,1]` input. There is no implicit temporal
alignment, resizing, or truncation in the tensor API.

The conventions are explicit, following the audited LVSM `origin/yibo_dev`
implementation where applicable:

- **PSNR:** compute `-10*log10(MSE)` independently for each RGB frame, then average
  the dB scores. This matches LVSM's per-view averaging; it is not PSNR computed
  from a single pooled video MSE. Exact matches have positive infinity, represented
  as the string `"inf"` in standard JSON rather than nonstandard `Infinity`.
- **SSIM:** `pytorch-msssim==1.0.0`, RGB channel average, 11x11 Gaussian window,
  sigma1.5, data_range1, valid border handling; the same settings as LVSM's
  `model/fancy_losses/ssim_loss.py`, reporting SSIM rather than `1-SSIM`.
- **LPIPS:** official `lpips==0.1.4`, calibrated metric version0.1, pretrained
  **VGG** by default (as in LVSM's `LossComputer`); `--lpips_net alex` is an explicit
  alternative. Results with different backbones should not be mixed. First use
  downloads torchvision backbone weights into the torch hub cache if absent;
  no random-weight fallback is used. See the [official LPIPS implementation](https://github.com/richzhang/PerceptualSimilarity).

These are documented metric choices, not a claim of exact numerical equivalence
to MiniWorld's unpublished evaluation implementation. Scores are absolute, not
the paper's normalized radar-chart ratios. LVSM's `eval/loss` combines pixel L2
and optionally weighted LPIPS; MiniWorld's flow-matching loss is a different
quantity. These metrics do not change the training objective. Optional periodic training
validation is described below.

### Evaluate while sampling a checkpoint

Add `--metrics` to `uv run --no-sync python -m miniworld.sample ...`, or set
`METRICS=1` in either dataset's sampling launcher. For example:

```bash
source /mnt/localssd/dataset/miniworld_paths.env
DATA_ROOT="$RE10K_EVAL_DATA_ROOT" POSE_DIR="$RE10K_EVAL_POSE_DIR" \
CKPT=/path/to/checkpoint.pt VAE_CKPT=/path/to/Wan2.2_VAE.pth \
GPU=0 METRICS=1 SAMPLE_NUM_VIDEOS=50 SEED=42 \
SAMPLE_DIR=outputs/eval/re10k_tas \
bash scripts/sample_re10k.sh
```

For DROID use `DATA_ROOT="$DROID_EVAL_DATA_ROOT"` with `scripts/sample_droid.sh`.
Select an authorized GPU. `LPIPS_NET=vgg` and `METRIC_FRAME_BATCH_SIZE=4` can be
set in either launcher; remaining CLI arguments are forwarded. Append
`--benchmark_no_save` to skip MP4 encoding while still calculating metrics.
PyAV is included in the uv lock for the existing torchvision video writer.

The sampler scores the generated RGB tensor **before lossy MP4 encoding**
against original dataset RGB, not VAE-reconstructed targets. It excludes the
observed prefix: `--history_len` is in latent frames, so the number of excluded
RGB frames is `1 + 4*(history_len-1)` (one frame by default). Only the generated
future is scored. Metric mode rejects custom camera trajectories with no GT and
throughput benchmarking in the same invocation. Dataset decoding errors fail
evaluation instead of silently substituting another sample.

Sampling uses `--seed` (default42), with seed+sample_index per rollout. Use the
same samples, seed, conditioning, resolution, horizon, CFG and sampling steps
for backbone comparisons. This reports one sample per condition, not best-of-N.
No W&B run is created by these evaluation entry points.

### Evaluate saved videos without loading a world model

Create a JSONL manifest with explicitly paired clips; relative paths resolve
against the manifest's directory:

```json
{"id":"clip_001","prediction":"pred/clip_001.mp4","target":"gt/clip_001.mp4"}
{"id":"clip_002","prediction":"pred/clip_002.mp4","target":"gt/clip_002.mp4"}
```

```bash
uv run --no-sync python -m miniworld.evaluate \
  --manifest /path/to/pairs.jsonl --output_dir outputs/eval/saved \
  --device cpu --context_frames 1 --lpips_net vgg
```

Use `CUDA_VISIBLE_DEVICES=0 ... --device cuda` on an authorized GPU for speed.
Optional `--num_frames 253` explicitly selects the first253 frames of both
clips; `--resize_hw 240 320` explicitly resizes both. Without these options,
lengths and dimensions must match. Clips must already correspond to the same
starting time and frame indices; no FPS resampling or alignment search occurs.
The offline path measures decoded/compressed video quality, so its scores can
differ from the online path that measures pre-encoding RGB.

### Reports and verification

Both paths write to the chosen output directory:

- `metrics_per_video.jsonl`: sample IDs, means, every scored frame's index and
  PSNR/SSIM/LPIPS; online reports also include source paths/frame IDs and seeds.
- `metrics_summary.json`: equal-weight video means, frame-index curves with
  contributing-video counts, metric/package settings and run provenance.

Each video's score is the mean over its evaluated frames; the dataset score
is the mean over videos, so a longer video does not receive more weight.
Reusing an output directory overwrites its metric reports. Keep separate
output directories for separate checkpoints and protocols.

Focused tests (including actual pretrained LPIPS and MP4 read/write):

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 uv run --no-sync python -m pytest -q tests/test_metrics.py
```

Verified on 2026-09-30: all8 metric tests pass, including a sampler integration
check with a simulated rollout and RGB-prefix exclusion. GPU0 FP32 metric
execution inside BF16 autocast was also checked on real DROID/RE10K RGB clips
with synthetic pixel noise; results are in ignored `outputs/metrics_smoke/`.
This validates the metric pipeline, not a trained model's generation quality.


## Periodic held-out evaluation and W&B

`--eval_every N` runs quality evaluation after every N optimizer updates using
EMA weights. Default0 disables it; `EVAL_EVERY=N` exposes it in both curriculum
launchers. Evaluation runs independently of W&B and `image_log_every`, so
`--no-wandb` still produces JSON reports. It evaluates complete autoregressive
rollouts against held-out RGB targets, excluding the observed prefix.

```bash
source /mnt/localssd/dataset/miniworld_paths.env
CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 MODEL=B BACKBONE=transformer \
RUN_NAME=re10k_B_transformer_s42_eval8 \
DATA_ROOT="$RE10K_TRAIN_DATA_ROOT" POSE_DIR="$RE10K_TRAIN_POSE_DIR" \
EVAL_DATA_ROOT="$RE10K_EVAL_DATA_ROOT" EVAL_POSE_DIR="$RE10K_EVAL_POSE_DIR" \
EVAL_EVERY=1000 EVAL_NUM_VIDEOS=8 \
EVAL_FILTER_CACHE_DIR="$RE10K_FILTER_CACHE_DIR" \
VAE_CKPT=/path/to/Wan2.2_VAE.pth \
OUTPUT_DIR=/path/to/experiments/re10k_B_transformer_s42_eval8 \
STAGE1_BATCH_SIZE=1 STAGE2_BATCH_SIZE=1 STAGE3_BATCH_SIZE=1 STAGE4_BATCH_SIZE=1 \
bash scripts/train_re10k.sh
```

For DROID set `DATA_ROOT="$DROID_TRAIN_DATA_ROOT"`,
`EVAL_DATA_ROOT="$DROID_EVAL_DATA_ROOT"` and use `train_droid.sh` without pose
paths. Use GPUs assigned to the experiment; these are launch examples, not
experiments started by this change.

| CLI | Launcher environment | Default |
| --- | --- | --- |
| `--eval_every` | `EVAL_EVERY` | 0 (disabled) |
| `--eval_num_videos` | `EVAL_NUM_VIDEOS` | 8 globally |
| `--eval_latent_frames` | `EVAL_LATENT_FRAMES` | 0: current stage length |
| `--eval_seed` | `EVAL_SEED` | 42 |
| `--eval_sampling_steps` | `EVAL_SAMPLING_STEPS` | 100 |
| `--eval_cfg_scale` | `EVAL_CFG_SCALE` | 2 |
| `--eval_ardiff_step` | `EVAL_ARDIFF_STEP` | 5 |
| `--eval_history_len` | pass as CLI argument | 1 latent frame |
| `--eval_lpips_net` | pass as CLI argument | vgg |
| `--eval_metric_frame_batch_size` | pass as CLI argument | 4 |

Eval paths must be explicit and the selected sample IDs must not overlap the
training set. The first N eligible samples are fixed; too few samples is an
error. Each sample uses eval_seed+sample_index for view selection and generation
noise, unchanged across checkpoints. Python, NumPy and PyTorch RNG state and
EMA modes/sampler settings are restored afterward. The sampling cache/in-flight
window fits the current trained window; an explicit evaluation horizon can be
longer. Default stage-dependent horizons produce different protocols between
stages, so compare checkpoints at the same horizon or set a fixed override.

For DDP, rank0 constructs the evaluation dataset once, then broadcasts it.
Ranks evaluate disjoint indices `rank, rank+world_size, ...`, without duplicate
padding when N is not divisible by GPU count. Results and errors are gathered
on all ranks; rank0 alone writes reports and W&B. Training resumes only after
the report completes. Evaluation and reporting failures stop all ranks rather
than silently skipping evaluation. JSON reports are saved under
`OUTPUT_DIR/eval/step_XXXXXXXX/`, including per-video scores, frame curves,
protocol, seed, source IDs and step. Evaluation duration is excluded from the
next training throughput interval.

W&B defaults are fixed to **https://wandb.ai/LVSM-Experiment/miniworld**.
The project was verified accessible using the existing public credential.
An inherited private-host API key is not sent to the public host: online mode
uses the existing `api.wandb.ai` netrc entry in that case, or fails with a setup
error. Global credentials and settings are not modified.

Give each experiment a descriptive `RUN_NAME`. All four stages now share one
run ID stored in `OUTPUT_DIR/wandb_run.json`, and use `resume=must` after stage1.
Direct CLI uses `--wandb_run_file` and `--curriculum_stage`. A fresh stage1 launch
refuses an existing identity file unless `--resume` is explicit; new experiments
must use new output roots. Multi-node launchers require the same RUN_NAME across
nodes. Direct standalone CLI runs without a shared identity file still create a
fresh uniquely named run.

The `train_step` chart axis is cumulative. Checkpoints keep the old stage-local
`global_step` for training budgets and additionally store `total_train_steps`
and `curriculum_stage`. Later stages infer their offset from the preceding
checkpoint; legacy stage1 checkpoints are supported. Scalars and videos also
record `curriculum/stage`, `curriculum/stage_step`, and
`curriculum/latent_frames`. Per-stage configs are stored under separate
`curriculum_stage_N` W&B config keys, preserving stage1's original config.
Stage directories, checkpoints and per-evaluation JSON files remain separate.
`START_STAGE=2` (or3/4) skips already-completed stages while loading the preceding
checkpoint and continuing the shared run. Stage budgets still count local steps.

Online tracking resumes one remote run between processes. In offline mode,
W&B does not implement server-side resume; each stage writes a separate local
session even though the recorded run ID is shared. Use online mode for the
continuous curriculum run described here.

The launchers ignore unrelated inherited W&B entity/project defaults. To
explicitly override their target use `MINIWORLD_WANDB_ENTITY` /
`MINIWORLD_WANDB_PROJECT`, or the CLI flags `--wandb_entity` / `--wandb_project`.
Each stage writes a copy of `wandb_run.json` with the shared ID, name, URL and mode. Scalars
and training videos all use the explicit `train_step` chart axis; eval PSNR/SSIM
also track their maximum and LPIPS its minimum in W&B summaries. The existing
`train/gen_video` remains a training-sample visualization, separate from these
held-out evaluation scores. W&B initialization failures no longer fall back
silently. `--wandb_mode offline` needs no public authentication and does not
create an online run; `--no-wandb` disables tracking entirely.

Verified with a two-update synthetic training loop, actual offline W&B logging,
and two-process CPU/Gloo evaluation with uneven sample counts and injected
rank failure. Together with backbone/metric checks, 40 tests pass. This does not
constitute a full GPU/DDP real-data training or online W&B run.


## Serial execution and chunk4 research protocol

User update2026-09-30: new Transformer, RTransformer and TaS comparisons all use
serial chunk processing with **4 latent frames =1200 tokens per chunk** at
240x320 resolution. Transformer uses the same chunk-outer/layer-inner loop as
the other two models while retaining its own layer's historical KV. It has no
extra parameters, memory bank or cross-layer KV blending. Gradients remain
attached through all earlier chunks; one optimizer update follows the complete
video batch. Optional `--transformer_execution parallel` preserves the original
full-clip masked Transformer reference.

With the same chunk size and weights, the two Transformer execution modes are
mathematically equivalent. CPU tests compare outputs, input gradients and every
parameter gradient for action/pose conditions, full/partial chunks and activation
checkpointing. Finite-precision CUDA kernels can introduce rounding differences.
The default trainer remains eager with activation checkpointing; fullgraph
compilation is validated in the smoke script, not implicitly enabled for runs.

The official [paper](https://arxiv.org/html/2608.01127#S4.SS1) and
[upstream launcher](https://github.com/Zhao-Yian/MiniWorld/blob/master/scripts/train_re10k.sh)
use chunk2. No official large-chunk preset was found in those sources. Chunk4 is
our research setting, not an established convergence result from the paper.

| Stage | Latent / RGB frames | Chunks/video | Batch/GPU | Update budget on downloaded RE10K |
| --- | --- | --- | --- | --- |
| 1 | 8 / 29 | 2 | 8 | 97200 steps (100 epochs) |
| 2 | 16 / 61 | 4 | 4 | 86950 steps (50 epochs) |
| 3 | 32 / 125 | 8 | 1 | 30000 steps |
| 4 | 64 / 253 | 16 | 1 | 30000 steps |

Total244150 optimizer updates on8 GPUs. Stage1 has62244 eligible training clips;
all later stage budgets and the default learning rates are retained. Two complete
chunks in stage1 allow one effective recurrent memory transition and leave room
for one committed-history chunk plus one in-flight chunk during evaluation.
Increasing chunk size changes block-causal visibility, noise grouping, automatic
timestep shift and update frequency; it is not just a performance switch.
Short-video convergence under the new protocol needs real training evidence.
TaS defaults remain256 slots with4 latent frames of preceding raw KV; matching
chunk size does not change the architectural distinction in retained history.

Launchers accept `DF_CHUNK_SIZE` (default4) and `TRANSFORMER_EXECUTION` (default
serial). They start from8 latent frames by default. Explicitly set chunk2,
first-stage6 and parallel Transformer to reproduce the original baseline, or
use its recorded immutable checkout. New recurrent training rejects clips with
only one chunk because cross-chunk state parameters would receive no outer-loss
gradient. Checkpoint loading rejects a recorded chunk-size mismatch.

Sampling automatically restores the recorded chunk size and Transformer execution
mode. Historical checkpoints without execution metadata use parallel Transformer
and legacy chunk2. The default streaming cache/in-flight counts adapt to the
checkpoint's trained window and chunk size:64 latent frames with chunk4 use
12 cached +4 in-flight chunks; stage1 uses1+1. Explicit inference overrides remain
available but must fit the trained window. Previously published checkpoints are
not silently reinterpreted as chunk4 models.

Validation:55 relevant tests pass; all three serial backbones pass BF16 forward,
backward, optimizer update and fullgraph Inductor compilation at chunk4. The
synthetic compile results are in ignored `outputs/serial_chunk4_smoke.json`.
No convergence or throughput equivalence is inferred from these checks.


## Live curriculum handoff and reconstruction correction (2026-09-30)

The already-running serial chunk4 stage1 remains on its original immutable
checkout and keeps training uninterrupted. Its existing W&B run is adopted as
the shared curriculum run. A separate controller pauses only the old shell
supervisor (not its GPU trainer), waits for the stage1 foreground process to
exit0 and its final97200-step checkpoint to exist, then starts stages2-4 from a
new immutable checkout. The controller never launches later stages after a
failed/cancelled stage1. Its source is `scripts/continue_after_stage.py`; runtime
configuration and state are recorded in the experiment output root. The old
source is not patched in place and no checkpoint rollback is used.

This handoff is specific to the active run; new experiments use the shared-run
launchers directly. The shared run keeps stage1's existing train_step history;
stage2 begins at97201, stage3 at184151, and stage4 at214151. Stage1's old logs do
not retroactively gain the new curriculum fields; its original config and an
explicit stage1 record identify that portion. Score discontinuities at stage
boundaries may reflect the changed video length, not just model improvement.

A separate visualization-only bug was found in the upstream denoiser:
`z=(1-t)*x+t*noise` and `v_target=x-noise` imply `x_hat=z+t*v_pred`.
The old `z+(1-t)*v_pred` line was incorrect. The corrected line affects only the
single-step `train/recon_video` branch. Training loss, model targets, iterative
rollout and held-out PSNR/SSIM/LPIPS do not use this reconstruction expression.
A test using an oracle velocity checks exact clean-latent recovery at multiple
noise levels. Captions now label the value as maximum t and state that observed
context is copied. The still-running old stage1 keeps its legacy visualization;
stages2-4 and new launches use the fix. Historical recon videos are not relabeled
as corrected results. The W&B metadata records this version boundary.

Validation includes two consecutive synthetic training stages sharing a mocked
W&B ID and continuous loss/evaluation steps, legacy-checkpoint offsets, the
oracle reconstruction test, and real OS-process tests of successful and failed
stage handoffs. No synthetic online runs are created.

### Corrected fresh-run restart

The user subsequently requested a fresh run with all fixes active from step0,
superseding the live-handoff plan above. The old trainer and its handoff controller
were stopped; no later stages will be attached to that old run. The replacement
uses a fresh identity/output root, corrected reconstruction, and shared-run
curriculum tracking from its first step. Old checkpoints/media remain unchanged.

`train/recon_t_max` now records the same maximum timestep shown in the recon
caption. Full-generation media records `train/gen_effective_sampling_steps`,
and periodic evaluation records `eval/effective_sampling_steps`; per-video
reports include the sampler's metadata. The existing pipeline caps effective
steps at `inflight_chunks * ar_step`, so a configured100-step limit is not always
100 executed updates. In the current stage1 eval (one in-flight chunk, stride5),
the effective count is5. This change makes the existing protocol explicit;
it does not change the scheduler, noise distribution or optimizer objective.


### Flow Matching audit and asynchronous time-label fix

See [the2026-09-30 formula audit](experiments/flow_matching_audit_20260930.md)
for the derivation, primary references and test evidence. No additional training
formula error was found. A separate inference bug was fixed: a completed chunk
that stays in a multi-chunk in-flight window must retain t=0, not the penultimate
positive timestep. The corrected schedule indexes its actual prior state.
There are78 passing tests, including analytical per-step path checks and exact
single-in-flight output equivalence. No noise/CFG/EMA hyperparameters changed.

### Common larger-batch B comparison

The prepared RE10K profile uses per-GPU batches32/16/8/2 across Transformer,
RTransformer and TaS, with common per-stage LRs and preserved video exposure.
See [the capacity record and launch instructions](experiments/re10k_B_shared_batch_20260930.md).
The existing launchers also accept STAGE1_LR through STAGE4_LR; their defaults
remain1e-4/2e-5/2e-5/2e-5. Capacity checks use the actual VAE/pose pipeline and
singleton DDP, not an eight-rank throughput or convergence benchmark.
