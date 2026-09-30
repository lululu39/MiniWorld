# MiniWorld AR video generation

Work on `main`. This repository trains action-conditioned DROID and
pose-conditioned RealEstate10K video diffusion models. Keep project instructions
local; do not copy unrelated LLM evaluations, LVSM datasets or machine setup.

## Git and credentials

- Commit and push completed changes unless the user asks otherwise. Push the
  exact revision before launching training, including remote experiments.
- Use this repository's ignored `tokens.txt` for GitHub authentication. Keep
  authentication non-interactive with a temporary private `GIT_ASKPASS` helper
  that reads the token at runtime; disable credential helpers and remove the
  temporary helper afterward. Never put credentials in URLs or Git config.
- Never print or commit tokens, `.env`, `.netrc`, or private keys. Stage explicit
  files and preserve unrelated edits and historical experiment configurations.

## Environment and verification

- Use the repo-local uv environment: Python 3.11, `uv sync --locked`, then
  `uv run --no-sync ...`. Commit `pyproject.toml` and `uv.lock` together when
  changing dependencies. Do not initialize from neighboring setup scripts.
- Reuse the main checkout's `.venv` for additional worktrees via
  `UV_PROJECT_ENVIRONMENT`; do not create a duplicate environment unless different
  dependencies require it. The currently running fixed-revision job predates this preference.
- Torch and the official FlashAttention wheel must have matching Python,
  CUDA, Torch and C++ ABI versions. See `REPRODUCING.md` for tested versions.
- Verify changed behavior with focused tests and a small forward/backward/update
  check. Verify `torch.compile` when changing model kernels. Distinguish synthetic
  checks from real-data training, reproduction, convergence and throughput claims.

## Experiments and W&B

- Check GPU occupancy before every launch. Set `CUDA_VISIBLE_DEVICES` explicitly;
  use idle GPUs or GPUs specifically authorized by the user. Never stop unrelated
  processes. A card used in an example is not a permanent reservation.
- Launch via the uv environment and the existing dataset-specific scripts. Set
  dataset, pose (RE10K), VAE checkpoint and output paths explicitly. Keep large
  checkpoints outside Git, preferably outside this repository.
- Preserve the MiniWorld recipe unless an experiment explicitly changes it:
  latent video chunks, block-causal attention, 3D RoPE, action/pose conditioning,
  rectified-flow objective, and separate conditional/unconditional streaming state.
- All new comparisons use serial chunk execution. Transformer retains its own
  per-layer historical KV; `--transformer_execution parallel` is the legacy
  equivalent reference. Use the same chunk size across all three backbones.
- Current research default: 4 latent frames/chunk (1200 tokens at240x320),
  curriculum8/16/32/64 latent frames. This is a user-selected change from the
  official chunk2, 6/16/32/64 recipe; do not relabel historical experiments.
  Keep at least two chunks when training state-writing models.
- RTransformer uses top-layer historical
  KV with per-head own/top blending for the preceding chunk. TaS uses fixed
  token banks; it is distinct from LaCT fast-weight/inner-optimizer models.
- TaS's slot identity, sigmoid EMA, write-from-last, state sharing and assigned
  write switches remain independently configurable. Read old banks throughout
  the layer sweep, write afterward, keep training gradients across chunks.
  TaS attention uses only the current chunk's raw KV. Only memory banks cross
  chunk boundaries; never retain previous-chunk/sliding-window/raw-history KV
  for TaS. Default memory is two chunks of tokens:2400 tokens at the current
  1200-token chunk size, in one shared bank of backbone width. Enable WFL,
  state sharing, assigned write, sigmoid EMA and slot identity by default.
  Persistent inference state is accepted only on clean chunk commit; tentative
  denoising calls must never mutate committed state or cross CFG branches.
- Record the command/config, Git revision, dependency lock, seed, precision,
  backbone, memory configuration, video/chunk lengths, batch size, GPU allocation,
  checkpoint and W&B URL in a run record under `experiments/` for real experiments.
- Public W&B defaults: `WANDB_BASE_URL=https://api.wandb.ai`, entity
  `LVSM-Experiment`, project `miniworld`:
  https://wandb.ai/LVSM-Experiment/miniworld. Keep API keys in private netrc or
  environment variables; never print them or copy them into run records.
- Include the backbone/model explicitly in every run name (for example
  `re10k_B_transformer_serial_c4_s42_...`, not just `re10k_B_serial_...`).
- Define a descriptive run name for each experiment (`RUN_NAME` in launchers,
  `--wandb_name` in the CLI). New experiments use a separate W&B run for every
  model/stage, with explicit model/stage names and stage-local identity files.
  The old shared-run mode is supported only when explicitly selected.
  Log cumulative `train_step` plus stage/local-step/latent-frame fields. Keep
  stage checkpoints in separate directories. Fresh experiments use fresh run
  identities; never attach an unrelated experiment to a historical run.
- Enable periodic held-out EMA evaluation with `--eval_every` and explicit eval
  data/pose paths. Use fixed sample IDs, noise seeds and protocol when comparing
  checkpoints. Keep PSNR/SSIM/LPIPS in `eval/*` and use `train_step` as the W&B
  chart axis. Evaluation must preserve training RNG state and synchronize ranks.
- Synthetic checks use offline W&B or mocks; never populate the real project
  with test runs. W&B startup failures stop training unless `--no-wandb` was
  explicitly selected. Read historical runs without mutating them.
- Keep testing minimal and specific to changed behavior/configuration. Once
  essential checks pass, launch the authorized experiment instead of adding
  broad or repeated checks and speculative edge-case tests.
- During experiment preparation, hold the GPUs authorized for that experiment
  with scripts/prepare_stage.py. Record its PID and release the relevant reservation
  before a check or training launch. Never reserve unrelated/occupied GPUs.
