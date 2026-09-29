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
- Transformer is the existing baseline. RTransformer uses top-layer historical
  KV with per-head own/top blending for the preceding chunk. TaS uses fixed
  token banks; it is distinct from LaCT fast-weight/inner-optimizer models.
- TaS's slot identity, sigmoid EMA, write-from-last, state sharing and assigned
  write switches remain independently configurable. Read old banks throughout
  the layer sweep, write afterward, keep training gradients across chunks.
  Persistent inference state is accepted only on clean chunk commit; tentative
  denoising calls must never mutate committed state or cross CFG branches.
- Record the command/config, Git revision, dependency lock, seed, precision,
  backbone, memory configuration, video/chunk lengths, batch size, GPU allocation,
  checkpoint and W&B URL in a run record under `experiments/` for real experiments.
- Public W&B host: `https://api.wandb.ai`; project: `miniworld`. Set
  `WANDB_ENTITY` explicitly for real runs; do not inherit another project's
  organization or silently redirect credentials. Keep keys in private netrc or
  environment variables. Do not print them or copy them into run records.
- Use unique run/output names for comparisons. Read historical runs without
  mutating them. Synthetic checks use no W&B run; real training logs on rank 0.
  Verify the intended W&B run actually started (the inherited trainer tolerates
  logging failures). Do not report unlogged work as a tracked experiment.
