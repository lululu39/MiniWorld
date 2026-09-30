# RE10K Transformer-B baseline, 2026-09-30

User authorized releasing the eight existing GPU reservation workers and starting
this baseline on GPUs0-7 (8x H10080GB). The reservation parent781657 and its
workers781685-781692 were stopped; no unrelated processes were stopped.

- Run group: `re10k_B_transformer_s42_8xh100_20260930T003734Z`.
- Launcher: `experiments/re10k_B_transformer_20260930.sh`.
- Initial implementation revision: `dc0c01f`; the launch revision includes this
  tracked launcher/record and is written to each stage's `config.json`.
- Dependency lock: repository `uv.lock`; its SHA256 is recorded by the trainer.
- Model: official `B`, Transformer, depth12, width768, heads12, patch1.
  RE10K pose-conditioned trainable parameter count is112,432,176 (the paper's
  architecture label is approximately0.12B). No parameter-count adjustment.
- Attention: dense block-causal across the full training clip, bidirectional
  within each two-latent-frame chunk; no TaS/SWA. Streaming uses rolling KV.
- Data: `/mnt/localssd/dataset/re10k/training_256` + `training_poses`;
  held-out: `test_256` + `test_poses`, original DFoT split. Dataset source is
  `kiwhansong/DFoT@0959defb4c4fe010f84791d732cc978ad7d49fef`.
- Resolution240x320; Wan2.2 latent spatial size15x20, 48 channels, pose dim720.
- VAE: `/mnt/localssd/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth`, official
  `Wan-AI/Wan2.2-TI2V-5B` download. Frozen; excluded from trainable count.
- PrecisionBF16, Muon, seed42, condition dropout0.1, chunk size2.
- Default curriculum is unchanged:

| Stage | Latent / RGB frames | Batch/GPU / global | Budget | LR |
| --- | --- | --- | --- | --- |
| 1 | 6 / 21 | 8 / 64 | 100 epochs | 1e-4 |
| 2 | 16 / 61 | 4 / 32 | 50 epochs | 2e-5 |
| 3 | 32 / 125 | 1 / 8 | 30000 steps | 2e-5 |
| 4 | 64 / 253 | 1 / 8 | 30000 steps | 2e-5 |

Evaluation: every1000 optimizer updates, 8 global held-out videos, EMA,
stage-matched horizon, seed42, one observed latent frame, 100 sampling steps,
CFG2, AR stride5, VGG-LPIPS. Scores exclude the observed RGB frame. Training
reconstruction/generation videos also use the existing1000-step interval.

Output root:
`/mnt/localssd/experiments/yibo/miniworld/re10k_B_transformer_s42_8xh100_20260930T003734Z`.
Each stage stores checkpoints, config, W&B run metadata and evaluation reports.
The top-level `train.log` records the curriculum process, and `launcher.json`
records its PID and launch revision. W&B project:
https://wandb.ai/LVSM-Experiment/miniworld (stage names append `_stageN_lfM`).

The length-filter cache was derived from the downloaded DFoT frame-timestamp
metadata for21/61/125/253 RGB frames, after matching file inventories and
checking32 actual video frame counts per split. Cache provenance and eligible
counts are in `re10k/filter_cache/metadata_cache_provenance.json`.
