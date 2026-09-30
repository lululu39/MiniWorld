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

## Verified startup

- Launched2026-09-30 00:40:20 UTC; curriculum supervisor PID1093287.
- Fixed source checkout: output root's `source/`, detached at
  `1d18919ddd02a0b01a725d3dbbbe7ff2ec398bd6`, with its own `uv sync --locked`
  environment. Later edits to the main workspace do not affect this run.
- VAE source revision: `Wan-AI/Wan2.2-TI2V-5B@921dbaf3f1674a56f47e83fb80a34bac8a8f203e`;
  download LFS SHA256: `20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36`.
  VAE has704,688,668 frozen parameters, separate from the112.43M trainable DiT.
- Stage1 training samples after length/pose filtering:63,681; selected eval
  samples:8. Eligible train counts for21/61/125/253 frames:
  63,681 / 55,660 / 29,530 / 9,785. Eligible test counts:
  6,930 / 6,098 / 3,205 / 1,099.
- All8 GPUs showed active training (95–100% at the startup observation),
  approximately14–15GiB allocated by the processes per card.
- Public W&B readback at00:42:12 UTC confirmed `running`, train_step100,
  train/loss1.5542407, and1.6181 step/s. Subsequent local log reached step140
  with finite loss1.256290. These are startup observations, not final results.
- Stage1 run: https://wandb.ai/LVSM-Experiment/miniworld/runs/ec2c3d7e57be439294aec99ee285dc92
- `startup_verified.json` in the output root stores the W&B readback. First
  periodic quality evaluation is scheduled at step1000; no quality result was
  claimed at startup.

## Superseded by the serial chunk4 experiment

On2026-09-30 at01:34 UTC the user-selected protocol changed to serial execution
and4 latent frames per chunk. This run was intentionally interrupted, not
completed to its original budget. Last logged step5140; last completed epoch
checkpoint4975. The original source snapshot, checkpoints, W&B run and evaluation
reports through step5000 remain intact. See the output root's
`intentional_stop.json` and `experiments/re10k_B_transformer_serial_chunk4_20260930.md`
for the fresh replacement run; no old result has been relabeled as chunk4.
