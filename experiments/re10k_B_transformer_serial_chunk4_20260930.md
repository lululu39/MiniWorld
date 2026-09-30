# RE10K B, serial Transformer, chunk4 — 2026-09-30

The user requested serial execution for Transformer/RTransformer/TaS and selected
4 latent frames (1200 tokens) per chunk. This run starts from scratch with seed42;
it does not warm-start from the earlier parallel chunk2 experiment. The earlier
run is intentionally stopped for this protocol change and its artifacts remain.

- Run/group: `re10k_B_transformer_serial_c4_s42_8xh100_20260930T012919Z`.
- Launcher: `experiments/re10k_B_transformer_serial_chunk4_20260930.sh`.
- Model: Transformer-B, serial per-layer historical KV, depth12, width768,
  12 heads,112432176 trainable parameters. No recurrent cross-layer KV or TaS.
- GPUs0-7:8x H10080GB, BF16, Muon, standard checkpointing, no compile override.
- Dataset/poses: `/mnt/localssd/dataset/re10k/{training_256,training_poses}`;
  held-out: `{test_256,test_poses}`. DFoT source revision
  `0959defb4c4fe010f84791d732cc978ad7d49fef`.
- Resolution240x320, frozen Wan2.2 VAE at
  `/mnt/localssd/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth`.
- Chunk4; curriculum8/16/32/64 latent frames (29/61/125/253 RGB frames),
  2/4/8/16 chunks/video. Batch/GPU8/4/1/1; global64/32/8/8.
- Budgets97200/86950/30000/30000 updates (total244150); learning rates
  1e-4/2e-5/2e-5/2e-5. Stage1 count62244 eligible videos;100 epochs.
- Standard automatic timestep shift retained;1200-token chunk resolves to
  `2.667*sqrt(2)` rather than the old600-token chunk's2.667.
- Eval every1000 steps,8 global held-out videos, EMA, stage-length horizon,
  eval seed42,100 denoising steps, CFG2, AR stride5, VGG-LPIPS.
- W&B project: https://wandb.ai/LVSM-Experiment/miniworld; stages have separate
  names/IDs under the run group. No historical W&B run is reused.
- Output root:
  `/mnt/localssd/experiments/yibo/miniworld/re10k_B_transformer_serial_c4_s42_8xh100_20260930T012919Z`.
  `launcher.json` and stage `config.json` record the exact launch revision and
  dependency lock hash; an immutable detached source checkout and repo-local uv
  environment are used. `train.log` records all four stages.

Unlike changing Transformer execution order alone, changing chunk size changes
the training protocol. This is an exploratory common configuration for later
RTransformer/TaS comparisons, with no advance claim about convergence quality.
