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

## Verified startup

- Started2026-09-30 01:34:37 UTC; supervisor PID2627266.
- Immutable source revision: `1e5c8460e4995b6cea419de5e7d47ff056b98f09`,
  checked out at the output root's `source/`, with an independently synced uv env.
- Source/runtime config confirms `backbone=transformer`,
  `transformer_execution=serial`, `df_chunk_size=4`, `latent_frames=8`.
- Public W&B readback at01:36:24 UTC: `running`, step80, loss1.6527309,
  1.2535 step/s. Local logs subsequently reached step110 with finite loss1.567333.
  All8 GPUs showed active compute; startup memory occupancy was about14–16GiB/card.
- Run URL: https://wandb.ai/LVSM-Experiment/miniworld/runs/f9c403e9cc4d441ead4a7dd006d8ac09
- The old parallel chunk2 run was intentionally interrupted at its last logged
  step5140; its4975-step checkpoint and five evaluation events are preserved.
  Its `intentional_stop.json` records the change reason. No old checkpoint was
  loaded into this new run. These are startup observations, not convergence or
  an apples-to-apples speed comparison: stage1 video length also changed21->29.

## One W&B run across the curriculum

The user requested a single run across all four stages. Stage1 continues without
restart on its original code. Its run ID `f9c403e9cc4d441ead4a7dd006d8ac09` will be
reused by stages2-4 with explicit W&B resume, cumulative train_step and separate
stage/local-step fields. The display name becomes the base experiment name.
Stage checkpoint/output directories remain unchanged.

A new immutable `continuation_source/` checkout and uv environment contain the
tracking changes. Only the old shell supervisor is paused; its foreground uv /
torchrun process and all GPU workers continue. A controller waits for successful
stage1 exit and `stage1_lf8/epoch_0100_step_00097200.pt`, removes the paused shell,
and launches the new curriculum script with `START_STAGE=2`. Failed stage1 does
not launch subsequent stages. Runtime identities, commit and status are recorded
in `continuation.json`, `continuation_status.json` and the augmented launcher
record under the output root. The controller is `scripts/continue_after_stage.py`.

Expected continuous boundaries: stage2 starts97201, stage3 starts184151,
stage4 starts214151, ending244150. Existing stage1 scalar/media history remains.
The earlier, intentionally stopped parallel chunk2 experiment is not merged.

During this change, the inherited reconstruction visualization formula was
corrected from `z+(1-t)*v_pred` to `z+t*v_pred`, consistent with
`z=(1-t)*x+t*noise` and `v_target=x-noise`. This affects only recon-video display,
not optimization or generated-video metrics. The live stage1 retains the old
visualization implementation; new stages use the correction. This version
boundary is explicitly recorded in W&B metadata; old videos are not relabeled.

## Fresh-restart request supersedes the handoff

The user subsequently requested a wholly fresh corrected run. The handoff
controller1642798 and old trainer were stopped on2026-09-30 at02:18 UTC.
Last logged step3030; checkpoint2916 preserved. The continuation is cancelled,
so stages2-4 will not be launched for this old run. See
`experiments/re10k_B_serial_c4_corrected_20260930.md` for the replacement, which
uses the reconstruction fix and single-run curriculum from step0.
