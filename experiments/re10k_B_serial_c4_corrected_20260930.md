# Corrected fresh RE10K Transformer-B run, 2026-09-30

The user requested stopping the old experiment and starting a new run after the
reconstruction and W&B corrections. Both the old stage1 trainer and its planned
handoff controller were stopped. No old checkpoint is loaded and no old W&B ID
is reused. Old source snapshots/checkpoints/media remain available.

- Run name: `re10k_B_serial_c4_s42_fixed_20260930T022125Z`.
- Launcher: `experiments/re10k_B_serial_c4_corrected_20260930.sh`.
- Model: serial Transformer-B,12 layers, width768,12 heads,112432176 trainable
  parameters; chunk4 latent frames =1200 tokens at240x320 resolution.
- Eight H10080GB GPUs0-7, BF16, Muon, seed42, activation checkpointing.
- Curriculum:8/16/32/64 latent frames (29/61/125/253 RGB); per-GPU batches8/4/1/1.
  Budgets97200/86950/30000/30000 updates; learning rates1e-4/2e-5/2e-5/2e-5.
- RE10K train/eval and poses under `/mnt/localssd/dataset/re10k/`, pinned DFoT
  revision `0959defb4c4fe010f84791d732cc978ad7d49fef`. Stage1 has62244 eligible
  train videos; fixed8-video held-out evaluation every1000 updates.
- Frozen VAE: `/mnt/localssd/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth`.
- Training/sampling protocol unchanged from the prior serial chunk4 experiment.
  Auto timestep shift remains `2.667*sqrt(2)`; evaluation configured sampling
  cap100, AR stride5, CFG2, stage-matched horizon, seed42, VGG-LPIPS.
- Output root:
  `/mnt/localssd/experiments/yibo/miniworld/re10k_B_serial_c4_s42_fixed_20260930T022125Z`.
  Immutable source checkout and a separately synced uv environment. Exact commit
  and dependency lock SHA256 are recorded by the launcher/trainer.

Corrections active from step0:

1. The reconstruction-only branch uses `x_hat=z+t*v_pred`, consistent with
   `z=(1-t)*x+t*noise`, `v_target=x-noise`. Training targets/loss, gradients,
   iterative generation and held-out metrics are unaffected by this correction.
2. All four stages share one new W&B run in `LVSM-Experiment/miniworld`;
   checkpoint-derived cumulative train_step, separate stage/local-step/latent
   frame fields, and per-stage config snapshots are retained.
3. Reconstructed-video captions explicitly identify maximum t and copied context.
   `train/recon_t_max` records that value numerically. Actual effective sampler
   steps are logged as well, since the pipeline may cap below the configured100:
   stage1 eval uses5 effective steps with one in-flight chunk and stride5.

Validation before launch:61 tests passed; focused reconstruction/tracking/metric
checks passed again after adding the diagnostic fields. The oracle-velocity
test verifies clean-latent recovery at multiple t values. Previous serial model
forward/backward/Inductor checks remain applicable; no model kernels changed.

The previous run stopped after its last logged3030 step, preserving its2916-step
checkpoint and evaluations through3000. Its continuation controller was cancelled;
there is no deferred launch left attached to that run.


## Verified startup and display-name correction

- Started2026-09-30 02:25:11 UTC; supervisor PID1853839.
- Immutable training revision: `10eb062f0d22eefad329a5420b2f70849c72e64a`.
- Public W&B readback at02:28:01 UTC confirmed `running`, cumulative step150,
  stage1/local step150,8 latent frames, loss1.1986213 and1.2505 step/s.
- Source/config verification confirms scratch initialization, serial Transformer,
  chunk4, offset0 and corrected `x_pred=z+t_view*v_pred` from the first step.
- Shared run ID: `08731ad6239140c1a8fcf27bef94914f`.
  https://wandb.ai/LVSM-Experiment/miniworld/runs/08731ad6239140c1a8fcf27bef94914f
- At the user's request the display name was corrected to include the backbone:
  `re10k_B_transformer_serial_c4_s42_fixed_20260930T022125Z`.
  Root/stage run-identity JSON files were updated so subsequent stages use the
  corrected name. The ID, history, checkpoint paths and active training process
  remain the same; `run_name_update.json` records the rename.
- This run uses the separately synced uv environment in its immutable checkout.
  Its `uv.lock` is byte-identical to `/data/yibo/MiniWorld/uv.lock` (SHA256
  `7a79e838806250bcb9329ad47251c45ec61940c406f34d4e9e9763b216a9d49e`).
  The user prefers reusing the main checkout's uv environment for future starts;
  this preference does not require interrupting the active process.

## Flow Matching audit follow-up

The later formula audit confirmed the training objective/integration sign and
found a separate completed-chunk time-label issue in multi-inflight sampling.
See `experiments/flow_matching_audit_20260930.md`. The current single-inflight
stage1 is unaffected and continues without restart. After its normal completion,
stages2-4 will use audited revision`8a99f40` through a controlled handoff, retaining
the same run ID, hyperparameters, checkpoints and continuous step axis. The
continuation reuses the main checkout's `.venv`; details and process identities
are in `flow_audit_handoff.json` and `continuation_status.json` under this run.
