# Common larger batches for the three serial B backbones

Prepared on 2026-09-30 for RE10K, one node with 8 H100 80GB GPUs. This is a
capacity-checked starting configuration for Transformer, RTransformer and TaS,
not a completed training run or a convergence/throughput result.

| Stage | Latent / RGB frames | Batch/GPU | Global batch | Common base LR | Training budget |
| --- | --- | --- | --- | --- | --- |
| 1 | 8 / 29 | 32 | 256 | 2e-4 | 100 epochs, 24,300 updates |
| 2 | 16 / 61 | 16 | 128 | 4e-5 | 50 epochs, 21,700 updates |
| 3 | 32 / 125 | 8 | 64 | 5.656854249e-5 | 3,750 updates |
| 4 | 64 / 253 | 2 | 16 | 2.828427125e-5 | 15,000 updates |

All backbones use the same table, serial execution, 4 latent frames/chunk
(1200 tokens at 240x320), BF16, activation checkpointing, Muon with auxiliary
Adam, weight decay 0, gradient clipping 1, seed 42, and EMA decay 0.9999 per update.
TaS uses 256 slots, 4 latent frames of raw history and the five default switches
enabled (slot identity, sigmoid EMA, write-from-last, shared banks, assigned
write). Transformer/RTransformer have 112,432,176/112,432,320 trainable parameters;
TaS has 147,056,689 because of its additional readers/writer. All use the B
backbone dimensions; they are not parameter-count-matched models.

The LR multiplier is sqrt(new global batch / previous global batch), relative
to the actual local recipe: batches 8/4/1/1 and LRs 1e-4/2e-5/2e-5/2e-5. It is a
conservative experimental heuristic for this Muon/Adam mixture. It is **not**
a theoretically justified Muon scaling law, an LR search, or a claim that the
same LR is optimal for each architecture. The
[Adam/RMSprop scaling paper](https://papers.nips.cc/paper_files/paper/2022/hash/32ac710102f0620d0f28d5d05a44fe08-Abstract-Conference.html)
does not establish that rule for Muon. Optimizer momentum/betas, EMA, and
evaluation frequency remain measured in updates; their sample-scale horizons
change when batch increases. The underlying training objective is unchanged.

The user selected preserving video exposure. Stages 3/4 each still process
exactly 240,000 videos: 3,750x64 and 15,000x16. Stages 1/2 preserve 100/50 epochs.
Stage 2 drops 96 more clips per epoch because of full-batch rounding
(55,552 vs 55,648); exact original exposure differs by 0.17% in that stage.
The total is 64,750 updates versus 244,150 previously. Fewer updates do not imply
the same ratio of wall-clock speedup or identical optimization trajectories.

## Verification and limits

Raw cases, Git revisions, lock hashes, sample ID, losses, timings and CUDA
memory peaks are in [the probe record](re10k_B_shared_batch_probe_20260930.json).
The GPU capacity check repeats one held-out RE10K clip, with strict actual
video/pose loading, the real frozen Wan2.2 VAE, actual pose conditioning,
singleton DDP buckets, forward/backward, clipping, Muon state, EMA and
reconstruction decode. Each selected configuration must pass three consecutive
updates with finite losses/gradient norms and changed weights. This is a
pipeline capacity check on repeated real data, not training on diverse clips.
No W&B run or training checkpoint is created.

GPU 0 was explicitly authorized. The existing eight-GPU run stays active, so
the probe allocator was capped at 65% of physical GPU memory (about 51.8 GiB).
RTransformer stage 4 batches 4 and 3 exceeded that protected test budget.
These failures **do not establish** that they cannot fit an otherwise idle
H100 80GB. The chosen batch 2 leaves room for multi-rank NCCL, LPIPS, sampling and
allocator variation. Eight-rank throughput, held-out streaming evaluation at
the increased batch and sustained training convergence have not been measured.
The test covers default memory switches only, not arbitrary TaS ablations.

Timings exclude disk/DataLoader costs and multi-rank communication and share
GPU 0 with the active run. Old-batch Transformer probes provide context only;
they do not justify a training speedup or MFU claim. The VAE/pose phase consumes
a substantial part of each step, so filling more memory alone may yield little
improvement in short-video throughput. Probe Torch RNG is reset to 42; the
initial probes did not fix NumPy's timestep RNG, so listed losses are not a
bit-exact reproducibility target.

The focused CPU curriculum/reconstruction checks passed (6 tests). All three
backbones' four-stage launcher arguments were checked using a stub uv command,
without starting training or W&B. No model kernel changed in this task.

## Launch the prepared profile

Use [scripts/train_re10k_B_shared_batch.sh](../scripts/train_re10k_B_shared_batch.sh).
It pins the table above and requires eight explicitly selected GPUs. Check
occupancy and push the exact revision before an actual launch. For example,
after those GPUs are available:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 BACKBONE=transformer \
  bash scripts/train_re10k_B_shared_batch.sh
```

Set BACKBONE=rtransformer or tas for the other two models. Names include the
backbone/B/chunk/seed and a unique timestamp. Each experiment's four stages
share one new W&B run under LVSM-Experiment/miniworld; seed 42, held-out 8 videos,
eval_every 1000, CFG 2 and the existing sampler protocol are preserved. All
launches run via the repo-local locked uv environment. Data/pose/cache paths
default to /mnt/localssd/dataset/re10k; checkpoints/output roots default to
/mnt/localssd/experiments/yibo/miniworld. VAE checkpoint is
/mnt/localssd/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth (SHA256
20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36).

This task prepares the new comparison profile. The active Transformer run
08731ad6239140c1a8fcf27bef94914f and its audited-stage handoff retain their
existing batch/LR/budgets and source revisions; they were not restarted or
relabelled as this profile.
