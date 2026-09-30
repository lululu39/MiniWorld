# Corrected RTransformer memory and bank-only TaS

The earlier MiniWorld port had two memory problems. RTransformer's own/top KV
blend and history concatenation ran outside each layer's checkpoint, retaining
an extra full-history tensor per layer/chunk. Commit `8a834ee` moves the identical
blend inside that checkpoint. Outputs and input/parameter gradients match the
legacy formula, including nonzero per-head blend gradients. History remains
attached to autograd; the model/checkpoint weights and mixing formula are unchanged.

TaS had an unintended preceding-chunk raw-KV bypass and archived the clip's
full KV even though reads used only a short window. This was a porting error,
not the user's LVSM architecture. Commit `fd764bd` removes that path: self-attention
uses only the current chunk, and only fixed token banks cross chunks. This
corrects the architecture and changes the former hybrid's behavior; it is not
an output-equivalent optimization of that hybrid. Its earlier capacity probes
must not be interpreted as measurements of the corrected TaS.

Reference: `../ttt_lvsm`, `origin/yibo_dev` at
f34753d69aeb0dba495aef738a38d2ad6f43eeff,
`model/blocks/memory_block_tokens.py`: self-attention receives the current x,
then cross-attention reads memory; the view/layer sweep writes memory afterward.
No previous-view raw-KV cache is part of that token-memory path. MiniWorld keeps
its own chunk schedule, 3D RoPE, pose/action conditioning, AdaLN gates and flow
matching; the five TaS switches remain independently configurable.

## State and inference

Default B TaS has one shared 256x768 bank. Actual CUDA inspection found BF16
state, exactly 393,216 bytes (384 KiB) per video/CFG branch, and zero cross-chunk
raw-KV entries. The bank's size does not grow with chunk count. With state
sharing disabled, each layer has its own fixed bank. Slow model parameters,
bank versions/activations needed for training, VAE and optimizer state are
additional memory, not part of the persistent bank payload.

TaS no longer returns raw chunk KV from a layer or appends it to VideoState.
Tentative denoising returns no accepted state and never mutates committed banks.
Clean t=0 calls commit functional candidate banks; conditional/unconditional
branches maintain distinct storage. Sampler cache_frames still tracks its
logical RoPE window; eviction is a no-op on bank-only state and does not rotate
or discard memory. Training gradients cross all bank writes without detach.

The old `memory_window_frames` field is retained only for metadata compatibility.
TaS requires 0 and rejects nonzero values/raw-KV streaming state. Loading a
legacy TaS checkpoint recording 4 rejects that different architecture; existing
Transformer/RTransformer checkpoints ignore this irrelevant field. All new
launchers/probes use 0. The active Transformer run uses its recorded immutable
source and is unaffected by this TaS correction.

## Validation and memory results

Raw records are in [the follow-up probe record](tas_bank_only_probe_20260930.json).
They include source revisions, lock SHA256, GPU, actual sample ID, fixed
Torch/NumPy seed 42 for the corrected probe, losses, step timings, phase peaks,
and the inspected bank's shape/dtype/bytes. No W&B run or checkpoint is created.

RTransformer stage 4 batch 2 drops from 40.02 GiB to about 25.06 GiB. A repeated
same-process warm pair found 25.06 GiB RTransformer versus 25.09 GiB Transformer.
Stage 3 batch 8 RTransformer drops from 44.06 GiB to 30.21 GiB. Those changes are from
checkpoint placement, not fewer tokens/history or reduced gradient coverage.

Corrected bank-only TaS stage 4 batch 2 has a training-phase peak of 10.03 GiB,
versus roughly 25.1 GiB for Transformer/RTransformer. The phase includes resident
VAE/model/EMA/optimizer/input tensors as well as forward/backward/update;
it is not the standalone memory-bank size. Cold VAE/cuDNN autotuning can produce
a transient larger peak (39.22 GiB for the first Transformer batch2 case), so the
record separates VAE/pose, training, decoding and state inspection instead of
attributing a startup workspace to backbone state.

Each capacity case runs three full B forward/backward/Muon/EMA updates and a
real VAE reconstruction, using a repeated strict-loaded held-out RE10K clip.
Torch allocator is capped at 65% of physical H100 80GB memory to protect the active
eight-GPU run. Timing shares GPU 0 with that run and excludes DataLoader and
multi-rank communication; it does not establish MFU or eight-rank speedup.

64 focused backbone/flow tests passed, plus 5 CPU curriculum tests. New checks
require current-chunk attention with no past KV, fixed bank size across 8 chunks,
memory as the sole cross-chunk information path, live cross-chunk gradients,
clean-only streaming commits, branch isolation, no tentative mutation, and
legacy checkpoint rejection. Tiny-model CUDA eager/fullgraph torch.compile
forward/backward/update checks passed for all three backbones. These are
correctness/capacity checks, not convergence or a trained quality result.

Same-stage batch 4 training-phase peaks: Transformer 45.42 GiB,
RTransformer 45.40 GiB, and corrected TaS 14.74 GiB. The TaS read/write state
inspection confirms 384 KiB per video at all four curriculum lengths.

## Current prepared common B profile

| Stage | Latent frames | Batch/GPU | Global batch on 8 GPUs | Common base LR | Budget |
| --- | --- | --- | --- | --- | --- |
| 1 | 8 | 32 | 256 | 2e-4 | 100 epochs / 24,300 updates |
| 2 | 16 | 16 | 128 | 4e-5 | 50 epochs / 21,700 updates |
| 3 | 32 | 8 | 64 | 5.656854249e-5 | 3,750 updates |
| 4 | 64 | 4 | 32 | 4e-5 | 7,500 updates |

All models use serial execution, 4 latent frames/chunk (1200 tokens), 240x320,
BF16, checkpointing, Muon with auxiliary Adam, weight decay 0, clipping 1,
EMA 0.9999/update, seed 42. TaS has 256 slots, one shared bank, no history KV,
slot identity, sigmoid EMA, write-from-last and assigned write enabled.
The models retain their B dimensions; TaS adds memory modules and therefore
has 147,056,689 parameters versus 112,432,176/112,432,320 in Transformer/RTransformer.

LRs use the previously recorded sqrt(batch multiplier) heuristic, with no claim
of a Muon scaling theorem or optimal LR. The user selected preserving video
exposure. Stages3/4 each see 240,000 videos, and stages1/2 keep their epochs
(the earlier note about stage2 full-batch rounding still applies). Total 57,250
updates. Fixed-per-update EMA and evaluation cadence have different sample-scale
horizons at the larger batches; convergence still needs training evidence.

Use [the prepared launcher](../scripts/train_re10k_B_shared_batch.sh) with 8
explicitly authorized GPUs and BACKBONE=transformer/rtransformer/tas. It uses
the repo-local locked uv environment, explicit dataset/pose/VAE/output paths,
fresh backbone-named runs, one W&B identity across stages, and held-out EMA
evaluation every 1000 steps under LVSM-Experiment/miniworld. VAE checkpoint,
dataset provenance, BF16 stack and original exposure arithmetic remain recorded
in [the initial record](re10k_B_shared_batch_20260930.md).

This prepares a new profile; it does not restart or change the active run
08731ad6239140c1a8fcf27bef94914f or its audited-stage controller.
