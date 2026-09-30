# EMA versus raw weights after increasing batch

The current Transformer run uses global batch256 / LR2e-4, versus64 /1e-4
previously. Both use EMA decay0.9999 per update. Matched8-video eval sample IDs,
frame indices, noise seeds, metric protocol, CFG2 and5 effective steps agree.
At8k steps the old EMA PSNR is14.89 and the new EMA PSNR14.25. New EMA PSNR
was higher at1k-7k, so the new run is not uniformly worse at equal update count.

One essential diagnostic reuses the new step8019 checkpoint, sampling protocol
and all8 clips. Raw weights obtain PSNR21.30 / SSIM0.7104 / LPIPS0.2174;
its EMA obtains14.24 /0.4064 /0.6498. Both gen_video and periodic eval use EMA.
This demonstrates the current EMA's lag relative to its trained model. It does
not establish that new raw weights outperform old raw weights, or isolate LR
from batch and update-budget changes. Only8 clips were evaluated.

With decay0.9999 the initial parameter contribution remains0.9999**8019,
about44.85%. At equal update count this contribution is the same in both runs;
it must not be described as uniquely caused by the larger batch. At equal
video exposure, however, the larger batch's EMA half-life is4 times longer.
To preserve the old half-life in videos, use decay_new=decay_old**4,
approximately0.99960006. This is a sample-scale equivalence calculation,
not a newly tuned optimal decay. Updating a live decay would not reconstruct
its previous EMA trajectory immediately.

Reference: NVIDIA EDM measures EMA half-life in training images and computes
ema_beta=0.5**(batch_size/ema_halflife_nimg):
https://github.com/NVlabs/edm/blob/main/training/training_loop.py#L129-L135

Raw diagnostic/protocol paths, the exact checkpoint and curves are in
ema_batch_diagnosis_20260930.json. The transient diagnostic used the frozen
training source and the main uv environment, CUDA_VISIBLE_DEVICES=0, an18%
allocator cap, cudnn.benchmark=False and no W&B run. Training was not stopped
or changed, and no further sweeps/tests were launched.
