# Flow Matching implementation audit — 2026-09-30

## Conclusion and scope

The current training noising formula, denoising-velocity target, masked loss,
reconstruction formula and Euler integration sign are mutually consistent.
The reconstructed-image coefficient bug was already fixed before the current
fresh run. This audit found and repaired a separate asynchronous sampler time-label
bug affecting completed chunks that remain in a multi-chunk in-flight window.
No training objective or hyperparameter was changed.

Sources checked:

- [MiniWorld, sections2,3.3 and3.5](https://arxiv.org/html/2608.01127#S2).
- [Original Rectified Flow paper](https://arxiv.org/abs/2209.03003).
- [Meta Flow Matching affine-path implementation](https://github.com/facebookresearch/flow_matching/blob/main/flow_matching/path/affine.py),
  [CondOT scheduler](https://github.com/facebookresearch/flow_matching/blob/main/flow_matching/path/scheduler/scheduler.py),
  and [ODE solver](https://github.com/facebookresearch/flow_matching/blob/main/flow_matching/solver/ode_solver.py).
- Local source, original repository snapshot `e484206`, and the active run's
  immutable `10eb062f0d22eefad329a5420b2f70849c72e64a` checkout.

## Time convention and formulas

MiniWorld uses t=0 for data and t=1 for noise. In the usual source-noise to
target-data Flow Matching convention, write s=1-t. The two parameterizations are:

```text
Meta/standard direction:
  y(s) = (1-s)*noise + s*x
  dy/ds = x-noise

MiniWorld noise-level coordinate:
  z(t) = (1-t)*x + t*noise
  v_target = x-noise = -dz/dt
  z_next = z - (t_next-t_current)*v_pred    # t_next <= t_current
  x_hat = z + t*v_pred
```

The network predicts the denoising-direction field, not dz/dt. Thus its positive
`x-noise` target and negative-dt Euler update are correct together. Reversing
only the target or only the update sign would be wrong. A time shift changes the
sampling density/grid; the sampler computes dt in the shifted physical t, so no
additional derivative of the shift is required in this implementation.

| Component | Actual implementation / result |
| --- | --- |
| Noising | Gaussian noise in the normalized VAE latent space; exact linear interpolation |
| Target | `latents-noise`, not epsilon-prediction or VP-diffusion v-prediction |
| Loss | FP operations on velocity residual; mean over channels/spatial positions, masked mean per video, then mean across batch |
| Condition mask | Condition frames contribute no loss; tests verify their prediction gradients are zero |
| Reconstruction | Corrected `z+t*v_pred`; oracle velocity recovers x at multiple t values |
| Integration | `z-dt*v`, with negative physical dt; oracle trajectories reach the target |
| CFG | `v_uncond + scale*(v_cond-v_uncond)`; an analytic conditional/unconditional pair reaches its exact guided endpoint |
| Time warp | Finite endpoints0/1 and monotone mapping; sampler uses actual shifted time differences |
| Latent scaling | VAE encode applies `(mu-mean)*inverse_std`; decode applies its inverse before RGB reconstruction |

## Newly found sampler defect

The old schedule obtained current t by indexing the preceding grid value with
`current_lookup[step_index]`. This works for a chunk actively advancing one
step, but fails after its index has reached the final step and stops advancing.
The latent is already clean while the model is still told the penultimate
positive timestep. This happens in the final multi-chunk in-flight window,
or when chunks finish before that window advances.

Minimal unshifted example: two chunks, two denoising steps, AR stride1.

```text
                    row1       row2       row3
next time           [0.5,1]    [0,0.5]    [0,0]
old current time    [1,1]      [0.5,1]    [0.5,0.5]  # first entry wrong
correct current     [1,1]      [0.5,1]    [0,0.5]
```

The fix indexes the actual preceding schedule row, including stationary
completed chunks. It enforces `current[k] == next[k-1]`; completed visible chunks
remain labelled0. Update counts, timestep grid, CFG scale, cache layout and all
training settings are unchanged. The same old expression is present in the
original `e484206` repository snapshot; it was not introduced by the serial port.

Although an oracle that ignores its timestep can still reach the correct final
value with the old code, a real network is conditioned on t. The added tests
therefore assert **the latent content matches its supplied t at every network
call**, not only that final outputs are finite. Before the fix9 audit cases
failed; after the fix all analytical checks pass.

## Existing policies noted, not silently changed

- Training condition-noise augmentation is hardcoded to at most0.05. The
  so-called clean context is therefore slightly noised during training. The
  sampled chunk timesteps are monotone, but overwriting context frames with a
  small positive t can break strict ordering relative to a neighboring t=0
  chunk. Inputs and supplied t remain consistent; this is separate from the
  inference labelling bug. Inference pins observed context at t=0.
- CFG interval selection uses the mean frame timestep of a chunk, including
  pinned context frames. For a partially observed chunk this is not identical
  to testing the unobserved frames' common t. The affine CFG formula is correct;
  this existing gate policy was retained.
- The configured sampling count is a cap. Unless the full sequence fits in
  flight, effective steps are capped by `inflight_chunks * ar_step`. Current
  stage1 train visualization uses10 effective steps; held-out eval uses5.
- `train/recon_video` uses current training weights and a noisy GT batch;
  `train/gen_video` uses EMA and a fixed first-frame-conditioned rollout.
  With EMA0.9999, the initial-weight term has coefficient0.90483 at step1000.
  These diagnostics are not interchangeable measures of unconditional rollout
  quality. This audit does not establish convergence or sampling quality.

## Verification and active-run handling

`tests/test_flow_matching_audit.py` covers the exact interpolation, analytic
masked MSE and gradients, conditional/unconditional CFG endpoints, shifted-grid
Euler trajectories, observed-context pinning, prefill, cache eviction, partial
final chunks,1/2/4 in-flight chunks and time-grid continuity. Together with the
existing tests, **78 tests pass** using the main repository uv environment.
An actual tiny serial Transformer produces bit-identical old/fixed outputs for
the active single-in-flight protocol. The current first two stages use that
protocol and do not trigger the defect. Later stages use multiple in-flight
chunks and need the repaired scheduler.

The active run is not restarted and its training trajectory/configuration is not
changed. Stage1 continues on its original immutable source. At its successful
completion, a controlled handoff runs stages2-4 from the audited source, reusing
`/data/yibo/MiniWorld/.venv` and the existing shared W&B run. Runtime handoff status,
process identities and exact code revision are recorded under the experiment
output root. No source file is hot-patched in the active Python processes.
