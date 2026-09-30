"""Framewise RGB fidelity metrics and explicit video/dataset aggregation.

PSNR follows LVSM's per-view dB averaging. SSIM uses the same Gaussian window
as its SsimLoss; LPIPS defaults to its VGG backbone. Inputs are RGB [-1, 1].
"""
from __future__ import annotations

import json
import math
import hashlib
import subprocess
from importlib.metadata import version
from pathlib import Path

import torch
from pytorch_msssim import ssim

NAMES = ('psnr', 'ssim', 'lpips')


def psnr_per_frame(prediction, target):
    """[N,3,H,W] FP32 RGB in [0,1]; exact matches have +inf PSNR."""
    mse = (prediction.float() - target.float()).square().mean(dim=(1, 2, 3))
    return -10 * torch.log10(mse)


def rgb_context_frames(latent_frames):
    if latent_frames < 1:
        raise ValueError('history_len must be positive (latent frames)')
    return 1 + 4 * (latent_frames - 1)


class VideoMetrics:
    def __init__(self, device='cpu', frame_batch_size=4, lpips_net='vgg'):
        if frame_batch_size < 1 or lpips_net not in ('vgg', 'alex'):
            raise ValueError('Positive frame_batch_size and vgg/alex LPIPS are required')
        import lpips
        self.device = torch.device(device)
        self.frame_batch_size = frame_batch_size
        self.lpips_net = lpips_net
        # Official pretrained features AND calibrated LPIPS v0.1 weights.
        self.lpips = lpips.LPIPS(net=lpips_net, version='0.1', verbose=False).to(self.device).float().eval()
        self.lpips.requires_grad_(False)

    def protocol(self):
        return dict(input='RGB float [-1,1], clamped then mapped to [0,1]',
                    psnr='mean of per-frame dB; exact matches are +inf',
                    ssim=dict(window=11, sigma=1.5, data_range=1, padding='valid', channels='RGB mean'),
                    lpips=dict(net=self.lpips_net, version='0.1', pretrained=True),
                    aggregation='equal frames within each video; equal videos in dataset mean',
                    frame_batch_size=self.frame_batch_size,
                    versions={name: version(name) for name in ('torch', 'lpips', 'pytorch-msssim')})

    @torch.inference_mode()
    def score(self, prediction, target, context_frames=1):
        """Score matching [T,3,H,W] clips, excluding the first context_frames RGB frames."""
        if prediction.shape != target.shape or prediction.ndim != 4 or prediction.shape[1] != 3:
            raise ValueError('Expected matching [T,3,H,W] RGB tensors; no implicit crop/resize/alignment')
        if not prediction.is_floating_point() or not target.is_floating_point():
            raise ValueError('Inputs must be floating-point RGB in [-1,1]')
        if min(prediction.shape[-2:]) < 32:
            raise ValueError('LPIPS evaluation requires spatial dimensions >= 32')
        if not 0 <= context_frames < prediction.shape[0]:
            raise ValueError('context_frames must leave at least one predicted frame')
        if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
            raise ValueError('Non-finite input video')
        scores = {name: [] for name in NAMES}
        # Never inherit BF16/autocast from training or sampling.
        with torch.autocast(device_type=self.device.type, enabled=False):
            for start in range(context_frames, prediction.shape[0], self.frame_batch_size):
                end = start + self.frame_batch_size
                p = prediction[start:end].to(self.device, dtype=torch.float32).clamp(-1, 1)
                t = target[start:end].to(self.device, dtype=torch.float32).clamp(-1, 1)
                p01, t01 = (p + 1) / 2, (t + 1) / 2
                scores['psnr'].extend(psnr_per_frame(p01, t01).cpu().tolist())
                scores['ssim'].extend(ssim(p01, t01, data_range=1, size_average=False,
                                           win_size=11, win_sigma=1.5).cpu().tolist())
                scores['lpips'].extend(self.lpips(p, t, normalize=False).flatten().cpu().tolist())
        return dict(context_frames=context_frames, evaluated_frames=prediction.shape[0]-context_frames,
                    frame_indices=list(range(context_frames, prediction.shape[0])), per_frame=scores,
                    mean={name: sum(values)/len(values) for name, values in scores.items()})


def summarize_scores(rows):
    """Equal-video mean plus horizon curves; never pool MSE before taking log."""
    if not rows:
        raise ValueError('No videos were evaluated')
    means = {name: sum(row['mean'][name] for row in rows)/len(rows) for name in NAMES}
    horizons = {}
    for row in rows:
        for j, frame in enumerate(row['frame_indices']):
            values = horizons.setdefault(frame, {name: [] for name in NAMES})
            for name in NAMES:
                values[name].append(row['per_frame'][name][j])
    curve = [dict(frame_index=i, num_videos=len(values['psnr']),
                  **{name: sum(v)/len(v) for name, v in values.items()})
             for i, values in sorted(horizons.items())]
    return dict(num_videos=len(rows), evaluated_frames=sum(r['evaluated_frames'] for r in rows),
                mean=means, per_frame=curve)


def json_safe(value):
    # Standard JSON has no Infinity; preserve perfect-match semantics explicitly.
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            raise ValueError('NaN metric cannot be reported')
        return 'inf' if value > 0 else '-inf'
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def write_report(output_dir, rows, protocol, run):
    """Overwrite this run's report; never append incompatible previous runs."""
    root = Path(output_dir)
    repo = Path(__file__).resolve().parents[1]
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    lock = repo / 'uv.lock'
    provenance = dict(git_revision=revision, lock_sha256=hashlib.sha256(lock.read_bytes()).hexdigest() if lock.exists() else None)
    summary = dict(protocol=protocol, run=run, provenance=provenance, **summarize_scores(rows))
    root.mkdir(parents=True, exist_ok=True)
    (root/'metrics_per_video.jsonl').write_text(''.join(
        json.dumps(json_safe(row), allow_nan=False)+'\n' for row in rows))
    (root/'metrics_summary.json').write_text(json.dumps(json_safe(summary), indent=2, allow_nan=False)+'\n')
    return summary
