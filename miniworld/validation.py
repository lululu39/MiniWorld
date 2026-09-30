"""Periodic held-out EMA rollouts, with fixed samples and distributed aggregation."""
from __future__ import annotations

import copy
from contextlib import contextmanager
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset

from miniworld.conditioning.actions import ConditioningConfig, build_cond_seq_for_batch
from miniworld.tracking import progress_fields
from miniworld.metrics import VideoMetrics, rgb_context_frames, write_report
from miniworld.vae.codec import StreamingVAEDecoder, vae_encode


def add_validation_args(parser):
    parser.add_argument('--eval_every', type=int, default=0, help='Optimizer steps between EMA evaluations; 0 disables')
    parser.add_argument('--eval_data_root', default=None)
    parser.add_argument('--eval_pose_dir', default=None)
    parser.add_argument('--eval_filter_cache_dir', default=None)
    parser.add_argument('--eval_num_videos', type=int, default=8, help='Global sample count, not per GPU')
    parser.add_argument('--eval_latent_frames', type=int, default=0, help='0 follows the current curriculum stage')
    parser.add_argument('--eval_history_len', type=int, default=1, help='Observed context in latent frames')
    parser.add_argument('--eval_seed', type=int, default=42)
    parser.add_argument('--eval_sampling_steps', type=int, default=100)
    parser.add_argument('--eval_cfg_scale', type=float, default=2.)
    parser.add_argument('--eval_ardiff_step', type=int, default=5)
    parser.add_argument('--eval_lpips_net', choices=['vgg', 'alex'], default='vgg')
    parser.add_argument('--eval_metric_frame_batch_size', type=int, default=4)


def validate_options(args):
    if args.eval_every < 0:
        raise ValueError('eval_every must be nonnegative')
    if not args.eval_every:
        return
    if not args.eval_data_root:
        raise ValueError('--eval_every requires --eval_data_root pointing to held-out data')
    if Path(args.eval_data_root).resolve() == Path(args.data_root).resolve():
        raise ValueError('Training and evaluation data roots must differ')
    if args.dataset == 're10k' and not args.eval_pose_dir:
        raise ValueError('RE10K evaluation requires --eval_pose_dir')
    frames = args.eval_latent_frames or args.latent_frames
    if not 1 <= args.eval_history_len < frames or args.eval_latent_frames < 0:
        raise ValueError('Evaluation requires 1 <= eval_history_len < eval_latent_frames (or stage length)')
    if min(args.eval_num_videos, args.eval_sampling_steps, args.eval_ardiff_step,
           args.eval_metric_frame_batch_size) < 1:
        raise ValueError('Evaluation sample count, sampling steps, stride and metric batch size must be positive')
    if args.latent_frames < args.df_chunk_size:
        raise ValueError('The trained window must contain at least one complete chunk for streaming evaluation')


def due_for_evaluation(step, interval):
    return interval > 0 and step > 0 and step % interval == 0


@contextmanager
def isolated_rng(device):
    """Preserve training RNGs, including only the CUDA device owned by this rank."""
    device = torch.device(device)
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def seed_sample(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.default_generator.manual_seed(seed)
    if device.type == 'cuda':
        with torch.cuda.device(device):
            torch.cuda.manual_seed(seed)


def gather_results(value):
    if not dist.is_initialized():
        return [value]
    values = [None] * dist.get_world_size()
    dist.all_gather_object(values, value)
    return values


class PeriodicEvaluator:
    def __init__(self, args, training_dataset, device):
        from miniworld.sample import build_dataset
        self.args = args
        self.device = torch.device(device)
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.frames = args.eval_latent_frames or args.latent_frames
        self.metric = None
        # Reuse the strict, augmentation-free evaluation loaders and metadata.
        eval_args = copy.copy(args)
        eval_args.total_len = self.frames
        eval_args.data_root = args.eval_data_root
        eval_args.pose_dir = args.eval_pose_dir
        eval_args.dataset_filter_cache_dir = args.eval_filter_cache_dir
        eval_args.sample_num_videos = args.eval_num_videos
        eval_args.metrics = True
        dataset, error = None, None
        if self.rank == 0:
            try:
                dataset = build_dataset(eval_args)
            except Exception as exc:
                error = f'{type(exc).__name__}: {exc}'
        payload = [dataset, error]
        if dist.is_initialized():
            # Avoid simultaneous scans/writes of the same length-filter cache.
            dist.broadcast_object_list(payload, src=0)
        self.dataset, error = payload
        if error:
            raise ValueError('Held-out dataset initialization failed: ' + error)
        if len(self.dataset) < args.eval_num_videos:
            raise ValueError(f'Only {len(self.dataset)} eligible evaluation videos; requested {args.eval_num_videos}')
        if args.dataset == 'droid':
            overlap = set(training_dataset.samples) & set(self.dataset.samples)
            if training_dataset.d_action != self.dataset.d_action:
                raise ValueError('Training and evaluation action dimensions differ')
        else:
            overlap = {p.stem for p in training_dataset.files} & {p.stem for p in self.dataset.files}
        if overlap:
            raise ValueError(f'Training/evaluation overlap: {len(overlap)} selected sample IDs')
        self.indices = list(range(self.rank, args.eval_num_videos, self.world_size))

    @torch.no_grad()
    def run(self, ema, vae, global_step, wandb_run=None):
        from miniworld.train import resolve_stream_chunks
        started = time.monotonic()
        args = self.args
        rows, error = [], None
        # Save all module modes; preserve the EMA sampler used by train videos.
        modes = {module: module.training for module in ema.modules()}
        names = ('steps', 'cfg_scale', 'cfg_interval_min', 'cfg_interval_max', 'df_ardiff_step')
        old = {name: getattr(ema, name) for name in names}
        try:
            with isolated_rng(self.device):
                ema.eval()
                ema.steps, ema.cfg_scale = args.eval_sampling_steps, args.eval_cfg_scale
                ema.cfg_interval_min, ema.cfg_interval_max = .2, 1.
                ema.df_ardiff_step = args.eval_ardiff_step
                cache, inflight = resolve_stream_chunks(args.latent_frames, args.df_chunk_size)
                full_history = (args.eval_history_len // args.df_chunk_size) * args.df_chunk_size
                if full_history > cache * args.df_chunk_size:
                    raise ValueError('Evaluation history exceeds the streaming cache capacity')
                if self.indices and self.metric is None:
                    self.metric = VideoMetrics(self.device, args.eval_metric_frame_batch_size, args.eval_lpips_net)
                # CPU generator keeps DataLoader construction out of CUDA RNG.
                loader = DataLoader(Subset(self.dataset, self.indices), batch_size=1, num_workers=0,
                                    generator=torch.Generator().manual_seed(args.eval_seed))
                batches = iter(loader)
                for idx in self.indices:
                    seed_sample(args.eval_seed + idx, self.device)
                    batch = next(batches)
                    video = batch['videos'].to(self.device)
                    rgb = video.permute(0, 4, 1, 2, 3).contiguous()
                    poses, actions = batch.get('poses'), batch.get('actions')
                    poses = None if poses is None else poses.to(self.device)
                    actions = None if actions is None else actions.to(self.device)
                    with torch.autocast(self.device.type, dtype=torch.bfloat16,
                                        enabled=self.device.type == 'cuda' and args.mixed_precision == 'bf16'):
                        latent = vae_encode(vae, rgb)
                        cond = build_cond_seq_for_batch(
                            cfg=ConditioningConfig(args.use_pose_cond, args.use_action_cond, args.pose_enc_freq),
                            poses=poses, actions=actions, t_latent=latent.shape[2],
                            h_lat=latent.shape[3], w_lat=latent.shape[4])
                        noise = torch.randn(latent.shape, device=self.device, dtype=latent.dtype,
                                            generator=torch.Generator(device=self.device).manual_seed(args.eval_seed + idx))
                        _, generated = ema.generate_eval_latents_streaming(
                            latent, cond, total_len=self.frames, history_len=args.eval_history_len,
                            max_cache_chunks=cache, inflight_chunks=inflight,
                            sink_frames=1 if cache else 0, noise=noise,
                            stream_decoder=StreamingVAEDecoder(vae))
                    row = self.metric.score(generated[0].permute(1, 0, 2, 3), video[0].permute(0, 3, 1, 2),
                                            context_frames=rgb_context_frames(args.eval_history_len))
                    row.update(sample_idx=idx, sample_id=batch['sample_id'][0],
                               source_path=batch['source_path'][0], source_frame_ids=batch['frame_ids'][0].tolist(),
                               seed=args.eval_seed + idx, sampling=dict(getattr(ema, 'last_eval_meta', {})))
                    rows.append(row)
        except Exception as exc:
            error = f'rank {self.rank}: {type(exc).__name__}: {exc}'
        finally:
            for name, value in old.items():
                setattr(ema, name, value)
            for module, mode in modes.items():
                module.training = mode
        parts = gather_results(dict(rows=rows, error=error))
        errors = [p['error'] for p in parts if p['error']]
        if errors:
            raise RuntimeError('Periodic evaluation failed: ' + '; '.join(errors))
        all_rows = sorted((row for part in parts for row in part['rows']), key=lambda r: r['sample_idx'])
        report, report_error = None, None
        if self.rank == 0:
            try:
                if len(all_rows) != args.eval_num_videos or len({r['sample_id'] for r in all_rows}) != len(all_rows):
                    raise ValueError('Evaluation sample count/identity mismatch')
                report = write_report(Path(args.output_dir)/'eval'/f'step_{global_step:08d}', all_rows,
                                      self.metric.protocol(), dict(mode='periodic_ema', global_step=global_step, progress=progress_fields(args, global_step),
                                      eval_latent_frames=self.frames, arguments=vars(args), world_size=self.world_size))
                scalars = {f'eval/{name}': value for name, value in report['mean'].items()}
                scalars.update({'eval/num_videos': report['num_videos'],
                                'eval/seconds': time.monotonic()-started, 'eval/latent_frames': self.frames})
                effective = all_rows[0].get('sampling', {}).get('effective_steps')
                if effective is not None:
                    scalars['eval/effective_sampling_steps'] = effective
                if wandb_run is not None:
                    # Explicit train_step avoids dropping eval values logged after train scalars at the same step.
                    wandb_run.log(dict(**progress_fields(args, global_step), **scalars))
                print(f'[Eval] step={global_step} {scalars}', flush=True)
            except Exception as exc:
                report_error = f'{type(exc).__name__}: {exc}'
        # Keep ranks synchronized until the report/W&B update has completed.
        report_errors = gather_results(report_error)
        if any(report_errors):
            raise RuntimeError('Evaluation reporting failed: ' + '; '.join(e for e in report_errors if e))
        return report
