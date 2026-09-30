#!/usr/bin/env python3
"""Capacity probe on repeated real RE10K clips, without W&B or checkpoints.

Includes the actual VAE, pose encoding, checkpointed serial backbone, singleton
DDP buckets, Muon state, EMA and reconstruction. Timings exclude disk loading,
multi-rank communication and evaluation; concurrent GPU users affect timings.
"""
import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from miniworld.conditioning import ConditioningConfig, build_cond_seq_for_batch
from miniworld.data.re10k import RealEstate10KDataset
from miniworld.denoiser import DenoiserConfig, build_denoiser_from_mode
from miniworld.train import build_optimizer, update_ema
from miniworld.vae.codec import load_wan22_vae, vae_decode, vae_encode


def probe(vae, sample, args, frames, batch, backbone):
    args.probe_phase = 'setup'
    torch.manual_seed(42)
    np.random.seed(42)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    cfg = DenoiserConfig(wm_model='B', backbone=backbone,
                         transformer_execution='serial', latent_size=(15, 20),
                         latent_frames=frames, df_chunk_size=4, cond_dim=720,
                         cond_per_token=True, cond_dropout_prob=.1,
                         wm_use_checkpoint=True, num_memory_tokens=256,
                         memory_window_frames=0)
    model = build_denoiser_from_mode(cfg).cuda()
    # Exercise recurrent/write gradients beyond the AdaLN-zero initialization.
    torch.nn.init.normal_(model.net.shared_mod[-1].weight, std=.005)
    torch.nn.init.normal_(model.net.final_layer.linear.weight, std=.005)
    ema = copy.deepcopy(model).requires_grad_(False)
    wrapped = DDP(model, device_ids=[0])
    optimizer = build_optimizer(argparse.Namespace(use_muon=True, lr=args.lr,
                                                  weight_decay=0.), wrapped)
    raw_frames = 4 * (frames - 1) + 1
    videos = sample['videos'][:raw_frames].unsqueeze(0).repeat(batch, 1, 1, 1, 1).cuda()
    poses = sample['poses'][:raw_frames].unsqueeze(0).repeat(batch, 1, 1).cuda()
    video_for_vae = videos.permute(0, 4, 1, 2, 3).contiguous()
    cond_cfg = ConditioningConfig(use_pose_cond=True, use_action_cond=False, pose_enc_freq=15)
    timings, vae_times, dit_times, losses = [], [], [], []
    state_info = None
    phase_peaks = {name: [] for name in ('vae_pose', 'training', 'decode', 'state_inspection')}
    allocated_peaks = [torch.cuda.max_memory_allocated()]
    reserved_peaks = [torch.cuda.max_memory_reserved()]
    def record_phase(name):
        allocated_peaks.append(torch.cuda.max_memory_allocated())
        reserved_peaks.append(torch.cuda.max_memory_reserved())
        phase_peaks[name].append(allocated_peaks[-1]/2**30)
    for step in range(args.steps):
        torch.cuda.synchronize()
        started = time.perf_counter()
        args.probe_phase = 'vae_and_pose'
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            latents = vae_encode(vae, video_for_vae)
            cond = build_cond_seq_for_batch(cfg=cond_cfg, poses=poses, actions=None,
                                           t_latent=frames, h_lat=15, w_lat=20)
        torch.cuda.synchronize()
        after_vae = time.perf_counter()
        record_phase('vae_pose')
        optimizer.zero_grad(set_to_none=True)
        args.probe_phase = 'forward'
        torch.cuda.reset_peak_memory_stats()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            # The last iteration also covers training's reconstruction tensors.
            outputs = wrapped(latents, cond, return_pred=step == args.steps - 1)
        loss, pred, noise_t = outputs if isinstance(outputs, tuple) else (outputs, None, None)
        args.probe_phase = 'backward'
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(wrapped.parameters(), 1.)
        if not torch.isfinite(loss) or not torch.isfinite(grad_norm):
            raise RuntimeError('Nonfinite loss or gradients')
        before = model.net.final_layer.linear.weight.detach().clone()
        args.probe_phase = 'optimizer_and_ema'
        optimizer.step()
        update_ema(model, ema, .9999)
        if torch.equal(before, model.net.final_layer.linear.weight):
            raise RuntimeError('Optimizer did not update weights')
        torch.cuda.synchronize()
        ended = time.perf_counter()
        record_phase('training')
        timings.append(ended - started)
        vae_times.append(after_vae - started)
        dit_times.append(ended - after_vae)
        losses.append(float(loss.detach()))
        if pred is not None:
            args.probe_phase = 'reconstruction_decode'
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                rgb = vae_decode(vae, pred[:1].float())
            if not torch.isfinite(rgb).all():
                raise RuntimeError('Nonfinite decoded reconstruction')
            del rgb
            record_phase('decode')
            if backbone == 'tas':
                args.probe_phase = 'bank_only_state_inspection'
                torch.cuda.reset_peak_memory_stats()
                model.net.eval()
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    length = min(frames, cfg.df_chunk_size)
                    _, state = model.net.forward_with_cache(
                        latents[:1, :, :length], latents.new_zeros((1, length)),
                        cond[:1, :length], return_kv=True, chunk_size=cfg.df_chunk_size)
                assert all(kv is None for kv in state.kv)
                state_info = dict(bank_shapes=[list(bank.shape) for bank in state.banks],
                                  bank_dtypes=[str(bank.dtype) for bank in state.banks],
                                  persistent_bytes_per_video=sum(bank.numel()*bank.element_size()
                                                                 for bank in state.banks),
                                  cross_chunk_kv_entries=0)
                model.net.train()
                del state
                record_phase('state_inspection')
        del latents, cond, outputs, pred, noise_t, loss, before
    warm = timings[1:] or timings
    return dict(status='passed', frames=frames, raw_frames=raw_frames, batch_per_gpu=batch,
                backbone=backbone, lr=args.lr, parameters=sum(p.numel() for p in model.parameters()),
                losses=losses, seconds=timings, median_seconds=statistics.median(warm),
                videos_per_second=batch/statistics.median(warm),
                vae_pose_seconds=vae_times, forward_backward_update_ema_seconds=dit_times,
                memory_state=state_info,
                phase_peak_allocated_gib=phase_peaks,
                peak_allocated_gib=max(allocated_peaks)/2**30,
                peak_reserved_gib=max(reserved_peaks)/2**30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--pose-dir', required=True)
    parser.add_argument('--filter-cache-dir', required=True)
    parser.add_argument('--vae-checkpoint', required=True)
    parser.add_argument('--cases', nargs='+', required=True,
                        help='backbone:latent_frames:batch, e.g. tas:64:4')
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--stage-lrs', type=float, nargs=4, default=None,
                        help='Override LR for the 8/16/32/64-latent-frame stages')
    parser.add_argument('--memory-fraction', type=float, default=.65,
                        help='Hard allocator limit; protect an existing GPU workload')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        parser.error('Select an authorized GPU with CUDA_VISIBLE_DEVICES')
    if args.steps < 2:
        parser.error('At least two iterations are required to exercise optimizer state')
    cases = []
    for value in args.cases:
        name, frames, batch = value.split(':')
        if name not in ('transformer', 'rtransformer', 'tas') or int(frames) < 8 or int(batch) < 1:
            parser.error(f'Invalid case {value}')
        cases.append((name, int(frames), int(batch)))
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[1]
    record = dict(config=vars(args), git_revision=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
        lock_sha256=hashlib.sha256((repo/'uv.lock').read_bytes()).hexdigest(),
        torch=torch.__version__, cuda=torch.version.cuda, torch_seed=42, numpy_seed=42,
        gpu=torch.cuda.get_device_name(), results=[],
        protocol='Repeated held-out clip; singleton DDP; real VAE/pose; no data-loader or multi-rank timing')
    dataset = RealEstate10KDataset([args.data_root], 4*(max(f for _, f, _ in cases)-1)+1, 1,
                                  randomize=False, color_aug=False, return_pose=True,
                                  pose_dir=args.pose_dir, filter_cache_dir=args.filter_cache_dir,
                                  strict_loading=True, return_metadata=True, max_keep=1)
    sample = dataset[0]
    record['sample_id'] = sample['sample_id']
    vae = load_wan22_vae(args)
    with tempfile.TemporaryDirectory(prefix='miniworld-batch-probe-') as directory:
        dist.init_process_group('nccl', init_method='file://' + directory + '/rendezvous',
                                rank=0, world_size=1)
        try:
            for name, frames, batch in cases:
                print(f'PROBE {name} frames={frames} batch={batch}', flush=True)
                case_args = copy.copy(args)
                if args.stage_lrs:
                    case_args.lr = args.stage_lrs[(8, 16, 32, 64).index(frames)]
                try:
                    result = probe(vae, sample, case_args, frames, batch, name)
                except torch.OutOfMemoryError as exc:
                    result = dict(status='out_of_memory', backbone=name, frames=frames,
                                  batch_per_gpu=batch, memory_fraction=args.memory_fraction,
                                  phase=getattr(case_args, 'probe_phase', 'setup'), error=str(exc),
                                  peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                                  peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
                record['results'].append(result)
                path.write_text(json.dumps(record, indent=2) + '\n')
                print(json.dumps(result), flush=True)
                vae.model.clear_cache()
                gc.collect()
                torch.cuda.empty_cache()
        finally:
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
