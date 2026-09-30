"""MiniWorld training entry point for action- and pose-conditioned datasets."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import uuid
import sys
import os
import random
import subprocess
import hashlib
import numpy as np
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
from einops import rearrange
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from miniworld.validation import add_validation_args, validate_options, due_for_evaluation, PeriodicEvaluator, gather_results
from miniworld.tracking import progress_fields, checkpoint_progress, resolve_step_offset, wandb_session, write_json
from miniworld.backbones import add_backbone_args, backbone_config
from miniworld.conditioning.actions import ConditioningConfig, build_cond_seq_for_batch
from miniworld.data.droid import LeRobotActionDataset
from miniworld.data.re10k import RealEstate10KDataset
from miniworld.denoiser import DenoiserConfig, build_denoiser_from_mode
from miniworld.vae.codec import StreamingVAEDecoder, load_wan22_vae, print0, vae_decode, vae_encode


def setup_distributed() -> Dict[str, int | bool]:
    """Initialize torch.distributed from torchrun environment variables."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
    return {"world_size": world_size, "rank": rank, "local_rank": local_rank, "distributed": distributed}


def cleanup_distributed(distributed: bool) -> None:
    """Destroy torch.distributed state when active."""
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """Return whether this rank should write logs/checkpoints."""
    return int(os.environ.get("RANK", "0")) == 0


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    """Return the checkpoint with the largest global step, if any."""
    import re

    root = Path(output_dir)
    if not root.exists():
        return None
    best_path, best_step = None, -1
    pattern = re.compile(r"epoch_(\d+)_step_(\d+)\.pt$")
    for path in root.iterdir():
        match = pattern.match(path.name)
        if match and int(match.group(2)) > best_step:
            best_path, best_step = path, int(match.group(2))
    return os.fspath(best_path) if best_path is not None else None


def update_ema(model: torch.nn.Module, ema_model: torch.nn.Module, decay: float) -> None:
    """Update EMA parameters in-place."""
    with torch.no_grad():
        model_state = model.state_dict()
        for key, value in ema_model.state_dict().items():
            if key in model_state and torch.is_floating_point(value):
                value.copy_(value * decay + model_state[key] * (1.0 - decay))
            elif key in model_state:
                value.copy_(model_state[key])


def build_dataset(args: argparse.Namespace, *, randomize: bool, color_aug: bool):
    """Build the configured training dataset."""
    num_raw_frames = 4 * (int(args.latent_frames) - 1) + 1
    if args.dataset == "droid":
        return LeRobotActionDataset(
            root=args.data_root,
            num_frames=num_raw_frames,
            frame_interval=args.frame_interval if randomize else 1,
            resize_hw=(args.resize_h, args.resize_w),
            camera_views=args.action_camera_views.split(","),
            action_keys=args.action_keys.split(","),
            action_norm="q01q99",
            randomize=randomize,
            color_aug=color_aug,
            require_success=True,
        )
    return RealEstate10KDataset(
        dataset_paths=[args.data_root],
        num_frames=num_raw_frames,
        frame_interval=args.frame_interval if randomize else 1,
        resize_hw=(args.resize_h, args.resize_w),
        randomize=randomize,
        color_aug=color_aug,
        filter_cache_dir=args.dataset_filter_cache_dir,
        return_pose=True,
        pose_dir=args.pose_dir,
    )


def resolve_conditioning(args: argparse.Namespace, dataset) -> None:
    """Populate conditioning dimensions on args."""
    if args.dataset == "droid":
        args.use_action_cond = True
        args.use_pose_cond = False
        args.cond_dim = 4 * int(dataset.d_action)
        args.cond_per_token = False
        print0(f"[Cond] DROID action dim={dataset.d_action}, cond_dim={args.cond_dim}")
    else:
        args.use_action_cond = False
        args.use_pose_cond = True
        args.cond_dim = 4 * 6 * 2 * int(args.pose_enc_freq)
        args.cond_per_token = True
        print0(f"[Cond] RE10K pose cond_dim={args.cond_dim}, poses_per_latent=4")


def build_denoiser(args: argparse.Namespace):
    """Create the MiniWorld denoiser from CLI args."""
    cfg_kwargs = vars(args).copy()
    cfg_kwargs.update(
        latent_size=(args.resize_h // args.spatial_downsample, args.resize_w // args.spatial_downsample),
    )
    cfg = DenoiserConfig(**cfg_kwargs)
    return build_denoiser_from_mode(cfg)


def load_pretrained(
    path: str,
    model: torch.nn.Module,
    ema_model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    progress: Optional[dict] = None,
) -> tuple[int, int]:
    """Load a MiniWorld checkpoint, allowing shape changes between curriculum stages."""
    ckpt = torch.load(path, map_location="cpu")
    if progress is not None:
        progress.update(checkpoint_progress(ckpt))
    recorded = ckpt.get("meta", {})
    if "backbone" in recorded:
        expected = backbone_config(model.cfg)
        mismatches = [key for key, value in expected.items() if key in recorded and recorded[key] != value]
        if mismatches:
            raise ValueError(f"Checkpoint backbone configuration differs: {mismatches}")
    if 'df_chunk_size' in recorded and recorded['df_chunk_size'] != model.cfg.df_chunk_size:
        raise ValueError('Checkpoint chunk size differs from the requested training protocol')
    model_state = model.state_dict()
    raw_model = ckpt.get("model", ckpt.get("ema_model", {}))
    filtered = {k: v for k, v in raw_model.items() if k in model_state and model_state[k].shape == v.shape}
    info = model.load_state_dict(filtered, strict=False)
    print0(f"[Checkpoint] loaded model from {path}: {info}")

    ema_state = ema_model.state_dict()
    raw_ema = ckpt.get("ema_model", raw_model)
    filtered_ema = {k: v for k, v in raw_ema.items() if k in ema_state and ema_state[k].shape == v.shape}
    ema_model.load_state_dict(filtered_ema, strict=False)
    if optimizer is not None and "optimizer" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except ValueError:
            print0("[Checkpoint] skipped optimizer state because parameter groups changed")
    return int(ckpt.get("epoch", 0)), int(ckpt.get("global_step", 0))


def save_checkpoint(
    *,
    args: argparse.Namespace,
    model: torch.nn.Module,
    ema_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
) -> None:
    """Save the current model, EMA, optimizer, and metadata."""
    if not is_main_process():
        return
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "ema_model": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "total_train_steps": progress_fields(args, global_step)["train_step"],
        "curriculum_stage": getattr(args, "curriculum_stage", 1),
        "config": vars(args).copy(),
        "meta": {
            **backbone_config(args),
            "wm_model": args.wm_model,
            "trained_num_frames": args.latent_frames,
            "df_chunk_size": args.df_chunk_size,
            "latent_frames": args.latent_frames,
            "use_pose_cond": bool(args.use_pose_cond),
            "use_action_cond": bool(args.use_action_cond),
            "cond_dim": int(args.cond_dim),
            "cond_per_token": bool(args.cond_per_token),
            "timestep_baseshift": float(args.timestep_baseshift),
            # Resolved shift, i.e. baseshift already scaled by the token count.
            "timestep_shift": float(getattr(model, "timestep_shift", -1.0)),
        },
    }
    epoch_path = output_dir / f"epoch_{epoch:04d}_step_{global_step:08d}.pt"
    torch.save(payload, epoch_path)
    torch.save(payload, output_dir / "last.pt")
    print0(f"[Checkpoint] saved {epoch_path}")


def public_wandb_settings(wandb, mode="online"):
    """Do not send an inherited private-host API key to the public W&B service."""
    import netrc
    settings = {'base_url': 'https://api.wandb.ai'}
    inherited_host = os.environ.get('WANDB_BASE_URL', settings['base_url']).rstrip('/')
    if mode == 'online' and inherited_host != settings['base_url'] and os.environ.get('WANDB_API_KEY'):
        try:
            credential = netrc.netrc().authenticators('api.wandb.ai')
        except (OSError, netrc.NetrcParseError):
            credential = None
        if not credential:
            raise RuntimeError('Inherited W&B credentials target another host; configure a public W&B login')
        settings['api_key'] = credential[2]
    return wandb.Settings(**settings)


def init_wandb(args: argparse.Namespace, global_batch_size: int) -> Optional[Any]:
    """Create one explicitly routed run; propagate rank-0 startup errors to DDP peers."""
    if not args.wandb:
        return None
    run, error = None, None
    if is_main_process():
        try:
            import wandb
            config = vars(args).copy()
            config.update(batch_size=global_batch_size, global_batch_size=global_batch_size)
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            run_id, run_name, resume_policy, run_file = wandb_session(args)
            stage = getattr(args, 'curriculum_stage', 1)
            # Keep each stage's hyperparameters separately; never replace stage1's
            # top-level config with a different video length/batch size on resume.
            session_config = {f'curriculum_stage_{stage}': config}
            if resume_policy == 'never':
                session_config.update(config)
            run = wandb.init(entity=args.wandb_entity, project=args.wandb_project,
                             name=run_name, group=args.wandb_group, config=session_config,
                             allow_val_change=True,
                             dir=args.output_dir, mode=args.wandb_mode,
                             id=run_id, resume=resume_policy,
                             settings=public_wandb_settings(wandb, args.wandb_mode))
            run.define_metric('train_step')
            run.define_metric('train/*', step_metric='train_step')
            run.define_metric('eval/*', step_metric='train_step')
            run.define_metric('curriculum/*', step_metric='train_step')
            for metric, direction in [('psnr', 'max'), ('ssim', 'max'), ('lpips', 'min')]:
                run.define_metric('eval/' + metric, step_metric='train_step', summary=direction)
            record = dict(entity=args.wandb_entity, project=args.wandb_project, name=run_name,
                          id=run.id, url=run.url, mode=args.wandb_mode)
            write_json(Path(args.output_dir)/'wandb_run.json', record)
            if run_file:
                write_json(run_file, record)
            print0(f'[W&B] {args.wandb_entity}/{args.wandb_project}: {run_name} ({run.url}; {resume_policy})')
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
            if run is not None:
                run.finish(exit_code=1)
    errors = gather_results(error)
    if any(errors):
        raise RuntimeError('W&B initialization failed; configure public credentials or use --no-wandb: '
                           + '; '.join(e for e in errors if e))
    return run


def resolve_stream_chunks(latent_frames: int, chunk_size: int) -> tuple[int, int]:
    window_chunks = max(1, int(latent_frames) // max(1, int(chunk_size)))
    inflight_chunks = max(1, window_chunks // 4)
    return window_chunks - inflight_chunks, inflight_chunks


def capture_fixed_inputs(
    latents: torch.Tensor,
    cond_seq: torch.Tensor,
    video_for_vae: torch.Tensor,
    seed: int = 42,
) -> Dict[str, torch.Tensor]:
    """Freeze one clip and its initial noise so rollouts stay comparable.

    Cloned because the training tensors are overwritten every step; the noise
    uses its own generator to leave the global RNG stream untouched.
    """
    fixed_latents = latents[:1].detach().clone()
    generator = torch.Generator(device=fixed_latents.device).manual_seed(seed)
    return {
        "latents": fixed_latents,
        "cond_seq": cond_seq[:1].detach().clone(),
        "video": video_for_vae[:1].detach().clone(),
        "noise": torch.randn(
            fixed_latents.shape,
            generator=generator,
            device=fixed_latents.device,
            dtype=fixed_latents.dtype,
        ),
    }


def side_by_side_uint8(gt: torch.Tensor, pred: torch.Tensor):
    """Lay out two ``(C, T, H, W)`` clips in ``[-1, 1]`` side by side.

    Returns the ``(T, C, H, 2W)`` uint8 array ``wandb.Video`` expects.
    """
    frames = min(gt.shape[1], pred.shape[1])
    pair = torch.cat([gt[:, :frames], pred[:, :frames]], dim=3).permute(1, 0, 2, 3)
    return ((pair.float().clamp(-1, 1) + 1.0) * 127.5).to(torch.uint8).cpu().numpy()


def log_train_videos(
    *,
    wandb_run: Any,
    ema_denoiser: torch.nn.Module,
    vae: Any,
    args: argparse.Namespace,
    fixed: Dict[str, torch.Tensor],
    x_pred: torch.Tensor,
    video_for_vae: torch.Tensor,
    t_noise: torch.Tensor,
    global_step: int,
    dtype: torch.dtype,
) -> None:
    """Log a one-step denoising video and an EMA rollout from frozen inputs."""
    import wandb

    max_cache_chunks, inflight_chunks = resolve_stream_chunks(args.latent_frames, args.df_chunk_size)
    autocast = torch.autocast(
        device_type="cuda",
        dtype=dtype,
        enabled=args.mixed_precision == "bf16" and torch.cuda.is_available(),
    )
    with torch.no_grad(), autocast:
        recon = side_by_side_uint8(video_for_vae[0], vae_decode(vae, x_pred[:1].float())[0])
        ema_denoiser.eval()
        _, generated_rgb = ema_denoiser.generate_eval_latents_streaming(
            fixed["latents"],
            fixed["cond_seq"],
            total_len=args.latent_frames,
            # Image-to-video from the first frame, as in miniworld.sample.
            history_len=1,
            max_cache_chunks=max_cache_chunks,
            inflight_chunks=inflight_chunks,
            sink_frames=1 if max_cache_chunks > 0 else 0,
            noise=fixed["noise"],
            stream_decoder=StreamingVAEDecoder(vae),
        )
        generated = side_by_side_uint8(fixed["video"][0], generated_rgb[0])

    sampling = getattr(ema_denoiser, 'last_eval_meta', {})
    sampling_scalars = ({'train/gen_effective_sampling_steps': sampling['effective_steps']}
                        if 'effective_steps' in sampling else {})
    wandb_run.log(
        {
            **progress_fields(args, global_step),
            **sampling_scalars,
            'train/recon_t_max': float(t_noise[0]),
            "train/recon_video": wandb.Video(
                recon, fps=args.video_log_fps, format="mp4", caption=f"Single-step reconstruction | max t={float(t_noise[0]):.4f} | observed context copied"
            ),
            "train/gen_video": wandb.Video(
                generated, fps=args.video_log_fps, format="mp4", caption=f"step={global_step}"
            ),
        },
    )


def build_optimizer(args: argparse.Namespace, model: torch.nn.Module) -> torch.optim.Optimizer:
    """Build AdamW or optional Muon optimizer."""
    if args.use_muon:
        from miniworld.muon import MuonWithAuxAdam, get_muon_param_groups

        return MuonWithAuxAdam(get_muon_param_groups(model, weight_decay=args.weight_decay, lr=args.lr))
    return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def parse_args() -> argparse.Namespace:
    """Parse public MiniWorld training arguments."""
    parser = argparse.ArgumentParser("Train MiniWorld")
    parser.add_argument("--dataset", choices=["droid", "re10k"], required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--pose_dir", default=None)
    parser.add_argument("--dataset_filter_cache_dir", default=None)
    parser.add_argument("--resize_h", type=int, default=240)
    parser.add_argument("--resize_w", type=int, default=320)
    parser.add_argument("--frame_interval", type=int, default=1)

    parser.add_argument("--action_camera_views", default="exterior_image_1_left")
    parser.add_argument("--action_keys", default="cartesian_position,gripper_position")
    parser.add_argument("--pose_enc_freq", type=int, default=15)

    parser.add_argument("--wm_model", choices=["B", "L", "0.5B", "1B", "3B"], default="1B")
    parser.add_argument("--wm_mlp_ratio", type=float, default=4.0)
    parser.add_argument("--wm_use_qknorm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wm_use_checkpoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wm_adaln_mode", dest="adaln_mode", choices=["adaln_lora", "fully_shared", "per_block"], default="adaln_lora")
    parser.add_argument("--wm_cond_dropout_prob", dest="cond_dropout_prob", type=float, default=0.1)

    parser.add_argument("--vae_checkpoint", required=True)
    parser.add_argument("--latent_channels", type=int, default=48)
    parser.add_argument("--spatial_downsample", type=int, default=16)

    parser.add_argument("--latent_frames", type=int, default=32)
    parser.add_argument(
        "--history_len",
        type=int,
        default=1,
        help="Kept for CLI/API compatibility; DF training samples clean context via Mode A/B.",
    )
    parser.add_argument("--df_chunk_size", type=int, default=4)
    parser.add_argument("--df_ardiff_step", type=int, default=10)
    parser.add_argument(
        "--timestep_baseshift",
        type=float,
        default=2.667,
        help="SD3-style timestep shift at 600 tokens/chunk (240x320, chunk_size=2); "
             "scaled by sqrt(chunk_size*H_lat*W_lat/600). Higher leans training "
             "towards noisier timesteps.",
    )
    parser.add_argument(
        "--timestep_shift",
        type=float,
        default=-1.0,
        help="Absolute timestep shift; overrides --timestep_baseshift when > 0.",
    )

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--mixed_precision", choices=["no", "bf16"], default="bf16")
    parser.add_argument("--use_muon", action="store_true")

    parser.add_argument("--output_dir", default="outputs/miniworld")
    parser.add_argument("--save_every_epoch", type=int, default=1)
    parser.add_argument("--load_pretrained", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--max_train_steps", type=int, default=0)

    parser.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Log to W&B; initialization errors stop training. --no-wandb disables logging.",
    )
    parser.add_argument("--wandb_entity", default="LVSM-Experiment")
    parser.add_argument("--wandb_project", default="miniworld")
    parser.add_argument("--wandb_group", default=None)
    parser.add_argument("--wandb_run_file", default=None, help="Shared curriculum run identity JSON")
    parser.add_argument("--curriculum_stage", type=int, choices=[1, 2, 3, 4], default=1)
    parser.add_argument("--step_offset", type=int, default=None, help="Default: inferred from preceding checkpoint")
    parser.add_argument("--wandb_mode", choices=['online', 'offline'], default='online')
    parser.add_argument("--wandb_name", default=None, help="Custom run name; defaults to dataset/backbone/model/stage/seed plus a unique suffix.")
    parser.add_argument(
        "--image_log_every",
        type=int,
        default=1000,
        help="Steps between logging denoising and generation videos to W&B. "
             "0 disables; each event costs one EMA rollout on rank 0.",
    )
    parser.add_argument("--video_log_fps", type=int, default=8, help="Frame rate of the videos logged to W&B.")
    parser.add_argument("--seed", type=int, default=42)
    add_validation_args(parser)
    add_backbone_args(parser)
    args = parser.parse_args()
    if args.df_chunk_size < 1:
        parser.error('--df_chunk_size must be positive')
    if args.backbone in ('rtransformer', 'tas') and args.latent_frames <= args.df_chunk_size:
        parser.error('Recurrent training requires more than one chunk so state writers receive gradients')
    if args.step_offset is not None and args.step_offset < 0:
        parser.error('--step_offset must be nonnegative')
    if args.curriculum_stage > 1 and not (args.load_pretrained or args.resume):
        parser.error('Later curriculum stages require --load_pretrained or --resume')
    validate_options(args)
    if args.wandb_name is None:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
        args.wandb_name = (f'{args.dataset}_{args.backbone}_{args.wm_model}_lf{args.latent_frames}'
                           f'_s{args.seed}_{stamp}_{uuid.uuid4().hex[:6]}')
    return args


def main() -> None:
    """Run MiniWorld training."""
    args = parse_args()
    dist_info = setup_distributed()
    device = torch.device(f"cuda:{dist_info['local_rank']}" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    args.git_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    lock_path = Path(__file__).resolve().parents[1] / "uv.lock"
    args.dependency_lock_sha256 = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    torch.backends.cudnn.benchmark = True
    wandb_run = None

    try:
        dataset = build_dataset(args, randomize=True, color_aug=True)
        resolve_conditioning(args, dataset)
        evaluator, eval_setup_error = None, None
        if args.eval_every:
            try:
                evaluator = PeriodicEvaluator(args, dataset, device)
            except Exception as exc:
                eval_setup_error = f'{type(exc).__name__}: {exc}'
            setup_errors = gather_results(eval_setup_error)
            if any(setup_errors):
                raise RuntimeError('Evaluation setup failed: ' + '; '.join(e for e in setup_errors if e))
        sampler = DistributedSampler(dataset, shuffle=True, drop_last=True) if dist_info["distributed"] else None
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
            prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
            drop_last=True,
        )

        vae = load_wan22_vae(args)
        denoiser = build_denoiser(args).to(device)
        print0(f"[MiniWorld] trainable params={sum(p.numel() for p in denoiser.parameters() if p.requires_grad):,}")

        if dist_info["distributed"]:
            denoiser = DDP(denoiser, device_ids=[dist_info["local_rank"]], output_device=dist_info["local_rank"])
        target = denoiser.module if hasattr(denoiser, "module") else denoiser
        ema_denoiser = copy.deepcopy(target).requires_grad_(False)
        optimizer = build_optimizer(args, denoiser)
        global_batch_size = args.batch_size * int(dist_info["world_size"])
        start_epoch, global_step = 0, 0
        restored_progress = None
        if args.load_pretrained:
            restored_progress = {}
            load_pretrained(args.load_pretrained, target, ema_denoiser, progress=restored_progress)
        elif args.resume:
            latest = find_latest_checkpoint(args.output_dir)
            if latest:
                restored_progress = {}
                start_epoch, global_step = load_pretrained(latest, target, ema_denoiser, optimizer,
                                                          progress=restored_progress)
        resolve_step_offset(args, restored_progress)
        if is_main_process():
            write_json(Path(args.output_dir)/'config.json', vars(args))
        wandb_run = init_wandb(args, global_batch_size)

        dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float32
        last_log_time, last_log_step = time.time(), global_step
        fixed_inputs: Optional[Dict[str, torch.Tensor]] = None
        for epoch in range(start_epoch, args.max_epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)
            denoiser.train()
            for step, batch in enumerate(dataloader):
                videos = batch["videos"].to(device, non_blocking=True)
                poses = batch.get("poses")
                actions = batch.get("actions")
                if poses is not None:
                    poses = poses.to(device, non_blocking=True)
                if actions is not None:
                    actions = actions.to(device, non_blocking=True)

                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype, enabled=args.mixed_precision == "bf16" and torch.cuda.is_available()):
                    video_for_vae = rearrange(videos, "b t h w c -> b c t h w").contiguous()
                    latents = vae_encode(vae, video_for_vae)
                    _, _, t_latent, h_lat, w_lat = latents.shape
                    cond_cfg = ConditioningConfig(
                        use_pose_cond=args.use_pose_cond,
                        use_action_cond=args.use_action_cond,
                        pose_enc_freq=args.pose_enc_freq,
                    )
                    cond_seq = build_cond_seq_for_batch(
                        cfg=cond_cfg,
                        poses=poses,
                        actions=actions,
                        t_latent=t_latent,
                        h_lat=h_lat,
                        w_lat=w_lat,
                    )

                # Step count only, so every rank runs the same forward.
                log_videos = args.wandb and args.image_log_every > 0 and (global_step + 1) % args.image_log_every == 0

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=dtype, enabled=args.mixed_precision == "bf16" and torch.cuda.is_available()):
                    outputs = denoiser(latents, cond_seq, history_len=args.history_len, return_pred=log_videos)
                loss, x_pred, t_noise = outputs if log_videos else (outputs, None, None)
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(denoiser.parameters(), args.grad_clip)
                optimizer.step()
                update_ema(target, ema_denoiser, args.ema_decay)
                global_step += 1

                if global_step % args.log_every == 0:
                    now = time.time()
                    steps_per_sec = (global_step - last_log_step) / max(now - last_log_time, 1e-6)
                    last_log_time, last_log_step = now, global_step
                    loss_value = float(loss.detach())
                    current_lr = float(optimizer.param_groups[0]["lr"])
                    print0(
                        f"[Train] epoch={epoch} step={step} global_step={global_step} "
                        f"total_step={progress_fields(args, global_step)['train_step']} "
                        f"stage={args.curriculum_stage} loss={loss_value:.6f} lr={current_lr:.2e} speed={steps_per_sec:.2f} step/s"
                    )
                    if wandb_run is not None:
                        wandb_run.log(
                            {
                                **progress_fields(args, global_step),
                                "train/loss": loss_value,
                                "train/lr": current_lr,
                                "train/step_per_sec": steps_per_sec,
                                # Comparable across curriculum stages, unlike
                                # step/s, since each stage uses its own batch size.
                                "train/sample_per_sec": steps_per_sec * global_batch_size,
                                "epoch": int(epoch),
                            },
                        )

                if log_videos and wandb_run is not None:
                    if fixed_inputs is None:
                        fixed_inputs = capture_fixed_inputs(latents, cond_seq, video_for_vae)
                    try:
                        log_train_videos(
                            wandb_run=wandb_run,
                            ema_denoiser=ema_denoiser,
                            vae=vae,
                            args=args,
                            fixed=fixed_inputs,
                            x_pred=x_pred,
                            video_for_vae=video_for_vae,
                            t_noise=t_noise,
                            global_step=global_step,
                            dtype=dtype,
                        )
                    except Exception as exc:  # noqa: BLE001 - never let logging break training
                        print0(f"[W&B] video logging failed at step {global_step} ({exc})")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                if due_for_evaluation(global_step, args.eval_every):
                    evaluator.run(ema_denoiser, vae, global_step, wandb_run)
                    # Evaluation time must not lower the reported training throughput.
                    last_log_time, last_log_step = time.time(), global_step

                if args.max_train_steps > 0 and global_step >= args.max_train_steps:
                    save_checkpoint(
                        args=args,
                        model=target,
                        ema_model=ema_denoiser,
                        optimizer=optimizer,
                        epoch=epoch + 1,
                        global_step=global_step,
                    )
                    return

            if (epoch + 1) % args.save_every_epoch == 0:
                save_checkpoint(
                    args=args,
                    model=target,
                    ema_model=ema_denoiser,
                    optimizer=optimizer,
                    epoch=epoch + 1,
                    global_step=global_step,
                )
    finally:
        if wandb_run is not None:
            wandb_run.finish(exit_code=1 if sys.exc_info()[0] is not None else 0)
        cleanup_distributed(bool(dist_info["distributed"]))


if __name__ == "__main__":
    main()

