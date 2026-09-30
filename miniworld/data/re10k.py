"""RealEstate10K video dataset with optional camera-pose loading."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from decord import VideoReader, cpu
from torch.utils.data import Dataset


def _print0(message: str) -> None:
    if int(os.environ.get("RANK", "0")) == 0:
        print(message, flush=True)


def _collect_mp4_files(dataset_paths: Sequence[str]) -> List[Path]:
    files: List[Path] = []
    for path in dataset_paths:
        root = Path(path)
        if root.is_file() and root.suffix.lower() == ".mp4":
            files.append(root)
        else:
            files.extend(sorted(root.rglob("*.mp4")))
    return sorted(files)


def _cache_key(files: Sequence[Path], required_frames: int) -> str:
    payload = {
        "files": [os.fspath(p) for p in files],
        "required_frames": int(required_frames),
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def _video_frame_count(path: Path) -> int:
    try:
        return int(len(VideoReader(os.fspath(path), ctx=cpu(0))))
    except Exception as exc:
        _print0(f"[Dataset] failed to read {path}: {exc}")
        return 0


def _filter_videos_by_length(
    files: Sequence[Path],
    *,
    required_frames: int,
    cache_dir: Optional[str],
    max_keep: Optional[int] = None,
) -> List[Path]:
    cache_path = None
    if cache_dir is not None:
        cache_root = Path(cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_path = cache_root / f"re10k_len_filter_{_cache_key(files, required_frames)}.json"
        if cache_path.exists():
            kept = [Path(p) for p in json.loads(cache_path.read_text())]
            return kept[:max_keep] if max_keep is not None else kept

    kept = [p for p in files if _video_frame_count(p) >= required_frames]
    if cache_path is not None:
        cache_path.write_text(json.dumps([os.fspath(p) for p in kept], indent=2))
    return kept[:max_keep] if max_keep is not None else kept


class RealEstate10KDataset(Dataset):
    """Sample clips from RealEstate10K videos and align optional pose tensors."""

    def __init__(
        self,
        dataset_paths: Sequence[str],
        num_frames: int,
        frame_interval: int,
        resize_hw: Tuple[int, int] = (240, 320),
        randomize: bool = True,
        color_aug: bool = True,
        filter_cache_dir: Optional[str] = None,
        max_keep: Optional[int] = None,
        return_pose: bool = False,
        pose_dir: Optional[str] = None,
        strict_loading: bool = False,
        return_metadata: bool = False,
    ) -> None:
        self.strict_loading = strict_loading
        self.return_metadata = return_metadata
        self.num_frames = int(num_frames)
        self.randomize = bool(randomize)
        self.resize_hw = tuple(resize_hw)
        self.color_aug = bool(color_aug)
        self.frame_interval = int(frame_interval)
        self.return_pose = bool(return_pose)
        self.pose_dir = Path(pose_dir) if pose_dir is not None else None

        if self.return_pose:
            if self.pose_dir is None or not self.pose_dir.exists():
                raise ValueError(f"return_pose=True requires an existing pose_dir, got {pose_dir}")
            if len(dataset_paths) != 1:
                raise ValueError("return_pose=True only supports a single RealEstate10K root")

        files = _collect_mp4_files(dataset_paths)
        if not files:
            raise ValueError(f"No mp4 files found in paths: {dataset_paths}")
        required_frames = (self.num_frames - 1) * self.frame_interval + 1
        self.files = _filter_videos_by_length(
            files,
            required_frames=required_frames,
            cache_dir=filter_cache_dir,
            max_keep=max_keep,
        )
        if not self.files:
            raise ValueError(f"All videos filtered out; required_frames={required_frames}")

        if self.return_pose:
            before = len(self.files)
            self.files = [p for p in self.files if (self.pose_dir / f"{p.stem}.pt").exists()]
            _print0(f"[RealEstate10K] kept {len(self.files)}/{before} videos with matching poses")
            if not self.files:
                raise ValueError(f"No matching pose files found under {self.pose_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def _decode_video(self, video_path: Path):
        required_frames = (self.num_frames - 1) * self.frame_interval + 1
        reader = VideoReader(os.fspath(video_path), ctx=cpu(0), width=self.resize_hw[1], height=self.resize_hw[0])
        total_frames = len(reader)

        if total_frames < required_frames:
            raise RuntimeError(f"Video too short: {video_path} has {total_frames}, need {required_frames}")
        max_start = total_frames - required_frames
        start = torch.randint(0, max_start + 1, ()).item() if self.randomize else 0
        frame_ids = [start + i * self.frame_interval for i in range(self.num_frames)]
        video = torch.from_numpy(reader.get_batch(frame_ids).asnumpy()).float() / 255.0
        if self.color_aug:
            video = (video + torch.rand(1) * 0.2 - 0.1).clamp(0, 1)
        return video * 2.0 - 1.0, frame_ids

    def _load_pose(self, video_path: Path, frame_ids: List[int]) -> torch.Tensor:
        pose_path = self.pose_dir / f"{video_path.stem}.pt"
        pose = torch.load(pose_path, weights_only=False)
        if pose.shape[0] < max(frame_ids) + 1:
            raise RuntimeError(f"Pose file too short: {pose_path}")
        pose = pose[frame_ids]
        pose = torch.cat([pose[:, :4], pose[:, 6:]], dim=-1).to(torch.float32)
        if pose.shape[-1] != 16:
            raise RuntimeError(f"Unexpected pose layout: {pose.shape}")
        return pose

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        for off in range(len(self.files)):
            path = self.files[(idx + off) % len(self.files)]
            try:
                video, frame_ids = self._decode_video(path)
                sample: Dict[str, torch.Tensor | str] = {"videos": video}
                if self.return_pose:
                    sample["poses"] = self._load_pose(path, frame_ids)
                if self.return_metadata:
                    sample.update(sample_id=path.stem, source_path=str(path), frame_ids=torch.tensor(frame_ids))
                return sample
            except Exception as exc:
                if self.strict_loading:
                    raise RuntimeError(f"Evaluation sample failed: {path}") from exc
                _print0(f"[RealEstate10K] skip {path}: {exc}")
        raise RuntimeError("All samples failed to decode")
