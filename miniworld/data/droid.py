"""LeRobot-v2.0 dataset for real action-conditioned world-model training.

Targets the DreamZero-DROID release (``GEAR-Dreams/DreamZero-DROID-Data``),
which stores, per episode:

* ``data/chunk-{c:03d}/episode_{e:06d}.parquet``  -- per-frame tabular data with
  a 28-dim ``action`` column and 14-dim ``observation.state`` column.
* ``videos/chunk-{c:03d}/{video_key}/episode_{e:06d}.mp4`` -- one mp4 per camera.

This dataset returns the real robot action aligned frame-by-frame with the
sampled video clip:

    __getitem__ -> {"videos": (T, H, W, C) in [-1, 1],
                    "actions": (T - 1, d_action)}   # d_action per --action_keys

The ``actions`` tensor plugs directly into
``miniworld.conditioning.actions.build_cond_seq_from_actions``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def _print0(*a, **k):
    if int(os.environ.get("RANK", "0")) == 0:
        print(*a, **k, flush=True)


# --------------------------------------------------------------------------- #
#                          meta / normalization helpers                       #
# --------------------------------------------------------------------------- #
def resolve_action_slices(
    modality: dict, action_keys: Sequence[str]
) -> List[Tuple[int, int]]:
    """Map named action modalities to (start, end) column slices via modality.json."""
    action_mod = modality["action"]
    slices: List[Tuple[int, int]] = []
    for key in action_keys:
        if key not in action_mod:
            raise KeyError(
                f"action key {key!r} not in modality.json action keys "
                f"{list(action_mod.keys())}"
            )
        slices.append((int(action_mod[key]["start"]), int(action_mod[key]["end"])))
    return slices


def _gather_cols(slices: Sequence[Tuple[int, int]]) -> np.ndarray:
    """Flatten (start, end) slices into an explicit column-index array."""
    cols: List[int] = []
    for s, e in slices:
        cols.extend(range(s, e))
    return np.asarray(cols, dtype=np.int64)


def build_action_normalizer(
    stats: dict,
    slices: Sequence[Tuple[int, int]],
    mode: str = "q01q99",
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Return (lo, hi/mean, mode) arrays for the selected action columns.

    * ``q01q99``  -> lo=q01, hi=q99 ; normalize as 2*(x-lo)/(hi-lo)-1, then clip.
    * ``meanstd`` -> lo=mean, hi=std ; normalize as (x-mean)/std.
    * ``none``    -> identity.
    """
    cols = _gather_cols(slices)
    a = stats["action"]
    if mode == "q01q99":
        lo = np.asarray(a["q01"], dtype=np.float64)[cols]
        hi = np.asarray(a["q99"], dtype=np.float64)[cols]
        span = hi - lo
        span[np.abs(span) < 1e-6] = 1.0  # avoid div-by-zero on constant dims
        return lo, span, "q01q99"
    if mode == "meanstd":
        mean = np.asarray(a["mean"], dtype=np.float64)[cols]
        std = np.asarray(a["std"], dtype=np.float64)[cols]
        std[np.abs(std) < 1e-6] = 1.0
        return mean, std, "meanstd"
    if mode == "none":
        n = len(cols)
        return np.zeros(n), np.ones(n), "none"
    raise ValueError(f"unknown action_norm mode: {mode}")


def _apply_norm(raw: np.ndarray, lo: np.ndarray, hi: np.ndarray, mode: str) -> np.ndarray:
    if mode == "q01q99":
        out = 2.0 * (raw - lo) / hi - 1.0
        return np.clip(out, -1.0, 1.0)
    if mode == "meanstd":
        return (raw - lo) / hi
    return raw


# --------------------------------------------------------------------------- #
#                                  dataset                                     #
# --------------------------------------------------------------------------- #
class LeRobotActionDataset(Dataset):
    """Real action-conditioned video clips from a LeRobot-v2.0 DROID dataset."""

    def __init__(
        self,
        root: str,
        num_frames: int,
        frame_interval: int,
        resize_hw: Tuple[int, int] = (240, 320),
        camera_views: Optional[Sequence[str]] = None,
        action_keys: Sequence[str] = ("cartesian_position", "gripper_position"),
        action_norm: str = "q01q99",
        randomize: bool = True,
        color_aug: bool = True,
        require_success: bool = True,
        max_keep: Optional[int] = None,
        strict_loading: bool = False,
        return_metadata: bool = False,
    ) -> None:
        self.root = Path(root)
        self.strict_loading = strict_loading
        self.return_metadata = return_metadata
        self.num_frames = num_frames
        self.frame_interval = frame_interval
        self.resize_hw = tuple(resize_hw)
        self.randomize = randomize
        self.color_aug = color_aug

        info = json.loads((self.root / "meta" / "info.json").read_text())
        modality = json.loads((self.root / "meta" / "modality.json").read_text())
        stats = json.loads((self.root / "meta" / "stats.json").read_text())

        self.chunks_size = int(info["chunks_size"])
        self.data_tmpl = info["data_path"]
        self.video_tmpl = info["video_path"]

        # camera view(s): map friendly key -> original observation key.
        vid_mod = modality["video"]
        all_views = {k: v["original_key"] for k, v in vid_mod.items()}
        if camera_views is None:
            camera_views = ["exterior_image_1_left"]
        self.video_keys: List[str] = []
        for cv in camera_views:
            if cv in all_views:
                self.video_keys.append(all_views[cv])  # e.g. observation.images.*
            elif cv in all_views.values():
                self.video_keys.append(cv)
            else:
                raise KeyError(
                    f"camera view {cv!r} not found. options: "
                    f"{list(all_views.keys())}"
                )

        # action columns + normalizer.
        self.action_keys = list(action_keys)
        self.slices = resolve_action_slices(modality, self.action_keys)
        self.action_cols = _gather_cols(self.slices)
        self.d_action = int(len(self.action_cols))
        self.norm_lo, self.norm_hi, self.norm_mode = build_action_normalizer(
            stats, self.slices, mode=action_norm
        )
        self.norm_lo_t = torch.from_numpy(self.norm_lo).float()
        self.norm_hi_t = torch.from_numpy(self.norm_hi).float()

        # enumerate + length-filter episodes from meta/episodes.jsonl (no file I/O).
        required_frames = (num_frames - 1) * frame_interval + 1
        episodes = self._load_episode_index(require_success)
        self.samples: List[int] = [
            ep for ep, length in episodes if length >= required_frames
        ]
        _print0(
            f"[LeRobot] {root}: {len(episodes)} episodes -> "
            f"{len(self.samples)} kept (required_frames={required_frames}, "
            f"require_success={require_success})"
        )
        if max_keep is not None:
            self.samples = self.samples[:max_keep]
        if len(self.samples) == 0:
            raise ValueError(
                f"No episode long enough in {root} (required {required_frames} "
                f"frames). Reduce --latent_frames / --frame_interval."
            )
        _print0(
            f"[LeRobot] action_keys={self.action_keys} d_action={self.d_action} "
            f"views={self.video_keys} norm={self.norm_mode}"
        )

    # ------------------------------------------------------------------ #
    def _load_episode_index(self, require_success: bool) -> List[Tuple[int, int]]:
        path = self.root / "meta" / "episodes.jsonl"
        out: List[Tuple[int, int]] = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if require_success and not rec.get("success", True):
                    continue
                out.append((int(rec["episode_index"]), int(rec["length"])))
        return out

    def _chunk(self, ep: int) -> int:
        return ep // self.chunks_size

    def _parquet_path(self, ep: int) -> Path:
        return self.root / self.data_tmpl.format(
            episode_chunk=self._chunk(ep), episode_index=ep
        )

    def _video_path(self, ep: int, video_key: str) -> Path:
        return self.root / self.video_tmpl.format(
            episode_chunk=self._chunk(ep), video_key=video_key, episode_index=ep
        )

    def __len__(self) -> int:
        return len(self.samples)

    # ------------------------------------------------------------------ #
    def _decode_clip(self, video_path: Path) -> Tuple[torch.Tensor, List[int]]:
        required_frames = (self.num_frames - 1) * self.frame_interval + 1
        from decord import VideoReader, cpu

        vr = VideoReader(
            os.fspath(video_path), ctx=cpu(0),
            width=self.resize_hw[1], height=self.resize_hw[0],
        )
        total = len(vr)

        if total < required_frames:
            raise RuntimeError(
                f"Video too short: {video_path} has {total} frames, "
                f"need {required_frames}"
            )
        max_start = total - required_frames
        start = torch.randint(0, max_start + 1, ()).item() if self.randomize else 0
        frame_ids = [start + i * self.frame_interval for i in range(self.num_frames)]

        arr = vr.get_batch(frame_ids).asnumpy()
        video = torch.from_numpy(arr).float() / 255.0

        if self.color_aug:
            video = (video + torch.rand(1) * 0.2 - 0.1).clamp(0, 1)
        video = video * 2.0 - 1.0  # to [-1, 1]
        return video, frame_ids

    def _load_actions(self, ep: int, frame_ids: List[int]) -> torch.Tensor:
        """Real per-frame action for the sampled frames, sliced + normalized.

        Returns ``(num_frames - 1, d_action)``: the action at each of the first
        ``T-1`` frames (frame i's command drives the transition to frame i+1),
        matching the causal VAE latent-frame convention.
        """
        import pandas as pd  # local import: keep worker startup cheap

        df = pd.read_parquet(self._parquet_path(ep), columns=["action"])
        act = np.stack(df["action"].values).astype(np.float64)  # (N, 28)
        need = frame_ids[:-1]  # T-1 actions
        if act.shape[0] <= max(need):
            raise RuntimeError(
                f"parquet too short for ep {ep}: {act.shape[0]} rows, "
                f"need > {max(need)}"
            )
        sel = act[np.asarray(need)][:, self.action_cols]  # (T-1, d_action)
        sel = _apply_norm(sel, self.norm_lo, self.norm_hi, self.norm_mode)
        return torch.from_numpy(sel).float()

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        n = len(self.samples)
        for off in range(n):
            ep = self.samples[(idx + off) % n]
            try:
                view = (
                    self.video_keys[torch.randint(0, len(self.video_keys), ()).item()]
                    if len(self.video_keys) > 1
                    else self.video_keys[0]
                )
                video, frame_ids = self._decode_clip(self._video_path(ep, view))
                actions = self._load_actions(ep, frame_ids)
                sample = {"videos": video, "actions": actions}
                if self.return_metadata:
                    sample.update(sample_id=f"episode_{ep:06d}:{view}",
                                  source_path=str(self._video_path(ep, view)), frame_ids=torch.tensor(frame_ids))
                return sample
            except Exception as e:  # noqa: BLE001
                if self.strict_loading:
                    raise RuntimeError(f"Evaluation episode failed: {ep}") from e
                print(f"[LeRobot][Skip] ep={ep} err={e}", flush=True)
                continue
        raise RuntimeError("All samples failed to decode in this dataset shard")
