"""Evaluate explicitly paired saved videos: python -m miniworld.evaluate --help."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from decord import VideoReader, cpu
from miniworld.metrics import VideoMetrics, write_report


def read_video(path, num_frames=None, resize_hw=None):
    kwargs = {} if resize_hw is None else dict(height=resize_hw[0], width=resize_hw[1])
    reader = VideoReader(str(path), ctx=cpu(0), **kwargs)
    count = len(reader) if num_frames is None else num_frames
    if count < 1 or count > len(reader):
        raise ValueError(f'{path}: requested {count} frames, available {len(reader)}')
    video = torch.from_numpy(reader.get_batch(list(range(count))).asnumpy()).permute(0, 3, 1, 2)
    return video.float() / 127.5 - 1


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--manifest', type=Path, required=True,
                        help='JSONL: {"id": "clip", "prediction": "pred.mp4", "target": "gt.mp4"}')
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--context_frames', type=int, default=1, help='Number of initial RGB frames to exclude')
    parser.add_argument('--num_frames', type=int, default=None, help='Explicitly use the first N frames of both videos')
    parser.add_argument('--resize_hw', type=int, nargs=2, default=None, metavar=('H', 'W'))
    parser.add_argument('--frame_batch_size', type=int, default=4)
    parser.add_argument('--lpips_net', choices=['vgg', 'alex'], default='vgg')
    args = parser.parse_args()
    entries = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    if not entries:
        raise ValueError('Empty evaluation manifest')
    ids = [str(e.get('id', i)) for i, e in enumerate(entries)]
    if len(ids) != len(set(ids)):
        raise ValueError('Manifest IDs must be unique')
    evaluator = VideoMetrics(args.device, args.frame_batch_size, args.lpips_net)
    rows = []
    for sample_id, entry in zip(ids, entries):
        paths = {k: (args.manifest.parent / entry[k]).resolve() for k in ('prediction', 'target')}
        pred, target = (read_video(paths[k], args.num_frames, args.resize_hw) for k in ('prediction', 'target'))
        row = evaluator.score(pred, target, args.context_frames)
        row.update(sample_id=sample_id, **{k: str(v) for k, v in paths.items()})
        rows.append(row)
        print(sample_id, row['mean'], flush=True)
    report = write_report(args.output_dir, rows, evaluator.protocol(),
                          dict(mode='offline_decoded_videos', manifest=str(args.manifest.resolve()),
                               num_frames=args.num_frames, resize_hw=args.resize_hw,
                               context_frames=args.context_frames, alignment='frame index; no FPS resampling'))
    print('Summary:', report['mean'])


if __name__ == '__main__':
    main()
