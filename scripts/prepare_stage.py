#!/usr/bin/env python3
"""Prepare-stage GPU reservation on explicitly authorized devices."""
import argparse
import json
import os
from pathlib import Path
import signal
import threading

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--memory-gib', type=float, default=64)
    parser.add_argument('--pid-file', required=True)
    args = parser.parse_args()
    devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not devices:
        parser.error('Set CUDA_VISIBLE_DEVICES to authorized idle GPUs')
    if args.memory_gib <= 0:
        parser.error('--memory-gib must be positive')
    stopped = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stopped.set())
    allocations = []
    try:
        size = int(args.memory_gib * 2**30)
        for index in range(torch.cuda.device_count()):
            free, _ = torch.cuda.mem_get_info(index)
            if free < size + 2**30:
                raise RuntimeError(f'GPU {index} lacks enough free memory for reservation')
            allocations.append(torch.empty(size, dtype=torch.uint8, device=f'cuda:{index}'))
        record = dict(pid=os.getpid(), cuda_visible_devices=devices, memory_gib_per_gpu=args.memory_gib)
        path = Path(args.pid_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2)+'\n')
        print(json.dumps(record), flush=True)
        while not stopped.wait(1):
            pass
    finally:
        allocations.clear()
        for index in range(torch.cuda.device_count()):
            with torch.cuda.device(index):
                torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
