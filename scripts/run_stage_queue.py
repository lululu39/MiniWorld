#!/usr/bin/env python3
"""Run the configured model/stage entries sequentially on the same GPUs."""
import argparse
import fcntl
import json
import netrc
import os
from pathlib import Path
import signal
import subprocess
import time


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    root = args.config.parent
    lock = (root/'queue.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = dict(status='starting', pid=os.getpid(), revision=config['revision'],
                 completed=[], order=[entry['backbone'] for entry in config['entries']])
    child = None
    def interrupted(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    env = os.environ | config['environment']
    credential = netrc.netrc().authenticators('api.wandb.ai')
    if not credential:
        raise RuntimeError('Public W&B credential is missing')
    # Runtime only; never put this key in config/status/command text.
    env.update(WANDB_BASE_URL='https://api.wandb.ai', WANDB_API_KEY=credential[2])
    try:
        for entry in config['entries']:
            occupied = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                                '--format=csv,noheader,nounits'], text=True).strip()
            if occupied:
                raise RuntimeError('GPUs have compute processes; refusing to launch over another job')
            log_path = Path(entry['log'])
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open('ab', buffering=0) as log:
                child = subprocess.Popen(entry['command'], cwd=config['source'],
                         env=env | entry['environment'], stdout=log,
                         stderr=subprocess.STDOUT, start_new_session=True)
            state.update(status='running', current=entry['backbone'], child_pid=child.pid,
                         log=str(log_path), started_at=time.time())
            write(root/'queue_status.json', state)
            code = child.wait()
            child = None
            if code != 0:
                raise RuntimeError(f"{entry['backbone']} exited with code {code}; queue stopped")
            if not Path(entry['expected_checkpoint']).is_file():
                raise RuntimeError('Final stage checkpoint missing; queue stopped')
            state['completed'].append(entry['backbone'])
        state.update(status='completed', finished_at=time.time())
        write(root/'queue_status.json', state)
    except BaseException as exc:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        state.update(status='cancelled' if isinstance(exc, KeyboardInterrupt) else 'failed',
                     error=f'{type(exc).__name__}: {exc}', finished_at=time.time())
        write(root/'queue_status.json', state)
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    main()
