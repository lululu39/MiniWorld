#!/usr/bin/env python3
"""Hand off an already-running stage without restarting its trainer.

The old shell supervisor is paused externally; its foreground child finishes
normally and remains waitable as a zombie. Only after exit0 and a final
checkpoint does this controller replace the obsolete shell with a new launcher.
"""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(') ', 1)[1].split()
    except FileNotFoundError:
        return None
    return dict(state=fields[0], starttime=int(fields[19]), wait_status=int(fields[49]))


def record(path, **value):
    value['time'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--poll_seconds', type=float, default=10)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    status = args.config.with_name('continuation_status.json')
    lock = args.config.with_suffix('.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        record(status, status='waiting_for_stage1', controller_pid=os.getpid(),
               trainer_pid=config['trainer_pid'], revision=config['revision'])
        while True:
            parent, trainer = identity(config['supervisor_pid']), identity(config['trainer_pid'])
            if not parent or parent['starttime'] != config['supervisor_starttime'] or parent['state'] not in ('T', 't'):
                raise RuntimeError('Original supervisor is missing, replaced or no longer paused; refusing automatic continuation')
            if not trainer or trainer['starttime'] != config['trainer_starttime']:
                raise RuntimeError('Original stage process disappeared/replaced; no verified exit status')
            if trainer['state'] == 'Z':
                code = os.waitstatus_to_exitcode(trainer['wait_status'])
                if code != 0:
                    raise RuntimeError(f'Stage1 exited with code {code}; later stages were not launched')
                break
            time.sleep(args.poll_seconds)
        if not Path(config['expected_checkpoint']).is_file():
            raise RuntimeError('Stage1 final checkpoint is missing; later stages were not launched')
        # Only the paused shell is removed; its trainer has already exited.
        os.kill(config['supervisor_pid'], signal.SIGKILL)
        with Path(config['log']).open('ab', buffering=0) as log:
            child = subprocess.Popen(config['command'], cwd=config['source'],
                                     env=os.environ | {'START_STAGE': '2'},
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        record(status, status='running_remaining_stages', controller_pid=os.getpid(),
               launcher_pid=child.pid, revision=config['revision'])
        code = child.wait()
        record(status, status='completed' if code == 0 else 'failed', exit_code=code,
               controller_pid=os.getpid(), launcher_pid=child.pid, revision=config['revision'])
    except Exception as exc:
        record(status, status='failed', error=f'{type(exc).__name__}: {exc}', controller_pid=os.getpid())
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    main()
