"""Continuous curriculum progress and a shared W&B run identity."""
import json
from pathlib import Path
import uuid


def progress_fields(args, stage_step):
    return {'train_step': int(getattr(args, 'step_offset', 0) or 0) + int(stage_step),
            'curriculum/stage': int(getattr(args, 'curriculum_stage', 1)),
            'curriculum/stage_step': int(stage_step),
            'curriculum/latent_frames': int(args.latent_frames)}


def checkpoint_progress(checkpoint):
    local = int(checkpoint.get('global_step', 0))
    config = checkpoint.get('config', {})
    total = int(checkpoint.get('total_train_steps', local + int(config.get('step_offset', 0) or 0)))
    return {'stage_step': local, 'total_train_steps': total,
            'curriculum_stage': int(checkpoint.get('curriculum_stage', config.get('curriculum_stage', 1)))}


def resolve_step_offset(args, progress=None):
    if getattr(args, 'step_offset', None) is not None:
        return
    if progress is None:
        args.step_offset = 0
    elif args.resume:
        args.step_offset = progress['total_train_steps'] - progress['stage_step']
    elif getattr(args, 'curriculum_stage', 1) > 1:
        if progress['curriculum_stage'] != args.curriculum_stage - 1:
            raise ValueError('Curriculum stage must initialize from the preceding stage checkpoint')
        args.step_offset = progress['total_train_steps']
    else:
        args.step_offset = 0  # Ordinary pretrained initialization starts a new experiment.


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def wandb_session(args):
    """Only explicit stage continuation/resume may reuse a persisted identity."""
    filename = getattr(args, 'wandb_run_file', None)
    stage = getattr(args, 'curriculum_stage', 1)
    path = Path(filename) if filename else None
    if path and path.exists():
        if stage == 1 and not getattr(args, 'resume', False):
            raise ValueError('W&B run file already exists; use a fresh output root or explicit --resume')
        saved = json.loads(path.read_text())
        for key, value in [('entity', args.wandb_entity), ('project', args.wandb_project), ('mode', args.wandb_mode)]:
            if saved.get(key) != value:
                raise ValueError(f'Curriculum W&B {key} differs from the recorded run')
        if not saved.get('id'):
            raise ValueError('Curriculum W&B run ID is missing')
        return saved['id'], saved['name'], 'must', path
    if stage > 1 and path:
        raise ValueError('Later curriculum stages require the preceding stage W&B run file')
    return uuid.uuid4().hex, args.wandb_name, 'never', path
