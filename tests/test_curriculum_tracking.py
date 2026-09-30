import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest
import torch
import wandb

from miniworld import train
from miniworld.tracking import checkpoint_progress, progress_fields, resolve_step_offset, wandb_session
from test_validation import Clips, TinyEMA, install_stubs


def test_legacy_and_new_checkpoint_cumulative_progress():
    old=checkpoint_progress({'global_step':97200})
    args=SimpleNamespace(step_offset=None,resume=False,curriculum_stage=2,latent_frames=16)
    resolve_step_offset(args,old)
    assert progress_fields(args,1)['train_step']==97201
    second=checkpoint_progress({'global_step':86950,'total_train_steps':184150,'curriculum_stage':2})
    args=SimpleNamespace(step_offset=None,resume=False,curriculum_stage=3,latent_frames=32)
    resolve_step_offset(args,second)
    assert progress_fields(args,30000)['train_step']==214150
    args=SimpleNamespace(step_offset=None,resume=True,curriculum_stage=2,latent_frames=16)
    resolve_step_offset(args,second)
    assert args.step_offset==97200


def test_run_identity_requires_explicit_continuation(tmp_path):
    path=tmp_path/'run.json'
    args=SimpleNamespace(wandb_run_file=str(path),curriculum_stage=2,resume=False,
                         wandb_entity='LVSM-Experiment',wandb_project='miniworld',wandb_mode='online',wandb_name='experiment')
    with pytest.raises(ValueError,match='preceding stage'):wandb_session(args)
    path.write_text(json.dumps(dict(id='existing-id',name='experiment',entity=args.wandb_entity,
                                    project=args.wandb_project,mode='online')))
    assert wandb_session(args)[:3]==('existing-id','experiment','must')
    args.curriculum_stage=1
    with pytest.raises(ValueError,match='already exists'):wandb_session(args)
    args.resume=True
    assert wandb_session(args)[0]=='existing-id'
    args.wandb_project='wrong'
    with pytest.raises(ValueError,match='project differs'):wandb_session(args)


def test_two_stages_share_id_and_continuous_metrics(tmp_path,monkeypatch):
    install_stubs(monkeypatch)
    monkeypatch.setattr(train,'build_dataset',lambda args,**kw:Clips((100,101),args.latent_frames))
    def model(args):
        net=TinyEMA();net.cfg.df_chunk_size=args.df_chunk_size
        return net
    monkeypatch.setattr(train,'build_denoiser',model)
    monkeypatch.setattr(train,'load_wan22_vae',lambda args:None)
    monkeypatch.setattr(train,'vae_encode',lambda vae,rgb:rgb[:,:,::4])
    monkeypatch.setattr(train,'build_cond_seq_for_batch',lambda **kw:torch.zeros(1,kw['t_latent'],4))
    calls=[];history=[]
    class Run:
        def __init__(self,kwargs):self.id=kwargs['id'];self.url='https://example.invalid/test'
        def define_metric(self,*a,**kw):pass
        def log(self,row):history.append(copy.deepcopy(row))
        def finish(self,**kw):pass
    def init(**kwargs):calls.append(kwargs);return Run(kwargs)
    monkeypatch.setattr(wandb,'init',init)
    for stage,frames,steps in [(1,4,2),(2,8,3)]:
        argv=['train','--dataset','droid','--data_root','/training','--vae_checkpoint','none',
              '--eval_data_root','/held_out','--eval_every','1','--eval_num_videos','3',
              '--latent_frames',str(frames),'--max_train_steps',str(steps),'--batch_size','1','--num_workers','0',
              '--mixed_precision','no','--image_log_every','0','--log_every','1','--output_dir',str(tmp_path/f'stage{stage}'),
              '--wandb_name','one-experiment','--wandb_run_file',str(tmp_path/'run.json'),'--curriculum_stage',str(stage)]
        if stage==2:argv+=['--load_pretrained',str(tmp_path/'stage1/last.pt')]
        monkeypatch.setattr(sys,'argv',argv)
        train.main()
    assert calls[0]['id']==calls[1]['id']
    assert [c['resume'] for c in calls]==['never','must']
    assert 'latent_frames' not in calls[1]['config']
    assert calls[1]['config']['curriculum_stage_2']['latent_frames']==8
    losses=[r for r in history if 'train/loss' in r]
    evals=[r for r in history if 'eval/psnr' in r]
    assert [r['train_step'] for r in losses]==[1,2,3,4,5]
    assert [r['train_step'] for r in evals]==[1,2,3,4,5]
    assert [r['curriculum/stage'] for r in losses]==[1,1,2,2,2]
    checkpoint=torch.load(tmp_path/'stage2/last.pt',weights_only=True)
    assert checkpoint['global_step']==3 and checkpoint['total_train_steps']==5


@pytest.mark.parametrize('exit_code',[0,7])
def test_existing_stage_handoff(tmp_path,exit_code):
    # A stopped shell keeps its foreground child waitable; verify the same
    # mechanism used for the live job, without launching training or touching GPUs.
    script=tmp_path/'old.sh'
    script.write_text(f"#!/bin/bash\npython3 -c 'import time;time.sleep(1);raise SystemExit({exit_code})'\necho wrong > {tmp_path/'wrong'}\n")
    parent=subprocess.Popen(['bash',str(script)],start_new_session=True)
    controller=None
    try:
        children_file=Path(f'/proc/{parent.pid}/task/{parent.pid}/children')
        for _ in range(100):
            children=children_file.read_text().split()
            if children:break
            time.sleep(.005)
        child=int(children[0])
        def identity(pid):
            fields=Path(f'/proc/{pid}/stat').read_text().rsplit(') ',1)[1].split()
            return int(fields[19])
        os.kill(parent.pid,signal.SIGSTOP)
        checkpoint=tmp_path/'last.pt';checkpoint.touch()
        config=dict(supervisor_pid=parent.pid,supervisor_starttime=identity(parent.pid),
                    trainer_pid=child,trainer_starttime=identity(child),expected_checkpoint=str(checkpoint),
                    source=str(tmp_path),revision='test',log=str(tmp_path/'log'),
                    command=[sys.executable,'-c',f"from pathlib import Path;Path({str(tmp_path/'continued')!r}).touch()"])
        path=tmp_path/'handoff.json';path.write_text(json.dumps(config))
        controller=subprocess.run([sys.executable,str(Path(__file__).parents[1]/'scripts/continue_after_stage.py'),
                                   '--config',str(path),'--poll_seconds','.02'],capture_output=True,text=True,timeout=15)
        status=json.loads((tmp_path/'continuation_status.json').read_text())
        if exit_code==0:
            assert controller.returncode==0 and status['status']=='completed'
            assert (tmp_path/'continued').exists()
        else:
            assert controller.returncode!=0 and status['status']=='failed'
            assert not (tmp_path/'continued').exists()
        assert not (tmp_path/'wrong').exists()
    finally:
        if parent.poll() is None:
            os.kill(parent.pid,signal.SIGKILL)
        parent.wait(timeout=5)
