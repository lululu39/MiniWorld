"""Periodic trainer control flow, RNG isolation, offline W&B and real Gloo collectives."""
import copy
import json
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from miniworld import train, validation, sample


class Clips(torch.utils.data.Dataset):
    d_action = 1
    def __init__(self, ids=(0,1,2), frames=4):
        self.samples = list(ids)
        self.frames = frames
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, index):
        n = 1+4*(self.frames-1)
        return dict(videos=torch.rand(n,32,32,3)*2-1, actions=torch.zeros(n-1,1),
                    sample_id=str(self.samples[index]), source_path=f'{self.samples[index]}.mp4',
                    frame_ids=torch.arange(n))


class TinyEMA(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.1))
        self.steps, self.cfg_scale = 3, 1.
        self.cfg_interval_min, self.cfg_interval_max, self.df_ardiff_step = .1, .9, 1
        self.cfg = SimpleNamespace(backbone='transformer')
    def forward(self, latent, cond, **kwargs):
        return (self.weight-latent.mean()).square()
    def generate_eval_latents_streaming(self, latent, cond, total_len, noise, **kwargs):
        # Consume all RNG families to verify restoration even with stochastic backends.
        random.random(); np.random.rand(); torch.rand(1)
        pred = torch.zeros(1,3,1+4*(total_len-1),32,32) + self.weight + noise.mean()
        return latent, pred


class TinyMetric:
    """Cheap metric stub: the real pretrained metrics have separate numerical tests."""
    def __init__(self,*args):
        torch.rand(1)  # initialization must also be RNG-isolated
    def score(self,pred,target,context_frames):
        error=(pred[context_frames:]-target[context_frames:]).square().mean((1,2,3))
        scores=dict(psnr=(-10*error.log10()).tolist(),ssim=(1-error).tolist(),lpips=error.tolist())
        return dict(mean={k:sum(v)/len(v) for k,v in scores.items()},per_frame=scores,
                    frame_indices=list(range(context_frames,pred.shape[0])),evaluated_frames=len(error),
                    context_frames=context_frames)
    def protocol(self):
        return {'test_stub':True}


def options(tmp_path, **overrides):
    values=dict(dataset='droid', data_root='/training', eval_data_root='/held_out', pose_dir=None,
                eval_pose_dir=None, eval_filter_cache_dir=None, eval_every=1, eval_num_videos=3,
                eval_latent_frames=0, latent_frames=4, eval_history_len=1, eval_seed=42,
                eval_sampling_steps=2, eval_cfg_scale=2., eval_ardiff_step=1,
                eval_metric_frame_batch_size=2, eval_lpips_net='vgg', df_chunk_size=2,
                mixed_precision='no', use_pose_cond=False, use_action_cond=True, pose_enc_freq=15,
                output_dir=str(tmp_path), wandb=True, wandb_entity='LVSM-Experiment',
                wandb_project='miniworld', wandb_name='periodic-eval-test', wandb_group=None, wandb_mode='offline')
    values.update(overrides)
    return SimpleNamespace(**values)


def install_stubs(monkeypatch=None):
    patches=[(sample,'build_dataset',lambda args:Clips(frames=args.total_len)),
             (validation,'VideoMetrics',TinyMetric),
             (validation,'vae_encode',lambda vae,rgb:rgb[:,:,::4]),
             (validation,'build_cond_seq_for_batch',lambda **kwargs:torch.zeros(1,kwargs['t_latent'],4)),
             (validation,'StreamingVAEDecoder',lambda vae:None)]
    for module,name,value in patches:
        if monkeypatch is None:setattr(module,name,value)
        else:monkeypatch.setattr(module,name,value)


def test_fixed_eval_restores_training_state(tmp_path,monkeypatch):
    install_stubs(monkeypatch)
    evaluator=validation.PeriodicEvaluator(options(tmp_path),Clips((100,101)), 'cpu')
    ema=TinyEMA().train()
    random.seed(8);np.random.seed(8);torch.manual_seed(8)
    before=(random.getstate(),np.random.get_state(),torch.get_rng_state().clone())
    first=evaluator.run(ema,None,1)
    assert random.getstate()==before[0]
    assert np.array_equal(np.random.get_state()[1],before[1][1])
    assert torch.equal(torch.get_rng_state(),before[2])
    assert ema.training and ema.steps==3 and ema.cfg_scale==1. and ema.df_ardiff_step==1
    second=evaluator.run(ema,None,2)
    assert first['mean']==second['mean']
    assert first['num_videos']==3 and (tmp_path/'eval/step_00000002/metrics_summary.json').exists()


def test_eval_overlap_and_failure(tmp_path,monkeypatch):
    install_stubs(monkeypatch)
    with pytest.raises(ValueError,match='overlap'):
        validation.PeriodicEvaluator(options(tmp_path),Clips((1,100)),'cpu')
    evaluator=validation.PeriodicEvaluator(options(tmp_path),Clips((100,)),'cpu')
    ema=TinyEMA()
    def fail(*args,**kwargs):
        torch.rand(1)
        raise ValueError('decode failed')
    monkeypatch.setattr(validation,'vae_encode',fail)
    before=torch.get_rng_state().clone()
    with pytest.raises(RuntimeError,match='decode failed'):evaluator.run(ema,None,1)
    assert ema.training and ema.steps==3 and torch.equal(torch.get_rng_state(),before)


def test_options_require_held_out_and_default_project(tmp_path,monkeypatch):
    with pytest.raises(ValueError,match='eval_data_root'):validation.validate_options(options(tmp_path,eval_data_root=None))
    with pytest.raises(ValueError,match='differ'):validation.validate_options(options(tmp_path,eval_data_root='/training'))
    monkeypatch.setattr(sys,'argv',['train','--dataset','droid','--data_root','x','--vae_checkpoint','x'])
    args=train.parse_args()
    assert args.eval_every==0 and args.wandb_entity=='LVSM-Experiment' and args.wandb_project=='miniworld'
    assert args.wandb_name.startswith('droid_transformer_1B_')
    assert train.parse_args().wandb_name!=args.wandb_name


def test_offline_wandb_routing_and_shared_step(tmp_path,monkeypatch):
    args=options(tmp_path)
    run=train.init_wandb(args,8)
    try:
        assert run.entity=='LVSM-Experiment' and run.project=='miniworld' and run.name==args.wandb_name
        assert run.settings.base_url=='https://api.wandb.ai' and run.settings.mode=='offline'
        run.log({'train_step':1,'train/loss':.5})
        run.log({'train_step':1,'eval/psnr':20.,'eval/ssim':.7,'eval/lpips':.3})
        assert run.step==2  # both records at the same train_step survive
        assert json.loads((tmp_path/'wandb_run.json').read_text())['project']=='miniworld'
    finally:
        run.finish()


def test_two_step_training_calls_periodic_evaluation(tmp_path,monkeypatch):
    install_stubs(monkeypatch)
    monkeypatch.setattr(train,'build_dataset',lambda *args,**kw:Clips((100,101),4))
    monkeypatch.setattr(train,'build_denoiser',lambda args:TinyEMA())
    monkeypatch.setattr(train,'load_wan22_vae',lambda args:None)
    monkeypatch.setattr(train,'vae_encode',lambda vae,rgb:rgb[:,:,::4])
    monkeypatch.setattr(train,'build_cond_seq_for_batch',lambda **kw:torch.zeros(1,kw['t_latent'],4))
    monkeypatch.setattr(sys,'argv',['train','--dataset','droid','--data_root','/training','--vae_checkpoint','none',
        '--eval_data_root','/held_out','--eval_every','1','--eval_num_videos','3','--latent_frames','4',
        '--max_train_steps','2','--batch_size','1','--num_workers','0','--mixed_precision','no',
        '--image_log_every','0','--log_every','1','--no-wandb','--output_dir',str(tmp_path)])
    train.main()
    for step in [1,2]:
        assert json.loads((tmp_path/f'eval/step_{step:08d}/metrics_summary.json').read_text())['num_videos']==3
    assert (tmp_path/'last.pt').exists()


def ddp_worker(rank,world,init_file,output,fail):
    os.environ['RANK']=str(rank)
    dist.init_process_group('gloo',init_method='file://'+init_file,rank=rank,world_size=world)
    try:
        install_stubs()
        evaluator=validation.PeriodicEvaluator(options(Path(output)),Clips((100,)),'cpu')
        if fail and rank==1:
            def bad(*args,**kwargs):raise RuntimeError('intentional rank1 failure')
            validation.vae_encode=bad
        try:
            evaluator.run(TinyEMA(),None,1)
            outcome='ok'
        except RuntimeError as exc:
            outcome=str(exc)
        (Path(output)/f'rank{rank}.txt').write_text(outcome)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('fail',[False,True])
def test_distributed_shards_and_failure_propagation(tmp_path,fail):
    mp.spawn(ddp_worker,args=(2,str(tmp_path/'init'),str(tmp_path),fail),nprocs=2,join=True)
    outcomes=[(tmp_path/f'rank{i}.txt').read_text() for i in range(2)]
    if fail:assert all('intentional rank1 failure' in out for out in outcomes)
    else:
        assert outcomes==['ok','ok']
        rows=[json.loads(line) for line in (tmp_path/'eval/step_00000001/metrics_per_video.jsonl').read_text().splitlines()]
        assert [r['sample_idx'] for r in rows]==[0,1,2]
        assert len({r['sample_id'] for r in rows})==3


def test_public_wandb_uses_public_credential_without_mutating_env(monkeypatch):
    import netrc
    monkeypatch.setenv('WANDB_BASE_URL','https://private.example')
    monkeypatch.setenv('WANDB_API_KEY','private-test-placeholder')
    monkeypatch.setattr(netrc,'netrc',lambda:SimpleNamespace(authenticators=lambda host:('u',None,'public-test-placeholder')))
    settings=train.public_wandb_settings(SimpleNamespace(Settings=lambda **kwargs:kwargs))
    assert settings['base_url']=='https://api.wandb.ai' and settings['api_key']=='public-test-placeholder'
    assert os.environ['WANDB_API_KEY']=='private-test-placeholder'


def test_wandb_failure_is_not_silently_ignored(tmp_path,monkeypatch):
    import wandb
    def fail(**kwargs):raise RuntimeError('intentional init failure')
    monkeypatch.setattr(wandb,'init',fail)
    with pytest.raises(RuntimeError,match='W&B initialization failed'):
        train.init_wandb(options(tmp_path),1)
    assert train.init_wandb(options(tmp_path,wandb=False),1) is None
