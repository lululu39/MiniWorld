"""Analytical checks of the actual training loss and streaming integration path."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from miniworld.denoiser import DiffusionForcingDenoiser, IncrementalTimesteps


def bare_denoiser(chunk=4):
    model=object.__new__(DiffusionForcingDenoiser)
    torch.nn.Module.__init__(model)
    model.df_chunk_size=chunk
    model.drop_cond=lambda x:x
    model.steps=7
    model.df_ardiff_step=2
    model.timestep_shift=3.7717
    model.cfg_scale=1.
    model.cfg_interval_min=0.
    model.cfg_interval_max=1.
    model.use_model_null_cfg=True
    model.P_mean=0.;model.P_std=1.
    model.condition_noise_max_t=.05
    model.df_train_time_bins=50
    model._df_train_step_samplers={}
    return model


def test_noising_masked_loss_and_gradient_match_analytic_values(monkeypatch):
    torch.manual_seed(12)
    clean=torch.randn(2,1,8,2,2)
    noise=torch.randn_like(clean)
    target=clean-noise
    errors=torch.tensor([[100.,100.,100.,100.,1.,2.,3.,4.],
                         [100.,1.,2.,3.,4.,5.,6.,7.]]).view(2,1,8,1,1).expand_as(clean)
    times=torch.tensor([[.02,.02,.02,.02,.8,.8,.8,.8],[.02,.2,.2,.2,.8,.8,.8,.8]])
    mask=torch.tensor([[1.,1.,1.,1.,0.,0.,0.,0.],[1.,0.,0.,0.,0.,0.,0.,0.]])
    class Prediction(torch.nn.Module):
        def __init__(self):
            super().__init__();self.pred=torch.nn.Parameter(target+errors)
        def forward(self,z,t,y,**kwargs):
            # Independent interpolation via the direction from the clean endpoint.
            torch.testing.assert_close(z,clean+t[:,None,:,None,None]*(noise-clean))
            return self.pred
    d=bare_denoiser();d.net=Prediction()
    d._build_diffusion_forcing_timesteps=lambda **kwargs:(times,None,None,mask)
    monkeypatch.setattr(torch,'randn_like',lambda x:noise)
    loss=d(clean,torch.zeros(2,8,1))
    torch.testing.assert_close(loss,torch.tensor(13.75))
    loss.backward()
    assert torch.count_nonzero(d.net.pred.grad[0,:,:4])==0
    assert torch.count_nonzero(d.net.pred.grad[1,:,:1])==0
    torch.testing.assert_close(d.net.pred.grad[0,:,4],torch.full((1,2,2),1/16))
    torch.testing.assert_close(d.net.pred.grad[1,:,1],torch.full((1,2,2),1/28))


class IdentityRope:
    def rope_shift_time(self,delta,tensor):return tensor


class OracleNet(torch.nn.Module):
    def __init__(self,trained,history,guidance,assert_path=True):
        super().__init__()
        self.x_embedder=SimpleNamespace(patch_size=(1,1,1),input_size=(trained,1,1))
        self.feat_rope=IdentityRope()
        self.use_abs_pos=False;self.depth=1
        self.history=history;self.guidance=guidance;self.assert_path=assert_path
    def forward_with_cache(self,z,t,y,past_kv_list=None,return_kv=False,cond_drop=None,**kwargs):
        target,noise,frame_id=y.unbind(-1)
        target,noise=target[:,None,:,None,None],noise[:,None,:,None,None]
        # An affine conditional/unconditional pair allows an exact CFG endpoint.
        uncond_target=.5*target
        guided=uncond_target+self.guidance*(target-uncond_target)
        guided=torch.where((frame_id<self.history)[:,None,:,None,None],target,guided)
        if self.assert_path:
            expected=guided+t[:,None,:,None,None]*(noise-guided)
            torch.testing.assert_close(z,expected,atol=1e-6,rtol=1e-6,
                                       msg='Sampler latent content and supplied noise-level t disagree')
        velocity=(uncond_target if cond_drop is not None else target)-noise
        kv=torch.zeros(z.shape[0],1,z.shape[2],1,dtype=z.dtype,device=z.device)
        return velocity,[(kv,kv)] if return_kv else None


@pytest.mark.parametrize('chunk,total,history,inflight,cache',[
    (4,8,1,1,1),       # active stage1 protocol
    (4,16,1,1,3),      # active stage2 protocol
    (4,32,1,2,6),      # stage3: finished chunks remain in final inflight window
    (4,64,1,4,12),     # stage4
    (4,8,1,2,0),       # entire rollout fits in flight
    (4,17,4,2,1),      # prefill, eviction and a partial final chunk
])
@pytest.mark.parametrize('guidance',[1.,2.])
def test_oracle_streaming_path_and_endpoint(chunk,total,history,inflight,cache,guidance):
    torch.manual_seed(9)
    target=torch.randn(1,1,total,1,1,dtype=torch.float64)
    noise=torch.randn_like(target)
    cond=torch.stack((target.flatten(2).squeeze(1),noise.flatten(2).squeeze(1),
                      torch.arange(total,dtype=torch.float64)[None]),dim=-1)
    d=bare_denoiser(chunk);trained=chunk*(inflight+cache)
    d.cfg=SimpleNamespace(latent_frames=trained);d.trained_num_frames=trained
    d.cfg_scale=guidance
    d.net=OracleNet(trained,history,guidance)
    result=d.generate_eval_latents_streaming(target,cond,total_len=total,history_len=history,
           inflight_chunks=inflight,max_cache_chunks=cache,sink_frames=min(1,cache*chunk),noise=noise)
    expected=.5*target+guidance*.5*target
    expected[:,:,:history]=target[:,:,:history]
    torch.testing.assert_close(result,expected,atol=1e-6,rtol=1e-6)


def test_scheduler_timesteps_match_previous_and_next_states():
    d=bare_denoiser()
    current,next_,update=d._build_chunk_sampling_schedule(4,torch.device('cpu'),torch.float64,
                                                        n_context_chunks=0,effective_steps=7)
    torch.testing.assert_close(current[0],torch.ones(4,dtype=torch.float64))
    torch.testing.assert_close(current[1:],next_[:-1])
    assert (next_<=current).all()
    torch.testing.assert_close((current-next_).sum(0),torch.ones(4,dtype=torch.float64))
    assert (update.sum(0)==7).all()


def test_monotone_time_warp_endpoints_and_cfg_interval():
    d=bare_denoiser();u=torch.linspace(0,1,51,dtype=torch.float64)
    mapped=d.shift_timestep(d.logit_normal_warp(u),d.timestep_shift)
    assert torch.isfinite(mapped).all() and mapped[0]==0 and mapped[-1]==1
    assert (mapped[1:]>=mapped[:-1]).all()
    assert mapped[25]>.5
    d.cfg_scale=2.;d.cfg_interval_min=.2
    torch.testing.assert_close(d._get_df_action_guidance_scale(torch.tensor([0.,.2,.3,1.])),
                               torch.tensor([1.,1.,2.,2.]))
    np.random.seed(42)
    for _ in range(50):
        indices=IncrementalTimesteps(16,50).sample_stepseq_from_mid()
        assert ((indices>=0)&(indices<50)).all() and (indices[1:]>=indices[:-1]).all()


def test_condition_noise_is_explicit_augmentation_not_always_zero(monkeypatch):
    d=bare_denoiser()
    monkeypatch.setattr(d,'_sample_df_chunk_timesteps',lambda n,device:torch.zeros(n,dtype=torch.long))
    monkeypatch.setattr(d,'sample_condition_t',lambda shape,device,dtype:torch.full(shape,.03,dtype=dtype))
    t,_,chunks,mask=d._build_diffusion_forcing_timesteps(1,8,torch.device('cpu'),torch.float32)
    assert torch.count_nonzero(chunks)==0
    assert torch.all(t[mask.bool()]==.03) and torch.all(t[~mask.bool()]==0)
    # The nondecreasing property is for sampled chunks, not every overwritten
    # condition frame when clean-context noise augmentation is enabled.
    assert (t[:,1:]<t[:,:-1]).any()


def test_current_single_inflight_rollout_unchanged_by_time_label_fix(monkeypatch):
    """Actual tiny Transformer: active stage1-style sampling is bit-identical."""
    from miniworld.denoiser import DenoiserConfig
    from miniworld.miniworld import MiniWorldModels, MiniWorldModel
    def tiny(**kwargs):
        cls=kwargs.pop('_model_class',MiniWorldModel)
        return cls(depth=2,hidden_size=64,num_heads=2,patch_size=1,adaln_lora_dim=8,**kwargs)
    monkeypatch.setitem(MiniWorldModels,'audit_tiny',tiny)
    torch.manual_seed(42)
    d=DiffusionForcingDenoiser(DenoiserConfig(wm_model='audit_tiny',latent_size=2,latent_channels=4,
        latent_frames=8,cond_dim=6,df_chunk_size=4,num_sampling_steps=100,df_ardiff_step=5,
        cfg_scale=2.,cond_dropout_prob=.1,wm_use_checkpoint=False)).eval()
    torch.nn.init.normal_(d.net.shared_mod[-1].weight,std=.03)
    torch.nn.init.normal_(d.net.final_layer.linear.weight,std=.03)
    data=torch.randn(1,4,12,2,2);condition=torch.randn(1,12,6);noise=torch.randn_like(data)
    options=dict(total_len=12,history_len=1,max_cache_chunks=1,inflight_chunks=1,sink_frames=1,noise=noise)
    fixed=d.generate_eval_latents_streaming(data,condition,**options)
    def legacy(total_chunks,device,dtype,n_context_chunks=1,effective_steps=None):
        steps=effective_steps if effective_steps is not None else d.steps
        times=d.shift_timestep(torch.linspace(1.,0.,steps+1,device=device,dtype=dtype),d.timestep_shift)
        index,mask=d._build_async_step_index_matrix(total_chunks,steps,device)
        current=torch.cat((times[:1],times[:-1]))[index]
        next_=times[index]
        current[:,:n_context_chunks]=0;next_[:,:n_context_chunks]=0;mask[:,:n_context_chunks]=False
        return current,next_,mask
    monkeypatch.setattr(d,'_build_chunk_sampling_schedule',legacy)
    before=d.generate_eval_latents_streaming(data,condition,**options)
    torch.testing.assert_close(fixed,before,atol=0,rtol=0)
