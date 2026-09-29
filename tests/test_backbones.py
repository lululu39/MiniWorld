import copy
import pytest
import torch
from miniworld.miniworld import MiniWorldModel
from miniworld.recurrent import RecurrentMiniWorldModel, VideoState
from miniworld.memory import assigned_attention
from miniworld.denoiser import DiffusionForcingDenoiser as Denoiser, DenoiserConfig


def model(kind, pose=False, **kwargs):
    torch.manual_seed(7)
    cls = MiniWorldModel if kind == 'transformer' else RecurrentMiniWorldModel
    extra = {} if kind == 'transformer' else dict(backbone=kind, num_memory_tokens=8)
    net = cls(in_channels=4, hidden_size=64, cond_dim=6, depth=2, num_heads=2,
              patch_size=1, input_size=2, num_frames=6, adaln_lora_dim=8,
              cond_per_token=pose, **extra, **kwargs)
    # Exercise memory and causality: zero-initialized DiT outputs would hide bugs.
    torch.nn.init.normal_(net.shared_mod[-1].weight, std=.03)
    torch.nn.init.normal_(net.final_layer.linear.weight, std=.03)
    if pose:
        torch.nn.init.normal_(net.pose_encoder.to_mod.weight, std=.03)
    return net


def inputs(pose=False):
    return torch.randn(1, 4, 6, 2, 2), torch.rand(1, 6), torch.randn((1, 6, 6, 2, 2) if pose else (1, 6, 6))


@pytest.mark.parametrize('kind', ['transformer', 'rtransformer', 'tas'])
@pytest.mark.parametrize('pose', [False, True])
def test_causal_stream_and_gradient(kind, pose):
    net = model(kind, pose)
    x, t, y = inputs(pose)
    x.requires_grad_()
    full = net(x, t, y, temporal_causal=True, chunk_size=2)
    changed = x.detach().clone(); changed[:, :, 4:] += 20
    future = net(changed, t, y, temporal_causal=True, chunk_size=2)
    torch.testing.assert_close(full[:, :, :4], future[:, :, :4])
    cache = None
    chunks = []
    for start in range(0, 6, 2):
        out, candidate = net.forward_with_cache(x[:, :, start:start+2], t[:, start:start+2],
                       y[:, start:start+2], past_kv_list=cache,
                       current_position_offset=start, return_kv=True, chunk_size=2)
        cache = candidate if cache is None else Denoiser._append_kv_cache(cache, candidate)
        chunks.append(out)
    torch.testing.assert_close(full, torch.cat(chunks, dim=2), atol=2e-6, rtol=2e-5)
    full[:, :, -2:].square().mean().backward()
    assert x.grad[:, :, :2].abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
    if kind == 'tas':
        assert net.banks[0].writer.q.weight.grad.abs().sum() > 0
        assert net.banks[0].initial.grad.abs().sum() > 0
    if kind == 'rtransformer':
        assert net.blocks[0].attn.prev_chunk_alpha.grad.abs().sum() > 0


@pytest.mark.parametrize('kind', ['rtransformer', 'tas'])
def test_checkpoint_gradients_and_no_cache_mutation(kind):
    net = model(kind)
    checked = copy.deepcopy(net); checked.use_checkpoint = True
    x, t, y = inputs()
    for current in (net, checked):
        current(x, t, y, temporal_causal=True, chunk_size=2).square().mean().backward()
    for p, q in zip(net.parameters(), checked.parameters()):
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad)
    _, state = net.forward_with_cache(x[:, :, :2], t[:, :2], y[:, :2], return_kv=True, chunk_size=2)
    before = copy.deepcopy(state)
    for _ in range(2):
        net.forward_with_cache(x[:, :, 2:4], t[:, 2:4], y[:, 2:4], state, 2, False, 2)
    for a, b in zip(state.kv, before.kv):
        torch.testing.assert_close(a, b)
    for a, b in zip(state.banks, before.banks):
        torch.testing.assert_close(a, b)
    shifted = Denoiser._evict_and_shift_cache(state, 1, 4, net.feat_rope)
    assert shifted.kv[0][0].shape[-2] == 4
    assert shifted.previous_tokens == 4
    if kind == 'tas':
        assert shifted.banks[0] is state.banks[0]


def test_assigned_write_formula():
    q, k, v = [torch.randn(1, 2, n, 4, dtype=torch.double, requires_grad=True) for n in (3, 5, 5)]
    a = (k @ q.transpose(-1, -2)).float().div(2).softmax(-1).transpose(-1, -2)
    expected = (a / (a.sum(-1, keepdim=True) + 1e-6)).double() @ v
    torch.testing.assert_close(assigned_attention(q, k, v), expected)
    assigned_attention(q, k, v).square().sum().backward()
    assert all(x.grad.abs().sum() > 0 for x in (q, k, v))


@pytest.mark.parametrize('feature', ['slot_embed', 'gated_ema', 'write_from_last', 'state_sharing', 'assigned_write'])
def test_tas_switches(feature):
    net = model('tas', **{feature: False})
    x, t, y = inputs()
    net(x, t, y, temporal_causal=True, chunk_size=2).square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)


@pytest.mark.parametrize('kind', ['transformer', 'rtransformer', 'tas'])
@pytest.mark.parametrize('history', [1, 4])
def test_actual_streaming_denoiser(kind, history, monkeypatch):
    from miniworld.miniworld import MiniWorldModels
    # Exercise the real Denoiser construction and sampler with a tiny model.
    def tiny(**kwargs):
        cls = kwargs.pop('_model_class', MiniWorldModel)
        return cls(depth=2, hidden_size=64, num_heads=2, patch_size=1, adaln_lora_dim=8, **kwargs)
    monkeypatch.setitem(MiniWorldModels, 'tiny', tiny)
    cfg = DenoiserConfig(wm_model='tiny', backbone=kind, latent_size=2, latent_channels=4,
                        latent_frames=8, trained_num_frames=8, cond_dim=6, num_memory_tokens=8,
                        num_sampling_steps=2, cfg_scale=2, cond_dropout_prob=.1, df_chunk_size=2,
                        wm_use_checkpoint=False)
    d = Denoiser(cfg).eval()
    torch.nn.init.normal_(d.net.final_layer.linear.weight, std=.03)
    x = torch.randn(1, 4, 12, 2, 2)
    cond = torch.randn(1, 12, 6)
    if history == 1:
        loss = d(x[:, :, :8], cond[:, :8])
        loss.backward()
        assert torch.isfinite(loss)
        assert d.net.final_layer.linear.weight.grad.abs().sum() > 0
    out = d.generate_eval_latents_streaming(x, cond, 12, history_len=history,
                   max_cache_chunks=2, inflight_chunks=2, sink_frames=1)
    assert out.shape == x.shape and torch.isfinite(out).all()
    torch.testing.assert_close(out[:, :, :history], x[:, :, :history])

@pytest.mark.parametrize('kind', ['transformer', 'rtransformer', 'tas'])
def test_checkpoint_backbone_metadata(kind, tmp_path, monkeypatch):
    from argparse import Namespace
    from miniworld.train import save_checkpoint, load_pretrained
    from miniworld.sample import read_checkpoint
    from miniworld.backbones import backbone_config
    from miniworld.miniworld import MiniWorldModels
    def tiny(**kwargs):
        cls = kwargs.pop('_model_class', MiniWorldModel)
        return cls(depth=2, hidden_size=64, num_heads=2, patch_size=1, adaln_lora_dim=8, **kwargs)
    monkeypatch.setitem(MiniWorldModels, 'tiny', tiny)
    cfg = DenoiserConfig(wm_model='tiny', backbone=kind, latent_size=2, latent_channels=4,
                        latent_frames=6, cond_dim=6, num_memory_tokens=8)
    net = Denoiser(cfg)
    ema = copy.deepcopy(net)
    optimizer = torch.optim.AdamW(net.parameters())
    args = Namespace(**vars(cfg), output_dir=str(tmp_path), use_pose_cond=False, use_action_cond=True)
    save_checkpoint(args=args, model=net, ema_model=ema, optimizer=optimizer, epoch=1, global_step=2)
    weights, meta = read_checkpoint(str(tmp_path/'last.pt'))
    assert {k: meta[k] for k in backbone_config(args)} == backbone_config(args)
    restored = Denoiser(cfg)
    restored.load_state_dict(weights, strict=True)
    assert load_pretrained(str(tmp_path/'last.pt'), restored, copy.deepcopy(restored)) == (1, 2)
    restored.cfg.backbone = 'tas' if kind != 'tas' else 'transformer'
    with pytest.raises(ValueError, match='backbone configuration'):
        load_pretrained(str(tmp_path/'last.pt'), restored, copy.deepcopy(restored))
