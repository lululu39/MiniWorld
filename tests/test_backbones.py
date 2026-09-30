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
    extra = {} if kind == 'transformer' else dict(backbone='transformer' if kind == 'serial_transformer' else kind, num_memory_tokens=8)
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


@pytest.mark.parametrize('kind', ['serial_transformer', 'rtransformer', 'tas'])
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
        if a is None:
            assert b is None
        else:
            torch.testing.assert_close(a, b)
    for a, b in zip(state.banks, before.banks):
        torch.testing.assert_close(a, b)
    shifted = Denoiser._evict_and_shift_cache(state, 1, 4, net.feat_rope)
    if kind == 'tas':
        assert shifted is state and all(kv is None for kv in shifted.kv)
        assert shifted.previous_tokens == 0
        assert shifted.banks[0] is state.banks[0]
    else:
        assert shifted.kv[0][0].shape[-2] == 4
        assert shifted.previous_tokens == 4


def legacy_rtransformer_forward(net, x, t, y, chunk_size):
    """Pre-optimization formula: blend/concatenate outside the layer call."""
    b, _, frames, height, width = x.shape
    per_frame = height * width  # This reference uses the test model's patch1.
    tokens = net.x_embedder(x)
    frame_ids = torch.arange(tokens.shape[1]) // per_frame
    emb, shared, pose = net._build_conditioning(
        t, y, b, frames, tokens.shape[1], frame_ids, tokens.device, tokens.dtype, None)
    history, previous_tokens, outputs = [None] * net.depth, 0, []
    for start in range(0, frames, chunk_size):
        count = min(chunk_size, frames - start)
        sl = slice(start * per_frame, (start + count) * per_frame)
        def rope(q, count=count, position=start):
            return net.feat_rope(q, num_frames_override=count, start_frame=position)
        chunk, current = tokens[:, sl], []
        for i in range(net.depth):
            past = history[i]
            if past is not None:
                alpha = net.blocks[i].attn.prev_chunk_alpha.sigmoid()[None, :, None, None]
                p = previous_tokens
                past = tuple(torch.cat((own[..., :-p, :],
                             (1-alpha).to(own.dtype)*own[..., -p:, :] +
                             alpha.to(own.dtype)*top[..., -p:, :]), dim=-2)
                             for own, top in zip(past, history[-1]))
            # No previous_top is passed: this reference already mixed the KV.
            chunk, kv, _ = net._layer(i, chunk, emb[:, sl],
                None if shared is None else shared[:, sl],
                None if pose is None else pose[:, sl], rope, past, None, None)
            current.append(kv)
        old = history[-1]
        history = [kv if old is None else tuple(torch.cat((a, v), dim=-2)
                   for a, v in zip(old, kv)) for kv in current]
        previous_tokens = count * per_frame
        outputs.append(net.final_layer(chunk, emb[:, sl]))
    return torch.cat(outputs, dim=1).reshape(b, frames, height, width, net.out_channels).permute(0, 4, 1, 2, 3)


@pytest.mark.parametrize('pose', [False, True])
@pytest.mark.parametrize('chunk_size', [2, 4])
def test_checkpointed_rtransformer_matches_legacy_blend_formula(pose, chunk_size):
    reference = model('rtransformer', pose, use_checkpoint=False)
    for block in reference.blocks:
        block.attn.prev_chunk_alpha.data.copy_(torch.tensor([-1.5, 2.0]))
    checked = copy.deepcopy(reference)
    checked.use_checkpoint = True
    frames = 16
    x = torch.randn(1, 4, frames, 2, 2)
    t = torch.rand(1, frames)
    y = torch.randn((1, frames, 6, 2, 2) if pose else (1, frames, 6))
    inputs_and_grads, outputs = [], []
    for net, legacy in ((reference, True), (checked, False)):
        z, times, cond = [v.detach().clone().requires_grad_() for v in (x, t, y)]
        out = (legacy_rtransformer_forward(net, z, times, cond, chunk_size) if legacy else
               net(z, times, cond, temporal_causal=True, chunk_size=chunk_size))
        out[:, :, -chunk_size:].square().mean().backward()
        outputs.append(out.detach())
        inputs_and_grads.append([v.grad for v in (z, times, cond)])
    torch.testing.assert_close(outputs[0], outputs[1])
    for a, b in zip(*inputs_and_grads):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-4)
    assert checked.blocks[0].attn.prev_chunk_alpha.grad.abs().sum() > 0
    for (name, p), (other, q) in zip(reference.named_parameters(), checked.named_parameters()):
        assert name == other and (p.grad is None) == (q.grad is None)
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=2e-6, rtol=2e-4, msg=name)


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


@pytest.mark.parametrize('sharing', [False, True])
def test_tas_only_banks_cross_chunk_boundaries(sharing):
    net = model('tas', state_sharing=sharing).eval()
    calls = []
    def attention_inputs(module, args, kwargs):
        calls.append(args[0].shape[1])
        assert kwargs['past_kv'] is None
        assert kwargs['return_kv'] is False
    handles = [block.attn.register_forward_pre_hook(attention_inputs, with_kwargs=True)
               for block in net.blocks]
    state = None
    for chunk in range(8):
        x = torch.randn(1, 4, 4, 2, 2)
        _, state = net.forward_with_cache(x, torch.zeros(1, 4), torch.randn(1, 4, 6),
                    past_kv_list=state, current_position_offset=chunk*4,
                    return_kv=True, chunk_size=4)
        assert all(kv is None for kv in state.kv) and state.previous_tokens == 0
        assert len(state.banks) == (1 if sharing else net.depth)
        assert all(bank.shape == (1, 8, 64) for bank in state.banks)
        assert sum(bank.numel() for bank in state.banks) == (1 if sharing else net.depth)*8*64
    assert calls == [16]*(8*net.depth)
    for handle in handles:
        handle.remove()


def test_tas_memory_is_the_only_cross_chunk_information_path(monkeypatch):
    net = model('tas')
    x, t, y = inputs()
    changed = x.clone(); changed[:, :, :2] += 20
    def no_read(x, state, identity):
        return torch.zeros_like(x), x
    for reader in net.readers:
        monkeypatch.setattr(reader, 'forward', no_read)
    a = net(x, t, y, temporal_causal=True, chunk_size=2)
    b = net(changed, t, y, temporal_causal=True, chunk_size=2)
    # Removing only memory reads must remove all dependence on prior chunks.
    torch.testing.assert_close(a[:, :, 2:], b[:, :, 2:], atol=0, rtol=0)


def test_tas_rejects_raw_history_configuration_and_state():
    with pytest.raises(ValueError, match='within-chunk KV'):
        model('tas', memory_window_frames=4)
    net = model('tas'); x, t, y = inputs()
    raw = torch.randn(1, 2, 4, 32)
    state = VideoState([(raw, raw)]*net.depth, (torch.randn(1, 8, 64),), 4)
    with pytest.raises(ValueError, match='cross-chunk KV'):
        net.forward_with_cache(x, t, y, state, return_kv=True, chunk_size=2)


def test_tas_streaming_cfg_commits_clean_banks_without_raw_kv(monkeypatch):
    from miniworld.miniworld import MiniWorldModels
    def tiny(**kwargs):
        cls = kwargs.pop('_model_class', MiniWorldModel)
        return cls(depth=2, hidden_size=64, num_heads=2, patch_size=1, adaln_lora_dim=8, **kwargs)
    monkeypatch.setitem(MiniWorldModels, 'bank_only_tiny', tiny)
    cfg = DenoiserConfig(wm_model='bank_only_tiny', backbone='tas', latent_size=2,
                        latent_channels=4, latent_frames=8, cond_dim=6,
                        num_memory_tokens=8, df_chunk_size=2, num_sampling_steps=3,
                        df_ardiff_step=1, cfg_scale=2, cond_dropout_prob=.1,
                        wm_use_checkpoint=False)
    d = Denoiser(cfg).eval()
    torch.nn.init.normal_(d.net.shared_mod[-1].weight, std=.03)
    torch.nn.init.normal_(d.net.final_layer.linear.weight, std=.03)
    committed = {'conditional': [], 'unconditional': []}
    original = d.net.forward_with_cache
    def tracked(x, t, y, **kwargs):
        state = kwargs.get('past_kv_list')
        before = [bank.clone() for bank in state.banks] if isinstance(state, VideoState) else []
        result, candidate = original(x, t, y, **kwargs)
        if before:
            for a, b in zip(before, state.banks):
                torch.testing.assert_close(a, b, atol=0, rtol=0)
        if kwargs.get('return_kv'):
            assert torch.count_nonzero(t) == 0
            assert all(kv is None for kv in candidate.kv)
            branch = 'unconditional' if kwargs.get('cond_drop') is not None else 'conditional'
            committed[branch].append(candidate.banks[0])
        else:
            assert candidate is None
        return result, candidate
    monkeypatch.setattr(d.net, 'forward_with_cache', tracked)
    data = torch.randn(1, 4, 14, 2, 2)
    result = d.generate_eval_latents_streaming(data, torch.randn(1, 14, 6),
                 total_len=14, history_len=1, max_cache_chunks=1,
                 inflight_chunks=2, sink_frames=1)
    assert torch.isfinite(result).all()
    assert len(committed['conditional']) == len(committed['unconditional']) > 1
    assert {b.data_ptr() for b in committed['conditional']}.isdisjoint(
           b.data_ptr() for b in committed['unconditional'])


@pytest.mark.parametrize('kind', ['transformer', 'rtransformer', 'tas'])
@pytest.mark.parametrize('history', [1, 4])
@pytest.mark.parametrize('chunk', [2, 4])
def test_actual_streaming_denoiser(kind, history, chunk, monkeypatch):
    from miniworld.miniworld import MiniWorldModels
    # Exercise the real Denoiser construction and sampler with a tiny model.
    def tiny(**kwargs):
        cls = kwargs.pop('_model_class', MiniWorldModel)
        return cls(depth=2, hidden_size=64, num_heads=2, patch_size=1, adaln_lora_dim=8, **kwargs)
    monkeypatch.setitem(MiniWorldModels, 'tiny', tiny)
    cfg = DenoiserConfig(wm_model='tiny', backbone=kind, latent_size=2, latent_channels=4,
                        latent_frames=4*chunk, trained_num_frames=4*chunk, cond_dim=6, num_memory_tokens=8,
                        num_sampling_steps=2, cfg_scale=2, cond_dropout_prob=.1, df_chunk_size=chunk,
                        wm_use_checkpoint=False)
    d = Denoiser(cfg).eval()
    torch.nn.init.normal_(d.net.final_layer.linear.weight, std=.03)
    x = torch.randn(1, 4, 6*chunk, 2, 2)
    cond = torch.randn(1, 6*chunk, 6)
    if history == 1:
        loss = d(x[:, :, :4*chunk], cond[:, :4*chunk])
        loss.backward()
        assert torch.isfinite(loss)
        assert d.net.final_layer.linear.weight.grad.abs().sum() > 0
    out = d.generate_eval_latents_streaming(x, cond, 6*chunk, history_len=history,
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
    legacy = torch.load(tmp_path/'last.pt', weights_only=False)
    legacy['meta']['memory_window_frames'] = 4
    torch.save(legacy, tmp_path/'legacy_history.pt')
    if kind == 'tas':
        with pytest.raises(ValueError, match='memory_window_frames'):
            load_pretrained(str(tmp_path/'legacy_history.pt'), restored, copy.deepcopy(restored))
    else:
        # The retired TaS-only field never changes Transformer/RTransformer.
        assert load_pretrained(str(tmp_path/'legacy_history.pt'), restored, copy.deepcopy(restored)) == (1, 2)
    restored.cfg.backbone = 'tas' if kind != 'tas' else 'transformer'
    with pytest.raises(ValueError, match='backbone configuration'):
        load_pretrained(str(tmp_path/'last.pt'), restored, copy.deepcopy(restored))


@pytest.mark.parametrize('pose', [False, True])
@pytest.mark.parametrize('chunk_size', [2, 4, 6])
def test_serial_transformer_matches_parallel_outputs_and_gradients(pose, chunk_size):
    parallel = model('transformer', pose)
    serial = model('serial_transformer', pose, use_checkpoint=True)
    assert set(serial.state_dict()) == set(parallel.state_dict())
    x, t, y = inputs(pose)
    outputs=[]
    grads=[]
    for net in (parallel, serial):
        z=x.clone().requires_grad_()
        output=net(z,t,y,temporal_causal=True,chunk_size=chunk_size)
        output.square().mean().backward()
        outputs.append(output.detach());grads.append(z.grad)
    torch.testing.assert_close(outputs[0],outputs[1],atol=2e-6,rtol=2e-5)
    torch.testing.assert_close(grads[0],grads[1],atol=2e-6,rtol=2e-4)
    for (name,p),(other,q) in zip(parallel.named_parameters(),serial.named_parameters()):
        assert name==other
        assert (p.grad is None)==(q.grad is None)
        if p.grad is not None:
            torch.testing.assert_close(p.grad,q.grad,atol=2e-6,rtol=2e-4,msg=name)


def test_transformer_factory_selects_serial_by_default(monkeypatch):
    from miniworld.backbones import build_video_model
    from miniworld.miniworld import MiniWorldModels
    def tiny(**kwargs):
        cls=kwargs.pop('_model_class',MiniWorldModel)
        return cls(depth=2,hidden_size=64,num_heads=2,patch_size=1,**kwargs)
    monkeypatch.setitem(MiniWorldModels,'tiny',tiny)
    kwargs=dict(in_channels=4,cond_dim=6,input_size=2,num_frames=6)
    serial=build_video_model('tiny',DenoiserConfig(),**kwargs)
    parallel=build_video_model('tiny',DenoiserConfig(transformer_execution='parallel'),**kwargs)
    assert isinstance(serial,RecurrentMiniWorldModel) and serial.backbone=='transformer'
    assert type(parallel) is MiniWorldModel
    parallel.load_state_dict(serial.state_dict(),strict=True)


def test_sampling_restores_chunk_and_legacy_execution():
    from argparse import Namespace
    from miniworld.sample import restore_execution_config
    def args(frames):
        return Namespace(df_chunk_size=None, latent_frames=frames, stream_inflight_chunks=None, stream_max_cache_chunks=None)
    old=args(64);restore_execution_config({},old)
    assert (old.transformer_execution,old.df_chunk_size,old.stream_inflight_chunks,old.stream_max_cache_chunks)==('parallel',2,8,24)
    new=args(64);restore_execution_config({'transformer_execution':'serial','df_chunk_size':4},new)
    assert (new.transformer_execution,new.df_chunk_size,new.stream_inflight_chunks,new.stream_max_cache_chunks)==('serial',4,4,12)
    short=args(8);restore_execution_config({'transformer_execution':'serial','df_chunk_size':4},short)
    assert (short.stream_inflight_chunks,short.stream_max_cache_chunks)==(1,1)
