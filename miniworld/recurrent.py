"""Chunk-recurrent video backbones; cache snapshots are functional, never mutated."""
from dataclasses import dataclass
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from miniworld.miniworld import MiniWorldModel, modulate
from miniworld.memory import MemoryBank, MemoryReader


@dataclass
class VideoState:
    # RTransformer: old entries are top-layer KV, newest chunk is per-layer KV.
    kv: list
    banks: tuple = ()
    previous_tokens: int = 0


def cat_kv(a, b):
    return b if a is None else tuple(torch.cat((x, y), dim=-2) for x, y in zip(a, b))


class RecurrentMiniWorldModel(MiniWorldModel):
    def __init__(self, *args, backbone='rtransformer', num_memory_tokens=2400,
                 memory_window_frames=0, slot_embed=True, gated_ema=True,
                 write_from_last=True, state_sharing=True, assigned_write=True,
                 **kwargs):
        if backbone not in ('transformer', 'rtransformer', 'tas'):
            raise ValueError('Expected transformer, rtransformer or tas')
        if num_memory_tokens < 1 or memory_window_frames < 0:
            raise ValueError('Memory slots must be positive; history window must be nonnegative')
        if backbone == 'tas' and memory_window_frames != 0:
            raise ValueError('TaS only keeps within-chunk KV; memory_window_frames must be 0')
        super().__init__(*args, **kwargs)
        self.backbone = backbone
        self.memory_window_frames = memory_window_frames
        self.write_from_last = write_from_last
        self.bank_for_layer = [0 if state_sharing else i for i in range(self.depth)]
        self.bank_owners = [self.depth - 1] if state_sharing else list(range(self.depth))
        if backbone == 'rtransformer':
            for block in self.blocks:
                block.attn.prev_chunk_alpha = nn.Parameter(torch.zeros(self.num_heads))
        elif backbone == 'tas':
            self.readers = nn.ModuleList([MemoryReader(self.hidden_size, self.num_heads) for _ in self.blocks])
            self.banks = nn.ModuleList([
                MemoryBank(self.hidden_size, self.num_heads, num_memory_tokens,
                           slot_embed, gated_ema, assigned_write) for _ in self.bank_owners])

    def _layer(self, idx, x, emb, shared, pose, rope, past, bank, identity,
               previous_top=None, previous_tokens=0):
        block = self.blocks[idx]
        # Recompute the mixed history with the layer instead of retaining a
        # second full-history tensor per layer/chunk outside its checkpoint.
        if self.backbone == 'rtransformer' and past is not None and previous_tokens:
            p = previous_tokens
            alpha = block.attn.prev_chunk_alpha.sigmoid()[None, :, None, None]
            past = tuple(torch.cat((own[..., :-p, :],
                         (1-alpha).to(own.dtype)*own[..., -p:, :] +
                         alpha.to(own.dtype)*top[..., -p:, :]), dim=-2)
                         for own, top in zip(past, previous_top))
        mod = block.modulation(emb, shared, pose)
        shift, scale, gate, mshift, mscale, mgate = mod.chunk(6, -1)
        attended = block.attn(modulate(block.norm1(x), shift, scale),
                              rope=rope, past_kv=past, return_kv=self.backbone != 'tas')
        attention, kv = (attended, None) if self.backbone == 'tas' else attended
        x = x + gate * attention
        if self.backbone == 'tas':
            read, h = self.readers[idx](x, bank, identity)
            # Preserve the DiT AdaLN-zero residual initialization.
            x = x + gate * read
        else:
            h = x
        x = x + mgate * block.mlp(modulate(block.norm2(x), mshift, mscale))
        return x, kv, h

    def _run(self, x, t, y, state, offset, chunk_size, cond_drop):
        if chunk_size < 1:
            raise ValueError('chunk_size must be positive')
        b, _, frames, height, width = x.shape
        if (height, width) != self.x_embedder.input_size[1:]:
            raise ValueError('Input spatial size must match model configuration')
        tokens = self.x_embedder(x)
        per_frame = (height // self.patch_size) * (width // self.patch_size)
        n = tokens.shape[1]
        frame_ids = torch.arange(n, device=x.device) // per_frame
        # Resolve once for the whole clip, not separately per chunk.
        emb, shared, pose = self._build_conditioning(
            t, y, b, frames, n, frame_ids, tokens.device, tokens.dtype, cond_drop, frame_offset=offset)
        if state is None or isinstance(state, list):
            if state is not None and any(kv is not None for kv in state):
                raise ValueError('Recurrent backbones require a VideoState cache')
            banks = (() if self.backbone != 'tas' else
                     tuple(bank.initial.unsqueeze(0).expand(b, -1, -1).to(tokens.dtype) for bank in self.banks))
            state = VideoState([None] * self.depth, banks)
        if self.backbone == 'tas' and any(kv is not None for kv in state.kv):
            raise ValueError('TaS streaming state cannot contain cross-chunk KV')
        outputs = []
        for start in range(0, frames, chunk_size):
            count = min(chunk_size, frames - start)
            sl = slice(start * per_frame, (start + count) * per_frame)
            # Bind positions: checkpoint recomputation must not capture loop variables by reference.
            def rope(q, count=count, position=offset + start):
                return self.feat_rope(q, num_frames_override=count, start_frame=position)
            chunk = tokens[:, sl]
            current, sources = [], []
            for i in range(self.depth):
                past = state.kv[i]
                bank_id = self.bank_for_layer[i]
                bank = state.banks[bank_id] if state.banks else None
                identity = self.banks[bank_id].identity(chunk.dtype) if state.banks else None
                args = (i, chunk, emb[:, sl], None if shared is None else shared[:, sl],
                        None if pose is None else pose[:, sl], rope, past, bank, identity,
                        state.kv[-1] if self.backbone == 'rtransformer' else None,
                        state.previous_tokens if self.backbone == 'rtransformer' else 0)
                chunk, kv, h = (checkpoint(self._layer, *args, use_reentrant=False)
                                if self.use_checkpoint and self.training else self._layer(*args))
                current.append(kv)
                sources.append(h)
            next_banks = []
            if self.backbone == 'tas':
                for j, owner in enumerate(self.bank_owners):
                    source = sources[-1 if self.write_from_last else owner]
                    bank = self.banks[j]
                    updated = (checkpoint(bank, source, state.banks[j], use_reentrant=False)
                               if self.use_checkpoint and self.training else bank(source, state.banks[j]))
                    next_banks.append(updated)
            next_kv = [None] * self.depth
            if self.backbone != 'tas':
                for i, kv in enumerate(current):
                    # Archive preceding top-layer K/V for RTransformer.
                    old = state.kv[-1] if self.backbone == 'rtransformer' else state.kv[i]
                    next_kv[i] = cat_kv(old, kv)
            # TaS persists only banks. Current-chunk KV stays inside attention.
            state = VideoState(next_kv, tuple(next_banks),
                               0 if self.backbone == 'tas' else count * per_frame)
            outputs.append(self.final_layer(chunk, emb[:, sl]))
        result = torch.cat(outputs, dim=1)
        p = self.patch_size
        result = result.reshape(b, frames, height//p, width//p, 1, p, p, self.out_channels)
        result = torch.einsum('nthwpqrc->nctphqwr', result).reshape(b, self.out_channels, frames, height, width)
        return result, state

    def forward(self, x, t=None, y=None, use_fp16=False, temporal_causal=False,
                chunk_size=None, cond_drop=None, frame_offset=0):
        if not temporal_causal:
            raise ValueError('Recurrent video backbones require temporal_causal=True')
        return self._run(x, t, y, None, frame_offset, chunk_size or 1, cond_drop)[0]

    @torch.no_grad()
    def forward_with_cache(self, x, t, y=None, past_kv_list=None,
                           current_position_offset=0, return_kv=False, chunk_size=1, cond_drop=None):
        prediction, candidate = self._run(x, t, y, past_kv_list,
                                          current_position_offset, chunk_size, cond_drop)
        # Caller accepts candidate only when committing clean chunks at t=0.
        return prediction, candidate if return_kv else None
