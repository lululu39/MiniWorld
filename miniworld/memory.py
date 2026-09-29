"""TaS fixed-slot primitives adapted from LVSM memory_block_tokens and LLM TaS.

No inner optimizer: all writes stay in the outer video diffusion autograd graph.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from miniworld.miniworld import RMSNorm, SwiGLUFFN


def unit_rms(x):
    x = x.float()
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-5)


def assigned_attention(q, k, v):
    """[B,H,slots,D] queries; assign each input token over slots, then mass-normalize."""
    scores = (k @ q.transpose(-1, -2)).float() / math.sqrt(q.shape[-1])
    assignment = scores.softmax(dim=-1)
    weights = assignment.transpose(-1, -2)
    weights = weights / (weights.sum(-1, keepdim=True) + 1e-6)
    return weights.to(v.dtype) @ v


class CrossAttention(nn.Module):
    def __init__(self, dim, heads, assigned=False):
        super().__init__()
        self.heads = heads
        self.assigned = assigned
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, query, content, identity=0):
        def split(x):
            return x.unflatten(-1, (self.heads, -1)).transpose(1, 2)
        q, k, v = split(self.q(query)), split(self.k(content + identity)), split(self.v(content))
        out = assigned_attention(q, k, v) if self.assigned else F.scaled_dot_product_attention(q, k, v)
        return self.out(out.transpose(1, 2).flatten(-2))


class MemoryReader(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm = RMSNorm(dim, eps=1e-5)
        self.state_norm = RMSNorm(dim, eps=1e-5)
        self.attn = CrossAttention(dim, heads)

    def forward(self, x, state, identity):
        h = self.norm(x)
        return self.attn(h, self.state_norm(state).to(h.dtype), identity), h


class MemoryBank(nn.Module):
    def __init__(self, dim, heads, slots, slot_embed=True, gated_ema=True, assigned_write=True):
        super().__init__()
        self.initial = nn.Parameter(torch.randn(slots, dim) * .02)
        self.slots = nn.Parameter(self.initial.detach().clone()) if slot_embed else None
        self.norm = RMSNorm(dim, eps=1e-5)
        self.mlp_norm = RMSNorm(dim, eps=1e-5)
        self.writer = CrossAttention(dim, heads, assigned=assigned_write)
        self.mlp = SwiGLUFFN(dim, 2 * dim, bias=False)
        self.gated_ema = gated_ema
        if gated_ema:
            self.gate_weight = nn.Parameter(torch.zeros(1, 2 * dim))
            self.gate_bias = nn.Parameter(torch.tensor([math.log(.1 / .9)]))

    def identity(self, dtype):
        return 0 if self.slots is None else unit_rms(self.slots).to(dtype)

    def forward(self, source, old):
        query = self.norm(old).to(source.dtype) + self.identity(source.dtype)
        update = self.writer(query, source)
        if self.gated_ema:
            candidate = update + self.mlp(self.mlp_norm(update))
            with torch.autocast(device_type=old.device.type, enabled=False):
                features = torch.cat((unit_rms(old), unit_rms(update)), dim=-1)
                gate = F.linear(features, self.gate_weight.float(), self.gate_bias.float()).sigmoid()
                result = (1 - gate) * old.float() + gate * candidate.float()
        else:
            summed = old.float() + update.float()
            result = summed + self.mlp(self.mlp_norm(summed).to(source.dtype)).float()
        return result.to(old.dtype)
