#!/usr/bin/env python3
"""Synthetic latent checks; no dataset, VAE weights or W&B run is created."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from miniworld.miniworld import MiniWorldModel
from miniworld.recurrent import RecurrentMiniWorldModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--output', default='outputs/backbone_smoke.json')
    args = parser.parse_args()
    if args.device == 'cuda' and not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('Select an authorized GPU with CUDA_VISIBLE_DEVICES')
    torch.manual_seed(42)
    results = {}
    for name in ('transformer', 'rtransformer', 'tas'):
        cls = MiniWorldModel if name == 'transformer' else RecurrentMiniWorldModel
        extra = {} if name == 'transformer' else dict(backbone=name, num_memory_tokens=8)
        net = cls(in_channels=4, hidden_size=64, cond_dim=6, depth=2, num_heads=2,
                  patch_size=1, input_size=2, num_frames=6, adaln_lora_dim=8, **extra).to(args.device)
        torch.nn.init.normal_(net.shared_mod[-1].weight, std=.03)
        torch.nn.init.normal_(net.final_layer.linear.weight, std=.03)
        optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)
        clean = torch.randn(1, 4, 6, 2, 2, device=args.device)
        noise = torch.randn_like(clean)
        timepoints = torch.tensor([[.1, .1, .5, .5, .9, .9]], device=args.device)
        t = timepoints[:, None, :, None, None]
        noisy = (1-t)*clean + t*noise
        cond = torch.randn(1, 6, 6, device=args.device)
        def call(module):
            with torch.autocast(args.device, dtype=torch.bfloat16, enabled=args.device == 'cuda'):
                return module(noisy, timepoints, cond, temporal_causal=True, chunk_size=2)
        started = time.monotonic()
        if args.device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        eager = call(net)
        eager_loss = (eager.float() - (clean-noise)).square().mean()
        eager_loss.backward()
        eager_grads = {n: p.grad.detach().clone() for n, p in net.named_parameters() if p.grad is not None}
        optimizer.zero_grad(set_to_none=True)
        print(f'{name}: eager forward/backward passed; compile={args.compile}', flush=True)
        runner = torch.compile(net, fullgraph=True) if args.compile else net
        output = call(runner)
        torch.testing.assert_close(output.float(), eager.float(), atol=.006, rtol=.04)
        loss = (output.float() - (clean-noise)).square().mean()
        loss.backward()
        assert torch.isfinite(loss)
        for n, p in net.named_parameters():
            if n in eager_grads:
                assert p.grad is not None and torch.isfinite(p.grad).all(), n
                torch.testing.assert_close(p.grad, eager_grads[n], atol=.005, rtol=.08)
        before = net.final_layer.linear.weight.detach().clone()
        optimizer.step()
        assert not torch.equal(before, net.final_layer.linear.weight)
        results[name] = dict(loss=loss.item(), fullgraph_compile=args.compile,
                            parameters=sum(p.numel() for p in net.parameters()),
                            seconds=time.monotonic()-started,
                            peak_allocated_mb=torch.cuda.max_memory_allocated()/2**20 if args.device=='cuda' else None)
        print(name, results[name], flush=True)
        del runner, net, optimizer, eager, output, loss, eager_loss, eager_grads
        if args.device == 'cuda':
            torch.cuda.empty_cache()
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(torch=torch.__version__, cuda=torch.version.cuda,
                    device=args.device, synthetic=True, results=results), indent=2)+'\n')


if __name__ == '__main__':
    main()
