"""Shared CLI, checkpoint metadata and model construction for video backbones."""
import argparse

BACKBONE_DEFAULTS = dict(
    backbone='transformer', transformer_execution='serial', num_memory_tokens=2400, memory_window_frames=0,
    slot_embed=True, gated_ema=True, write_from_last=True,
    state_sharing=True, assigned_write=True,
)


def add_backbone_args(parser):
    parser.add_argument('--backbone', choices=['transformer', 'rtransformer', 'tas'], default='transformer')
    parser.add_argument('--transformer_execution', choices=['serial', 'parallel'], default='serial',
                        help='Transformer training execution; serial shares the recurrent chunk loop')
    parser.add_argument('--num_memory_tokens', type=int, default=2400)
    parser.add_argument('--memory_window_frames', type=int, default=0,
                        help='Legacy metadata field; TaS requires 0 (only memory crosses chunks)')
    for name in ('slot_embed', 'gated_ema', 'write_from_last', 'state_sharing', 'assigned_write'):
        parser.add_argument('--' + name, action=argparse.BooleanOptionalAction, default=True)


def backbone_config(args):
    return {key: getattr(args, key, value) for key, value in BACKBONE_DEFAULTS.items()}


def build_video_model(size, cfg, **kwargs):
    from miniworld.miniworld import MiniWorldModels
    config = backbone_config(cfg)
    backbone = config.pop('backbone')
    execution = config.pop('transformer_execution')
    if backbone == 'transformer' and execution == 'parallel':
        return MiniWorldModels[size](**kwargs)
    from miniworld.recurrent import RecurrentMiniWorldModel
    return MiniWorldModels[size](_model_class=RecurrentMiniWorldModel,
                                 backbone=backbone, **config, **kwargs)
