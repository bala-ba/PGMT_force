"""PGMT training on mjlab / MuJoCo Warp, without Isaac imports.

The public policy and motion tensors retain PMT's BFS joint/body order. Only
simulator reads and writes cross the named MJCF permutation boundary.
"""


def build_stage1_env(cfg):
    from .config import build_env
    return build_env(cfg, stage=1)


def build_stage2_env(cfg):
    from .config import build_env
    return build_env(cfg, stage=2)
