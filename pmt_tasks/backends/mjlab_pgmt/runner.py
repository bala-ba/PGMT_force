"""PGMT training/resume on MJWarp, with separate flat EMD evaluation worlds."""
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from functools import partial
from enum import Enum
from importlib.metadata import version
from pathlib import Path
import hashlib

import torch

from motion_tracking_rl.runners.on_policy_runner import OnPolicyRunner
from motion_tracking_rl.utils.pgmt_checkpoint import (
    canonical, migrate_stage1, restore_training, training_payload, validate_tensors, capture_rng, restore_rng,
)


def serialize(value):
    if isinstance(value, Enum):
        return serialize(value.value)
    if is_dataclass(value):
        return serialize(asdict(value))
    if isinstance(value, partial):
        return dict(function=serialize(value.func), args=serialize(value.args), keywords=serialize(value.keywords))
    if isinstance(value, dict):
        return {str(k): serialize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize(v) for v in value]
    # mjlab configs also contain IntEnum values, which canonical handles as ints.
    return canonical(value)


class MJLabPGMTRunner(OnPolicyRunner):
    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        cfg = deepcopy(train_cfg)
        self.training_contract = dict(backend="mjlab", version=1, environment=serialize(env.cfg),
            data=deepcopy(env.command.identity), policy=deepcopy(cfg["policy"]),
            algorithm=deepcopy(cfg["algorithm"]), obs_groups=deepcopy(cfg["obs_groups"]),
            num_steps_per_env=cfg["num_steps_per_env"],
            packages={name: version(name) for name in ("mjlab", "mujoco", "mujoco-warp")})
        self.training_contract = canonical(self.training_contract)
        self.completed_iterations = 0
        self.initialization = dict(kind="scratch", backend="mjlab")
        self._eval_env = None
        super().__init__(env, cfg, log_dir, device)
        if self.is_distributed:
            raise ValueError("Use one process for PGMT mjlab training; distributed checkpointing is not wired")
        update = self.alg.update
        def counted_update():
            result = update()
            self.completed_iterations += 1
            return result
        self.alg.update = counted_update
        if env.cfg.pgmt_stage == 2:
            self._emd_evaluation = self

    def initialize(self, path, *, instanttrack_weights="ema"):
        if self.completed_iterations or self.alg.optimizer.state:
            raise ValueError("Weights initialization requires a fresh runner")
        saved = torch.load(path, map_location="cpu", weights_only=False)
        family = saved.get("policy_metadata", {}).get("policy_family")
        target_family = self.alg.policy.get_checkpoint_metadata()["policy_family"]
        if saved.get("format") == "instanttrack.pgmt_force_cop.v1":
            from motion_tracking_rl.utils.pgmt_force_checkpoint import migrate_instanttrack_force
            self.initialization = migrate_instanttrack_force(self.alg.policy, path, weights=instanttrack_weights,
                                                            critic_init="source")
        elif family == "pgmt_stage1_force_v1" and target_family == "pgmt_stage2_force_v1":
            from motion_tracking_rl.utils.pgmt_force_checkpoint import migrate_force_stage2
            self.initialization = migrate_force_stage2(self.alg.policy, path)
        elif family == "pgmt_stage1" and target_family == "pgmt_stage1_force_v1":
            from motion_tracking_rl.utils.pgmt_force_checkpoint import migrate_stage1_force
            self.initialization = migrate_stage1_force(self.alg.policy, path)
        elif family == "pgmt_stage1" and self.env.cfg.pgmt_stage == 2:
            from motion_tracking_rl.utils.pgmt_checkpoint import migrate_three_head
            self.initialization = migrate_three_head(self.alg.policy, path, terrain_initialization="random")
        else:
            if saved.get("policy_metadata") != self.alg.policy.get_checkpoint_metadata():
                raise ValueError("Weights initialization requires matching PGMT policy metadata")
            validate_tensors(saved["model_state_dict"], self.alg.policy.state_dict())
            self.alg.policy.load_state_dict(saved["model_state_dict"])
            self.initialization = dict(kind="policy_weights_only", source=str(Path(path).resolve()),
                                       sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())

    def save(self, path, infos=None):
        payload = training_payload(self, infos)
        payload["backend"] = "mjlab"
        path = Path(path)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    def load(self, path, **kwargs):
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved.get("backend") != "mjlab":
            raise ValueError("Cross-simulator optimizer resume is forbidden; use --initialize-from")
        return restore_training(self, saved)

    def learn(self, *args, **kwargs):
        result = super().learn(*args, **kwargs)
        self.current_learning_iteration = self.completed_iterations
        return result

    def after_update(self, completed_iterations, *, force=False):
        """Evaluate all complete clips in isolated flat mjlab worlds for EMD.

        The training simulation, observation histories, running normalizers,
        curriculum and RNG are not advanced by evaluation.
        """
        sampler = self.env.command.sampler
        step = completed_iterations * self.num_steps_per_env * self.env.num_envs
        if not force and not sampler.due(step):
            return
        from .evaluation import evaluate_emd
        rng = capture_rng(self.device)
        was_training = self.alg.policy.training
        try:
            self.alg.policy.eval()
            with torch.inference_mode():
                values = evaluate_emd(self, seed=sampler.config["evaluation_seed"])
            sampler.update_emd(values, step)
        finally:
            self.alg.policy.train(was_training)
            restore_rng(rng, self.device)
        if self.writer:
            self.writer.add_scalar("EMD/mean", float(sampler.emd.mean()), completed_iterations)
        print(f"[MJLab PGMT EMD] step={step}, clips={len(values)}, mean={sum(values) / len(values):.6f}", flush=True)

    def close(self):
        if self.writer:
            self.writer.close()
        if self._eval_env:
            self._eval_env.close()
