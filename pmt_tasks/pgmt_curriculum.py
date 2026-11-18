"""PGMT terrain progression, independent of the motion sampler's success counts."""
import copy
import hashlib
from pathlib import Path

import torch
from pmt_tasks.pgmt_survival import NormalEpisodeSurvival


OUTCOMES = ("success", "failure", "map_exit", "timeout", "incomplete_motion", "other_truncation")
SAMPLER_COUNTS = ("failed_motion_count", "success_motion_count",
                  "current_failed_motion_count", "current_success_motion_count")


class TerrainCurriculum:
    """Settle each birth-assigned episode once, before selecting its next motion.

    Legacy completion requires frame zero. EMD/random-start training requires
    completing the remaining reference, with the corresponding real steps. The
    one-step allowance matches Isaac's post-reset command advance. This clock
    is independent of the runner's randomized episode_length_buf.
    """

    def __init__(self, terrain, identity, *, horizon_steps=None):
        self.terrain, self.identity = terrain, copy.deepcopy(identity)
        levels = terrain.terrain_levels
        self.device = levels.device
        self.num_envs = len(levels)
        self.num_levels, self.num_families = terrain.terrain_origins.shape[:2]
        self.active = torch.zeros_like(levels, dtype=torch.bool)
        self.start_step = torch.zeros_like(levels)
        self.start_frame = torch.zeros_like(levels)
        self.motion_ids = torch.zeros_like(levels)
        self.lengths = torch.ones_like(levels)
        self.birth_levels, self.birth_types = levels.clone(), terrain.terrain_types.clone()
        self.settled_step = torch.full_like(levels, -1)
        self.sampler_pending = torch.zeros_like(self.active)
        self.last_outcome = torch.zeros_like(levels)  # 0 = no settled episode, else OUTCOMES index + 1
        shape = (self.num_families, self.num_levels)
        self.counts = torch.zeros((*shape, len(OUTCOMES)), dtype=torch.long, device=self.device)
        self.promotions = torch.zeros(shape, dtype=torch.long, device=self.device)
        self.demotions = torch.zeros_like(self.promotions)
        self.horizon_steps = horizon_steps
        self.survival = NormalEpisodeSurvival()
        self.recovery = torch.zeros_like(self.active)
        self.recovery_counts = torch.zeros_like(self.counts)
        self.transition_counts = torch.zeros(2, device=self.device, dtype=torch.long)  # all, recovery
        self.transition_step = None

    def account_transitions(self, step):
        """Close elapsed control steps before replacing any episode's birth flag."""
        if self.horizon_steps is None:
            return
        if self.transition_step is not None and step > self.transition_step:
            elapsed = step - self.transition_step
            self.transition_counts += elapsed * torch.stack((self.active.sum(), (self.active & self.recovery).sum()))
        self.transition_step = step

    def begin(self, ids, step, motion_ids, frame_ids, lengths, *, recovery=False):
        self.account_transitions(step)
        self.active[ids] = True
        self.recovery[ids] = recovery
        self.sampler_pending[ids] = False
        self.start_step[ids] = step
        self.motion_ids[ids], self.start_frame[ids], self.lengths[ids] = motion_ids, frame_ids, lengths
        self.birth_levels[ids] = self.terrain.terrain_levels[ids]
        self.birth_types[ids] = self.terrain.terrain_types[ids]

    def settle(self, ids, *, step, motion_ids, frame_ids, failure, finished, map_exit, timeout, other):
        self.account_transitions(step)
        elapsed = step - self.start_step[ids]
        eligible = self.active[ids] & (elapsed > 0) & (failure | finished | map_exit | timeout | other)
        random_start = self.identity.get("sampler", {}).get("kind") == "pbfm_emd"
        starts = self.start_frame[ids]
        valid_start = ((starts >= 0) & (starts < self.lengths[ids] - 1) if random_start else starts == 0)
        required_steps = self.lengths[ids] - 1 - (starts if random_start else 0)
        complete = (valid_start & (motion_ids == self.motion_ids[ids])
                    & (frame_ids == self.lengths[ids] - 1)
                    & (elapsed >= required_steps.clamp_min(1)))
        # Failure wins if it coincides with the last frame or leaving the map.
        # A completed clip may coincide with the ordinary episode time limit.
        reason = torch.where(failure, 2, torch.where(map_exit, 3, torch.where(other, 6,
                 torch.where(finished, torch.where(complete, 1, 5), 4))))
        chosen, reason = ids[eligible], reason[eligible]
        old_level, family = self.birth_levels[chosen], self.birth_types[chosen]
        ordinary = ~self.recovery[chosen]
        if self.horizon_steps is not None:
            normal_ids = chosen[ordinary].sort().values
            # L frames have L-1 playable control intervals. Use the actual birth frame,
            # even when completion and timeout happen on the same control step.
            remaining = self.lengths[normal_ids] - 1 - self.start_frame[normal_ids]
            horizon = remaining.clamp(min=1, max=self.horizon_steps)
            ratios = ((step - self.start_step[normal_ids]).double() / horizon).clamp(0, 1)
            self.survival.observe(ratios.tolist())
        change = (reason == 1).long() - (reason == 2).long()
        new_level = (old_level + ordinary * change).clamp(0, self.num_levels - 1)
        cells = family * self.num_levels + old_level
        bins = cells * len(OUTCOMES) + reason - 1
        by_mode = torch.bincount(bins + (~ordinary).long() * self.counts.numel(),
                                 minlength=2 * self.counts.numel()).reshape(2, *self.counts.shape)
        self.counts += by_mode[0]
        self.recovery_counts += by_mode[1]
        self.promotions += torch.bincount(cells[new_level > old_level], minlength=self.promotions.numel()).reshape_as(self.promotions)
        self.demotions += torch.bincount(cells[new_level < old_level], minlength=self.demotions.numel()).reshape_as(self.demotions)
        self.terrain.terrain_levels[chosen] = new_level
        self.terrain.terrain_types[chosen] = family
        self.terrain.env_origins[chosen] = self.terrain.terrain_origins[new_level, family]
        self.last_outcome[chosen], self.settled_step[chosen] = reason, step
        self.sampler_pending[chosen] = True  # Motion sampling retains its own outcome policy.
        # Every reset abandons its old episode, including manual/initial resets.
        self.active[ids] = False

    def metrics(self):
        totals = self.counts.sum((0, 1))
        result = {name: totals[i] for i, name in enumerate(OUTCOMES)}
        result.update(mean_level=self.terrain.terrain_levels.float().mean(),
                      success_rate=totals[0].float() / (totals[0] + totals[1]).clamp_min(1),
                      promotions=self.promotions.sum(), demotions=self.demotions.sum())
        for level in range(self.num_levels):
            result[f"level_{level}_fraction"] = (self.terrain.terrain_levels == level).float().mean()
        return result

    def state_dict(self):
        data = dict(version=1, identity=copy.deepcopy(self.identity),
                    levels=self.terrain.terrain_levels.detach().cpu().clone(),
                    families=self.terrain.terrain_types.detach().cpu().clone(),
                    **{key: getattr(self, key).detach().cpu().clone() for key in ("counts", "promotions", "demotions")})
        if self.horizon_steps is not None:
            data.update(version=2, horizon_steps=self.horizon_steps, horizon_rule="remaining_reference_capped",
                        normal_survival=self.survival.state_dict(), recovery_counts=self.recovery_counts.cpu().clone(),
                        transition_counts=self.transition_counts.cpu().clone())
        return data

    def upgrade_checkpoint(self, state):
        if state.get("version") == 1:
            return dict(state, version=2, horizon_steps=self.horizon_steps, horizon_rule="remaining_reference_capped",
                        normal_survival=NormalEpisodeSurvival().state_dict(),
                        recovery_counts=torch.zeros_like(self.recovery_counts, device="cpu"),
                        transition_counts=torch.zeros(2, dtype=torch.long))
        return state

    def validate_state_dict(self, state):
        state = self.upgrade_checkpoint(state)
        if (state.get("version") != 2 or state.get("horizon_rule") != "remaining_reference_capped"
                or state.get("identity") != self.identity
                or state.get("horizon_steps") != self.horizon_steps):
            raise ValueError("PGMT curriculum checkpoint schema/assets/motions/sampling settings differ")
        expected = dict(levels=(self.num_envs,), families=(self.num_envs,), counts=tuple(self.counts.shape),
                        promotions=tuple(self.promotions.shape), demotions=tuple(self.demotions.shape))
        expected.update(recovery_counts=tuple(self.recovery_counts.shape), transition_counts=(2,))
        result = {"normal_survival": NormalEpisodeSurvival.from_state(state.get("normal_survival"))}
        for key, shape in expected.items():
            value = state.get(key)
            if not isinstance(value, torch.Tensor) or value.dtype != torch.long or tuple(value.shape) != shape or (value < 0).any():
                raise ValueError(f"Invalid PGMT curriculum {key}; resume requires matching environment count and nonnegative int64 state")
            result[key] = value.to(self.device).clone()
        if (result["levels"] >= self.num_levels).any() or (result["families"] >= self.num_families).any():
            raise ValueError("PGMT curriculum checkpoint contains out-of-range terrain assignments")
        if result["transition_counts"][1] > result["transition_counts"][0]:
            raise ValueError("Invalid recovery transition counts")
        return result

    def load_state_dict(self, state):
        values = self.validate_state_dict(state)
        self.terrain.terrain_levels.copy_(values["levels"])
        self.terrain.terrain_types.copy_(values["families"])
        self.terrain.env_origins[:] = self.terrain.terrain_origins[values["levels"], values["families"]]
        for key in ("counts", "promotions", "demotions", "recovery_counts", "transition_counts"):
            getattr(self, key).copy_(values[key])
        self.survival = values["normal_survival"]
        self.transition_step = None
        self.recovery.zero_()
        # Simulator trajectories are restarted; an interrupted rollout is not
        # a completed/failed episode and must not enter either set of counters.
        self.active.zero_()
        self.sampler_pending.zero_()
        self.settled_step.fill_(-1)
        self.last_outcome.zero_()


def _identity(env, command):
    manifests = sorted({Path(m.source_file).parent / "manifest.json" for m in command.data_store.motions})
    sampler_identity = (dict(kind=command.sampler.kind, config=copy.deepcopy(command.sampler.config))
                        if getattr(command.sampler, "kind", None) == "pbfm_emd" else
                        {key: getattr(command.sampler, key) for key in
                         ("alpha", "beta", "uniform_ratio", "update_interval", "enable_frame_sampling")})
    result = dict(terrain_sha256=env.scene.terrain.manifest["asset_sha256"],
                families=list(command.terrain_families), compatibility_sha256=command.compatibility_sha256,
                motion_names=[Path(m.source_file).name for m in command.data_store.motions],
                motion_lengths=command.data_store.motion_lengths.tolist(),
                motion_manifests=[hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None for p in manifests],
                sampler=sampler_identity,
                reset={key: getattr(command.cfg, key) for key in ("fixed_start_frame", "require_valid_support_start")})
    proportions = getattr(env.scene.terrain, "family_proportions", None)
    if proportions is not None:
        result["terrain_family_proportions"] = dict(proportions)
    # Keep historical random-XY identities intact; center resets must not
    # silently resume a curriculum saved with the old reset distribution.
    xy_mode = getattr(command.cfg, "placement_xy_mode", "random")
    if xy_mode != "random":
        result["reset"]["placement_xy_mode"] = xy_mode
    if getattr(command, "placement_mode", "origin") == "patch":
        result["reset"].update(placement_mode="patch", boundary_margin=command.cfg.boundary_margin,
                               yaw_range=list(command.cfg.placement_yaw_range),
                               spawn_clearance=command.cfg.spawn_clearance,
                               max_placement_attempts=command.cfg.max_placement_attempts)
        spawn_grid = getattr(command.cfg, "placement_patch_grid", None)
        if spawn_grid is not None:
            result["reset"]["placement_patch_grid"] = list(spawn_grid)
    perception = getattr(env.cfg, "pgmt_terrain_observation", None)
    if perception is not None:
        from pmt_tasks.pgmt_perception import validate_terrain_observation_contract
        validate_terrain_observation_contract(perception)
        result["terrain_observation"] = copy.deepcopy(perception)
    relaxation = getattr(env.cfg, "pgmt_tracking_relaxation", None)
    if relaxation is not None and relaxation.get("enabled", False):
        from pmt_tasks.pgmt_math import body_groups
        from pmt_tasks.pgmt_tracking_relaxation import TrackingRelaxation
        robot = env.scene["robot"]
        result["tracking_relaxation"] = TrackingRelaxation(
            relaxation, robot.body_names, robot.joint_names,
            body_groups(robot.body_names, robot.joint_names), device="cpu").contract()
    return result


def pgmt_terrain_progress(env, env_ids):
    """Native CurriculumManager hook: called after old rewards, before command reset."""
    command = env.command_manager.get_term("motion")
    state = getattr(env, "_pgmt_terrain_curriculum", None)
    if state is None:
        state = env._pgmt_terrain_curriculum = TerrainCurriculum(env.scene.terrain, _identity(env, command),
                                                                horizon_steps=env.max_episode_length)
    ids = torch.arange(env.num_envs, device=env.device)[env_ids] if isinstance(env_ids, slice) else torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    manager = env.termination_manager
    def flag(name):
        return manager.get_term(name)[ids] if name in manager.active_terms else torch.zeros_like(ids, dtype=torch.bool)
    other = torch.zeros_like(ids, dtype=torch.bool)
    for name in manager.active_terms:
        if name not in ("motion_finished", "map_out_of_bounds", "time_out") and manager.get_term_cfg(name).time_out:
            other |= flag(name)
    state.settle(ids, step=env.common_step_counter, motion_ids=command.motion_ids[ids], frame_ids=command.frame_ids[ids],
                 failure=manager.terminated[ids], finished=flag("motion_finished"), map_exit=flag("map_out_of_bounds"),
                 timeout=flag("time_out"), other=other)
    return state.metrics()


def training_state_dict(env):
    state = getattr(env, "_pgmt_terrain_curriculum", None)
    if state is None:
        return None
    sampler = env.command_manager.get_term("motion").sampler
    if getattr(sampler, "kind", None) == "pbfm_emd":
        return dict(kind="pgmt_terrain", curriculum=state.state_dict(), sampler=sampler.state_dict())
    payload = {k: getattr(sampler, k).detach().cpu().clone() for k in SAMPLER_COUNTS}
    payload["steps_since_update"] = sampler._steps_since_update
    if getattr(sampler, "kind", None) == "terrain_adaptive":
        payload.update(kind=sampler.kind, families=list(sampler.families))
    return dict(kind="pgmt_terrain", curriculum=state.state_dict(), sampler=payload)


def terrain_sampler_checkpoint(sampler, payload):
    """Read per-family state or apportion a legacy pooled prior without multiplying its mass."""
    if "kind" in payload:
        if payload["kind"] != sampler.kind or payload.get("families") != list(sampler.families):
            raise ValueError("PGMT sampler checkpoint kind or terrain family order differs")
        return payload
    eligible = sampler.eligible.cpu()
    # Entirely excluded motions have no observable history for the new sampler.
    weights = eligible.float() / eligible.sum(0).clamp_min(1)
    converted = dict(kind=sampler.kind, families=list(sampler.families), steps_since_update=payload.get("steps_since_update"))
    for key in SAMPLER_COUNTS:
        value = payload.get(key)
        if (not isinstance(value, torch.Tensor) or value.shape != (sampler.num_motions,)
                or value.dtype != weights.dtype or not torch.isfinite(value).all() or (value < 0).any()):
            raise ValueError(f"Invalid legacy PGMT sampler checkpoint field: {key}")
        converted[key] = weights * value.cpu()
    return converted


def validate_training_state_dict(env, saved):
    state = getattr(env, "_pgmt_terrain_curriculum", None)
    if state is None or saved.get("kind") != "pgmt_terrain":
        raise ValueError("Checkpoint requires the PGMT terrain curriculum; use load_environment=False for weights-only loading")
    # Validate the entire payload before changing any live environment state.
    state.validate_state_dict(saved["curriculum"])
    sampler = env.command_manager.get_term("motion").sampler
    payload = saved["sampler"]
    if getattr(sampler, "kind", None) == "pbfm_emd":
        sampler.validate_state_dict(payload)
        return state, sampler, payload, None
    if getattr(sampler, "kind", None) == "terrain_adaptive":
        payload = terrain_sampler_checkpoint(sampler, payload)
    tensors = {}
    for key in SAMPLER_COUNTS:
        value = payload.get(key)
        target = getattr(sampler, key)
        if (not isinstance(value, torch.Tensor) or value.shape != target.shape or value.dtype != target.dtype
                or not torch.isfinite(value).all() or (value < 0).any()):
            raise ValueError(f"Invalid PGMT sampler checkpoint field: {key}")
        tensors[key] = value.to(target.device).clone()
    steps = payload.get("steps_since_update")
    if not isinstance(steps, int) or not 0 <= steps < sampler.update_interval:
        raise ValueError("Invalid PGMT sampler EMA step counter")
    return state, sampler, tensors, steps


def load_training_state_dict(env, saved):
    state, sampler, tensors, steps = validate_training_state_dict(env, saved)
    state.load_state_dict(saved["curriculum"])
    if getattr(sampler, "kind", None) == "pbfm_emd":
        sampler.load_state_dict(tensors)
    else:
        for key, value in tensors.items():
            getattr(sampler, key).copy_(value)
        sampler._steps_since_update = steps
        sampler._update_metrics()
    # With active=False and sampler_pending=False, this reset neither settles
    # a phantom episode nor changes the restored sampling counts.
    env.reset()
