"""PhysX foot wrenches / equivalent sole-plane CoP, sampled at control rate.

Use the existing explicitly terrain-filtered foot views. Normal and friction
manifolds have separate points/counts; sum both moments about the ankle link.
No body-net-force approximation, contact centroid, or reference contact label.
"""
from copy import deepcopy

import torch

from motion_tracking_rl.pgmt_force_feedback import validate_force_feedback_contract
from pmt_tasks.pgmt_contact_reference import CONTACT_FOOT_NAMES
from pmt_tasks.pgmt_contact_state import TERRAIN_FOOT_SENSORS
from pmt_tasks.pgmt_math import inverse_rotate


def manifold_wrench(forces, points, counts, starts, origins, *, slots=64, magnitudes=None, diagnostics=None):
    """GPU ragged-buffer reduction, including disjoint/out-of-order ranges.

    Returns force and ankle-origin moment in world axes, normal scalar sum,
    and per-environment validity. A saturated/truncated buffer is never valid.
    All unused buffer entries may be NaN and are masked before arithmetic.
    PhysX pools the capacity across environments: `slots` is an allocation
    budget, not a per-foot maximum. A foot may use more than that budget while
    the global buffer still contains its complete manifold.
    """
    n = len(origins)
    counts, starts = counts.reshape(n, -1).long(), starts.reshape(n, -1).long()
    # Compact sensor/filter pairs, not contact points. Sort their intervals to
    # assign every allocated point to its owning environment without N*max(K)
    # padding or an artificial per-environment contact cap.
    pair = (counts > 0).nonzero()
    begin = starts[pair[:, 0], pair[:, 1]]
    end = begin + counts[pair[:, 0], pair[:, 1]]
    owners = pair[:, 0]
    # The sentinel also handles an entirely empty contact stream.
    begin = torch.cat((begin, begin.new_tensor([len(points)])))
    end = torch.cat((end, end.new_tensor([len(points)])))
    owners = torch.cat((owners, owners.new_zeros(1)))
    order = begin.argsort()
    begin, end, owners = begin[order], end[order], owners[order]
    index = torch.arange(len(points), device=counts.device)
    interval = torch.searchsorted(begin.contiguous(), index, right=True) - 1
    selected = interval.clamp_min(0)
    owner = owners[selected]
    used = (interval >= 0) & (index < end[selected])
    f, p = forces, points
    finite = torch.isfinite(f).all(-1) & torch.isfinite(p).all(-1)
    f = torch.where(used[..., None], f, 0.)
    r = torch.where(used[..., None], p - origins[owner], 0.)
    normal = torch.zeros(n, device=f.device, dtype=f.dtype)
    if magnitudes is not None:
        m = magnitudes.flatten()
        finite &= torch.isfinite(m) & (m >= 0)
        normal.index_add_(0, owner, torch.where(used, m, 0.))
    represented = counts.new_zeros(n).index_add_(0, owner, used.long())
    bad_points = counts.new_zeros(n).index_add_(0, owner, (used & ~finite).long())
    total = counts.sum(-1)
    out_of_bounds = ((counts > 0) & ((starts < 0) | (starts + counts > len(points)))).any(-1)
    valid = ((bad_points == 0) & (represented == total) & (counts >= 0).all(-1) & ~out_of_bounds)
    valid &= total.sum() < len(points)
    if diagnostics is not None:
        diagnostics.update(total_points=total.sum(), capacity=len(points), per_env_points=total,
                           represented_points=represented, bad_points=bad_points, out_of_bounds=out_of_bounds)
    force = f.new_zeros(n, 3).index_add_(0, owner, f)
    moment = f.new_zeros(n, 3).index_add_(0, owner, torch.cross(r, f, dim=-1))
    return force, moment, normal, valid


def wrench_features(force_w, moment_w, normal, quat_w, config):
    force = inverse_rotate(quat_w, force_w)
    moment = inverse_rotate(quat_w, moment_w)
    z = config["plane_z"]
    denominator = torch.where(force[..., 2].abs() > 1e-8, force[..., 2], 1.)
    cop = torch.stack(((z * force[..., 0] - moment[..., 1]) / denominator,
                       (moment[..., 0] + z * force[..., 1]) / denominator), -1)
    valid = ((normal >= config["min_normal_force"]) & (force[..., 2] >= config["min_local_fz"])
             & torch.isfinite(cop).all(-1))
    return torch.cat((force, torch.where(valid[..., None], cop, 0.), valid[..., None].to(force.dtype)), -1)


class IsaacFeetForceFeedback:
    def __init__(self, env, config, slots=64):
        self.env, self.config, self.slots = env, validate_force_feedback_contract(config), slots
        self.robot = env.scene["robot"]
        self.body_ids = [self.robot.body_names.index(name) for name in CONTACT_FOOT_NAMES]
        self.sensors = [env.scene[name] for name in TERRAIN_FOOT_SENSORS]
        for sensor, foot in zip(self.sensors, CONTACT_FOOT_NAMES):
            if sensor.body_names != [foot] or sensor.contact_physx_view.sensor_count != env.num_envs:
                raise ValueError("Force feedback requires one terrain-filtered foot sensor per environment")
            if sensor.contact_physx_view.max_contact_data_count < env.num_envs * slots:
                raise ValueError("Force feedback contact buffer is too small")
        self.reset_step = torch.full((env.num_envs,), env.common_step_counter,
                                     device=env.device, dtype=torch.long)
        self._key = None
        self.output = torch.zeros(env.num_envs, 12, device=env.device)
        self.normal_force_w = torch.zeros(env.num_envs, 2, 3, device=env.device)
        self.friction_force_w = torch.zeros_like(self.normal_force_w)

    def reset(self, ids):
        self.reset_step[ids] = self.env.common_step_counter
        self.output[ids] = 0
        self.normal_force_w[ids] = 0
        self.friction_force_w[ids] = 0
        self._key = None

    @torch.no_grad()
    def compute(self):
        env = self.env
        if self._key == env.common_step_counter:
            return self.output
        features, valid, diagnostics = [], [], []
        for foot, (sensor, body) in enumerate(zip(self.sensors, self.body_ids)):
            view = sensor.contact_physx_view
            origin = self.robot.data.body_pos_w[:, body]
            quat = self.robot.data.body_quat_w[:, body]
            magnitude, points, normals, _, counts, starts = view.get_contact_data(dt=env.physics_dt)
            diagnostic = dict(normal={}, friction={})
            nf, nm, normal, nv = manifold_wrench(magnitude * normals, points, counts, starts,
                origin, slots=self.slots, magnitudes=magnitude, diagnostics=diagnostic['normal'])
            # PhysX reuses the count/start buffers in these two API calls.
            # The reduction above consumes them before requesting friction data.
            forces, points, counts, starts = view.get_friction_data(dt=env.physics_dt)
            ff, fm, _, fv = manifold_wrench(forces, points, counts, starts, origin, slots=self.slots,
                                           diagnostics=diagnostic['friction'])
            diagnostics.append(diagnostic)
            self.normal_force_w[:, foot] = nf
            self.friction_force_w[:, foot] = ff
            features.append(wrench_features(nf + ff, nm + fm, normal, quat, self.config))
            valid.append(nv & fv)
        result = torch.stack(features, 1)
        reset = self.reset_step == env.common_step_counter
        result[reset] = 0
        self.normal_force_w[reset] = 0
        self.friction_force_w[reset] = 0
        if not ((torch.stack(valid, -1) | reset[:, None]).all() and torch.isfinite(result).all()):
            details = [{stream: {k: v.tolist() if isinstance(v, torch.Tensor) else v
                                  for k, v in values.items()} for stream, values in foot.items()}
                       for foot in diagnostics]
            raise RuntimeError(f"Invalid/saturated PhysX force-feedback manifold at step {env.common_step_counter}; "
                               f"refusing truncated observations: {details}")
        self.output.copy_(result.flatten(1))
        self._key = env.common_step_counter
        return self.output


def foot_force_observation(env, config, slots=64):
    feedback = getattr(env, "_pgmt_force_feedback", None)
    if feedback is None:
        feedback = env._pgmt_force_feedback = IsaacFeetForceFeedback(env, config, slots)
    return feedback.compute()


def force_history_observation(env, config, slots=64):
    from pmt_tasks.pgmt_runtime import get_pgmt_state
    return torch.cat((get_pgmt_state(env).observations()["proprio"], foot_force_observation(env, config, slots)), -1)


def capture_critic_sim_context(env, env_ids):
    """Cache per-environment PhysX properties once, after all startup randomizations."""
    from pmt_tasks.pgmt_runtime import get_pgmt_state
    state = get_pgmt_state(env)
    robot = state.robot
    friction = torch.cat([env.scene[name].body_physx_view.get_material_properties()[..., 1]
                          for name in TERRAIN_FOOT_SENSORS], dim=1).mean(1, keepdim=True)
    bias = robot.data.default_joint_pos - state.nominal_critic_joint_pos
    com_delta = robot.root_physx_view.get_coms()[:, state.anchor, :3] - state.nominal_torso_com
    sim_context = torch.cat(((friction.to(env.device) - .75) / .45,
                             bias[:, state.sdk_joint_order] / .01, com_delta.to(env.device) / .05), dim=-1)
    state.critic_sim_context.copy_(sim_context)
    state.invalidate_observations()


def enable_isaac_force_feedback(cfg, agent, config=None, *, slots=64, proprio_layout="pgmt96",
                                current_force_feedback=False, terrain_privileged_critic=False):
    """Opt-in observation/network change; preserve the archived PPO reward heads."""
    from isaaclab.managers import EventTermCfg, ObservationTermCfg
    if slots < 4 or agent.policy.class_name != "PGMTActorCritic":
        raise ValueError("Expected PGMT policy and at least four contact slots per foot")
    config = validate_force_feedback_contract(config)
    params = dict(config=deepcopy(config), slots=slots)
    if current_force_feedback and proprio_layout != "instanttrack99":
        raise ValueError("Current force feedback requires the InstantTrack99 base fields")
    if proprio_layout == "instanttrack99":
        cfg.pgmt_actor_proprio_layout = agent.policy.proprio_layout = proprio_layout
        group = deepcopy(cfg.observations.proprio_history)
        group.proprio = ObservationTermCfg(func=foot_force_observation, params=params)
        group.enable_corruption = False
        cfg.observations.force_history = group
        agent.obs_groups["force_history"] = ["force_history"]
        if current_force_feedback:
            # Both groups read the same cached post-control PhysX measurement.
            cfg.observations.proprio.feet = ObservationTermCfg(func=foot_force_observation, params=params)
            agent.policy.current_force_feedback = True
    elif proprio_layout == "pgmt96":
        cfg.observations.proprio_history.proprio = ObservationTermCfg(func=force_history_observation, params=params)
    else:
        raise ValueError(f"Unknown actor proprio layout: {proprio_layout}")
    if terrain_privileged_critic:
        if proprio_layout != "instanttrack99" or cfg.scene.height_scanner is None:
            raise ValueError("Terrain privileged critic requires the InstantTrack99 layout and a height scanner")
        from pmt_tasks.env_cfgs.pgmt.placement import PGMTTerrainObservationsCfg
        cfg.pgmt_terrain_privileged_critic = agent.policy.terrain_privileged_critic = True
        cfg.observations.critic.feet = ObservationTermCfg(func=foot_force_observation, params=params)
        cfg.observations.critic_terrain_map = PGMTTerrainObservationsCfg.TerrainMapCfg()
        agent.obs_groups["critic_terrain_map"] = ["critic_terrain_map"]
        cfg.events.critic_sim_context = EventTermCfg(func=capture_critic_sim_context, mode="startup")
    for name in TERRAIN_FOOT_SENSORS:
        getattr(cfg.scene, name).max_contact_data_count_per_prim = slots
    agent.policy.class_name = "PGMTForceActorCritic"
    agent.policy.force_feedback_cfg = deepcopy(config)
