"""Batched GPU foot wrenches and CoP sampled after MJWarp's final forward."""
from copy import deepcopy
from contextlib import nullcontext

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import torch
import warp as wp

from motion_tracking_rl.pgmt_force_feedback import validate_force_feedback_contract


@wp.kernel
def _accumulate(
    nacon: wp.array(dtype=int), geom: wp.array(dtype=wp.vec2i),
    worldid: wp.array(dtype=int), positions: wp.array(dtype=wp.vec3),
    frames: wp.array(dtype=wp.mat33), force: wp.array(dtype=wp.spatial_vector),
    geom_foot: wp.array(dtype=int), terrain: wp.array(dtype=int),
    foot_bodies: wp.array(dtype=int), xpos: wp.array2d(dtype=wp.vec3),
    xmat: wp.array2d(dtype=wp.mat33),
    force_sum: wp.array2d(dtype=wp.vec3), moment_sum: wp.array2d(dtype=wp.vec3),
    normal_sum: wp.array2d(dtype=float),
):
    i = wp.tid()
    if i >= nacon[0]:
        return
    pair = geom[i]
    if pair[0] < 0 or pair[1] < 0:
        return
    foot = int(-1)
    sign = float(1.)
    if geom_foot[pair[0]] >= 0 and terrain[pair[1]] == 1:
        foot = geom_foot[pair[0]]
        sign = -1.
    elif geom_foot[pair[1]] >= 0 and terrain[pair[0]] == 1:
        foot = geom_foot[pair[1]]
    if foot < 0:
        return
    w = worldid[i]
    b = foot_bodies[foot]
    raw_force = wp.vec3(force[i][0], force[i][1], force[i][2])
    rotation = wp.transpose(xmat[w, b])
    f = rotation * (sign * raw_force)
    r = rotation * (positions[i] - xpos[w, b])
    wp.atomic_add(force_sum, w, foot, f)
    wp.atomic_add(moment_sum, w, foot, wp.cross(r, f))
    wp.atomic_add(normal_sum, w, foot, wp.max(wp.dot(frames[i][0], raw_force), 0.))


@wp.kernel
def _features(
    force: wp.array2d(dtype=wp.vec3), moment: wp.array2d(dtype=wp.vec3),
    normal: wp.array2d(dtype=float), reset_step: wp.array(dtype=wp.int64),
    step: int, plane_z: float, min_normal: float, min_fz: float,
    nacon: wp.array(dtype=int), nefc: wp.array(dtype=int), naconmax: int, njmax: int,
    output: wp.array2d(dtype=float),
):
    w, foot = wp.tid()
    base = foot * 6
    for j in range(6):
        output[w, base + j] = 0.
    if reset_step[w] == wp.int64(step):
        return
    f, m = force[w, foot], moment[w, foot]
    for j in range(3):
        output[w, base + j] = f[j]
    # A saturated buffer cannot provide reliable features. The configured
    # observation NaN policy raises, instead of silently training on truncation.
    if nacon[0] >= naconmax or nefc[w] >= njmax:
        output[w, base] = wp.nan
        return
    if normal[w, foot] >= min_normal and f[2] >= min_fz:
        x = (plane_z * f[0] - m[1]) / f[2]
        y = (m[0] + plane_z * f[1]) / f[2]
        if wp.isfinite(x) and wp.isfinite(y):
            output[w, base + 3] = x
            output[w, base + 4] = y
            output[w, base + 5] = 1.


class FeetForceFeedback:
    def __init__(self, env, config):
        self.config = validate_force_feedback_contract(config)
        self.env = env
        m, d = env.sim.mj_model, env.sim.wp_data
        self.device = d.nacon.device
        geom_foot = np.full(m.ngeom, -1, dtype=np.int32)
        body_ids = []
        for index, side in enumerate(self.config["feet"]):
            body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"robot/{side}_ankle_roll_link")
            if body < 0:
                raise ValueError(f"Missing {side} foot body")
            body_ids.append(body)
            ids = np.flatnonzero((m.geom_bodyid == body) & ((m.geom_contype != 0) | (m.geom_conaffinity != 0)))
            if not len(ids) or np.any(m.geom_condim[ids] != 3):
                raise ValueError("Force feedback V1 requires condim=3 foot collision geometry")
            geom_foot[ids] = index
        terrain = ((m.body_weldid[m.geom_bodyid] == 0)
                   & ((m.geom_contype != 0) | (m.geom_conaffinity != 0))).astype(np.int32)
        if not terrain.any():
            raise ValueError("No static terrain collision geometry")
        self.geom_foot = wp.array(geom_foot, device=self.device)
        self.terrain = wp.array(terrain, device=self.device)
        self.body_ids = wp.array(body_ids, dtype=int, device=self.device)
        self.contact_ids = wp.array(np.arange(d.naconmax, dtype=np.int32), device=self.device)
        self.contact_forces = wp.zeros(d.naconmax, dtype=wp.spatial_vector, device=self.device)
        self.forces = wp.zeros((env.num_envs, 2), dtype=wp.vec3, device=self.device)
        self.moments = wp.zeros_like(self.forces)
        self.normal = wp.zeros((env.num_envs, 2), dtype=float, device=self.device)
        self.output = wp.zeros((env.num_envs, 12), dtype=float, device=self.device)
        self.tensor = wp.to_torch(self.output)

    def compute(self):
        env, cfg = self.env, self.config
        d = env.sim.wp_data
        command = env.command_manager.get_term("motion")
        stream = (wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(env.device)))
                  if self.device.is_cuda else nullcontext())
        with wp.ScopedDevice(self.device), stream:
            for array in (self.contact_forces, self.forces, self.moments, self.normal):
                array.zero_()
            mjwarp.contact_force(env.sim.wp_model, d, self.contact_ids, True, self.contact_forces)
            wp.launch(_accumulate, dim=d.naconmax, inputs=[d.nacon, d.contact.geom, d.contact.worldid,
                d.contact.pos, d.contact.frame, self.contact_forces, self.geom_foot, self.terrain,
                self.body_ids, d.xpos, d.xmat, self.forces, self.moments, self.normal], device=self.device)
            wp.launch(_features, dim=(env.num_envs, 2), inputs=[self.forces, self.moments, self.normal,
                wp.from_torch(command.reset_step), env.common_step_counter, cfg["plane_z"],
                cfg["min_normal_force"], cfg["min_local_fz"], d.nacon, d.nefc, d.naconmax,
                d.njmax, self.output], device=self.device)
        return self.tensor


def force_history_observation(env, config):
    feedback = getattr(env, "_pgmt_force_feedback", None)
    if feedback is None:
        feedback = env._pgmt_force_feedback = FeetForceFeedback(env, config)
    command = env.command_manager.get_term("motion")
    return torch.cat((command.observations()["proprio"], feedback.compute()), dim=-1)


def enable_force_feedback_v1(cfg, agent, config=None):
    """Opt-in assembly; default configs/checkpoint contracts stay unchanged."""
    from mjlab.managers import ObservationTermCfg
    if cfg.pgmt_stage not in (1, 2) or agent["policy"]["class_name"] != "PGMTActorCritic":
        raise ValueError("Force feedback V1 requires a PGMT Stage1/Stage2 configuration")
    config = validate_force_feedback_contract(config)
    history = cfg.observations["proprio_history"]
    history.terms = {"state": ObservationTermCfg(func=force_history_observation,
                                               params={"config": deepcopy(config)})}
    history.nan_policy = "error"
    agent["policy"].update(class_name="PGMTForceActorCritic", force_feedback_cfg=deepcopy(config))
