"""MJWarp adapter for Isaac T1 touchdown events and T5 geometry_v3.

Reward contacts belong to the last solved physics substep, before reset and
reference advancement. Force/CoP actor observations keep their separate final
forward sampling. Normal forces (not friction-inclusive wrenches) match the
PhysX contact sensor's force convention used by the Isaac reward kernels.
"""
from contextlib import nullcontext
from copy import deepcopy

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import torch
import warp as wp

from pmt_tasks.pgmt_contact_geometry import CONTACT_GEOMETRY_CONFIG, TerrainSupportState, reference_switch_candidates
from pmt_tasks.pgmt_contact_state import ContactEventState
from pmt_tasks.pgmt_contact_match import evaluate_reference_contact_match
from pmt_tasks.pgmt_slip import evaluate_slip
from pmt_tasks.pgmt_stumble import STUMBLE_CONFIG, evaluate_stumble
from pmt_tasks.pgmt_contact_force import evaluate_contact_force
from pmt_tasks.pgmt_touchdown import evaluate_touchdown, validate_touchdown_config
from .motion import BFS_BODIES


@wp.kernel
def _collect(nacon: wp.array(dtype=int), pairs: wp.array(dtype=wp.vec2i),
             world: wp.array(dtype=int), positions: wp.array(dtype=wp.vec3),
             frames: wp.array(dtype=wp.mat33), forces: wp.array(dtype=wp.spatial_vector),
             feet: wp.array(dtype=int), terrain: wp.array(dtype=int), slots: int,
             net: wp.array2d(dtype=wp.vec3), ground: wp.array2d(dtype=wp.vec3),
             points: wp.array3d(dtype=wp.vec3), counts: wp.array2d(dtype=int)):
    i = wp.tid()
    if i >= nacon[0]:
        return
    pair = pairs[i]
    if pair[0] < 0 or pair[1] < 0:
        return
    normal = frames[i][0]
    raw = wp.vec3(forces[i][0], forces[i][1], forces[i][2])
    magnitude = wp.dot(raw, normal)
    normal_force = wp.max(magnitude, 0.) * normal
    w = world[i]
    # Both feet can participate in a self-contact: accumulate each side.
    for side in range(2):
        foot = feet[pair[side]]
        if foot >= 0:
            sign = float(1.)
            if side == 0:
                sign = -1.
            wp.atomic_add(net, w, foot, sign * normal_force)
            if terrain[pair[1-side]] == 1:
                wp.atomic_add(ground, w, foot, sign * normal_force)
                if magnitude > 0.:
                    slot = wp.atomic_add(counts, w, foot, 1)
                    if slot < slots:
                        points[w, foot, slot] = positions[i]


@wp.kernel
def _invalidate(nacon: wp.array(dtype=int), nefc: wp.array(dtype=int),
                capacity: int, njmax: int, slots: int,
                net: wp.array2d(dtype=wp.vec3), ground: wp.array2d(dtype=wp.vec3),
                counts: wp.array2d(dtype=int)):
    w, foot = wp.tid()
    if nacon[0] >= capacity or nefc[w] >= njmax:
        net[w, foot] = wp.vec3(wp.nan, wp.nan, wp.nan)
        ground[w, foot] = wp.vec3(wp.nan, wp.nan, wp.nan)
        counts[w, foot] = slots + 1


class TerrainContactReader:
    """Read solved contacts on GPU, keeping up to 64 raw points per foot."""
    def __init__(self, env, slots):
        self.env, self.slots = env, slots
        m, d = env.sim.mj_model, env.sim.wp_data
        self.device = d.nacon.device
        foot_map = np.full(m.ngeom, -1, dtype=np.int32)
        for index, side in enumerate(("left", "right")):
            body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"robot/{side}_ankle_roll_link")
            ids = np.flatnonzero((m.geom_bodyid == body) & ((m.geom_contype != 0) | (m.geom_conaffinity != 0)))
            if body < 0 or not len(ids):
                raise ValueError(f"Missing {side} foot collision geometry")
            foot_map[ids] = index
        # Explicit imported terrain only; exclude other static scene objects.
        terrain = np.array([(m.geom(i).name or "").startswith("pgmt_terrain_")
                            for i in range(m.ngeom)], dtype=np.int32)
        if not terrain.any():
            raise ValueError("Terrain rewards require the audited native terrain")
        self.feet = wp.array(foot_map, device=self.device)
        self.terrain = wp.array(terrain, device=self.device)
        self.ids = wp.array(np.arange(d.naconmax, dtype=np.int32), device=self.device)
        self.forces = wp.zeros(d.naconmax, dtype=wp.spatial_vector, device=self.device)
        self.net = wp.zeros((env.num_envs, 2), dtype=wp.vec3, device=self.device)
        self.ground = wp.zeros_like(self.net)
        self.points = wp.zeros((env.num_envs, 2, slots), dtype=wp.vec3, device=self.device)
        self.counts = wp.zeros((env.num_envs, 2), dtype=int, device=self.device)
        self.net_tensor, self.ground_tensor = wp.to_torch(self.net), wp.to_torch(self.ground)
        self.points_tensor, self.counts_tensor = wp.to_torch(self.points), wp.to_torch(self.counts)
        self.slot_ids = torch.arange(slots, device=env.device)

    def read(self):
        env, d = self.env, self.env.sim.wp_data
        stream = (wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(env.device)))
                  if self.device.is_cuda else nullcontext())
        with wp.ScopedDevice(self.device), stream:
            for array in (self.forces, self.net, self.ground, self.points, self.counts):
                array.zero_()
            mjwarp.contact_force(env.sim.wp_model, d, self.ids, True, self.forces)
            wp.launch(_collect, dim=d.naconmax, inputs=[d.nacon, d.contact.geom, d.contact.worldid,
                d.contact.pos, d.contact.frame, self.forces, self.feet, self.terrain, self.slots,
                self.net, self.ground, self.points, self.counts], device=self.device)
            wp.launch(_invalidate, dim=(env.num_envs, 2), inputs=[d.nacon, d.nefc, d.naconmax,
                d.njmax, self.slots, self.net, self.ground, self.counts], device=self.device)

    def cloud(self):
        valid = (self.slot_ids < self.counts_tensor[..., None]) & torch.isfinite(self.points_tensor).all(-1)
        return self.points_tensor, valid, self.counts_tensor > self.slots


class TerrainContactRewards:
    def __init__(self, command):
        self.command, self.env = command, command._env
        env = self.env
        validate_touchdown_config(env.cfg.pgmt_touchdown_quality)
        if env.cfg.pgmt_touchdown_quality["settlement"] != "touchdown_event_v1":
            raise ValueError("MJWarp Stage2 T1 requires touchdown_event_v1")
        self.contract = deepcopy(dict(mode="touchdown_event_v1", switching_mode="geometry_v3", version=2,
            contacts=env.cfg.pgmt_contact_events, touchdown=env.cfg.pgmt_touchdown_quality,
            geometry=CONTACT_GEOMETRY_CONFIG, stumble=STUMBLE_CONFIG,
            force="all-object normal force world vector", terrain_force="native terrain only",
            timing="last solved substep before reset", switching_force="mean of all physics substeps"))
        self.reader = TerrainContactReader(env, self.contract["geometry"]["contact_slots"])
        self.support = TerrainSupportState(env.num_envs, device=env.device, dt=env.step_dt,
                                          config=self.contract["geometry"])
        self.contact = ContactEventState(env.num_envs, device=env.device, dt=env.step_dt,
                                        config=self.contract["contacts"])
        self.feet = [BFS_BODIES.index(f"{side}_ankle_roll_link") for side in ("left", "right")]
        self.force_sum = torch.zeros(env.num_envs, 2, device=env.device)
        self.sample_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.snapshot = None

    def reset(self, ids):
        self.support.reset(ids)
        self.contact.reset(ids)
        self.force_sum[ids] = 0
        self.sample_count[ids] = 0
        # The settled reward snapshot survives automatic reset for PPO/logging.

    def capture_substep(self):
        self.reader.read()
        force_z = self.reader.ground_tensor[..., 2]
        self.force_sum.add_(force_z)
        self.sample_count.add_(1)
        return force_z.sum(-1)

    @torch.no_grad()
    def compute(self, state):
        c, step = self.command, self.env.common_step_counter
        mean = self.force_sum / self.sample_count.clamp_min(1)[:, None]
        mean = torch.where((self.sample_count == self.env.cfg.decimation)[:, None], mean, float("nan"))
        labels = c.data_store.contact_labels
        switching = self.support.update(step, mean, **reference_switch_candidates(
            labels, c.motion_ids, c.frame_ids, self.support.window))
        self.contact.update(step, self.reader.net_tensor, self.reader.ground_tensor)
        contact = self.contact.snapshot()
        reference = labels.gather(c.motion_ids, c.frame_ids)
        touchdown = evaluate_touchdown(contact["touchdown"],
            state["body_pos"][:, self.feet], state["body_quat"][:, self.feet],
            state["body_pos"][:, c.anchor], state["body_quat"][:, c.anchor],
            c.terrain.query, self.contract["touchdown"])
        match = evaluate_reference_contact_match(contact["contact"], contact["contact_valid"],
                                                 reference["contacts"], reference["contact_valid"])
        slip = evaluate_slip(contact["contact"], contact["contact_valid"], state["body_lin_vel"][:, self.feet])
        stumble = evaluate_stumble(contact["net_forces_w"], contact["contact_valid"], self.contract["stumble"])
        force = evaluate_contact_force(contact["net_forces_w"], contact["contact_valid"])
        self.snapshot = dict(touchdown_quality=touchdown, reference_contact_match=match, slip=slip,
                             stumble=stumble, contact_switching=switching, contact_force=force)
        rates = {"terrain_" + name: result["rate"] for name, result in self.snapshot.items()}
        self.snapshot.update(contact=contact, reference=reference, physics_samples=self.sample_count.clone(),
                             step=step, motion_ids=c.motion_ids.clone(), frame_ids=c.frame_ids.clone())
        self.force_sum.zero_()
        self.sample_count.zero_()
        return rates
