"""mjlab lifecycle adapters around PGMT's simulator-independent tensor math."""
from dataclasses import dataclass, field
from pathlib import Path
import hashlib

import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.envs.mdp.actions.actions import JointPositionAction, JointPositionActionCfg

from pmt_tasks.pgmt_math import (TRACKING_CONFIG, body_groups, build_observations, reward_terms,
                                 future_offsets, inverse_rotate, rotate, yaw_quat)
from pmt_tasks.pgmt_perception import terrain_observation_from_hits, pack_terrain_observation
from pmt_tasks.pgmt_reference import initial_transform, transform_reference, place_on_surface
from pmt_tasks.pgmt_terminations import ConsecutiveFailureTimer
from pmt_tasks.pgmt_tracking_relaxation import TrackingRelaxation
from .motion import BFS_BODIES, BFS_JOINTS, MotionLibrary, BinSampler


class BFSPositionAction(JointPositionAction):
    def _find_targets(self, cfg):
        return self._entity.find_joints(BFS_JOINTS, preserve_order=True)


@dataclass(kw_only=True)
class BFSPositionActionCfg(JointPositionActionCfg):
    def build(self, env):
        return BFSPositionAction(self, env)


@dataclass(kw_only=True)
class MotionCfg(CommandTermCfg):
    motion_files: list[str]
    stage: int = 1
    chunk_length: int = 4000
    terrain_asset: str | None = None
    compatibility_file: str | None = None
    placement_xy_mode: str = "center"
    fixed_start_frame: int = -1
    emd: dict = field(default_factory=dict)
    body_names: tuple[str, ...] = tuple(BFS_BODIES)
    noise: bool = True
    evaluation: bool = False
    tracking: dict = field(default_factory=lambda: dict(TRACKING_CONFIG))

    def build(self, env):
        return Motion(self, env)


class Motion(CommandTerm):
    """Owns references, sampler, placement and cached PGMT rates/observations."""
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.robot = env.scene["robot"]
        self.joints = torch.tensor(self.robot.find_joints(BFS_JOINTS, preserve_order=True)[0], device=self.device)
        self.bodies = torch.tensor(self.robot.find_bodies(BFS_BODIES, preserve_order=True)[0], device=self.device)
        self.anchor = BFS_BODIES.index("torso_link")
        self.data_store = MotionLibrary(cfg.motion_files, device=self.device, dt=env.step_dt,
                                        chunk_length=cfg.chunk_length, contact_labels=cfg.stage == 2 and not cfg.evaluation)
        self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.frame_ids = self.motion_ids.clone()
        self.start_frames = self.motion_ids.clone()
        self.started = torch.zeros_like(self.motion_ids, dtype=torch.bool)
        self.reset_step = torch.full_like(self.motion_ids, -1)
        self.reference_rotation = torch.zeros(self.num_envs, 4, device=self.device)
        self.reference_rotation[:, 0] = 1
        self.reference_translation = env.scene.env_origins.clone()
        self.groups = body_groups(BFS_BODIES, BFS_JOINTS, device=self.device,
            end_effector_names=("left_wrist_yaw_link", "right_wrist_yaw_link") if cfg.stage == 2 else None)
        self.previous_target = self.qtarget().clone()
        self.previous_velocity = torch.zeros(self.num_envs, 30, 3, device=self.device)
        self.previous_ref_velocity = torch.zeros_like(self.previous_velocity)
        self.pending_reset = torch.ones_like(self.started)
        self._obs = None
        self._reward_step = -1
        self._rewards = None
        self.timers = {name: ConsecutiveFailureTimer(self.num_envs, dt=env.step_dt,
                          duration_s=.2 if cfg.stage == 2 else 0., device=self.device)
                       for name in ("anchor_pos", "anchor_ori", "ee_body_pos", "root_xy")}
        self.terrain = self.curriculum = None
        self.identity = dict(motions=self.data_store.identity, stage=cfg.stage)
        if cfg.stage == 2 and not cfg.evaluation:
            from .terrain import Terrain
            from pmt_tasks.pgmt_sampling import load_terrain_compatibility
            from pmt_tasks.pgmt_curriculum import TerrainCurriculum
            self.terrain = Terrain(cfg.terrain_asset, self.num_envs, self.device, env.scene.env_origins)
            self.compatible = load_terrain_compatibility(cfg.compatibility_file, self.data_store.files,
                               families=self.terrain.compatibility_families, device=self.device)
            self.identity.update(terrain=hashlib.sha256((Path(cfg.terrain_asset) / "manifest.json").read_bytes()).hexdigest(),
                                 compatibility=hashlib.sha256(Path(cfg.compatibility_file).read_bytes()).hexdigest(),
                                 sampler=dict(kind="pbfm_emd"))
            self.curriculum = TerrainCurriculum(self.terrain, self.identity)
            from .geometry import collision_cover
            cover = collision_cover(env.sim.mj_model)
            self.cover = tuple(torch.as_tensor(v, device=self.device, dtype=torch.long if i == 0 else torch.float32)
                               for i, v in enumerate(cover))
        if cfg.stage == 2:
            from pmt_tasks.pgmt_emd_sampling import EMDMotionSampler
            self.sampler = EMDMotionSampler(self.data_store.num_motions, self.device, cfg.emd)
        else:
            self.sampler = BinSampler(self.data_store.motion_lengths, fps=1 / env.step_dt)
        axis = torch.linspace(-1, 1, 21, device=self.device)
        x, y = torch.meshgrid(axis, axis, indexing="xy")
        self.rays = torch.stack((x.flatten(), y.flatten(), torch.zeros(441, device=self.device)), -1)
        self.terrain_rewards = None
        if self.terrain is not None:
            from .terrain_rewards import TerrainContactRewards
            self.terrain_rewards = TerrainContactRewards(self)
            self.identity["terrain_rewards"] = self.terrain_rewards.contract
        self._configure_tracking_relaxation()

    def _configure_tracking_relaxation(self):
        self.tracking_relaxation = None
        self._tracking_relaxation_reward = None
        if self.terrain is None:
            return  # Stage1 and isolated flat EMD evaluation have no tile map.
        resolved = TrackingRelaxation(self._env.cfg.pgmt_tracking_relaxation, BFS_BODIES, BFS_JOINTS,
                                      self.groups, device=self.device, dtype=self.previous_target.dtype)
        self.identity["tracking_relaxation"] = resolved.contract()
        if resolved.config["enabled"]:
            self.tracking_relaxation = resolved

    def _tracking_relaxation_sample(self, state):
        # Metadata lookup only: no extra height rays, birth-tile lookup or CPU sync.
        xy = state["body_pos"][:, self.anchor, :2]
        tile = self.terrain.tile_map.locate(xy)
        margins, scale, applicable = self.tracking_relaxation.margins(tile)
        snapshot = {key: value.clone() for key, value in tile.items()}
        snapshot.update(sample_step=torch.full_like(tile["tile_level"], self._env.common_step_counter),
                        anchor_xy_w=xy.clone(), level_scale=scale, applicable=applicable,
                        birth_level=self.terrain.terrain_levels.clone(),
                        birth_family_index=self.terrain.terrain_types.clone(),
                        reference_motion_ids=self.motion_ids.clone(), reference_frame_ids=self.frame_ids.clone())
        return margins, snapshot

    def tracking_relaxation_state(self):
        """Read a copy of this step's settled reward snapshot, including after reset.

        This does not advance the state or query terrain. The pre-reset reference
        and tile remain attached to the reward collected by PPO.
        """
        if self._tracking_relaxation_reward is None or self._reward_step != self._env.common_step_counter:
            raise RuntimeError("No settled tracking relaxation for this control step")
        return {key: value.clone() for key, value in self._tracking_relaxation_reward.items()}

    @property
    def command(self):
        ref = self.reference()
        return torch.cat((ref["joint_pos"], ref["joint_vel"]), -1)

    def reference(self, future=False, ids=None):
        ids = slice(None) if ids is None else ids
        motions, frames = self.motion_ids[ids], self.frame_ids[ids]
        if future:
            frames = frames[:, None] + future_offsets(device=self.device)
            motions = motions[:, None].expand_as(frames)
        ref = self.data_store.gather(motions, frames)
        return transform_reference(ref, self.reference_rotation[ids], self.reference_translation[ids])

    def robot_state(self):
        d = self.robot.data
        return dict(joint_pos=d.joint_pos[:, self.joints], joint_vel=d.joint_vel[:, self.joints],
                    body_pos=d.body_link_pos_w[:, self.bodies], body_quat=d.body_link_quat_w[:, self.bodies],
                    body_lin_vel=d.body_link_lin_vel_w[:, self.bodies], body_ang_vel=d.body_link_ang_vel_w[:, self.bodies],
                    root_quat=d.root_link_quat_w, root_lin_vel=d.root_link_lin_vel_w, root_ang_vel=d.root_link_ang_vel_w)

    def qtarget(self):
        return self.robot.data.joint_pos_target[:, self.joints]

    def _update_metrics(self):
        pass

    def compute(self, dt):
        # reset() invokes compute(dt=0); do not advance the reference. Auto
        # resets skip advancement too, preserving the selected start frame.
        if dt > 0:
            active = self.reset_step != self._env.common_step_counter
            self.frame_ids[active] = torch.minimum(self.frame_ids[active] + 1,
                                       self.data_store.motion_lengths[self.motion_ids[active]] - 1)
            self.sampler.step()
        self._obs = None

    def _update_command(self):
        pass  # compute above owns advancement explicitly.

    def _settle(self, ids):
        active = self.started[ids] & (self._env.common_step_counter > self.reset_step[ids])
        chosen = ids[active]
        if not len(chosen) or self.cfg.evaluation:
            return
        tm = self._env.termination_manager
        failure = tm.terminated[chosen]
        finished = self.frame_ids[chosen] >= self.data_store.motion_lengths[self.motion_ids[chosen]] - 1
        if self.curriculum is not None:
            self.curriculum.settle(chosen, step=self._env.common_step_counter,
                motion_ids=self.motion_ids[chosen], frame_ids=self.frame_ids[chosen], failure=failure,
                finished=finished, map_exit=tm.get_term("map_out_of_bounds")[chosen],
                timeout=tm.time_outs[chosen], other=torch.zeros_like(failure))
        else:
            valid = failure | finished | tm.time_outs[chosen]
            self.sampler.update(self.motion_ids[chosen][valid], self.start_frames[chosen][valid], failure[valid])

    def _resample_command(self, env_ids):
        ids = env_ids
        self._settle(ids)
        if self.cfg.stage == 2:
            from pmt_tasks.pgmt_emd_sampling import sample_emd_starts
            compatibility = (self.compatible if self.terrain is not None else
                             torch.ones(1, self.data_store.num_motions, dtype=torch.bool, device=self.device))
            types = self.terrain.terrain_types[ids] if self.terrain is not None else torch.zeros_like(ids)
            motions, frames = sample_emd_starts(self.sampler.weights, compatibility,
                             self.data_store.motion_lengths, types, fixed_start_frame=self.cfg.fixed_start_frame)
        else:
            motions, frames = self.sampler.sample(len(ids))
        if self.cfg.evaluation:
            motions = self.evaluation_motion_ids[ids] if hasattr(self, "evaluation_motion_ids") else ids % self.data_store.num_motions
            frames = torch.zeros_like(ids)
        self.motion_ids[ids], self.frame_ids[ids], self.start_frames[ids] = motions, frames, frames
        self.place(ids)
        self.started[ids] = True
        self.reset_step[ids] = self._env.common_step_counter
        self.pending_reset[ids] = True
        if self.terrain_rewards is not None:
            self.terrain_rewards.reset(ids)
        self.previous_target[ids] = self.qtarget()[ids]
        for timer in self.timers.values():
            timer.reset(ids, step=self._env.common_step_counter)
        self._obs = None
        if self.curriculum is not None:
            self.curriculum.begin(ids, self._env.common_step_counter, motions, frames,
                                  self.data_store.motion_lengths[motions])

    def place(self, ids):
        ref = self.data_store.gather(self.motion_ids[ids], self.frame_ids[ids])
        self.reference_rotation[ids] = ref["body_quat"].new_tensor([1., 0., 0., 0.])
        self.reference_translation[ids] = self._env.scene.env_origins[ids]
        if self.terrain is not None:
            pending = torch.arange(len(ids), device=self.device)
            for _ in range(128):
                selected = ids[pending]
                source = {k: v[pending] for k, v in ref.items()}
                xy, bounds, _ = self.terrain.sample(selected, self.cfg.placement_xy_mode)
                yaw = torch.empty(len(selected), device=self.device).uniform_(-torch.pi, torch.pi)
                rotation, translation = initial_transform(source["body_pos"][:, 0], source["body_quat"][:, 0], xy, yaw)
                translation, valid, _ = place_on_surface(source, rotation, translation, bounds, self.terrain, *self.cover)
                frames = self.frame_ids[selected, None] + future_offsets(device=self.device)
                future = self.data_store.gather(self.motion_ids[selected, None].expand_as(frames), frames)
                positions = transform_reference(future, rotation, translation)["body_pos"]
                valid &= self.terrain.contains(positions[..., :2]).flatten(1).all(-1)
                self.reference_rotation[selected[valid]] = rotation[valid]
                self.reference_translation[selected[valid]] = translation[valid]
                pending = pending[~valid]
                if not len(pending):
                    break
            if len(pending):
                raise RuntimeError(f"No valid terrain placement for environments {ids[pending].tolist()}")
        placed = self.reference(ids=ids)
        self.robot.write_joint_state_to_sim(placed["joint_pos"], placed["joint_vel"], joint_ids=self.joints, env_ids=ids)
        root = torch.cat([placed[k][:, 0] for k in ("body_pos", "body_quat", "body_lin_vel", "body_ang_vel")], -1)
        self.robot.write_root_state_to_sim(root, env_ids=ids)
        self.robot.set_joint_position_target(placed["joint_pos"], joint_ids=self.joints, env_ids=ids[:, None])

    def initialize_previous(self, state, ref):
        ids = self.pending_reset
        self.previous_velocity[ids] = state["body_lin_vel"][ids]
        self.previous_ref_velocity[ids] = ref["body_lin_vel"][ids]
        self.pending_reset.zero_()

    def observations(self):
        if self._obs is None:
            state, ref = self.robot_state(), self.reference()
            self.initialize_previous(state, ref)
            self._obs = build_observations(state, ref, self.reference(future=True), self.qtarget(),
                       self._env.scene.env_origins, anchor=self.anchor, config=self.cfg.tracking)
            if self.cfg.noise:
                p = self._obs["proprio"]
                scale = p.new_tensor([.05] * 6 + [.2] * 3 + [.01] * 29 + [.025] * 29 + [0.] * 29)
                self._obs["proprio"] = p + (2 * torch.rand_like(p) - 1) * scale
        return self._obs

    def terrain_observation(self):
        state = self.robot_state()
        torso = state["body_pos"][:, self.anchor]
        hits = rotate(yaw_quat(state["body_quat"][:, self.anchor])[:, None], self.rays[None]) + torso[:, None]
        hits[..., 2] = self.terrain.height(hits[..., :2]) if self.terrain is not None else 0.
        valid = (self.terrain.contains(hits[..., :2]) if self.terrain is not None else
                 torch.ones(hits.shape[:-1], dtype=torch.bool, device=self.device))
        return pack_terrain_observation(terrain_observation_from_hits(hits, torso[:, 2], valid,
                                              self._env.cfg.pgmt_terrain_observation))

    def rewards(self):
        if self._reward_step != self._env.common_step_counter:
            state, ref = self.robot_state(), self.reference()
            self.initialize_previous(state, ref)
            # The sensor uses literal robot/body names in BFS order.
            forces = self._env.scene["pgmt_contacts"].data.force
            relaxation_kwargs = {}
            if self.tracking_relaxation is not None:
                margins, snapshot = self._tracking_relaxation_sample(state)
                relaxation_kwargs = dict(tracking_margins=margins, tracking_debug=snapshot)
            self._rewards = reward_terms(state, ref, self.qtarget(), self.previous_target,
                  self.previous_velocity, self.previous_ref_velocity, forces,
                  self.robot.data.soft_joint_pos_limits[:, self.joints], self.groups,
                  anchor=self.anchor, dt=self._env.step_dt, config=self.cfg.tracking, **relaxation_kwargs)
            if relaxation_kwargs:
                self._tracking_relaxation_reward = snapshot
            if self.terrain_rewards is not None:
                self._rewards.update(self.terrain_rewards.compute(state))
            self.previous_target.copy_(self.qtarget())
            self.previous_velocity.copy_(state["body_lin_vel"])
            self.previous_ref_velocity.copy_(ref["body_lin_vel"])
            self._reward_step = self._env.common_step_counter
        return self._rewards


def observation(env, key):
    return env.command_manager.get_term("motion").observations()[key]


def terrain_observation(env):
    return env.command_manager.get_term("motion").terrain_observation()


def reward(env, name):
    return env.command_manager.get_term("motion").rewards()[name]


def capture_terrain_contacts(env):
    return env.command_manager.get_term("motion").terrain_rewards.capture_substep()


def time_out(env):
    return env.episode_length_buf >= env.max_episode_length


def motion_finished(env):
    c = env.command_manager.get_term("motion")
    return c.frame_ids >= c.data_store.motion_lengths[c.motion_ids] - 1


def map_out_of_bounds(env):
    c = env.command_manager.get_term("motion")
    return (~c.terrain.contains(c.robot_state()["body_pos"][..., :2]).all(-1)
            | ~c.terrain.contains(c.reference()["body_pos"][..., :2]).all(-1))


def tracking_failure(env, kind):
    c = env.command_manager.get_term("motion")
    state, ref = c.robot_state(), c.reference()
    if kind == "anchor_ori":
        gravity = state["root_quat"].new_tensor([0., 0., -1.]).expand(env.num_envs, -1)
        actual = inverse_rotate(state["body_quat"][:, c.anchor], gravity)
        target = inverse_rotate(ref["body_quat"][:, c.anchor], gravity)
        error, threshold = (actual[:, 2] - target[:, 2]).abs(), .8
    elif kind == "root_xy":
        error, threshold = (state["body_pos"][:, 0, :2] - ref["body_pos"][:, 0, :2]).norm(dim=-1), 1.
    elif kind == "anchor_pos":
        target = ref["body_pos"][:, c.anchor, 2]
        if c.terrain is not None:
            source = c.data_store.gather(c.motion_ids, c.frame_ids)
            target = c.terrain.height(state["body_pos"][:, c.anchor, :2]) + source["body_pos"][:, c.anchor, 2]
        error, threshold = (state["body_pos"][:, c.anchor, 2] - target).abs(), .4 if c.cfg.stage == 2 else .25
    else:
        names = ["left_wrist_yaw_link", "right_wrist_yaw_link"]
        if c.cfg.stage == 1:
            names += ["left_ankle_roll_link", "right_ankle_roll_link"]
        bodies = [BFS_BODIES.index(name) for name in names]
        actual, target = state["body_pos"][:, bodies, 2], ref["body_pos"][:, bodies, 2]
        if c.cfg.stage == 2:
            actual = actual - state["body_pos"][:, 0, 2:3]
            target = target - ref["body_pos"][:, 0, 2:3]
        error, threshold = (actual - target).abs().max(-1).values, .4 if c.cfg.stage == 2 else .25
    return c.timers[kind].update(~torch.isfinite(error) | (error > threshold), env.common_step_counter)
