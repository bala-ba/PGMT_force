"""Read-only, pre-reset native reward and physics-substep diagnostics."""
import numpy as np
import torch
from pmt_tasks.pgmt_runtime import get_pgmt_state


class RewardProbe:
    def __init__(self, env, recording):
        self.env, self.recording = env, recording
        self.state = get_pgmt_state(env)
        self.clear()
        native_update = env.scene.update
        native_reward = env.reward_manager.compute

        def update(dt):
            result = native_update(dt)
            if recording["enabled"]:
                self.physics.append(torch.stack((self.state.robot_foot_contact_forces(),
                                                 self.state.robot_foot_terrain_forces()), dim=1).clone())
            return result

        def compute(dt):
            result = native_reward(dt)
            if recording["enabled"]:
                s = self.state
                row = {"reward_rates": torch.stack([s.rewards()[n] for n in env.reward_manager.active_terms], -1),
                       "weighted_rates": env.reward_manager._step_reward.clone()}
                row["all_body_forces"] = env.scene["contact_forces"].data.net_forces_w[:, s.contact_order].clone()
                for key, value in s.contact_state(for_reward=True).items():
                    row["contact_" + key] = value
                t1 = s.touchdown_state(for_reward=True)
                keys = (("rate", "rms", "coverage", "margin", "per_foot_cost", "plane_source", "contact_count",
                         "contact_overflow", "support", "support_valid", "physics_samples") if s.geometry_rewards else
                        ("rate", "quality", "variance", "event", "eligible", "pattern_matched", "reference_support_count"))
                for key in keys:
                    row["t1_" + key] = t1[key]
                t2 = s.reference_contact_match_state(for_reward=True)
                for key in ('rate', 'actual_contact', 'actual_valid', 'matched'):
                    row['t2_' + key] = t2[key]
                if s.support_switching:
                    for key, value in s._contact_switching_reward.items():
                        row["t5_" + key] = value.clone()
                row["foot_velocity"] = s.slip_state(for_reward=True)["velocity_w"]
                physical, reference = s.robot_state(), s.reference()
                for key in ("body_pos", "body_lin_vel", "body_quat", "joint_pos", "joint_vel"):
                    row["actual_" + key], row["reference_" + key] = physical[key].clone(), reference[key].clone()
                row["physics_forces"] = torch.stack(self.physics, 1)
                if len(self.physics) != env.cfg.decimation:
                    raise RuntimeError("Unexpected number of force substeps")
                self.physics.clear()
                self.rows.append({k: v.detach().clone() for k,v in row.items()})
            return result

        env.scene.update, env.reward_manager.compute = update, compute

    def clear(self):
        self.rows, self.physics = [], []

    def save(self, path, counts, trial_ids):
        arrays = {k: torch.stack([row[k] for row in self.rows]).cpu().numpy()[:, :len(counts)] for k in self.rows[0]}
        arrays.update(control_steps=np.asarray(counts), trial_ids=np.asarray(trial_ids),
                      term_names=np.asarray(self.env.reward_manager.active_terms),
                      control_dt=self.env.step_dt, physics_dt=self.env.physics_dt,
                      body_names=np.asarray(self.state.robot.body_names),
                      foot_body_indices=self.state.foot_body_indices.cpu().numpy(),
                      anchor_index=self.state.anchor)
        arrays['terrain_reward_mode'] = np.asarray(self.state.terrain_reward_mode)
        arrays['contact_switching_mode'] = np.asarray(self.state.contact_switching_mode)
        arrays['touchdown_settlement'] = np.asarray(self.state.touchdown_config['settlement'])
        threshold = self.state.reference_contact_threshold_n
        arrays['reference_contact_threshold_n'] = np.asarray(
            self.state.contact_tracker.config['contact_threshold_n'] if threshold is None else threshold)
        np.savez_compressed(path, **arrays)
