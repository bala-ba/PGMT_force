"""Opt-in CPU replay of T5 from independently captured Isaac substep forces."""
import numpy as np


class TerrainSwitchingAudit:
    def __init__(self, env):
        from pmt_tasks.pgmt_contact_state import TERRAIN_FOOT_SENSORS

        self.env, self.state = env, env._pgmt_state
        tracker = self.state.support_tracker
        self.config = tracker.config
        self.previous = tracker.state.cpu().numpy().copy()
        self.valid = tracker.valid.cpu().numpy().copy()
        self.last_switch = tracker.last_switch.cpu().numpy().copy()
        self.used = [[set(v[v >= 0].tolist()) for v in row] for row in tracker.used.cpu().numpy()]
        self.episode = self.state.contact_tracker.snapshot()['episode_id'].cpu().numpy().copy()
        self.substeps, self.rows = [], []
        self.physics_step = env._sim_step_counter
        original_update, original_compute = env.scene.update, env.reward_manager.compute

        def update(*args, **kwargs):
            result = original_update(*args, **kwargs)
            if env._sim_step_counter != self.physics_step:
                self.substeps.append(np.stack([
                    env.scene.sensors[name].data.force_matrix_w.sum((1, 2))[:, 2].cpu().numpy().copy()
                    for name in TERRAIN_FOOT_SENSORS], axis=-1))
                self.physics_step = env._sim_step_counter
            return result

        def compute(*args, **kwargs):
            result = original_compute(*args, **kwargs)
            self.check()
            return result

        env.scene.update, env.reward_manager.compute = update, compute

    def check(self):
        env, state, c = self.env, self.state, self.config
        assert len(self.substeps) == env.cfg.decimation
        mean = np.mean(np.asarray(self.substeps, dtype=np.float64), axis=0)
        self.substeps.clear()
        actual = {k: v.cpu().numpy().copy() for k, v in state._contact_switching_reward.items()}
        np.testing.assert_allclose(actual['mean_force_z'], mean, atol=.001, rtol=5e-6)
        np.testing.assert_array_equal(actual['physics_samples'], env.cfg.decimation)
        episode = state.contact_tracker.snapshot()['episode_id'].cpu().numpy()
        reset = episode != self.episode
        self.episode = episode.copy()
        self.valid[reset] = False
        self.previous[reset] = False
        self.last_switch[reset] = -1
        for i in np.flatnonzero(reset):
            self.used[i] = [set(), set()]
        valid = np.isfinite(mean)
        support = self.previous.copy()
        support[mean >= c['force_on_n']] = True
        support[mean <= c['force_off_n']] = False
        event = valid & self.valid & (support != self.previous)
        history = event & (self.last_switch >= 0)
        duration = (env.common_step_counter - self.last_switch) * env.step_dt
        soft = np.where(history, np.maximum(1-duration/c['minimum_state_s'], 0.), 0.)
        soft[duration >= c['minimum_state_s']-1e-7] = 0
        exempt = np.zeros_like(event)
        store, command = state.contact_label_store, state.command
        window = state.support_tracker.window
        # Reconstruct candidate edges independently from reference contact labels.
        candidates = []
        for offset in range(-window, window+1):
            now = store.gather(command.motion_ids, command.frame_ids + offset)
            before = store.gather(command.motion_ids, command.frame_ids + offset - 1)
            edge = (now['contact_valid'] & before['contact_valid'] & (now['contacts'] != before['contacts']))
            identity = store.offsets[command.motion_ids] + command.frame_ids + offset
            candidates.append((offset, identity.cpu().numpy(), now['contacts'].cpu().numpy(), edge.cpu().numpy()))
        for i, foot in np.argwhere(event):
            for offset, identity, labels, edge in sorted(candidates, key=lambda x: abs(x[0])):
                if edge[i, foot] and bool(labels[i, foot]) == support[i, foot] and identity[i] not in self.used[i][foot]:
                    exempt[i, foot] = True
                    self.used[i][foot].add(identity[i])
                    break
        expected = np.sum(soft * ~exempt, axis=-1) / env.step_dt
        for name, value in [('support', support), ('event', event), ('history_valid', history),
                            ('reference_exempt', exempt)]:
            np.testing.assert_array_equal(actual[name], value)
        np.testing.assert_allclose(actual['soft_cost'], soft, atol=1e-6)
        np.testing.assert_allclose(actual['rate'], expected, atol=1e-5)
        index = env.reward_manager.active_terms.index('terrain_contact_switching')
        weighted = env.reward_manager._step_reward[:, index].cpu().numpy()
        weight = env.reward_manager.get_term_cfg('terrain_contact_switching').weight
        np.testing.assert_allclose(weighted, expected * weight, atol=1e-5)
        self.rows.append(dict(step=env.common_step_counter, physics_samples=env.cfg.decimation,
            max_mean_force_error_N=float(np.max(np.abs(actual['mean_force_z']-mean))),
            events=int(event.sum()), penalized_events=int((soft * ~exempt > 0).sum()),
            reference_exemptions=int(exempt.sum()), reset_rows=int(reset.sum()),
            integrated_penalty_sum=float(np.sum(expected) * weight * env.step_dt)))
        self.last_switch[event] = env.common_step_counter
        self.last_switch[~(valid & self.valid)] = -1
        self.previous, self.valid = support, valid
