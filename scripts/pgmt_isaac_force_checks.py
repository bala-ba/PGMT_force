"""Opt-in independent float64 CPU audit of live PhysX force observations."""
import numpy as np
import torch


def audit_force_observation(env, obs):
    feedback = env._pgmt_force_feedback
    cfg = feedback.config
    group = 'force_history' if 'force_history' in obs else 'proprio_history'
    history = obs[group]
    forces = history if group == 'force_history' else history[..., 96:]
    actual = forces[:, -1].reshape(env.num_envs, 2, 6)
    torch.testing.assert_close(actual.flatten(1), feedback.compute(), rtol=0, atol=0)
    critic_audit = {}
    if "critic_terrain_map" in obs:
        state = env._pgmt_state
        critic = obs["critic"]
        assert critic.shape == (env.num_envs, 197) and torch.isfinite(critic).all()
        torch.testing.assert_close(critic[:, 185:], actual.flatten(1), rtol=0, atol=0)
        pelvis = state.pelvis
        for col, pos in ((0, state.robot.data.body_pos_w[:, pelvis]), (145, state.reference()["body_pos"][:, pelvis])):
            sample = env.scene.terrain.query(pos[:, None, :2])
            expected = torch.where(sample["valid"][:, 0], pos[:, 2] - sample["height"][:, 0], 0.)
            torch.testing.assert_close(critic[:, col], expected)
        torch.testing.assert_close(critic[:, 144], critic[:, 0], rtol=0, atol=0)
        scan = state.terrain_scan()
        torch.testing.assert_close(obs["critic_terrain_map"], scan["height"], rtol=0, atol=0)
        # Check source-unit DR features against live PhysX values, including the per-env bias.
        mu = torch.cat([s.body_physx_view.get_material_properties()[..., 1] for s in feedback.sensors], 1).mean(1)
        torch.testing.assert_close(critic[:, 152], (mu.to(critic) - .75) / .45)
        nominal = getattr(state.robot.data, "default_joint_pos_nominal", state.nominal_critic_joint_pos)
        bias = state.robot.data.default_joint_pos - nominal
        torch.testing.assert_close(critic[:, 153:182], bias[:, state.sdk_joint_order] / .01)
        com = state.robot.root_physx_view.get_coms()[:, state.anchor, :3] - state.nominal_torso_com
        torch.testing.assert_close(critic[:, 182:185], com.to(critic) / .05)
        critic_audit = dict(critic_scalar_features=197, critic_valid_terrain_cells=int(scan["valid"].sum()),
                            critic_com_max_m=float(com.abs().max()), critic_friction_mean=float(mu.mean()))
    reread = env.observation_manager.compute_group(group, update_history=False)
    torch.testing.assert_close(reread, history, rtol=0, atol=0)
    if group == 'force_history':
        body = obs['proprio_history'][:, -1]
        current = obs['proprio']
        torch.testing.assert_close(current[:, :body.shape[-1]], body, rtol=0, atol=0)
        if getattr(env.cfg.observations.proprio, 'feet', None) is not None:
            torch.testing.assert_close(current[:, body.shape[-1]:], forces[:, -1], rtol=0, atol=0)
    reset = (feedback.reset_step == env.common_step_counter).cpu().numpy()
    assert forces[torch.as_tensor(reset, device=env.device)].eq(0).all()
    expected = np.zeros((env.num_envs, 2, 6), dtype=np.float64)
    max_normal_error = 0.
    normal_points = friction_points = 0
    for side, (sensor, body) in enumerate(zip(feedback.sensors, feedback.body_ids)):
        view = sensor.contact_physx_view
        # Copy before the next call: the native view shares count/start buffers.
        magnitude, point, normal, _, count, start = [
            v.detach().cpu().numpy().copy() for v in view.get_contact_data(dt=env.physics_dt)]
        shear, shear_point, shear_count, shear_start = [
            v.detach().cpu().numpy().copy() for v in view.get_friction_data(dt=env.physics_dt)]
        filtered = sensor.data.force_matrix_w.sum((1, 2)).cpu().numpy()
        positions = feedback.robot.data.body_pos_w[:, body].cpu().numpy().astype(np.float64)
        rotations = feedback.robot.data.body_quat_w[:, body].cpu().numpy().astype(np.float64)
        for i in range(env.num_envs):
            if reset[i]:
                continue
            force, moment, normal_force, load = np.zeros(3), np.zeros(3), np.zeros(3), 0.
            for j in range(count.shape[1]):
                for k in range(int(start[i, j]), int(start[i, j] + count[i, j])):
                    f = float(magnitude[k, 0]) * normal[k].astype(np.float64)
                    force += f
                    normal_force += f
                    moment += np.cross(point[k].astype(np.float64) - positions[i], f)
                    load += float(magnitude[k, 0])
                    normal_points += 1
                for k in range(int(shear_start[i, j]), int(shear_start[i, j] + shear_count[i, j])):
                    f = shear[k].astype(np.float64)
                    force += f
                    moment += np.cross(shear_point[k].astype(np.float64) - positions[i], f)
                    friction_points += 1
            np.testing.assert_allclose(normal_force, filtered[i], rtol=5e-5, atol=.005)
            max_normal_error = max(max_normal_error, float(np.max(np.abs(normal_force-filtered[i]))))
            w, x, y, z = rotations[i]
            rotation = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                                 [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                                 [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
            force, moment = rotation.T @ force, rotation.T @ moment
            expected[i, side, :3] = force
            if load >= cfg['min_normal_force'] and force[2] >= cfg['min_local_fz']:
                expected[i, side, 3:] = [(cfg['plane_z']*force[0]-moment[1])/force[2],
                                         (moment[0]+cfg['plane_z']*force[1])/force[2], 1.]
    actual = actual.cpu().numpy()
    np.testing.assert_allclose(actual[..., :3], expected[..., :3], rtol=5e-5, atol=.005)
    np.testing.assert_allclose(actual[..., 3:], expected[..., 3:], rtol=5e-5, atol=2e-5)
    return dict(step=env.common_step_counter, reset_rows=int(reset.sum()),
        valid_cop=int(expected[..., 5].sum()), normal_points=normal_points, friction_points=friction_points,
        max_force_error_N=float(np.max(np.abs(actual[..., :3]-expected[..., :3]))),
        max_cop_error_m=float(np.max(np.abs(actual[..., 3:5]-expected[..., 3:5]))),
        max_normal_sensor_error_N=max_normal_error, repeated_read_and_reset_history=True, **critic_audit)
