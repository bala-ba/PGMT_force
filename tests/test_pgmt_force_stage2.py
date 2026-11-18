"""Force Stage2 import contracts and learning through both sensory branches."""
from copy import deepcopy

import pytest
import torch
from tensordict import TensorDict

from motion_tracking_rl.networks import PGMTForceActorCritic
from motion_tracking_rl.pgmt_force_feedback import force_feedback_contract
from motion_tracking_rl.utils.pgmt_force_checkpoint import InstantTrack99Transfer, migrate_instanttrack_force, migrate_force_stage2

GROUPS = dict(policy=["proprio"], policy_history=["proprio_history"],
              command_window=["command_window"], critic=["critic"])


@pytest.fixture
def setup(tmp_path):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(18)
    obs = TensorDict(dict(proprio=torch.randn(3, 96), proprio_history=torch.randn(3, 10, 108),
        command_window=torch.randn(3, 8, 61), critic=torch.randn(3, 251),
        terrain_map=torch.randn(3, 21, 21), terrain_valid=torch.ones(3, 21, 21)), [3])
    kwargs = dict(actor_obs_normalization=True, critic_obs_normalization=True)
    source = PGMTForceActorCritic(obs.exclude("terrain_map", "terrain_valid"), GROUPS, 29, **kwargs).eval()
    torch.nn.init.normal_(source.force_history_projection.weight, std=.1)
    target = PGMTForceActorCritic(obs, dict(GROUPS, terrain_map=["terrain_map"], terrain_valid=["terrain_valid"]),
        29, terrain_glimpse_cfg={}, terrain_mode="enabled", **kwargs).eval()
    saved = dict(format="instanttrack.pgmt_force_cop.v1", iteration=25500,
        args=dict(robot="g1", action_noise=.1),
        contract=dict(policy_arch="pgmt_stage1_force_v1", force_feedback=force_feedback_contract(),
            observation_shapes=dict(proprio=(96,), proprio_history=(10, 108), command_window=(8, 61), critic=(251,)),
            action="q0 + xi_joint_offset + scale*a", value_groups=["body_tracking", "root_tracking", "regularization"]),
        model=deepcopy(source.state_dict()), ema=deepcopy(source.state_dict()))
    path = tmp_path / "source.pt"
    torch.save(saved, path)
    yield obs, source, target, saved, path
    torch.set_num_threads(previous)


def test_instanttrack_preserves_actor_resets_critic_and_uses_collection_noise(setup):
    obs, source, target, saved, path = setup
    critic = deepcopy(target.critic.state_dict())
    result = migrate_instanttrack_force(target, path, critic_init="fresh")
    assert result["source_weights"] == "ema" and result["optimizer"] == "fresh"
    for key in result["copied"]:
        torch.testing.assert_close(target.state_dict()[key], saved["ema"][key], rtol=0, atol=0)
    for key, value in critic.items():
        torch.testing.assert_close(target.critic.state_dict()[key], value, rtol=0, atol=0)
    torch.testing.assert_close(target.std, torch.full((29,), .1))
    masked = obs.clone()
    masked["terrain_valid"].zero_()
    torch.testing.assert_close(target.act_inference(masked), source.act_inference(obs), rtol=1e-5, atol=1e-6)
    target.terrain_mode = "disabled"
    torch.testing.assert_close(target.act_inference(obs), source.act_inference(obs), rtol=0, atol=0)
    assert target.get_checkpoint_metadata()["policy_family"] == "pgmt_stage2_force_v1"


def test_force_and_terrain_receive_finite_nonzero_gradients(setup):
    obs, _, target, _, path = setup
    migrate_instanttrack_force(target, path)
    target.train()
    (target.act_inference(obs).square().mean() + target.evaluate(obs).square().mean()).backward()
    for module in (target.force_history_projection, target.terrain_glimpse):
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(float(g.abs().sum()) for g in grads) > 0


def test_instanttrack99_fields_match_reference_mdp():
    from scipy.spatial.transform import Rotation
    from pmt_tasks.pgmt_math import instanttrack_proprio99

    def quaternion(rot):
        return torch.tensor(rot.as_quat()[:, [3, 0, 1, 2]])
    actual = Rotation.from_rotvec([[.4, -.7, .2], [.1, .5, 1.3]])
    reference = Rotation.from_rotvec([[-.3, .1, .9], [1.4, .5, .2]])
    order = torch.randperm(29)
    q, qd, default, action = torch.randn(4, 2, 29, dtype=torch.float64)
    w = torch.randn(2, 3, dtype=torch.float64)
    state = dict(joint_pos=q, joint_vel=qd, root_quat=quaternion(actual), root_ang_vel=w)
    obs = instanttrack_proprio99(state, quaternion(reference), default, action, order)
    expected = torch.cat(((q-default)[:, order], qd[:, order], torch.tensor(actual.inv().apply(w.numpy())),
        torch.tensor(actual.inv().apply([0., 0., -1.])), action[:, order],
        torch.tensor((actual.inv()*reference).as_matrix()[:, :, :2]).reshape(2, 6)), -1)
    torch.testing.assert_close(obs, expected)


def test_privileged_fields_match_instanttrack_reference_with_two_heights():
    import numpy as np
    from scipy.spatial.transform import Rotation
    from pmt_tasks.pgmt_math import instanttrack_privileged

    def heading(rotation):
        axis = rotation.as_matrix()[:, 0]
        return Rotation.from_euler("z", np.arctan2(axis[1], axis[0]))

    def quat(rotation):
        return torch.from_numpy(rotation.as_quat()[[3, 0, 1, 2]])

    actual = Rotation.from_rotvec([.5, .7, 1.2])
    target = Rotation.from_rotvec([-.7, .8, -.9])
    pos, ref_pos = torch.randn(2, 1, 30, 3, dtype=torch.float64)
    v, w = torch.randn(2, 1, 3, dtype=torch.float64)
    q, qd, rq = torch.randn(3, 1, 29, dtype=torch.float64)
    pelvis, joints = 7, torch.randperm(29)
    bodies = torch.tensor([7, 6, 2, 14, 12, 13, 9, 1, 8, 26, 27, 17, 19, 11])
    heights = torch.tensor([[.86, 1.04]], dtype=torch.float64)
    xi = torch.randn(1, 33, dtype=torch.float64)
    ref = dict(body_quat=quat(target).expand(1, 30, -1), body_pos=ref_pos, joint_pos=rq)
    state = dict(root_quat=quat(actual)[None], root_lin_vel=v, root_ang_vel=w, body_pos=pos, joint_pos=q, joint_vel=qd)
    out = instanttrack_privileged(state, ref, pelvis, bodies, joints, heights, xi)
    H = heading(actual)
    aligned = heading(actual * target.inv()).apply((ref_pos[0, bodies] - ref_pos[0, pelvis]).numpy())
    anchor = np.r_[pos[0, pelvis, :2], ref_pos[0, pelvis, 2]]
    err = H.inv().apply(aligned + anchor - pos[0, bodies].numpy())
    xy = H.inv().apply((ref_pos[0, pelvis] - pos[0, pelvis]).numpy())[:2]
    expected = np.concatenate([heights[0, :1], (H.inv()*actual).as_matrix()[:, :2].ravel(),
        actual.inv().apply(v[0]), actual.inv().apply(w[0]), q[0, joints], qd[0, joints], (rq-q)[0, joints],
        err.ravel(), xy, heights[0], (actual.inv()*target).as_matrix()[:, :2].ravel(), xi[0]])
    assert out.shape == (1, 185)
    torch.testing.assert_close(out[0], torch.from_numpy(expected))


@pytest.mark.parametrize("critic_init", ["source", "fresh"])
def test_privileged_map_and_intent_in_critic(setup, critic_init):
    obs, source, _, saved, path = setup
    source.train()
    source.update_normalization(obs)
    saved["ema"] = source.state_dict()
    torch.save(saved, path)
    transfer = InstantTrack99Transfer(torch.randperm(29), torch.randn(29), torch.ones(29))
    inputs = obs.clone()
    inputs["proprio"], inputs["proprio_history"] = torch.randn(3, 111), torch.randn(3, 10, 99)
    inputs["force_history"], inputs["critic"] = torch.randn(3, 10, 12), torch.randn(3, 197)
    inputs["critic_terrain_map"] = torch.randn(3, 21, 21)
    groups = dict(GROUPS, **{k: [k] for k in ("force_history", "terrain_valid", "terrain_map", "critic_terrain_map")})
    policy = PGMTForceActorCritic(inputs, groups, 29, proprio_layout="instanttrack99", current_force_feedback=True,
        terrain_privileged_critic=True, terrain_glimpse_cfg={}, actor_obs_normalization=True, critic_obs_normalization=True,
        value_groups=["upper", "lower", "terrain", "auxiliary"])
    report = migrate_instanttrack_force(policy, path, critic_init=critic_init, proprio_transfer=transfer)
    assert (policy.actor[0].in_features, policy.critic[0].in_features) == (239, 389)
    assert policy.get_checkpoint_metadata()["signature"]["terrain_privileged_critic"]["terrain_mlp"] == [441, 128, 64]
    if critic_init == "source":
        # Native SDK permutation and velocity units reconstructed independently of the transfer helper.
        old = transfer.source_observations(inputs, source.actor_obs_normalizer._mean)
        raw = source.critic_obs_normalizer.mean.expand(3, -1).clone()
        raw[:, 6:12] = inputs["critic"][:, 7:13]
        native = (transfer.critic_old[6:35] - 12).argsort()
        raw[:, 12:41] = inputs["critic"][:, 13:42][:, native]
        raw[:, 41:70] = .05 * inputs["critic"][:, 42:71][:, native]
        old["critic"] = raw
        torch.testing.assert_close(policy.evaluate(inputs)[:, [0, 1, 3]], source.evaluate(old))
        assert report["copied_critic_scalar_columns"] == 64
        assert (policy.critic_obs_normalizer.count > 0).sum() == 64
    policy.update_normalization(inputs)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    for _ in range(2):
        optimizer.zero_grad()
        policy.evaluate(inputs).square().sum().backward()
        optimizer.step()
    for module in (policy.critic_terrain_encoder, policy.intent_attention):
        assert sum(p.grad.abs().sum() for p in module.parameters() if p.grad is not None) > 0
    changed = inputs.clone()
    changed["critic_terrain_map"].add_(1.)
    assert (policy.evaluate(changed) - policy.evaluate(inputs)).abs().max() > 1e-6
    torch.testing.assert_close(policy.act_inference(inputs.exclude("critic", "critic_terrain_map")),
                               policy.act_inference(changed), atol=0, rtol=0)


@pytest.mark.parametrize("current_force", [False, True])
def test_instanttrack99_transfer_and_split_history(setup, current_force):
    obs, source, _, saved, path = setup
    order = torch.randperm(29)
    q0 = torch.randn(29)
    scale = torch.rand(29) + .1
    # Use nontrivial source statistics to exercise unit/offset conversions.
    source.train()
    source.update_normalization(obs)
    saved["ema"] = source.state_dict()
    torch.save(saved, path)
    transfer = InstantTrack99Transfer(order, q0, scale)
    sdk = obs.clone()
    sdk["force_history"] = obs["proprio_history"][..., 96:].clone()
    sdk["proprio_history"] = torch.randn(3, 10, 99)
    sdk["proprio"] = sdk["proprio_history"][:, -1].clone()
    if current_force:
        sdk["proprio"] = torch.cat((sdk["proprio"], sdk["force_history"][:, -1]), -1)
    groups = dict(GROUPS, force_history=["force_history"], terrain_map=["terrain_map"], terrain_valid=["terrain_valid"])
    policy = PGMTForceActorCritic(sdk, groups, 29, proprio_layout="instanttrack99", current_force_feedback=current_force,
        terrain_glimpse_cfg={}, actor_obs_normalization=True, critic_obs_normalization=True,
        value_groups=["upper", "lower", "terrain", "auxiliary"])
    report = migrate_instanttrack_force(policy, path, proprio_transfer=transfer)
    assert report["proprio_layout"] == "instanttrack99"
    counts = policy.actor_obs_normalizer.count
    assert (counts[:, transfer.new] == source.actor_obs_normalizer.count).all()
    assert (counts == 0).sum() == 9
    assert (policy.actor[0].in_features, policy.critic[0].in_features) == (239 if current_force else 227, 379)
    if current_force:
        for layer in (policy.proprio_embed, policy.actor[0]):
            assert layer.weight[:, 99:111].count_nonzero() == 0
        assert "current_foot_feedback" in report["zero_input_columns"]
    # Rebuild the matched source history independently, in native joint order.
    x = sdk["proprio_history"]
    native = order.argsort()
    torso = source.actor_obs_normalizer.mean[:6].expand(3, 10, -1)
    legacy = torch.cat((torso, x[..., 58:61], x[..., :29][..., native] + q0,
        .05 * x[..., 29:58][..., native], q0 + scale * x[..., 64:93][..., native]), -1)
    obs["proprio"] = legacy[:, -1]
    obs["proprio_history"] = torch.cat((legacy, sdk["force_history"]), -1)
    torch.testing.assert_close(policy.act_inference(sdk), source.act_inference(obs))
    torch.testing.assert_close(policy.evaluate(sdk)[:, [0, 1, 3]], source.evaluate(obs))
    initial_mean = policy.actor_obs_normalizer.mean
    policy.update_normalization(sdk)
    torch.testing.assert_close(policy.actor_obs_normalizer.mean[61:64], sdk["proprio"][:, 61:64].mean(0))
    expected_mean = (initial_mean + sdk["proprio"][:, :99].mean(0)) / 2  # both batches have 3 samples
    torch.testing.assert_close(policy.actor_obs_normalizer.mean[transfer.new], expected_mean[transfer.new])
    policy.terrain_mode = "enabled"
    (policy.act_inference(sdk).square().mean() + policy.evaluate(sdk).square().mean()).backward()
    for layer in (policy.proprio_embed, policy.actor[0]):
        assert layer.weight.grad[:, 61:64].abs().sum() > 0
        assert layer.weight.grad[:, 93:99].abs().sum() > 0
        if current_force:
            assert layer.weight.grad[:, 99:111].abs().sum() > 0
    assert policy.force_history_projection.weight.grad.abs().sum() > 0
    meta = policy.get_checkpoint_metadata()
    restored = PGMTForceActorCritic(sdk, meta["obs_groups"], 29, **meta["policy_kwargs"])
    restored.load_state_dict(policy.state_dict(), strict=True)
    assert restored.get_checkpoint_metadata() == meta
    torch.testing.assert_close(restored.act_inference(sdk), policy.act_inference(sdk))


def test_current_foot_feedback_reaches_actor_directly(setup):
    obs, _, _, _, _ = setup
    groups = dict(GROUPS, force_history=["force_history"])
    obs["proprio_history"] = torch.zeros(3, 10, 99)
    obs["force_history"] = torch.zeros(3, 10, 12)
    feet = torch.tensor([80., 0., 200., .03, -.02, 1., 12., 8., 0., float("nan"), float("nan"), 0.])
    obs["proprio"] = torch.cat((obs["proprio_history"][:, -1], feet.expand(3, -1)), -1)
    policy = PGMTForceActorCritic(obs, groups, 29, proprio_layout="instanttrack99", current_force_feedback=True)
    body, _ = policy._encode(obs)
    cfg = force_feedback_contract()
    scaled = torch.tensor([80/cfg["force_scale"], 0., 200/cfg["force_scale"], .03/cfg["cop_scale"][0],
                           -.02/cfg["cop_scale"][1], 1., 12/cfg["force_scale"], 8/cfg["force_scale"], 0., 0., 0., 0.])
    torch.testing.assert_close(body[:, 99:], scaled.expand(3, -1))
    # Fix the query and history while perturbing the direct MLP input.
    with torch.no_grad():
        policy.proprio_embed.weight[:, 99:].zero_()
    before = policy.act_inference(obs)
    before_intent = policy._encode(obs)[1]
    obs["proprio"][:, 99] += 80.
    torch.testing.assert_close(policy._encode(obs)[1], before_intent)
    assert not torch.equal(policy.act_inference(obs), before)


def test_explicit_live_weights_and_force_stage2_reload(setup):
    obs, source, target, saved, path = setup
    saved["ema"]["actor.6.bias"] += 1.
    torch.save(saved, path)
    migrate_instanttrack_force(target, path, weights="model")
    target.terrain_mode = "disabled"
    torch.testing.assert_close(target.act_inference(obs), source.act_inference(obs), rtol=0, atol=0)
    target.terrain_mode = "enabled"
    metadata = target.get_checkpoint_metadata()
    restored = PGMTForceActorCritic(obs, metadata["obs_groups"], 29, **metadata["policy_kwargs"]).eval()
    restored.load_state_dict(target.state_dict(), strict=True)
    assert metadata == restored.get_checkpoint_metadata()
    torch.testing.assert_close(restored.act_inference(obs), target.act_inference(obs), rtol=0, atol=0)


@pytest.mark.parametrize("corruption", ["force_contract", "history", "nan", "extra_tensor", "noise"])
def test_incompatible_import_is_atomic(setup, corruption):
    _, _, target, saved, path = setup
    if corruption == "force_contract":
        saved["contract"]["force_feedback"]["frame"] = "world"
    elif corruption == "history":
        saved["contract"]["observation_shapes"]["proprio_history"] = [10, 120]
    elif corruption == "nan":
        saved["ema"]["actor.0.weight"][0, 0] = float("nan")
    elif corruption == "extra_tensor":
        saved["ema"]["unknown"] = torch.ones(1)
    else:
        saved["args"]["action_noise"] = 0
    torch.save(saved, path)
    before = deepcopy(target.state_dict())
    with pytest.raises(ValueError):
        migrate_instanttrack_force(target, path)
    for key, value in target.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_pmt_force_migration_keeps_matching_critic(setup):
    obs, source, target, _, path = setup
    torch.save(dict(policy_metadata=source.get_checkpoint_metadata(), model_state_dict=source.state_dict(), iter=3), path)
    migrate_force_stage2(target, path)
    target.terrain_mode = "disabled"
    torch.testing.assert_close(target.evaluate(obs), source.evaluate(obs), rtol=0, atol=0)


def test_instanttrack_to_four_heads_preserves_actor_and_trains_all_branches(setup):
    obs, source, _, saved, path = setup
    groups = dict(GROUPS, terrain_map=["terrain_map"], terrain_valid=["terrain_valid"])
    target = PGMTForceActorCritic(obs, groups, 29, terrain_glimpse_cfg={}, terrain_mode="enabled",
        actor_obs_normalization=True, critic_obs_normalization=True,
        value_groups=["upper", "lower", "terrain", "auxiliary"]).eval()
    critic = deepcopy(target.critic.state_dict())
    report = migrate_instanttrack_force(target, path, critic_init="fresh")
    assert report["target_value_groups"] == ["upper", "lower", "terrain", "auxiliary"]
    for key, value in critic.items():
        torch.testing.assert_close(target.critic.state_dict()[key], value, rtol=0, atol=0)
    target.terrain_mode = "disabled"
    torch.testing.assert_close(target.act_inference(obs), source.act_inference(obs), rtol=0, atol=0)
    target.terrain_mode = "enabled"
    values = target.evaluate(obs)
    assert values.shape == (3, 4)
    (values.square().mean() + target.act_inference(obs).square().mean()).backward()
    for module in (target.force_history_projection, target.terrain_glimpse):
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(float(g.abs().sum()) for g in grads) > 0
    final = [m for m in target.critic.modules() if isinstance(m, torch.nn.Linear)][-1]
    assert (final.weight.grad.abs().sum(-1) > 0).all()
    meta = target.get_checkpoint_metadata()
    clone = PGMTForceActorCritic(obs, meta['obs_groups'], 29, **meta['policy_kwargs']).eval()
    clone.load_state_dict(target.state_dict(), strict=True)
    assert clone.get_checkpoint_metadata() == meta
    torch.testing.assert_close(clone.act_inference(obs), target.act_inference(obs), rtol=0, atol=0)
    torch.testing.assert_close(clone.evaluate(obs), values, rtol=0, atol=0)


def test_four_head_import_rejects_wrong_source_critic(setup):
    obs, _, _, saved, path = setup
    target = PGMTForceActorCritic(obs, dict(GROUPS, terrain_map=["terrain_map"], terrain_valid=["terrain_valid"]),
        29, terrain_glimpse_cfg={}, terrain_mode="enabled", actor_obs_normalization=True,
        critic_obs_normalization=True, value_groups=["upper", "lower", "terrain", "auxiliary"])
    saved['ema']['critic.6.weight'] = torch.zeros(4, 128)
    torch.save(saved, path)
    before = deepcopy(target.state_dict())
    with pytest.raises(ValueError, match='critic.6.weight'):
        migrate_instanttrack_force(target, path)
    for key, value in before.items():
        torch.testing.assert_close(target.state_dict()[key], value, rtol=0, atol=0)


@pytest.mark.parametrize('weights', ['ema', 'model'])
def test_source_critic_copies_normalizer_trunk_and_three_rows_only(setup, weights):
    obs, source, _, saved, path = setup
    if weights == 'model':
        saved['contract']['value_groups'] = tuple(saved['contract']['value_groups'])
    saved[weights]['critic_obs_normalizer._mean'].fill_(.75)
    saved[weights]['critic_obs_normalizer._var'].fill_(4.)
    saved[weights]['critic_obs_normalizer._std'].fill_(2.)
    saved[weights]['critic_obs_normalizer.count'].fill_(1234)
    saved[weights]['critic.6.bias'].copy_(torch.tensor([3., 7., 11.]))
    torch.save(saved, path)
    source.load_state_dict(saved[weights])
    target = PGMTForceActorCritic(obs, dict(GROUPS, terrain_map=['terrain_map'], terrain_valid=['terrain_valid']),
        29, terrain_glimpse_cfg={}, terrain_mode='enabled', actor_obs_normalization=True,
        critic_obs_normalization=True, value_groups=['upper', 'lower', 'terrain', 'auxiliary']).eval()
    terrain_row = {k: v[2].clone() for k, v in target.state_dict().items() if k in ('critic.6.weight', 'critic.6.bias')}
    report = migrate_instanttrack_force(target, path, weights=weights)
    state = target.state_dict()
    for key in report['copied']:
        torch.testing.assert_close(state[key], saved[weights][key], rtol=0, atol=0)
    for key in report['partially_copied']:
        torch.testing.assert_close(state[key][[0, 1, 3]], saved[weights][key], rtol=0, atol=0)
        torch.testing.assert_close(state[key][2], terrain_row[key], rtol=0, atol=0)
    assert report['initialized_rows'] == {'critic.6.bias': [2], 'critic.6.weight': [2]}
    assert [m['target_row'] for m in report['critic_head_mapping']] == [0, 1, 3]
    assert 'critic.0.weight' in report['copied'] and 'critic_obs_normalizer.count' in report['copied']
    assert state['critic.6.weight'][2].abs().sum() > 0
    target.terrain_mode = 'disabled'
    torch.testing.assert_close(target.act_inference(obs), source.act_inference(obs), rtol=0, atol=0)
    torch.testing.assert_close(target.evaluate(obs)[:, [0, 1, 3]], source.evaluate(obs), rtol=2e-6, atol=2e-5)


def test_source_critic_unknown_head_order_rejected_atomically(setup):
    _, _, target, saved, path = setup
    saved['contract']['value_groups'].reverse()
    torch.save(saved, path)
    before = deepcopy(target.state_dict())
    with pytest.raises(ValueError, match='head order'):
        migrate_instanttrack_force(target, path, critic_init='source')
    for key, value in before.items():
        torch.testing.assert_close(target.state_dict()[key], value, rtol=0, atol=0)


def test_pmt_force_three_to_four_retains_random_terrain_row(setup):
    obs, source, _, _, path = setup
    torch.save(dict(policy_metadata=source.get_checkpoint_metadata(), model_state_dict=source.state_dict(), iter=3), path)
    target = PGMTForceActorCritic(obs, dict(GROUPS, terrain_map=['terrain_map'], terrain_valid=['terrain_valid']),
        29, terrain_glimpse_cfg={}, terrain_mode='enabled', actor_obs_normalization=True,
        critic_obs_normalization=True, value_groups=['upper', 'lower', 'terrain', 'auxiliary']).eval()
    before = deepcopy(target.state_dict())
    report = migrate_force_stage2(target, path)
    for key in report['copied']:
        torch.testing.assert_close(target.state_dict()[key], source.state_dict()[key], rtol=0, atol=0)
    for key in report['partially_copied']:
        torch.testing.assert_close(target.state_dict()[key][[0, 1, 3]], source.state_dict()[key], rtol=0, atol=0)
        torch.testing.assert_close(target.state_dict()[key][2], before[key][2], rtol=0, atol=0)
    assert target.state_dict()['critic.6.weight'][2].count_nonzero() > 0


def test_plain_pmt_stage1_random_terrain_migration(setup):
    from motion_tracking_rl.networks import PGMTActorCritic
    from motion_tracking_rl.utils.pgmt_checkpoint import migrate_three_head
    obs, _, _, _, path = setup
    obs = obs.clone()
    obs['proprio_history'] = obs['proprio_history'][..., :96]
    kwargs = dict(actor_obs_normalization=True, critic_obs_normalization=True)
    source = PGMTActorCritic(obs, GROUPS, 29, **kwargs).eval()
    torch.save(dict(policy_metadata=source.get_checkpoint_metadata(), model_state_dict=source.state_dict(), iter=3), path)
    target = PGMTActorCritic(obs, dict(GROUPS, terrain_map=['terrain_map'], terrain_valid=['terrain_valid']),
        29, terrain_glimpse_cfg={}, terrain_mode='enabled',
        value_groups=['upper', 'lower', 'terrain', 'auxiliary'], **kwargs).eval()
    before = deepcopy(target.state_dict())
    report = migrate_three_head(target, path, terrain_initialization='random')
    for key in report['remapped']:
        torch.testing.assert_close(target.state_dict()[key][[0, 1, 3]], source.state_dict()[key], rtol=0, atol=0)
        torch.testing.assert_close(target.state_dict()[key][2], before[key][2], rtol=0, atol=0)
    assert report['terrain_row']['initialization'] == 'random'
