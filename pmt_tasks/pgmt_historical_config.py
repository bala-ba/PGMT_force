"""Restore archived Isaac PGMT settings without executing arbitrary YAML tags.

The archive is applied after config construction: calling __post_init__ again
would silently replace the historical adaptive sampler with today's EMD preset.
"""
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path

import yaml

from motion_tracking_rl.utils.pgmt_checkpoint import canonical


class ArchiveLoader(yaml.SafeLoader):
    pass


ArchiveLoader.add_constructor("tag:yaml.org,2002:python/tuple",
                              lambda loader, node: tuple(loader.construct_sequence(node, deep=True)))
ArchiveLoader.add_constructor("tag:yaml.org,2002:python/object/apply:builtins.slice",
                              lambda loader, node: slice(*loader.construct_sequence(node, deep=True)))


def read_archive(path):
    value = yaml.load(Path(path).read_text(), Loader=ArchiveLoader)
    if not isinstance(value, dict):
        raise ValueError(f"Expected an archived configuration mapping: {path}")
    return value


def _restore(current, saved, path=""):
    if isinstance(saved, dict):
        if isinstance(current, dict):
            # Replace mappings, including empty ones. Keeping new defaults here
            # can change term params, reward scales, or randomization ranges.
            return {k: _restore(current.get(k), v, path + "/" + k) for k, v in saved.items()}
        if current is None:
            return deepcopy(saved)
        for key, value in saved.items():
            if not hasattr(current, key):
                raise ValueError(f"Unknown archived configuration field: {path}/{key}")
            setattr(current, key, _restore(getattr(current, key), value, path + "/" + key))
        return current
    if callable(current):
        module, name = saved.split(":", 1)
        value = importlib.import_module(module)
        for component in name.split("."):
            value = getattr(value, component)
        if not callable(value):
            raise ValueError(f"Archive does not identify a callable: {path}")
        return value
    return deepcopy(saved)


def archived_differences(expected, actual, path=""):
    """Compare every archived field; new explicit compatibility fields are separate."""
    differences = []
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key, value in expected.items():
            if key not in actual:
                differences.append(dict(path=path + "/" + key, expected=value, actual="<missing>"))
            else:
                differences.extend(archived_differences(value, actual[key], path + "/" + key))
    elif expected != actual:
        differences.append(dict(path=path, expected=expected, actual=actual))
    return differences


def restore_historical_stage2(reference_run):
    from pmt_tasks.env_cfgs.pgmt.terrain_training import PGMTTerrainTrainingEnvCfg
    from pmt_tasks.agent_cfgs.pgmt import PGMTTerrainPPORunnerCfg
    from motion_tracking_rl.utils.pgmt_checkpoint import digest
    import torch

    root = Path(reference_run).resolve()
    env_path, agent_path = root / "params/env.yaml", root / "params/agent.yaml"
    old_env, old_agent = read_archive(env_path), read_archive(agent_path)
    if (old_env.get("pgmt_reward_groups") != ("upper", "lower", "terrain", "auxiliary")
            or old_env["commands"]["motion"]["sampler_type"] != "adaptive"
            or old_env["commands"]["motion"]["fixed_start_frame"] != 0):
        raise ValueError("Historical profile requires four-head PGMT, adaptive sampling and frame-zero starts")
    old_agent.pop("pgmt_training_contract", None)  # Bind a fresh, actual runtime contract later.
    cfg = PGMTTerrainTrainingEnvCfg()
    # The production run replaced the default URDF importer with a prepared USD.
    from dataclasses import fields
    from isaaclab.sim import UsdFileCfg
    original_spawn = cfg.scene.robot.spawn
    shared = {f.name: deepcopy(getattr(original_spawn, f.name)) for f in fields(UsdFileCfg)
              if f.name not in ('func', 'usd_path') and hasattr(original_spawn, f.name)}
    cfg.scene.robot.spawn = UsdFileCfg(usd_path=old_env['scene']['robot']['spawn']['usd_path'], **shared)
    cfg = _restore(cfg, old_env)
    agent = _restore(PGMTTerrainPPORunnerCfg(), old_agent)
    # These switches were added after train_03. Select their historical meaning.
    cfg.pgmt_terrain_reward_mode = "legacy"
    cfg.pgmt_placement_xy_mode = cfg.commands.motion.placement_xy_mode = "random"
    cfg.pgmt_emd_sampling = {}
    cfg.commands.motion.emd_sampling = {}
    cfg.scene.terrain.family_proportions = None
    differences = archived_differences(canonical(old_env), canonical(cfg.to_dict()))
    differences += archived_differences(canonical(old_agent), canonical(agent.to_dict()), "/agent")
    if differences:
        raise ValueError(f"Historical configuration differs: {differences[:10]}")
    manifest = json.loads((Path(cfg.pgmt_terrain_asset) / "manifest.json").read_text())
    archived = torch.load(root / "initial.pt", weights_only=False, map_location="cpu")
    identity = archived["environment_state_dict"]["curriculum"]["identity"]
    if manifest["asset_sha256"] != identity["terrain_sha256"]:
        raise ValueError("Historical terrain asset differs from the archived checkpoint")
    report = dict(reference_run=str(root), archive_sha256={
        "env": hashlib.sha256(env_path.read_bytes()).hexdigest(),
        "agent": hashlib.sha256(agent_path.read_bytes()).hexdigest()},
        archive_fields_match=True, differences=[], terrain_sha256=manifest["asset_sha256"],
        expected_curriculum_identity=identity, baseline_config_sha256=digest(dict(env=old_env, agent=old_agent)),
        compatibility=dict(terrain_reward_mode="legacy", placement_xy_mode="random", emd_sampling={},
                           terrain_family_proportions=None),
        state="fresh optimizer, adaptive counts and curriculum; no Stage2 training state imported")
    return cfg, agent, report


def audit_historical_runtime(env, report):
    """Check asset/motion hashes, reset distribution and sampler after construction."""
    expected = deepcopy(report["expected_curriculum_identity"])
    if report.get("terrain_override"):
        expected['terrain_sha256'] = report['terrain_override']['asset_sha256']
    actual = env._pgmt_terrain_curriculum.identity
    differences = archived_differences(expected, actual)
    if differences:
        raise ValueError(f"Runtime historical curriculum differs: {differences[:10]}")
    report["runtime_curriculum_identity_match"] = True


def override_historical_terrain(cfg, report, asset_directory):
    """Change terrain geometry explicitly, keeping archived layout/reset/sampling."""
    from pmt_tasks.pgmt_contact_state import TERRAIN_FOOT_SENSORS, terrain_contact_filters
    old_root, new_root = Path(cfg.pgmt_terrain_asset), Path(asset_directory).resolve()
    old = json.loads((old_root / 'manifest.json').read_text())
    new = json.loads((new_root / 'manifest.json').read_text())
    # A geometry/difficulty replacement must not change the curriculum axes,
    # family allocation or the allowed spawn patches.
    for key in ('num_rows', 'num_cols', 'families', 'region_size', 'patch_grid',
                'patch_size', 'family_proportions'):
        if old['spec'].get(key) != new['spec'].get(key):
            raise ValueError(f'Historical terrain override changes layout: {key}')
    # Older v6 manifests omit this field; the actual archive command owns it.
    spawn_grid = cfg.commands.motion.placement_patch_grid
    if canonical(new['spec'].get('spawn_patch_grid', spawn_grid)) != canonical(spawn_grid):
        raise ValueError('Historical terrain override changes layout: spawn_patch_grid')
    if old['schema'] != new['schema']:
        raise ValueError('Historical terrain override changes terrain schema')
    filters = terrain_contact_filters(new_root, cfg.scene.terrain.prim_path)
    cfg.pgmt_terrain_asset = cfg.scene.terrain.asset_directory = str(new_root)
    cfg.scene.terrain.usd_path = None  # Resolved and hash-checked by the importer.
    for name in TERRAIN_FOOT_SENSORS:
        getattr(cfg.scene, name).filter_prim_paths_expr = filters.copy()
    report['terrain_override'] = dict(source_asset=str(old_root), asset=str(new_root),
        source_asset_sha256=old['asset_sha256'], asset_sha256=new['asset_sha256'],
        layout_unchanged=True, sampling_reset_curriculum_unchanged=True)
