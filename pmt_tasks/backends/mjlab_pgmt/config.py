"""Isaac-free environment and PPO configuration for the two PGMT stages."""
from dataclasses import dataclass, field, fields
from functools import partial
from pathlib import Path

from omegaconf import OmegaConf
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import ObservationGroupCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg

from pmt_tasks.pgmt_math import REWARD_WEIGHTS, VALUE_GROUPS
from pmt_tasks.pgmt_perception import terrain_observation_contract
from pmt_tasks.pgmt_contact_state import CONTACT_STATE_CONFIG
from pmt_tasks.pgmt_touchdown import TOUCHDOWN_QUALITY_CONFIG, TOUCHDOWN_EVENT_WEIGHT
from pmt_tasks.pgmt_tracking_relaxation import tracking_relaxation_config
from motion_tracking_rl.pgmt_value_contract import TERRAIN_VALUE_GROUPS
from . import mdp

ROOT = Path(__file__).resolve().parents[3]


@dataclass(kw_only=True)
class PGMTEnvCfg(ManagerBasedRlEnvCfg):
    pgmt_stage: int = 1
    pgmt_reward_groups: tuple[str, ...] = VALUE_GROUPS
    pgmt_terrain_observation: dict = field(default_factory=terrain_observation_contract)


@dataclass(kw_only=True)
class PGMTStage2EnvCfg(PGMTEnvCfg):
    """Stage2 reward settings; preserve the existing Stage1 checkpoint schema."""
    pgmt_touchdown_quality: dict | None = None
    pgmt_contact_events: dict | None = None
    pgmt_tracking_relaxation: dict = field(default_factory=tracking_relaxation_config)


def build_env(task, *, stage):
    if stage not in (1, 2):
        raise ValueError("Only PGMT Stage1 and Stage2 perception are supported")
    if OmegaConf.is_config(task):
        task = OmegaConf.to_container(task, resolve=True)
    base = unitree_g1_flat_tracking_env_cfg()
    env_class = PGMTStage2EnvCfg if stage == 2 else PGMTEnvCfg
    cfg = env_class(**{f.name: getattr(base, f.name) for f in fields(base)}, pgmt_stage=stage)
    motion = task["motion"]
    expected_sampler = "bin_adaptive" if stage == 1 else "emd"
    if motion.get("sampler") != expected_sampler:
        raise ValueError(f"PGMT mjlab Stage{stage} requires motion.sampler={expected_sampler}")
    if int(motion.get("chunk_length", 0)) < 0:
        raise ValueError("chunk_length must be nonnegative")
    if stage == 2:
        relaxation = dict(task.get("tracking_relaxation", {}))
        relaxation.update(task.get("mjlab_tracking_relaxation", {"enabled": True}))
        cfg.pgmt_tracking_relaxation = tracking_relaxation_config(relaxation)
    cfg.decimation = int(motion.get("decimation", task["robot"].get("decimation", 4)))
    cfg.sim.mujoco.timestep = float(motion.get("sim_dt", task["robot"].get("sim_dt", .005)))
    cfg.sim.nconmax, cfg.sim.njmax = 128, 512
    cfg.episode_length_s = 10.
    cfg.scale_rewards_by_dt = True
    cfg.scene.env_spacing = 3.
    action = cfg.actions["joint_pos"]
    cfg.actions = {"joint_pos": mdp.BFSPositionActionCfg(entity_name="robot", actuator_names=(".*",),
                                         scale=action.scale, use_default_offset=True)}
    # Literal names preserve BFS contact order independently of regex lookup.
    cfg.scene.sensors = (ContactSensorCfg(name="pgmt_contacts", primary=ContactMatch(mode="body",
        pattern=tuple("robot/" + name for name in mdp.BFS_BODIES)), fields=("force",), reduce="netforce"),)
    # Retain physical friction/COM randomization and pushes. Encoder bias is
    # omitted because PGMT observes the actual PD targets it sends to physics.
    cfg.events.pop("encoder_bias", None)
    paths = motion["motion_files"]
    if isinstance(paths, str):
        paths = [paths]
    cfg.commands = {"motion": mdp.MotionCfg(resampling_time_range=(1e9, 1e9),
        motion_files=list(paths), stage=stage, chunk_length=int(motion.get("chunk_length", 0)),
        terrain_asset=str(Path(task.get("mjlab_terrain_asset", task.get("terrain_asset", ""))).resolve()) if stage == 2 else None,
        compatibility_file=str(ROOT / "assets/motions/pgmt_clipped_50hz_tcrs_v1/motion_terrain_compatibility.csv") if stage == 2 else None,
        placement_xy_mode=motion.get("placement_xy_mode", "center"),
        fixed_start_frame=int(motion.get("fixed_start_frame", -1)), emd=motion.get("emd", {}),
        tracking=dict(mdp.TRACKING_CONFIG, **task.get("reward", {}).get("tracking", {})))}
    cfg.observations = {}
    for name, key in (("proprio", "proprio"), ("proprio_history", "proprio"),
                      ("command_window", "command_window"), ("critic", "critic")):
        cfg.observations[name] = ObservationGroupCfg(
            terms={key: ObservationTermCfg(func=mdp.observation, params={"key": key})},
            concatenate_terms=True, enable_corruption=False,
            history_length=10 if name == "proprio_history" else 0, flatten_history_dim=False)
    weights = dict(task.get("reward_weights", REWARD_WEIGHTS))
    if stage == 2:
        # Isaac's six terrain terms, with the requested train_03 T1 event
        # override. T5 retains geometry_v3 and T2/T3 retain their existing gate.
        terrain_weights = OmegaConf.to_container(OmegaConf.load(ROOT / "configs/reward/pgmt_stage2_terrain.yaml"))["weights"]
        weights.update({name: weight for name, weight in terrain_weights.items() if name.startswith("terrain_")})
        weights["terrain_touchdown_quality"] = TOUCHDOWN_EVENT_WEIGHT
        cfg.pgmt_touchdown_quality = dict(TOUCHDOWN_QUALITY_CONFIG, settlement="touchdown_event_v1")
        cfg.pgmt_contact_events = dict(CONTACT_STATE_CONFIG, contact_threshold_n=1., support_threshold_n=1.)
        cfg.pgmt_reward_groups = TERRAIN_VALUE_GROUPS
        cfg.metrics["terrain_support_force_z"] = MetricsTermCfg(func=mdp.capture_terrain_contacts, per_substep=True)
    expected = set(REWARD_WEIGHTS) | (set(terrain_weights) if stage == 2 else set())
    if set(weights) != expected:
        raise ValueError("PGMT mjlab requires 21 Stage1 or 27 Stage2 reward terms")
    cfg.rewards = {name: RewardTermCfg(func=mdp.reward, weight=float(weight), params={"name": name})
                   for name, weight in weights.items()}
    cfg.terminations = dict(time_out=TerminationTermCfg(func=mdp.time_out, time_out=True),
                           motion_finished=TerminationTermCfg(func=mdp.motion_finished, time_out=True))
    for kind in ("anchor_pos", "anchor_ori", "ee_body_pos") + (("root_xy",) if stage == 2 else ()):
        cfg.terminations[kind] = TerminationTermCfg(func=mdp.tracking_failure, params={"kind": kind})
    if stage == 2:
        if cfg.commands["motion"].chunk_length:
            raise ValueError("Stage2 EMD uses complete clips; chunk_length must be zero")
        from .terrain import add_native_terrain
        cfg.scene.terrain = None
        cfg.scene.env_spacing = 0.
        cfg.scene.spec_fn = partial(add_native_terrain, asset=cfg.commands["motion"].terrain_asset)
        cfg.observations["pgmt_terrain"] = ObservationGroupCfg(
            terms={"height_and_valid": ObservationTermCfg(func=mdp.terrain_observation)},
            concatenate_terms=True, enable_corruption=False)
        cfg.terminations["map_out_of_bounds"] = TerminationTermCfg(func=mdp.map_out_of_bounds, time_out=True)
    return cfg


def build_agent(task, *, stage):
    if OmegaConf.is_config(task):
        task = OmegaConf.to_container(task, resolve=True)
    network = dict(task["network"])
    network["class_name"] = network.pop("name")
    network.update(actor_obs_normalization=True, critic_obs_normalization=True)
    groups = dict(policy=["proprio"], policy_history=["proprio_history"],
                  command_window=["command_window"], critic=["critic"])
    if stage == 2:
        network["terrain_observation_cfg"] = terrain_observation_contract()
        network["value_groups"] = list(TERRAIN_VALUE_GROUPS)
        groups.update(terrain_map=["terrain_map"], terrain_valid=["terrain_valid"])
    algorithm = dict(class_name="PPO", value_loss_coef=1., use_clipped_value_loss=True,
        clip_param=.2, entropy_coef=.005, num_learning_epochs=5, num_mini_batches=4,
        learning_rate=1e-3 if stage == 1 else 1e-4, schedule="adaptive", gamma=.99,
        lam=.95, desired_kl=.01, max_grad_norm=1.)
    for key in algorithm.keys() & task["algorithm"].keys():
        algorithm[key] = task["algorithm"][key]
    return dict(policy=network, algorithm=algorithm, obs_groups=groups, num_steps_per_env=24,
                save_interval=200, logger="tensorboard", clip_actions=None,
                empirical_normalization=False, seed=42)
