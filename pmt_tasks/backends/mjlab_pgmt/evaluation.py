"""Full-clip flat MJWarp rollouts used only for Stage2 EMD priorities."""
from copy import deepcopy
import numpy as np
import torch

from pmt_tasks.pgmt_emd_sampling import joint_trajectory_emd
from .vecenv import PGMTVectorEnv


def evaluate_emd(runner, *, seed):
    if runner._eval_env is None:
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.terrains import TerrainEntityCfg
        cfg = deepcopy(runner.env.cfg)
        cfg.seed = seed
        cfg.scene.num_envs = min(runner.env.command.sampler.config["evaluation_num_envs"],
                                 runner.env.command.data_store.num_motions)
        cfg.scene.terrain = TerrainEntityCfg(terrain_type="plane")
        cfg.scene.spec_fn = None
        cfg.scene.env_spacing = 3.
        cfg.commands["motion"].evaluation = True
        cfg.commands["motion"].noise = False
        cfg.pgmt_terrain_observation["augmentation_enabled"] = False
        cfg.events, cfg.terminations, cfg.rewards, cfg.metrics = {}, {}, {}, {}
        cfg.auto_reset = False
        runner._eval_env = PGMTVectorEnv(ManagerBasedRlEnv(cfg, device=runner.device))
    vector = runner._eval_env
    env, command = vector.env, vector.command
    library = command.data_store
    values = []
    env.seed(seed)
    for start in range(0, library.num_motions, env.num_envs):
        stop = min(start + env.num_envs, library.num_motions)
        selected = torch.arange(start, stop, device=env.device)
        command.evaluation_motion_ids = torch.cat((selected, selected[-1:].expand(env.num_envs - len(selected))))
        command.started.zero_()
        vector.reset()
        lengths = library.motion_lengths[selected].tolist()
        actual = []
        for frame in range(max(lengths)):
            actual.append(command.robot_state()["joint_pos"][:len(selected)].cpu().numpy().copy())
            if frame + 1 < max(lengths):
                actions = runner.alg.policy.act_inference(vector.get_observations())
                env.step(actions)
        actual = np.stack(actual)
        for i, length in enumerate(lengths):
            ids = selected[i].expand(length)
            frames = torch.arange(length, device=env.device)
            reference = library.gather(ids, frames)["joint_pos"].cpu().numpy()
            values.append(joint_trajectory_emd(actual[:length, i], reference))
        print(f"[MJLab PGMT EMD] evaluated {stop}/{library.num_motions} clips", flush=True)
    return values
