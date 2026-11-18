"""PMT grouped-PPO interface and validated sampler/curriculum checkpoints."""
from copy import deepcopy
import torch
from tensordict import TensorDict

from pmt_tasks.pgmt_perception import split_terrain_observation
from pmt_tasks.pgmt_tracking_relaxation import RELAXATION_TERMS


class PGMTVectorEnv:
    def __init__(self, env):
        self.unwrapped = self.env = env
        self.num_envs, self.device = env.num_envs, env.device
        self.num_actions = env.action_manager.total_action_dim
        self.max_episode_length = env.max_episode_length
        self.cfg = env.cfg
        self.command = env.command_manager.get_term("motion")
        self.reset()

    @property
    def episode_length_buf(self):
        return self.env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        self.env.episode_length_buf = value

    def _obs(self, values):
        return TensorDict(split_terrain_observation(values), batch_size=[self.num_envs])

    def get_observations(self):
        # Return the realized policy observations; do not advance history or
        # resample terrain dropout when the runner asks for the current state.
        return self._obs(self.env.obs_buf)

    def reset(self):
        values, extras = self.env.reset()
        return self._obs(values), extras

    def step(self, actions):
        obs, scalar, terminated, truncated, info = self.env.step(actions)
        # reward_terms were frozen before auto-reset; do not query the new
        # episode here. mjlab's private reward buffer uses integrated values,
        # unlike Isaac's rate buffer, so reconstruct from the frozen raw rates.
        terms = self.command._rewards
        groups = torch.stack([sum(terms[name] * term.weight for name, term in self.cfg.rewards.items()
                              if name.startswith(group + "_")) * self.env.step_dt
                              for group in self.cfg.pgmt_reward_groups], -1)
        extras = dict(info, log=dict(info.get("log", {})), reward_groups=groups,
                      time_outs=truncated & ~terminated)
        for i, name in enumerate(self.cfg.pgmt_reward_groups):
            extras["log"][f"Reward_Groups/{name}_per_step"] = groups[:, i].mean()
        relaxation = getattr(self.command, "_tracking_relaxation_reward", None)
        if relaxation is not None:
            extras["log"]["TrackingRelaxation/level_scale"] = relaxation["level_scale"].mean()
            extras["log"]["TrackingRelaxation/applicable_fraction"] = relaxation["applicable"].float().mean()
            for name in RELAXATION_TERMS:
                extras["log"][f"TrackingRelaxation/{name}_margin"] = relaxation[f"{name}_margin"].mean()
                gain = relaxation[f"{name}_relaxed_rate"] - relaxation[f"{name}_strict_rate"]
                extras["log"][f"TrackingRelaxation/{name}_reward_gain"] = gain.mean() * self.cfg.rewards[name].weight * self.env.step_dt
        feedback = getattr(self.env, "_pgmt_force_feedback", None)
        if feedback is not None:
            current = feedback.tensor.reshape(self.num_envs, 2, 6)
            for i, foot in enumerate(("left", "right")):
                extras["log"][f"ForceFeedback/{foot}_force_norm_N"] = current[:, i, :3].norm(dim=-1).mean()
                extras["log"][f"ForceFeedback/{foot}_cop_valid_fraction"] = current[:, i, 5].mean()
        return self._obs(obs), scalar, (terminated | truncated).long(), extras

    def get_checkpoint_state(self):
        return dict(version=1, backend="mjlab", identity=deepcopy(self.command.identity),
                    num_envs=self.num_envs, sampler=self.command.sampler.state_dict(),
                    curriculum=self.command.curriculum.state_dict() if self.command.curriculum else None)

    def validate_checkpoint_state(self, state):
        if (state.get("version") != 1 or state.get("backend") != "mjlab"
                or state.get("identity") != self.command.identity or state.get("num_envs") != self.num_envs):
            raise ValueError("MJLab checkpoint backend/assets/motion library/environment count differ")
        self.command.sampler.validate_state_dict(state["sampler"])
        if self.command.curriculum:
            self.command.curriculum.validate_state_dict(state["curriculum"])
        elif state.get("curriculum") is not None:
            raise ValueError("Unexpected terrain curriculum in flat checkpoint")

    def load_checkpoint_state(self, state):
        self.validate_checkpoint_state(state)
        self.command.sampler.load_state_dict(state["sampler"])
        if self.command.curriculum:
            self.command.curriculum.load_state_dict(state["curriculum"])
        with torch.inference_mode():
            self.command.started.zero_()
            self.reset()

    def close(self):
        self.env.close()
