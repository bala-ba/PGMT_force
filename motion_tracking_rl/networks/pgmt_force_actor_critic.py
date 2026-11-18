"""PGMT with force/CoP history and optional Stage2 terrain perception."""
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from motion_tracking_rl.pgmt_force_feedback import validate_force_feedback_contract
from motion_tracking_rl.networks.layers import EmpiricalNormalization, MLP
from motion_tracking_rl.registry import register_network
from motion_tracking_rl.utils import build_obs_schema
from .pgmt_actor_critic import PGMTActorCritic


@register_network("PGMTForceActorCritic", compat_name="pgmt")
class PGMTForceActorCritic(PGMTActorCritic):
    history_features = 108

    def __init__(self, obs, obs_groups, num_actions, force_feedback_cfg=None, proprio_layout="pgmt96",
                 current_force_feedback=False, terrain_privileged_critic=False, **kwargs):
        config = validate_force_feedback_contract(force_feedback_cfg)
        if proprio_layout not in ("pgmt96", "instanttrack99"):
            raise ValueError(f"Unknown actor proprio layout: {proprio_layout}")
        self.proprio_layout = proprio_layout
        if current_force_feedback:
            if proprio_layout != "instanttrack99":
                raise ValueError("Current force feedback requires the InstantTrack99 base fields")
            self.extra_current_features = 12
        if proprio_layout == "instanttrack99":
            self.history_features = self.proprio_features = 99
            if len(obs_groups.get("force_history", [])) != 1:
                raise ValueError("InstantTrack99 requires an independent force_history observation")
        if len(obs_groups.get("policy_history", [])) != 1:
            raise ValueError("Force feedback V1 requires one proprio history observation")
        if terrain_privileged_critic:
            keys = obs_groups.get("critic_terrain_map", [])
            if (proprio_layout != "instanttrack99" or len(keys) != 1 or obs[keys[0]].shape[1:] != (21, 21)
                    or any(keys[0] in v for k, v in obs_groups.items() if k != "critic_terrain_map")):
                raise ValueError("Expected an independent clean 21x21 critic_terrain_map with the InstantTrack99 layout")
            self.critic_terrain_features = 64
            self.critic_normalized_features = 185
        super().__init__(obs, obs_groups, num_actions, **kwargs)
        if self.num_value_heads not in (3, 4) or (self.num_value_heads == 4 and self.terrain_glimpse is None):
            raise ValueError("Force feedback V1 requires Stage1 three-head or Stage2 three/four-head PPO")
        self.force_feedback_cfg = config
        # Equivalent to a 108 -> embedding projection with separate normalization.
        # Preserve all old tensors and make weights-only migration action-exact.
        self.force_history_projection = nn.Linear(12, self.proprio_embed.out_features, bias=False)
        nn.init.zeros_(self.force_history_projection.weight)
        self.register_buffer("force_feature_scale", torch.tensor(
            ([config["force_scale"]] * 3 + config["cop_scale"] + [1.]) * 2), persistent=False)
        if terrain_privileged_critic:
            if self.privileged_dim != self.critic_normalized_features + 12:
                raise ValueError("Terrain privileged critic expects 185 scalar features and 12 current foot features")
            self.critic_terrain_encoder = MLP(21 * 21, self.critic_terrain_features, [128], "elu")
            if self.critic_obs_normalization:
                self.critic_obs_normalizer = EmpiricalNormalization(self.critic_normalized_features,
                                                                    per_feature_count=True)
        if proprio_layout == "instanttrack99":
            self._sequence(obs, "force_history", self.history_frames, 12)
            if self.actor_obs_normalization:
                self.actor_obs_normalizer = EmpiricalNormalization(self.proprio_features, per_feature_count=True)
            return
        legacy_obs = obs.clone(False)
        key = obs_groups["policy_history"][0]
        legacy_obs[key] = obs[key][..., :96]
        self._legacy_metadata = super().get_checkpoint_metadata()
        self._legacy_metadata["policy_class"] = "PGMTActorCritic"
        self._legacy_metadata["obs_schema"] = build_obs_schema(legacy_obs, self.obs_groups)

    def _normalized_foot_feedback(self, force):
        raw = force.unflatten(-1, (2, 6))
        valid = raw[..., 5:6] > .5
        # Invalid CoP does not mean that the foot force is invalid.
        contact = torch.cat((raw[..., :3], torch.where(valid, raw[..., 3:5], 0.),
                             valid.to(raw.dtype)), dim=-1).flatten(-2)
        return contact / self.force_feature_scale

    def _normalize_current_proprio(self, raw):
        if self.extra_current_features:
            body, feet = raw.split((self.proprio_features, self.extra_current_features), dim=-1)
            return torch.cat((self.actor_obs_normalizer(body), self._normalized_foot_feedback(feet)), dim=-1)
        return super()._normalize_current_proprio(raw)

    def _history_embeddings(self, obs):
        history = self._sequence(obs, "policy_history", self.history_frames, self.history_features)
        proprio = history[..., :self.proprio_features]
        proprio = self.actor_obs_normalizer(proprio.reshape(-1, self.proprio_features)).reshape_as(proprio)
        force = (self._sequence(obs, "force_history", self.history_frames, 12)
                 if self.proprio_layout == "instanttrack99" else history[..., 96:])
        # Share the body columns with the current query. Historical foot feedback
        # already has its own trained projection.
        body = F.linear(proprio, self.proprio_embed.weight[:, :self.proprio_features], self.proprio_embed.bias)
        return body + self.force_history_projection(self._normalized_foot_feedback(force))

    def _evaluate_from_encoding(self, obs, intent):
        if not self.critic_terrain_features:
            return super()._evaluate_from_encoding(obs, intent)
        body, feet = self.get_critic_obs(obs).split((self.critic_normalized_features, 12), dim=-1)
        terrain = obs[self.obs_groups["critic_terrain_map"][0]].flatten(-2)
        return self.critic(torch.cat((self.critic_obs_normalizer(body), self._normalized_foot_feedback(feet),
                                     self.critic_terrain_encoder(terrain), intent), dim=-1))

    def get_checkpoint_metadata(self):
        meta = super().get_checkpoint_metadata()
        meta["policy_family"] = ("pgmt_stage2_force_v1" if self.terrain_glimpse is not None
                                 else "pgmt_stage1_force_v1")
        meta["policy_kwargs"]["force_feedback_cfg"] = deepcopy(self.force_feedback_cfg)
        meta["signature"].update(history_features=self.history_features, force_feedback_cfg=deepcopy(self.force_feedback_cfg))
        if self.proprio_layout == "instanttrack99":
            meta["policy_kwargs"]["proprio_layout"] = self.proprio_layout
            meta["signature"]["proprio_layout"] = self.proprio_layout
        if self.extra_current_features:
            meta["policy_kwargs"]["current_force_feedback"] = True
            meta["signature"]["current_force_feedback"] = True
        if self.critic_terrain_features:
            meta["policy_kwargs"]["terrain_privileged_critic"] = True
            meta["signature"]["terrain_privileged_critic"] = dict(body=185, feet=12, terrain_mlp=[441, 128, 64],
                input_order=["body", "feet", "terrain_embedding", "intent"])
        return meta

    def legacy_checkpoint_metadata(self):
        if self.proprio_layout != "pgmt96":
            raise ValueError("Legacy Stage1 migration requires the PGMT 96-D layout")
        return deepcopy(self._legacy_metadata)
