"""Explicit old Stage1 -> force feedback V1 weights-only migration."""
import hashlib
import math
from pathlib import Path

import torch

from .pgmt_checkpoint import canonical, validate_tensors
from motion_tracking_rl.pgmt_force_feedback import force_feedback_contract


def migrate_stage1_force(policy, path):
    path = Path(path)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if policy.get_checkpoint_metadata()["policy_family"] != "pgmt_stage1_force_v1":
        raise ValueError("Expected a force feedback V1 target")
    if saved.get("policy_metadata") != policy.legacy_checkpoint_metadata():
        raise ValueError("Stage1 -> force V1 architecture/observation contract differs")
    current = policy.state_dict()
    added = {"force_history_projection.weight"}
    base = {k: v for k, v in current.items() if k not in added}
    validate_tensors(saved["model_state_dict"], base)
    # Validate everything before mutating. New branch starts at exactly zero.
    target = {k: torch.zeros_like(v) if k in added else saved["model_state_dict"][k]
              for k, v in current.items()}
    policy.load_state_dict(target, strict=True)
    return dict(kind="stage1_to_force_v1", source=str(path.resolve()),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(), source_iteration=saved.get("iter"),
        copied=sorted(base), initialized=sorted(added), optimizer="fresh",
        completed_iterations=0, sampler_curriculum="fresh")


def migrate_force_stage2(policy, path):
    """PMT Force Stage1 -> Force Stage2; keep the existing PMT value semantics."""
    path = Path(path)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    old, new = saved.get("policy_metadata", {}), policy.get_checkpoint_metadata()
    if (old.get("policy_family") != "pgmt_stage1_force_v1"
            or new["policy_family"] != "pgmt_stage2_force_v1"):
        raise ValueError("Expected Force Stage1 -> Force Stage2")
    expanded = old.get("value_groups") == ["upper", "lower", "auxiliary"] and new.get("value_groups") == ["upper", "lower", "terrain", "auxiliary"]
    if expanded and old.get("signature", {}).get("num_value_heads") != 3:
        raise ValueError("Force source critic signature must contain three heads")
    for name in ("num_actions", "action_contract", "value_groups", "policy_class"):
        if name == "value_groups" and expanded:
            continue
        if old.get(name) != new.get(name):
            raise ValueError(f"Force Stage2 migration contract differs: {name}")
    for section in ("signature", "policy_kwargs", "obs_groups"):
        for name, value in old[section].items():
            if expanded and section in ("signature", "policy_kwargs") and name in ("num_value_heads", "value_groups"):
                continue
            if new[section].get(name) != value:
                raise ValueError(f"Force Stage2 migration differs: {section}/{name}")
    for name, schema in old["obs_schema"]["available_obs"].items():
        if new["obs_schema"]["available_obs"].get(name) != schema:
            raise ValueError(f"Force Stage2 observation differs: {name}")
    current = policy.state_dict()
    base = {k: v for k, v in current.items() if not k.startswith("terrain_glimpse.")}
    final_name = [name for name, layer in policy.critic.named_modules() if isinstance(layer, torch.nn.Linear)][-1]
    partial = [f"critic.{final_name}.{suffix}" for suffix in ("weight", "bias")] if expanded else []
    for key in partial:
        base[key] = base[key].new_empty((3, *base[key].shape[1:]))
    validate_tensors(saved["model_state_dict"], base)
    target = {k: saved["model_state_dict"].get(k, v) for k, v in current.items()}
    for key in partial:
        target[key] = current[key].clone()
        target[key][[0, 1, 3]] = saved["model_state_dict"][key].to(target[key].device)
    policy.load_state_dict(target, strict=True)
    return dict(kind="force_stage1_to_stage2", source=str(path.resolve()),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(), source_iteration=saved.get("iter"),
        copied=sorted(set(base) - set(partial)), partially_copied=partial,
        initialized_rows={key: [2] for key in partial},
        initialized=sorted(set(current) - set(base)), optimizer="fresh",
        completed_iterations=0, sampler_curriculum="fresh")


class InstantTrack99Transfer:
    """Affine correspondence of 90 input columns at the nominal joint calibration.

    For each paired field, old = gain * new + offset. The old torso error has no
    pelvis-frame equivalent. New gravity and pelvis-error columns start at zero.
    """
    def __init__(self, sdk_order, nominal_joint_pos, action_scale):
        order = sdk_order.detach().cpu()
        if not torch.equal(order.sort().values, torch.arange(29)) or order.shape != (29,):
            raise ValueError("Expected a permutation of 29 G1 joints")
        q0 = nominal_joint_pos.detach().cpu()[order]
        scale = action_scale.detach().cpu().expand(29)[order]
        if not torch.isfinite(q0).all() or not torch.isfinite(scale).all() or not (scale > 0).all():
            raise ValueError("Invalid nominal G1 action transform")
        self.old = torch.cat((order + 9, order + 38, torch.arange(6, 9), order + 67))
        self.new = torch.cat((torch.arange(61), torch.arange(64, 93)))
        self.gain = torch.cat((torch.ones(29), torch.full((29,), .05), torch.ones(3), scale))
        self.offset = torch.cat((q0, torch.zeros(32), q0))
        self.critic_old = torch.cat((torch.arange(6, 12), 12 + order, 41 + order))
        self.critic_new = torch.arange(7, 71)
        self.critic_gain = torch.cat((torch.ones(35), torch.full((29,), .05)))

    def source_observations(self, obs, source_mean, critic_mean=None):
        """Build matched old-format observations for the startup migration audit."""
        def convert(x):
            old = source_mean.to(x).expand(*x.shape[:-1], 96).clone()
            old[..., self.old] = x[..., self.new] * self.gain.to(x) + self.offset.to(x)
            return old
        matched = obs.exclude("force_history").clone(False)
        matched["proprio"] = convert(obs["proprio"])
        matched["proprio_history"] = torch.cat((convert(obs["proprio_history"]), obs["force_history"]), -1)
        if critic_mean is not None:
            scalar = obs["critic"]
            old = critic_mean.to(scalar).expand(len(scalar), -1).clone()
            old[:, self.critic_old] = scalar[:, self.critic_new] * self.critic_gain.to(scalar)
            matched["critic"] = old
        return matched

    @staticmethod
    def _remap_statistics(target, source, eps, norm, old, new, gain, offset):
        # Source weights are CPU tensors. Materialize converted tensors on CPU
        # before the single strict load, including when the target is on GPU.
        std = source[norm + "_std"][:, old]
        new_std = std / gain
        remapped = []
        for field, mapped in (
            ("_mean", (source[norm + "_mean"][:, old] - offset) / gain),
            ("_var", source[norm + "_var"][:, old] / gain.square()),
            ("_std", new_std),
        ):
            key = norm + field
            value = torch.full_like(target[key], 0. if field == "_mean" else 1., device="cpu")
            value[:, new] = mapped
            target[key] = value
            remapped.append(key)
        count_key = norm + "count"
        counts = torch.zeros_like(target[count_key], device="cpu")
        counts[:, new] = source[count_key]
        target[count_key] = counts
        remapped.append(count_key)
        # Account for EmpiricalNormalization adding epsilon *after* sqrt.
        return remapped, gain * (new_std + eps) / (std + eps)

    def remap(self, target, source, eps):
        remapped, factor = self._remap_statistics(target, source, eps, "actor_obs_normalizer.",
                                                   self.old, self.new, self.gain, self.offset)
        for key in ("proprio_embed.weight", "actor.0.weight"):
            value = torch.zeros_like(target[key], device="cpu")
            value[:, self.new] = source[key][:, self.old] * factor
            if key == "actor.0.weight":
                intent = source[key][:, 96:]
                value[:, -intent.shape[1]:] = intent
            target[key] = value
            remapped.append(key)
        return remapped

    def remap_critic(self, target, source, eps):
        remapped, factor = self._remap_statistics(target, source, eps, "critic_obs_normalizer.",
                                                  self.critic_old, self.critic_new, self.critic_gain, 0.)
        key = "critic.0.weight"
        value = torch.zeros_like(target[key], device="cpu")
        value[:, self.critic_new] = source[key][:, self.critic_old] * factor
        intent = source[key][:, 251:]
        value[:, -intent.shape[1]:] = intent
        target[key] = value
        return remapped + [key]


def migrate_instanttrack_force(policy, path, *, weights="ema", critic_init="source", proprio_transfer=None):
    """Import the versioned InstantTrack Force policy, never its BPTT optimizer.

    InstantTrack's critic predicts different reward groups. Default `source`
    initialization copies its trunk, normalizer and three outputs as a warm
    start (source rows 0/1/2 -> upper/lower/auxiliary); a new terrain row retains
    its random initialization. `fresh` preserves the original mjlab behavior.
    Use the source collection action noise for PPO exploration, since `std` is
    an unused BPTT compatibility tensor. New terrain parameters
    follow the ordinary PMT Stage2 initialization. With terrain disabled, the
    96-D actor preserves the source output; the 99-D actor matches the source
    on the nominal, paired inputs described by InstantTrack99Transfer.
    """
    if weights not in ("ema", "model"):
        raise ValueError("InstantTrack weights must be ema or model")
    if critic_init not in ("source", "fresh"):
        raise ValueError("InstantTrack critic initialization must be source or fresh")
    path = Path(path)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    meta = policy.get_checkpoint_metadata()
    sdk99 = meta["policy_kwargs"].get("proprio_layout") == "instanttrack99"
    terrain_critic = meta["policy_kwargs"].get("terrain_privileged_critic", False)
    if sdk99 != isinstance(proprio_transfer, InstantTrack99Transfer):
        raise ValueError("The InstantTrack99 layout requires its G1 proprio transfer")
    if (saved.get("format") != "instanttrack.pgmt_force_cop.v1"
            or meta["policy_family"] != "pgmt_stage2_force_v1"):
        raise ValueError("InstantTrack Force migration requires a Force Stage2 target")
    contract, args = saved.get("contract", {}), saved.get("args", {})
    expected_shapes = dict(proprio=[96], proprio_history=[10, 108], command_window=[8, 61], critic=[251])
    expected = dict(policy_arch="pgmt_stage1_force_v1", observation_shapes=expected_shapes,
                    force_feedback=force_feedback_contract(), action="q0 + xi_joint_offset + scale*a")
    for name, value in expected.items():
        if canonical(contract.get(name)) != canonical(value):
            raise ValueError(f"InstantTrack Force contract differs: {name}")
    if args.get("robot") != "g1" or args.get("foot_force_history", False):
        raise ValueError("InstantTrack Force migration requires the 29-DoF G1 with 108-D history")
    architecture = dict(embedding_dim=128, num_heads=4, command_frames=8, history_frames=10,
        rope_base=10000., actor_hidden_dims=[512, 256, 128], critic_hidden_dims=[512, 256, 128],
        activation="elu", noise_std_type="scalar", state_dependent_std=False,
        actor_obs_normalization=True, critic_obs_normalization=True,
        force_feedback_cfg=force_feedback_contract())
    for name, value in architecture.items():
        if meta["policy_kwargs"].get(name) != value:
            raise ValueError(f"InstantTrack Force target architecture differs: {name}")
    std = float(args.get("action_noise", float("nan")))
    if not math.isfinite(std) or std <= 0:
        raise ValueError("InstantTrack collection action_noise must be finite and positive")
    current = policy.state_dict()
    base = {k: v for k, v in current.items() if not k.startswith(("terrain_glimpse.", "critic_terrain_encoder."))}
    source = saved[weights]
    # InstantTrack always has three value outputs. Validate its entire state
    # against a three-head schema even when the fresh PMT critic has four heads.
    # The legacy actor schema also fixes its input order and normalizer shape.
    source_schema = dict(base)
    if terrain_critic:
        source_schema["critic_obs_normalizer.count"] = base["critic_obs_normalizer.count"].new_empty(())
        for suffix in ("_mean", "_var", "_std"):
            key = "critic_obs_normalizer." + suffix
            source_schema[key] = base[key].new_empty((1, 251))
        key = "critic.0.weight"
        source_schema[key] = base[key].new_empty((base[key].shape[0], 251 + 128))
    if sdk99:
        source_schema["actor_obs_normalizer.count"] = base["actor_obs_normalizer.count"].new_empty(())
        for key, width in (("actor.0.weight", 224), ("proprio_embed.weight", 96),
                           ("actor_obs_normalizer._mean", 96), ("actor_obs_normalizer._std", 96),
                           ("actor_obs_normalizer._var", 96)):
            source_schema[key] = base[key].new_empty((base[key].shape[0], width))
    final_name, final_layer = [(name, layer) for name, layer in policy.critic.named_modules()
                              if isinstance(layer, torch.nn.Linear)][-1]
    for suffix in ("weight", "bias"):
        key = f"critic.{final_name}.{suffix}"
        source_schema[key] = base[key].new_empty((3, *base[key].shape[1:]))
    validate_tensors(source, source_schema)
    copied = {k for k in base if k != "std" and not k.startswith(("critic.", "critic_obs_normalizer."))}
    partial = []
    head_mapping = []
    if critic_init == "source":
        if canonical(contract.get("value_groups")) != ["body_tracking", "root_tracking", "regularization"]:
            raise ValueError("Unknown InstantTrack source critic head order")
        final_keys = {f"critic.{final_name}.{suffix}" for suffix in ("weight", "bias")}
        copied.update(k for k in base if k.startswith(("critic.", "critic_obs_normalizer.")))
        if policy.num_value_heads == 4:
            copied.difference_update(final_keys)
            partial = sorted(final_keys)
        rows = [list(policy.value_groups).index(name) for name in ("upper", "lower", "auxiliary")]
        head_mapping = [dict(source_row=i, source_group=contract['value_groups'][i],
                             target_row=row, target_group=policy.value_groups[row]) for i, row in enumerate(rows)]
    target = {k: source[k] if k in copied else v for k, v in current.items()}
    remapped = []
    remapped_critic = []
    if terrain_critic and critic_init == "source":
        for key in ("critic.0.weight", "critic_obs_normalizer._mean", "critic_obs_normalizer._var",
                    "critic_obs_normalizer._std", "critic_obs_normalizer.count"):
            target[key] = current[key]
        remapped_critic = proprio_transfer.remap_critic(target, source, policy.critic_obs_normalizer.eps)
        copied.difference_update(remapped_critic)
    if sdk99:
        # Remap both weights and statistics before the single load_state_dict.
        for key in ("proprio_embed.weight", "actor.0.weight", "actor_obs_normalizer._mean",
                    "actor_obs_normalizer._var", "actor_obs_normalizer._std", "actor_obs_normalizer.count"):
            target[key] = current[key]
            copied.remove(key)
        remapped = proprio_transfer.remap(target, source, policy.actor_obs_normalizer.eps)
    for key in partial:
        target[key] = current[key].clone()
        target[key][rows] = source[key].to(device=target[key].device)
    target["std"] = torch.full_like(current["std"], std)
    policy.load_state_dict(target, strict=True)
    return dict(kind="instanttrack_force_to_stage2_ppo", source=str(path.resolve()),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(), source_iteration=saved["iteration"],
        source_weights=weights, copied=sorted(copied), partially_copied=partial,
        initialized=sorted(set(current) - copied - set(partial) - set(remapped) - set(remapped_critic)),
        **(dict(remapped_critic=remapped_critic, copied_critic_scalar_columns=64 if remapped_critic else 0)
           if terrain_critic else {}),
        **(dict(proprio_layout="instanttrack99", remapped_proprio=remapped,
                zero_input_columns=["projected_gravity", "pelvis_reference_orientation"] +
                    (["current_foot_feedback"] if meta["policy_kwargs"].get("current_force_feedback") else []),
                actor_normalizer_count="source on mapped columns, zero on new columns") if sdk99 else {}),
        initialized_rows={key: [2] for key in partial}, critic_head_mapping=head_mapping,
        source_value_groups=contract.get("value_groups"), target_value_groups=list(policy.value_groups),
        critic=("source matched scalar and intent columns, trunk and mapped heads" if remapped_critic else
                "source trunk, normalizer and three heads; terrain head random" if partial else
                "source critic and normalizer" if critic_init == "source" else "fresh: reward semantics differ"),
        critic_init=critic_init, exploration_std=std,
        optimizer="fresh", world_model="not imported", completed_iterations=0, sampler_curriculum="fresh")
