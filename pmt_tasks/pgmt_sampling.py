"""PGMT motion-level sampling conditioned on terrain compatibility and reset validity."""
import csv
import math
from pathlib import Path
import re

import torch

from pmt_tasks.pgmt_terrain import FAMILIES, FAMILY_GROUPS


def load_terrain_compatibility(path, motion_files, *, families=FAMILIES, device="cpu"):
    """Read an exclusion table in loader motion order; blank cells allow all terrains.

    A semantic group such as ``stairs`` also excludes ``up_stairs``. An explicit
    variant such as ``up_stairs`` only excludes that variant. CSV supersets are
    allowed for inspecting a subset of the library; missing selected motions,
    duplicate names and unknown terrain labels fail instead of allowing silently.
    """
    names = [Path(file).stem for file in motion_files]
    if len(names) != len(set(names)):
        raise ValueError("PGMT compatibility requires unique motion file stems")
    if any(family not in FAMILY_GROUPS for family in families):
        raise ValueError(f"Unknown PGMT terrain families: {families}")
    excluded = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or not {"motion_name", "dismatched terrain"}.issubset(reader.fieldnames):
            raise ValueError("Compatibility CSV requires motion_name and dismatched terrain columns")
        for line, row in enumerate(reader, 2):
            if None in row or row["dismatched terrain"] is None:
                raise ValueError(f"Malformed compatibility CSV row {line}; quote multi-terrain cells")
            name = row["motion_name"].strip()
            if name.endswith(".npz"):
                name = name[:-4]
            if not name or name in excluded:
                raise ValueError(f"Empty or duplicate motion_name on CSV row {line}: {name!r}")
            terrain_names = set(filter(None, re.split(r"[,;|\s]+", row["dismatched terrain"].strip())))
            unknown = terrain_names - set(FAMILIES)
            if unknown:
                raise ValueError(f"Unknown terrain labels on CSV row {line}: {sorted(unknown)}")
            excluded[name] = terrain_names
    missing = sorted(set(names) - excluded.keys())
    if missing:
        raise ValueError(f"Motions missing from compatibility CSV: {missing}")
    return torch.tensor([
        [family not in excluded[name] and FAMILY_GROUPS[family] not in excluded[name] for name in names]
        for family in families
    ], dtype=torch.bool, device=device)


def masked_motion_probabilities(p_fail, allowed, *, beta, uniform_ratio):
    """Normalize difficulty weights and the uniform component *within* each pool."""
    if beta < 0 or not 0 <= uniform_ratio <= 1:
        raise ValueError("Require beta >= 0 and uniform_ratio in [0, 1]")
    if allowed.ndim != 2 or p_fail.shape not in (allowed.shape, allowed.shape[1:]):
        raise ValueError("Expected failure probabilities per motion or per terrain and motion")
    if not allowed.any(-1).all():
        raise ValueError("Cannot sample from an empty compatible motion pool")
    weights = p_fail.clamp_min(1e-6).pow(beta) * allowed
    adaptive = weights / weights.sum(-1, keepdim=True)
    uniform = allowed.float() / allowed.sum(-1, keepdim=True)
    probabilities = (1 - uniform_ratio) * adaptive + uniform_ratio * uniform
    return probabilities / probabilities.sum(-1, keepdim=True)


def marginal_motion_probabilities(probabilities, terrain_types):
    """Project conditional distributions onto the current mix of environments."""
    mix = torch.bincount(terrain_types, minlength=probabilities.shape[0]).to(probabilities)
    return (probabilities * (mix / mix.sum())[:, None]).sum(0)


def sample_motion_starts(p_fail, compatible, valid_starts, terrain_ids, *, beta, uniform_ratio,
                         fixed_start_frame=-1, families=FAMILIES):
    """Select motions adaptively, then use a fixed or uniformly drawn eligible frame.

    Keeping these two draws separate prevents clips with more valid support
    frames from gaining unintended motion-level probability. No rejection loop
    can reintroduce an incompatible motion. A matrix of rates selects the
    corresponding terrain row even for reordered or partial environment resets.
    """
    if not len(terrain_ids):
        return terrain_ids.clone(), terrain_ids.clone()
    if fixed_start_frame >= valid_starts.shape[1]:
        raise ValueError("Requested start frame is outside all loaded segments")
    available = (valid_starts[:, fixed_start_frame] if fixed_start_frame >= 0 else valid_starts.any(-1))
    groups, inverse = terrain_ids.unique(return_inverse=True)
    eligible = compatible[groups] & available[None]
    empty = ~eligible.any(-1)
    if empty.any():
        names = [families[i] for i in groups[empty].tolist()]
        raise ValueError(f"No compatible motion with a valid reset start for terrain(s): {names}")
    rates = p_fail[groups] if p_fail.ndim == 2 else p_fail
    probabilities = masked_motion_probabilities(rates, eligible, beta=beta, uniform_ratio=uniform_ratio)
    motions = torch.empty_like(terrain_ids)
    for group in range(len(groups)):
        ids = (inverse == group).nonzero().flatten()
        motions[ids] = torch.multinomial(probabilities[group], len(ids), replacement=True)
    if fixed_start_frame >= 0:
        frames = torch.full_like(motions, fixed_start_frame)
    else:
        frames = torch.multinomial(valid_starts[motions].float(), 1).squeeze(-1)
    return motions, frames


class TerrainAdaptiveSampler:
    """Motion-level EMA counters indexed by episode birth family and clip."""
    kind = "terrain_adaptive"
    enable_frame_sampling = False

    def __init__(self, families, eligible, *, alpha, beta, uniform_ratio, update_interval):
        self.families = tuple(families)
        self.eligible = eligible
        self.num_motions = eligible.shape[1]
        self.alpha, self.beta = alpha, beta
        self.uniform_ratio, self.update_interval = uniform_ratio, update_interval
        self.failed_motion_count = eligible.float()
        self.success_motion_count = self.failed_motion_count.clone()
        self.current_failed_motion_count = torch.zeros_like(self.failed_motion_count)
        self.current_success_motion_count = torch.zeros_like(self.failed_motion_count)
        self._steps_since_update = 0
        self._update_metrics()

    def record_outcomes(self, families, motions, terminated):
        pairs = families * self.num_motions + motions
        failed = terminated.to(self.failed_motion_count.dtype)
        self.current_failed_motion_count.view(-1).scatter_add_(0, pairs, failed)
        self.current_success_motion_count.view(-1).scatter_add_(0, pairs, 1 - failed)

    def step(self):
        self._steps_since_update += 1
        if self._steps_since_update == self.update_interval:
            self.failed_motion_count.mul_(1 - self.alpha).add_(self.current_failed_motion_count, alpha=self.alpha)
            self.success_motion_count.mul_(1 - self.alpha).add_(self.current_success_motion_count, alpha=self.alpha)
            self._reset_counts()
            self._steps_since_update = 0
            self._update_metrics()

    def _reset_counts(self):
        self.current_failed_motion_count.zero_()
        self.current_success_motion_count.zero_()

    def get_failure_probabilities(self):
        return self.failed_motion_count / (self.failed_motion_count + self.success_motion_count + 1e-8)

    def get_sampling_probabilities(self):
        # A motion subset used for evaluation can leave an unused family empty.
        rates = self.get_failure_probabilities()
        active = self.eligible.any(-1)
        probabilities = torch.zeros_like(rates)
        probabilities[active] = masked_motion_probabilities(rates[active], self.eligible[active],
            beta=self.beta, uniform_ratio=self.uniform_ratio)
        return probabilities

    def _update_metrics(self):
        probabilities = self.get_sampling_probabilities()
        entropy = -(probabilities * probabilities.clamp_min(1e-20).log()).sum(-1)
        active = self.eligible.any(-1)
        self._metrics = dict(
            adaptive_entropy=entropy.sum() / (active.sum().clamp_min(1) * math.log(max(2, self.num_motions))),
            adaptive_pfail_mean=self.get_failure_probabilities()[self.eligible].sum() / self.eligible.sum().clamp_min(1),
            adaptive_top1_prob=probabilities.max(), total_failed=self.failed_motion_count.sum(),
            total_success=self.success_motion_count.sum())

    def get_metrics(self):
        return self._metrics
