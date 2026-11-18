"""Validated packed PMT clips and checkpointable temporal-bin sampling."""
from pathlib import Path
from types import SimpleNamespace
import hashlib

import numpy as np
import torch
import torch.nn.functional as F

from scripts.pmt_npz_to_mjlab import BFS_BODIES, BFS_JOINTS

FIELDS = dict(joint_pos="joint_pos", joint_vel="joint_vel", body_pos="body_pos_w",
              body_quat="body_quat_w", body_lin_vel="body_lin_vel_w", body_ang_vel="body_ang_vel_w")


class MotionLibrary:
    """Pack frames without padding every clip to the longest motion.

    Unnamed files follow PMT's documented BFS convention; named files are
    explicitly remapped. Future indices clamp inside the selected segment.
    """
    def __init__(self, paths, *, device, dt, chunk_length=0, contact_labels=False):
        if not np.isfinite(dt) or dt <= 0 or chunk_length < 0:
            raise ValueError("Control dt must be positive and chunk length nonnegative")
        if isinstance(paths, (str, Path)):
            paths = [paths]
        files = set()
        for path in paths:
            path = Path(path).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(path)
            files.update(path.rglob("*.npz") if path.is_dir() else [path])
        if not files:
            raise ValueError("No motion NPZ files found")
        arrays = {key: [] for key in FIELDS}
        lengths, records, labels = [], [], []
        self.files = [str(path) for path in sorted(files)]
        self.file_hashes = []
        for path in sorted(files):
            self.file_hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
            with np.load(path, allow_pickle=False) as data:
                fps = float(np.asarray(data["fps"]).reshape(-1)[0])
                if not np.isclose(fps * dt, 1., atol=1e-6, rtol=0):
                    raise ValueError(f"{path}: motion fps {fps} != control rate {1 / dt}")
                n = len(data["joint_pos"])
                if n < 2:
                    raise ValueError(f"{path}: a motion needs at least two frames")
                permutations = {}
                for kind, expected in (("joint", BFS_JOINTS), ("body", BFS_BODIES)):
                    names = data.get(kind + "_names", np.asarray(expected)).tolist()
                    names = [v.decode() if isinstance(v, bytes) else v for v in names]
                    if len(names) != len(expected) or set(names) != set(expected):
                        raise ValueError(f"{path}: incompatible {kind} names")
                    permutations[kind] = [names.index(name) for name in expected]
                clip = {}
                for key, source in FIELDS.items():
                    value = np.asarray(data[source], dtype=np.float32)
                    shape = (n, 29) if key.startswith("joint") else (n, 30, 4 if key == "body_quat" else 3)
                    if value.shape != shape or not np.isfinite(value).all():
                        raise ValueError(f"{path}: invalid {source}; expected finite {shape}")
                    clip[key] = value[:, permutations["joint" if key.startswith("joint") else "body"]]
                if not np.allclose(np.linalg.norm(clip["body_quat"], axis=-1), 1., atol=2e-3):
                    raise ValueError(f"{path}: body quaternions must be unit wxyz")
                for start in range(0, n, chunk_length or n):
                    stop = min(n, start + (chunk_length or n))
                    if stop - start < 2:
                        continue
                    records.append(dict(file=str(path), start=start, stop=stop))
                    lengths.append(stop - start)
                    if contact_labels:
                        required = {"contacts", "contact_valid", "foot_names"}
                        if not required <= set(data.files):
                            raise ValueError(f"{path}: terrain rewards require contact metadata {sorted(required)}")
                        labels.append(SimpleNamespace(source_file=str(path), num_frames=stop-start,
                            metadata={key: np.asarray(data[key])[start:stop] if key != "foot_names"
                                      else np.asarray(data[key]) for key in required}))
                    for key in arrays:
                        arrays[key].append(clip[key][start:stop])
        self.identity = dict(files=self.files, sha256=self.file_hashes, segments=records,
                             order="pmt_bfs", dt=dt)
        self.records = records
        self.motion_lengths = torch.tensor(lengths, dtype=torch.long, device=device)
        self.offsets = self.motion_lengths.cumsum(0) - self.motion_lengths
        self.arrays = {key: torch.as_tensor(np.concatenate(value), device=device) for key, value in arrays.items()}
        self.num_motions = len(lengths)
        self.contact_labels = None
        if contact_labels:
            from pmt_tasks.pgmt_contact_reference import ContactLabelStore
            self.contact_labels = ContactLabelStore(labels, device=device)

    def gather(self, motion_ids, frames):
        frames = torch.minimum(frames.clamp_min(0), self.motion_lengths[motion_ids] - 1)
        indices = self.offsets[motion_ids] + frames
        return {key: value[indices] for key, value in self.arrays.items()}


class BinSampler:
    """PMT's motion -> smoothed temporal-bin -> uniform-frame sampling.

    Counts belong to episode START bins, and EMA advances once per control
    step. The last frame is excluded so every reset has a real transition.
    """
    def __init__(self, lengths, *, fps, config=None):
        self.config = dict(bin_duration=1., beta=.5, alpha=.001, uniform_ratio=.1,
                           update_interval=240, kernel_size=5, kernel_lambda=.8)
        self.config.update(config or {})
        self.lengths = lengths
        self.bin_size = max(1, round(fps * self.config["bin_duration"]))
        counts = (lengths - 1 + self.bin_size - 1) // self.bin_size
        self.valid = torch.arange(int(counts.max()), device=lengths.device)[None] < counts[:, None]
        self.failed = torch.ones_like(self.valid, dtype=torch.float32)
        self.succeeded = torch.ones_like(self.failed)
        self.pending_failed = torch.zeros_like(self.failed)
        self.pending_succeeded = torch.zeros_like(self.failed)
        self.steps = 0

    def probabilities(self):
        c = self.config
        rates = self.failed / (self.failed + self.succeeded).clamp_min(1e-8)
        rates = rates * self.valid
        if c["kernel_size"] > 1:
            k = c["kernel_lambda"] ** torch.arange(c["kernel_size"] - 1, -1, -1,
                                                    device=rates.device, dtype=rates.dtype)
            rates = F.conv1d(F.pad(rates[:, None], (len(k) - 1, 0)), (k / k.sum())[None, None])[:, 0]
        rates = rates * self.valid
        score = rates.max(-1).values.clamp_min(1e-6).pow(c["beta"])
        motion = (1 - c["uniform_ratio"]) * score / score.sum() + c["uniform_ratio"] / len(score)
        bins = rates.clamp_min(1e-6).pow(c["beta"]) * self.valid
        bins = ((1 - c["uniform_ratio"]) * bins / bins.sum(-1, keepdim=True)
                + c["uniform_ratio"] * self.valid / self.valid.sum(-1, keepdim=True))
        return motion, bins

    def sample(self, count):
        motion, bins = self.probabilities()
        ids = torch.multinomial(motion, count, replacement=True)
        starts = torch.multinomial(bins[ids], 1).flatten() * self.bin_size
        width = (self.lengths[ids] - 1 - starts).clamp(1, self.bin_size)
        frames = starts + (torch.rand(count, device=ids.device) * width).long()
        return ids, frames

    def update(self, motion_ids, start_frames, failure):
        indices = motion_ids * self.failed.shape[1] + start_frames // self.bin_size
        self.pending_failed.flatten().scatter_add_(0, indices, failure.float())
        self.pending_succeeded.flatten().scatter_add_(0, indices, (~failure).float())

    def step(self):
        self.steps += 1
        if self.steps >= self.config["update_interval"]:
            a = self.config["alpha"]
            self.failed.lerp_(self.pending_failed, a)
            self.succeeded.lerp_(self.pending_succeeded, a)
            self.pending_failed.zero_()
            self.pending_succeeded.zero_()
            self.steps = 0

    def state_dict(self):
        return dict(kind="mjlab_bin_adaptive", config=dict(self.config), steps=self.steps,
                    **{key: getattr(self, key).cpu().clone() for key in
                       ("failed", "succeeded", "pending_failed", "pending_succeeded")})

    def validate_state_dict(self, state):
        if state.get("kind") != "mjlab_bin_adaptive" or state.get("config") != self.config:
            raise ValueError("Bin sampler configuration differs")
        if type(state.get("steps")) is not int or not 0 <= state["steps"] < self.config["update_interval"]:
            raise ValueError("Invalid bin sampler update clock")
        for key in ("failed", "succeeded", "pending_failed", "pending_succeeded"):
            v = state.get(key)
            if (not isinstance(v, torch.Tensor) or v.shape != self.failed.shape or v.dtype != self.failed.dtype
                    or not torch.isfinite(v).all() or (v < 0).any()):
                raise ValueError(f"Invalid sampler counts: {key}")

    def load_state_dict(self, state):
        self.validate_state_dict(state)
        for key in ("failed", "succeeded", "pending_failed", "pending_succeeded"):
            getattr(self, key).copy_(state[key])
        self.steps = state["steps"]
