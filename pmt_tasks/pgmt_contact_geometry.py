"""Versioned terrain-support state and T5 event cost (no Isaac dependency)."""
import math
import torch


CONTACT_GEOMETRY_CONFIG = dict(force_on_n=1.0, force_off_n=0.5,
                               minimum_state_s=0.1, reference_window_s=0.04,
                               contact_slots=64)


def configure_contact_switching(cfg, mode):
    """Override T5 alone, preserving T1 and the shared T2/T3 contact thresholds."""
    from pmt_tasks.pgmt_contact_switching import CONTACT_SWITCHING_WEIGHT

    if mode not in ('legacy', 'geometry_v3'):
        raise ValueError(f'Unknown contact switching mode: {mode}')
    term = getattr(cfg.rewards, 'terrain_contact_switching', None)
    if not getattr(cfg, 'pgmt_contact_switching', False) or term is None:
        raise ValueError('Contact switching override requires an enabled T5 reward')
    previous_mode = getattr(cfg, 'pgmt_contact_switching_mode', None) or getattr(
        cfg, 'pgmt_terrain_reward_mode', 'legacy')
    previous_weight = term.weight
    cfg.pgmt_contact_switching_mode = mode
    term.weight = -0.05 if mode == 'geometry_v3' else CONTACT_SWITCHING_WEIGHT
    if mode == 'geometry_v3':
        cfg.pgmt_geometry_contacts = dict(getattr(cfg, 'pgmt_geometry_contacts', CONTACT_GEOMETRY_CONFIG))
    return dict(term='terrain_contact_switching', previous_mode=previous_mode,
                mode=mode, previous_weight=previous_weight, weight=term.weight,
                config=dict(cfg.pgmt_geometry_contacts) if mode == 'geometry_v3' else None)


class TerrainSupportState:
    """One sample per control step; unknown samples break interval history.

    A first observation establishes a baseline, not a transition. The first
    subsequent transition starts a known interval but cannot score the unknown
    interval before it. Reference event identities are consumed once per episode.
    """
    def __init__(self, num_envs, *, device, dt, config=None):
        self.config = dict(CONTACT_GEOMETRY_CONFIG if config is None else config)
        c = self.config
        if (set(c) != set(CONTACT_GEOMETRY_CONFIG) or not math.isfinite(dt) or dt <= 0
                or not all(math.isfinite(v) for v in c.values())
                or not 0 <= c['force_off_n'] < c['force_on_n']
                or c['minimum_state_s'] <= 0 or c['reference_window_s'] < 0
                or type(c['contact_slots']) is not int or c['contact_slots'] < 3):
            raise ValueError('Invalid geometry contact configuration')
        self.dt = dt
        self.state = torch.zeros(num_envs, 2, dtype=torch.bool, device=device)
        self.valid = torch.zeros_like(self.state)
        self.last_switch = torch.full_like(self.state, -1, dtype=torch.long)
        self.last_step = None
        # At most one edge per frame; retain more identities than the whole window.
        self.window = int(math.floor(c['reference_window_s'] / dt + 1e-7))
        self.used = torch.full((num_envs, 2, 2*self.window+3), -1, dtype=torch.long, device=device)
        self.cursor = torch.zeros(num_envs, 2, dtype=torch.long, device=device)
        self.snapshot = None

    def reset(self, ids):
        self.state[ids] = False
        self.valid[ids] = False
        self.last_switch[ids] = -1
        self.used[ids] = -1
        self.cursor[ids] = 0

    @torch.no_grad()
    def update(self, step, mean_force_z, *, reference_ids=None, reference_states=None,
               reference_valid=None, reference_distance=None):
        if self.last_step == step:
            return self.snapshot
        if self.last_step is not None and step <= self.last_step:
            raise ValueError('Support samples must advance in time')
        valid = torch.isfinite(mean_force_z)
        previous_valid = self.valid & (self.last_step is not None and step == self.last_step + 1)
        c = self.config
        new = torch.where(mean_force_z >= c['force_on_n'], True,
                          torch.where(mean_force_z <= c['force_off_n'], False, self.state))
        # Unknown force cannot manufacture flight or a switch.
        new = torch.where(valid, new, self.state)
        event = valid & previous_valid & (new != self.state)
        history = event & (self.last_switch >= 0)
        duration = (step - self.last_switch).to(mean_force_z.dtype) * self.dt
        q = torch.where(history, (1 - duration / c['minimum_state_s']).clamp(0, 1), 0.)
        q = torch.where(duration >= c['minimum_state_s']-1e-7, 0., q)
        exempt = torch.zeros_like(event)
        if reference_ids is not None:
            # Candidate tensors [N,2,W]; identity is packed motion-offset+frame.
            unused = ~(reference_ids[..., None] == self.used[..., None, :]).any(-1)
            eligible = reference_valid & unused & (reference_states == new[..., None]) & event[..., None]
            distance = reference_distance.expand_as(reference_ids).abs()
            eligible &= distance <= self.window
            choice = torch.where(eligible, distance, self.window+1).argmin(-1)
            exempt = eligible.gather(-1, choice[..., None]).squeeze(-1)
            identity = reference_ids.gather(-1, choice[..., None]).squeeze(-1)
            slot = self.cursor.remainder(self.used.shape[-1])
            old = self.used.gather(-1, slot[..., None]).squeeze(-1)
            self.used.scatter_(-1, slot[..., None], torch.where(exempt, identity, old)[..., None])
            self.cursor += exempt.long()
        per_foot = q * (~exempt)
        self.last_switch = torch.where(event, step, self.last_switch)
        self.last_switch = torch.where(valid & previous_valid, self.last_switch,
                                       torch.full_like(self.last_switch, -1))
        self.state, self.valid = new, valid
        self.last_step = step
        self.snapshot = dict(support=new.clone(), valid=valid.clone(), event=event,
                             history_valid=history, duration_s=torch.where(history, duration, float('nan')),
                             soft_cost=q, reference_exempt=exempt, per_foot=per_foot,
                             mean_force_z=mean_force_z.clone(), rate=per_foot.sum(-1) / self.dt)
        return self.snapshot


def reference_switch_candidates(store, motion_ids, frame_ids, window):
    offsets = torch.arange(-window, window+1, device=frame_ids.device)
    frames = frame_ids[:, None] + offsets
    motions = motion_ids[:, None].expand_as(frames)
    now, before = store.gather(motions, frames), store.gather(motions, frames-1)
    valid = now['contact_valid'] & before['contact_valid'] & (now['contacts'] != before['contacts'])
    identity = store.offsets[motions] + frames
    return dict(reference_ids=identity[:, None].expand(-1, 2, -1),
                reference_states=now['contacts'].bool().transpose(1, 2),
                reference_valid=valid.transpose(1, 2), reference_distance=offsets[None, None])


def pack_contact_points(data, num_envs, slots):
    """Preserve raw PhysX manifold points, never the per-collider mean.

    Count/start buffers index one global point buffer. Overflow invalidates the
    contact-plane branch so a truncated cloud cannot masquerade as complete.
    """
    forces, points, normals, distances, counts, starts = data
    counts, starts = counts.reshape(num_envs, -1).long(), starts.reshape(num_envs, -1).long()
    ends = counts.cumsum(-1)
    slot = torch.arange(slots, device=counts.device)[None].expand(num_envs, -1).contiguous()
    column = torch.searchsorted(ends.contiguous(), slot, right=True).clamp_max(counts.shape[-1]-1)
    prior = (ends - counts).gather(-1, column)
    index = starts.gather(-1, column) + slot - prior
    total = ends[:, -1]
    valid = (slot < total[:, None]) & (index >= 0) & (index < len(points))
    index = index.clamp(0, len(points)-1)
    cloud = points[index]
    valid &= torch.isfinite(cloud).all(-1) & torch.isfinite(forces[index, 0]) & (forces[index, 0] > 0)
    overflow = (total > slots) | ((starts + counts > len(points)) & (counts > 0)).any(-1)
    overflow |= total.sum() >= len(points)
    return cloud, valid, overflow
