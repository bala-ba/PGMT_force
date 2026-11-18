"""T2: continuous, validity-masked match to the same-frame reference contacts.

Actual contact uses the all-object normal-force norm. Its threshold defaults
to the shared state, with an optional T2-only override that leaves T3 intact.
Only RewardManager applies the +1.5 weight and the control dt.
"""
import math
import torch


REFERENCE_CONTACT_MATCH_WEIGHT = 1.5


def validate_reference_contact_threshold(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError('Reference contact threshold must be finite and positive')


def configure_t1_t2_contact_threshold(cfg, threshold_n):
    """Change T1 terrain support and T2 contact thresholds without changing T3."""
    from pmt_tasks.pgmt_contact_state import validate_contact_config

    validate_reference_contact_threshold(threshold_n)
    contacts = getattr(cfg, 'pgmt_contact_events', None)
    if (contacts is None or getattr(cfg, 'pgmt_touchdown_quality', None) is None
            or not getattr(cfg, 'pgmt_reference_contact_match', False)):
        raise ValueError('T1/T2 threshold override requires enabled T1, T2 and shared contact state')
    if (getattr(cfg, 'pgmt_terrain_reward_mode', 'legacy') == 'geometry_v3'
            and cfg.pgmt_touchdown_quality['settlement'] != 'touchdown_event_v1'):
        raise ValueError('T1 threshold override requires variance T1; select --touchdown-event or legacy terrain rewards')
    updated = dict(contacts, support_threshold_n=float(threshold_n))
    validate_contact_config(updated)
    previous_t2 = getattr(cfg, 'pgmt_reference_contact_threshold_n', None)
    report = dict(terms=['terrain_touchdown_quality', 'terrain_reference_contact_match'],
                  previous_support_threshold_n=contacts['support_threshold_n'],
                  previous_reference_contact_threshold_n=(contacts['contact_threshold_n'] if previous_t2 is None else previous_t2),
                  support_threshold_n=float(threshold_n), reference_contact_threshold_n=float(threshold_n),
                  shared_contact_threshold_n=contacts['contact_threshold_n'], comparison='strictly_greater')
    cfg.pgmt_contact_events = updated
    cfg.pgmt_reference_contact_threshold_n = float(threshold_n)
    return report


def empty_contact_match(num_envs, *, device):
    result = {key: torch.zeros(num_envs, 2, dtype=torch.bool, device=device) for key in (
        "actual_contact", "actual_valid", "reference_contact", "reference_valid", "compared_valid", "matched")}
    result["reference_count"] = torch.zeros(num_envs, dtype=torch.long, device=device)
    result["rate"] = torch.zeros(num_envs, device=device)
    return result


@torch.no_grad()
def evaluate_reference_contact_match(contact, contact_valid, reference_contact, reference_valid):
    """Return [N,2] diagnostics and an unweighted [N] rate in [0,1].

    The denominator counts valid reference labels, as in the recorded formula.
    An unknown physical measurement cannot earn a match or shrink this
    denominator; it remains explicitly unknown in compared_valid. Invalid
    reference labels are excluded entirely, never interpreted as swing.
    """
    inputs = (contact, contact_valid, reference_contact, reference_valid)
    if (contact.ndim != 2 or contact.shape[1] != 2
            or any(value.shape != contact.shape or value.device != contact.device for value in inputs)
            or any(value.dtype != torch.bool for value in (contact, contact_valid, reference_valid))):
        raise ValueError("Contact match requires paired [N,2] states/labels and boolean masks on one device")
    # The label store validates binary metadata at load time. Keep the kernel
    # finite even if a diagnostic caller supplies NaN/Inf/unknown state codes.
    ref_valid = reference_valid & ((reference_contact == 0) | (reference_contact == 1))
    ref = reference_contact == 1
    valid = contact_valid & ref_valid
    matched = valid & (contact == ref)
    count = ref_valid.sum(-1)
    rate = matched.sum(-1).float() / count.clamp_min(1)
    return dict(actual_contact=contact.clone(), actual_valid=contact_valid.clone(),
                reference_contact=ref, reference_valid=ref_valid, compared_valid=valid,
                matched=matched, reference_count=count, rate=rate)
