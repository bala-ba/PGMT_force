"""Versioned, simulator-independent contract for PGMT foot force feedback."""
from copy import deepcopy
import math


def force_feedback_contract():
    return dict(version=1, feet=["left", "right"],
        fields=["force_x", "force_y", "force_z", "cop_x", "cop_y", "cop_valid"],
        frame="ankle_roll_link", force_units="N", cop_units="m", plane_z=-.035,
        min_normal_force=10., min_local_fz=10.,
        force_scale=327.07660302, cop_scale=[.15, .04],
        sampling="post_forward_control_step", reset="zero_feedback_and_reset_history")


def validate_force_feedback_contract(config=None):
    expected = force_feedback_contract()
    config = expected if config is None else deepcopy(config)
    if set(config) != set(expected):
        raise ValueError("Incomplete or unknown force feedback contract fields")
    adjustable = {"min_normal_force", "min_local_fz", "force_scale", "cop_scale"}
    for key in set(expected) - adjustable:
        if config[key] != expected[key]:
            raise ValueError(f"Unsupported force feedback contract: {key}")
    if len(config["cop_scale"]) != 2:
        raise ValueError("CoP normalization requires two scales")
    for value in [config[k] for k in adjustable - {"cop_scale"}] + list(config["cop_scale"]):
        if not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
            raise ValueError("Force feedback thresholds and scales must be finite and positive")
    return config
