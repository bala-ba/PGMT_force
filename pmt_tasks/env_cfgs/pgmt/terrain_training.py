"""Formal four-value PGMT terrain task; recovery remains disabled."""
from isaaclab.utils import configclass

from motion_tracking_rl.pgmt_value_contract import TERRAIN_VALUE_GROUPS
from pmt_tasks.env_cfgs.pgmt.stage2 import PGMTPerceptionEnvCfg
from pmt_tasks.env_cfgs.pgmt.terrain_rewards import PGMTGeometryRewardsCfg
from pmt_tasks.pgmt_tracking_relaxation import tracking_relaxation_config
from pmt_tasks.pgmt_contact_geometry import CONTACT_GEOMETRY_CONFIG
from pmt_tasks.pgmt_support_risk import SUPPORT_RISK_CONFIG
from pmt_tasks.pgmt_contact_state import CONTACT_STATE_CONFIG, TERRAIN_FOOT_SENSORS


@configclass
class PGMTTerrainTrainingEnvCfg(PGMTPerceptionEnvCfg):
    rewards: PGMTGeometryRewardsCfg = PGMTGeometryRewardsCfg()
    pgmt_reward_groups: tuple[str, ...] = TERRAIN_VALUE_GROUPS
    pgmt_tracking_relaxation: dict = tracking_relaxation_config({"enabled": True})
    pgmt_terrain_reward_mode: str = "geometry_v3"
    pgmt_geometry_contacts: dict = CONTACT_GEOMETRY_CONFIG.copy()
    pgmt_support_risk: dict = SUPPORT_RISK_CONFIG.copy()
    pgmt_contact_events: dict = dict(CONTACT_STATE_CONFIG, contact_threshold_n=1., support_threshold_n=1.)

    def __post_init__(self):
        super().__post_init__()
        if self.pgmt_terrain_reward_mode == "geometry_v3":
            for name in TERRAIN_FOOT_SENSORS:
                getattr(self.scene, name).max_contact_data_count_per_prim = self.pgmt_geometry_contacts['contact_slots']
