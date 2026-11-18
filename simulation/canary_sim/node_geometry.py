"""Node geometry and sign conventions (SYNTHETIC / SIMULATION).

FRAME
    Right-handed local frame, metres: x = east, y = north, z = up.
    Origin at NODE_01.

LAYOUT
    NODE_01 (0,0)   NODE_02 (20,0)   NODE_03 (20,20)   NODE_04 (0,20)
    HUB (10,10): LiDAR hub. The HUB is NOT a field-following reference; the four
    corner nodes are its reference. HUB rows in the timeline are ground truth of
    the surface at the hub only.
    NETWORK_RADIUS = distance from network centre (10,10) to a corner = 10*sqrt(2).
    All "radii" in the config are multiples of this length.

TILT CONVENTIONS (right-hand rule, angles in degrees)
    Surface height is z = w(x, y, t) (negative w = settlement / sinking).
    Slopes are taken in the NODE's own sensor frame, which is the world frame
    rotated CCW (seen from above) by the node's mounting yaw psi:
        [gx_l, gy_l] = Rot(-psi) @ [dw/dx, dw/dy]
    true_tilt_x_deg = +atan(gy_l)   rotation about the sensor +x axis;
                                    positive => sensor +y side is HIGHER.
    true_tilt_y_deg = -atan(gx_l)   rotation about the sensor +y axis;
                                    positive => sensor +x side is LOWER.
    Both are right-handed rotations (thumb along the axis, fingers curl positive).
    Mounting yaw psi: CCW positive, per-node uniform in [0, 360) deg, seeded.

MISALIGNMENT (seeded, per node, constant over the run)
    * yaw_error_deg: extra yaw added to the nominal mounting yaw (Gaussian).
    * tilt_x_offset_deg / tilt_y_offset_deg: static sensor-level offsets added
      to the tilt (Gaussian, small). They are part of what the sensor truly
      sees, so they ARE included in true_tilt_*.
    The effective yaw used for the frame rotation is psi + yaw_error.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

SIDE_M = 20.0
NODE_IDS = ("NODE_01", "NODE_02", "NODE_03", "NODE_04")
HUB_ID = "HUB"
NODE_XY: dict[str, tuple[float, float]] = {
    "NODE_01": (0.0, 0.0),
    "NODE_02": (SIDE_M, 0.0),
    "NODE_03": (SIDE_M, SIDE_M),
    "NODE_04": (0.0, SIDE_M),
}
HUB_XY = (SIDE_M / 2, SIDE_M / 2)
NETWORK_CENTRE = HUB_XY
NETWORK_RADIUS_M = float(np.hypot(SIDE_M / 2, SIDE_M / 2))  # centre -> corner


@dataclass(frozen=True)
class Mounting:
    node_id: str
    yaw_deg: float          # nominal mounting yaw
    yaw_error_deg: float    # misalignment added to yaw
    tilt_x_offset_deg: float
    tilt_y_offset_deg: float

    @property
    def effective_yaw_rad(self) -> float:
        return float(np.deg2rad(self.yaw_deg + self.yaw_error_deg))

    def to_dict(self) -> dict:
        return asdict(self)


def sample_mountings(
    seed: int, yaw_error_sd_deg: float = 1.0, level_offset_sd_deg: float = 0.02
) -> dict[str, Mounting]:
    """Seeded per-node mounting. Independent stream from scenario parameters."""
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 1]))
    out = {}
    for nid in NODE_IDS:
        out[nid] = Mounting(
            node_id=nid,
            yaw_deg=float(rng.uniform(0.0, 360.0)),
            yaw_error_deg=float(rng.normal(0.0, yaw_error_sd_deg)),
            tilt_x_offset_deg=float(rng.normal(0.0, level_offset_sd_deg)),
            tilt_y_offset_deg=float(rng.normal(0.0, level_offset_sd_deg)),
        )
    return out


def hub_mounting() -> Mounting:
    """HUB: ideal, yaw 0, no misalignment (truth reference only)."""
    return Mounting(HUB_ID, 0.0, 0.0, 0.0, 0.0)


def world_to_sensor_slopes(gx, gy, yaw_rad):
    """Rotate world slopes (dw/dx, dw/dy) into the sensor frame."""
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    gx_l = c * gx + s * gy
    gy_l = -s * gx + c * gy
    return gx_l, gy_l


def slopes_to_tilt_deg(gx_l, gy_l):
    """Sensor-frame slopes -> (tilt_x_deg, tilt_y_deg) per the convention above."""
    return np.rad2deg(np.arctan(gy_l)), -np.rad2deg(np.arctan(gx_l))
