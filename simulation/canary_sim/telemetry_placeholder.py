"""PLACEHOLDER telemetry: true tilt + Gaussian noise. Not a sensor model; no faults."""
from __future__ import annotations

import numpy as np
import pandas as pd

PLACEHOLDER = True


def make_telemetry(df: pd.DataFrame, noise_sd_deg: float, seed: int) -> pd.DataFrame:
    """Return wide per-node telemetry (nodes only; HUB excluded)."""
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 2]))
    d = df[df.node_id != "HUB"]
    out = d[["simulation_id", "time_s", "node_id", "active_class"]].copy()
    out["tilt_x_deg"] = d.true_tilt_x_deg.to_numpy() + rng.normal(0, noise_sd_deg, len(d))
    out["tilt_y_deg"] = d.true_tilt_y_deg.to_numpy() + rng.normal(0, noise_sd_deg, len(d))
    out["placeholder_telemetry"] = True
    return out.reset_index(drop=True)
