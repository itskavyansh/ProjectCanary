"""Windowing (SYNTHETIC / SIMULATION). Config-driven: window_s (default 5 s) and overlap (default 0.5)."""
from __future__ import annotations

import numpy as np


def make_windows(duration_s: float, window_s: float = 5.0, overlap: float = 0.5) -> np.ndarray:
    """Return array (n, 2) of [t_start, t_end) windows fully inside [0, duration_s]."""
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1)")
    step = window_s * (1.0 - overlap)
    n = int(np.floor((duration_s - window_s) / step + 1e-9)) + 1
    t0 = np.arange(n) * step
    return np.column_stack([t0, t0 + window_s])


def sample_slice(t: np.ndarray, t0: float, t1: float) -> slice:
    """Rows of a sorted time vector with t0 <= t < t1 (tolerant to float error)."""
    return slice(int(np.searchsorted(t, t0 - 1e-9)), int(np.searchsorted(t, t1 - 1e-9)))
