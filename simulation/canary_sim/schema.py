"""FROZEN state-timeline schema v1 (see docs/schema_v1.md).

DO NOT change columns, names, dtypes or units without explicit approval.
"""
from __future__ import annotations

from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

SCHEMA_VERSION = "1.0"
SCHEMA_FROZEN = True

COLUMNS = (
    "simulation_id", "time_s", "node_id", "true_x", "true_y", "true_z",
    "true_settlement", "true_tilt_x_deg", "true_tilt_y_deg",
    "active_class", "fault_flags",
)

ActiveClass = Literal["normal", "gradual", "rapid", "aftermath"]
NodeId = Literal["NODE_01", "NODE_02", "NODE_03", "NODE_04", "HUB"]


class StateRow(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    simulation_id: str = Field(min_length=1)
    time_s: float = Field(ge=0)
    node_id: NodeId
    true_x: float
    true_y: float
    true_z: float
    true_settlement: float        # w(x,y,t); negative = downward
    true_tilt_x_deg: float
    true_tilt_y_deg: float
    active_class: ActiveClass
    fault_flags: int = Field(ge=0)  # bitmask; 0 = no fault (no fault models in v1)


_ADAPTER = TypeAdapter(list[StateRow])


def validate_frame(df: pd.DataFrame) -> int:
    """Validate a timeline DataFrame; returns row count. Raises on any violation."""
    if tuple(df.columns) != COLUMNS:
        raise ValueError(f"columns must be exactly {COLUMNS}, got {tuple(df.columns)}")
    _ADAPTER.validate_python(df.to_dict("records"))
    if df[list(COLUMNS[1:2] + COLUMNS[3:9])].isna().any().any():
        raise ValueError("NaN in numeric columns")
    return len(df)
