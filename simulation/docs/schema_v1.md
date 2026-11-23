# Schema v1 — state timeline (**FROZEN**)

Status: **FROZEN** (`canary_sim/schema.SCHEMA_FROZEN = True`, version `1.0`).
Any change to columns, names, dtypes, units or enum values requires explicit approval first.

One file per run: `outputs/<simulation_id>/state_timeline.csv` (CSV; parquet not used because pyarrow is not installed),
plus `metadata.json` (all sampled parameters, mountings, full config, compression factor).
Rows are sorted by `time_s`, then `node_id`. Rows exist for NODE_01..NODE_04 and HUB.

| column | type | unit | meaning |
|---|---|---|---|
| simulation_id | str | – | `sim_<mode>_<seed>` |
| time_s | float ≥ 0 | s (simulated) | time since run start |
| node_id | enum | – | NODE_01..NODE_04, HUB |
| true_x, true_y | float | m | node position, local frame (no horizontal displacement modelled in v1) |
| true_z | float | m | nominal z0 = 0 plus w (equals true_settlement in v1) |
| true_settlement | float | m | w(x,y,t); negative = downward |
| true_tilt_x_deg | float | deg | rotation about sensor +x (right-hand): + ⇒ sensor +y side higher |
| true_tilt_y_deg | float | deg | rotation about sensor +y (right-hand): + ⇒ sensor +x side lower |
| active_class | enum | – | `normal`, `gradual`, `rapid` (during the ramp t_on..t_on+tau), `aftermath` (after ramp) |
| fault_flags | int ≥ 0 | bitmask | 0 = none; no fault models in v1 |

Validation: `canary_sim.schema.validate_frame(df)` (pydantic `StateRow`, `extra="forbid"`, exact column order).
