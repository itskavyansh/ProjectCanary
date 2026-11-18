# Report: sensor, fault and confounder models (SYNTHETIC / SIMULATION)

Frozen schema v1 (`docs/schema_v1.md`, `canary_sim/schema.py`) is untouched. Canary-Analyst is neither touched nor imported.
Telemetry is written to separate files: `telemetry_<tag>.csv` and `telemetry_<tag>_metadata.json` next to `state_timeline.csv`.
Ground-truth fault records live only in the telemetry metadata, never in the telemetry columns.
No new dependencies were added (numpy, pandas, PyYAML, matplotlib, pytest were already in `requirements.txt`).

## Values from the real noise profile (`configs/noise_profile.json`, MPU6050/ADS1115 prototype)
| Use | Derivation |
|---|---|
| `window_s = 1.0` | `sampling_interval_sec` = 1.013, rounded |
| IMU white-noise sd (tilt_x â‰ˆ 0.138Â°, tilt_y â‰ˆ 0.126Â° at multiplier 1) | mean over NODE_01..03 of `continuous_step_noise` Ã· âˆš2 (**assumption**: the step noise is the sd of the first difference) |
| Piezo rms baseline median 0.0075 | `overall.piezo_rms.median` |
| Piezo rms log-sigma â‰ˆ 0.55 | âˆšln(1+(std/mean)Â²) from `overall.piezo_rms` |
| Piezo rms AR(1) phi = 0.78 | `piezo_rms.autocorr_lag1` |
| peak/rms crest median â‰ˆ 1.72 | `piezo_peak.median / piezo_rms.median` |
| p2p/peak median â‰ˆ 1.02 | `piezo_peak_to_peak.median / piezo_peak.median` |

**NODE_04**: `continuous_step_noise` only has NODE_01..NODE_03. NODE_04 (like every node) uses the same pooled IMU white-noise sd, i.e. the mean over NODE_01..03 per field. The piezo baseline uses the pooled `overall` statistics for all nodes, plus a per-node placeholder gain (sd 0.15). Per-node bias and drift are placeholders. Nothing in the model is NODE_04-specific from the profile.

Not used: the per-node `nodes.*` statistics. They mix real mounting tilt (NODE_03 â‰ˆ 15Â°) with noise, and NODE_04 duplicates the pooled values. The HC-SR04 `distance` statistics are not used because the hub is LiDAR.

## Placeholders (invented, not calibrated)
- **LiDAR** (`configs/sensor_default.yaml`): `sweep_period_s=20` (one directional pass), `range_noise_sd_m=0.03`, `quantisation_m=0.01`, `max_range_m=40`, `dwell_s=1.5`.
- **IMU**: `node_bias_sd_deg=0.05`, `drift_sd_deg_per_sqrt_s=0.002`.
- **Piezo**: node gain sd 0.15, crest scatter 0.15, p2p scatter 0.10, impulse rate 0.01/s, impulse amplitude median 0.03 (sigma_ln 0.5), impulse rms fraction 0.35.
- **Comms**: node loss 0.01, hub loss 0.005, latency median 0.4 s (sigma_ln 0.4).
- **Faults** (`DEFAULT_PARAMS` in `faults.py`):
  - IMU bias step 0.5Â°, IMU drift 0.01Â°/s, IMU spike 3Â°.
  - Piezo bias 0.02, piezo drift 0.0005/s, piezo spike 0.2.
  - 3 spikes, 3 drops, burst drop probability 0.8, added latency 5 s.
  - Disconnect `p_off=0.15`, `p_on=0.3`.
  - Implausible tilt 120â€“175Â°, implausible piezo 5.0 (above the ADS1115 4.096 V range).
  - Wrong-dwell offset 0.5 of the way to the next corner.
  - Hub shift 0.30 m with a 3 s ramp.
- **Confounders** (`configs/confounder_default.yaml`): every value, including the `abnormal_reference` block that stands in for the future abnormal-vibration class.
- **User-specified**: noise multiplier 0.5â€“2Ã— (log-uniform, one per run, scales IMU white noise and drift step only); sigma cap 1.5, broad-basin limit 2.5.

## Assumptions
1. **Sigma cap**: it applies to `sx`, the sampled config sigma. `sy = sx Ã— aspect` (up to 2) can still reach 3.0 radii. If you want the cap on max(sx, sy), say so.
2. **Regenerated outputs**: `outputs/sim_gradual_20260001/` was regenerated with `scripts/run_one.py --seed 20260001` after the cap fix, and the demo outputs were regenerated too. The draw order changed, so every seed now samples different parameters than before.
3. **Sigma cap enforcement**: `engine.py` raises if the main `sigma_radii` upper bound is above 1.5. The flag is `ranges.broad_basin_subset` (default false), with `ranges.sigma_radii_broad_max = 2.5`.
4. **LiDAR schedule**: one pass is one sweep. Passes alternate clockwise (N1â†’N4â†’N3â†’N2) and anticlockwise, giving one reading per node per pass. Each reading is stamped at the end of its dwell. It is attached to the window containing that time.
5. **LiDAR range**: 3-D hub-to-node distance from `true_x`, `true_y`, `true_z`. Range change is relative to each node's median reading before the event starts. This baseline is computed from received data, so an early fault can contaminate it. Absolute range is not reported.
6. **Timing**: sensor and sweep times are in simulated seconds and are not rescaled by the compression factor (360Ã—).
7. **Packets**: a node packet carries IMU and piezo. LiDAR readings travel on a separate hub link. A node outage does not remove that node's range, and a network outage removes everything. A lost packet is a row with NaN values. The row stays in the file with `packet_received = 0`.
8. **Latency**: tracked per node packet only, as `arrival_time_s`. The hub link has no latency.
9. **Faults**:
   - `motor_stall`: the hub keeps ranging on schedule but every reading is taken toward the stalled node. Readings are still labelled by the schedule.
   - `stale_readings`: re-sends each node's last reading.
   - `intermittent_disconnect`: gives NaN sensor values, with the packet still arriving. This is distinct from packet loss.
   - `hub_self_motion`: an additive common-mode shift via the hub-motion hook (`sensors.add_hub_motion`). It is not a geometric hub translation. It returns to zero after the fault ends.
10. **Fault plans**: `sample_fault_plan` draws one node, or two with probability 0.2.
11. **Faults on the same channel**: faults are applied in start-time order and compose. For example, `stale_readings` freezes whatever earlier faults produced.
12. **Confounders**: temperature drift is common-mode on all 4 nodes because temperature is. It is limited in amplitude only. Wind touches at most 2 nodes and a vehicle touches 1. `check_margins` enforces a â‰¥3Ã— amplitude and duration margin and strictly fewer nodes than the abnormal class. Node spread cannot reach 3Ã— with only 4 nodes, so I used "strictly fewer".
13. **Piezo**: no ADC quantisation is applied, because the real profile values are not LSB multiples. Piezo does not respond to settlement. Abnormal vibration is a later class.
