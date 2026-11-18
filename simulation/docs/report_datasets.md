# Report: 6-Class Definitions, Windowing, Features, Label Policy & Dataset Generation (SYNTHETIC / SIMULATION)

Status of frozen schema: `docs/schema_v1.md` (`canary_sim/schema.py`) is **FROZEN** and was not modified.
`Canary-Analyst` was neither touched nor imported.
No models or baselines were trained (`scripts/build_datasets.py` generates datasets, labels, features, metadata, and diagnostic plots only).

---

## 1. Six Classes & Run Structure (`canary_sim/classes.py`)
Fixed class names: `normal`, `gradual`, `rapid`, `abnormal_vibration`, `sensor_fault`, `comm_fault`.
Class labels live in separate files (`window_labels.csv`, `labels_<split>.csv`) derived from the state timeline plus fault/event metadata, never written into `state_timeline.csv`.

- **`normal`**: Null settlement field (`null_field=True` in `engine.run_scenario`, keeping `s(t)=0` and `active_class="normal"`) plus benign confounders (`temperature_drift`, `vibration_burst` from wind/vehicles, and `harmless_spikes`) injected into telemetry as hard negatives.
- **`gradual` & `rapid`**: Gaussian settlement basin runs (`t_on`, `tau`, `s_max`), stratified across `peak_tilt_deg` in `[0.05, 3.0]` deg.
- **`abnormal_vibration`**: Null settlement field plus sustained multi-node vibration (`>= 3` nodes, duration `35–90 s`, added piezo RMS `0.09–0.22`, tilt jitter `0.16–0.35°`, and repeated impulse spikes `0.22–0.55`). Checked by `check_abnormal_vibration_margins` against `configs/confounder_default.yaml` (`abnormal_reference`) so amplitude and duration are at least `3×` larger than benign confounders and node spread is strictly wider (`>= 3` vs `<= 2`).
- **`sensor_fault`**: Null settlement field plus one IMU or hub fault (`bias_step`, `slow_drift`, `stuck_at`, `isolated_spikes`, `intermittent_disconnect`, `implausible_reading`, `motor_stall`, `wrong_dwell_position`, `stale_readings`, `hub_self_motion`), cycled evenly across runs with normal lead-in, active fault, and normal aftermath.
- **`comm_fault`**: Null settlement field plus one communication fault (`packet_drops`, `burst_loss`, `added_latency`, `node_unavailable`, `network_outage`), cycled evenly across runs with normal lead-in, active fault, and normal aftermath.
- **Combined-condition stress set (`stress`)**: 100 runs combining two conditions (e.g. `gradual + sensor_fault`, `abnormal_vibration + comm_fault`), stored and reported separately.

---

## 2. Windowing & Label Policy (`canary_sim/windows.py`, `canary_sim/labels.py`)
- **Windowing**: `window_s = 5.0` s simulated, `overlap = 0.5` (step `2.5` s), yielding `95` windows per `240` s run.
- **50% Overlap Rule**: A window `[t_start, t_end)` of length $L=5\text{ s}$ receives an event or fault class label if its overlap with that condition's active interval is $\ge 0.5 L$ (`min_label_overlap = 0.5`).
- **Aftermath Policy**:
  - For `gradual` and `rapid` basin runs, the active interval runs from `t_on` to the end of the run (`240.0` s) because the ground remains permanently tilted in the aftermath.
  - For transient conditions (`abnormal_vibration`, `sensor_fault`, `comm_fault`), the active interval is `[start_s, end_s)`; windows after `end_s` return to `normal`.
- **Transition Windows**: Windows straddling an onset or end boundary with $0 < \text{overlap} < 0.5 L$ receive label `"transition"`. They are excluded from training/scoring (`labels.trainable(df)`), but kept in the tables with `t_rel_event_start_s` (`t_mid - first_event_start_s`) so detection delay can be computed later.

---

## 3. Windowed Features & Explicit NaN Handling (`canary_sim/features.py`)
Every window produces **63 model features** (`feature_columns(df)`) and **5 diagnostic columns** (`diagnostic_columns(df)`, prefixed `diag_`):

| Feature Group | Columns | Missing / NaN Handling |
|---|---|---|
| **1. Magnitude (existing)** | `{NODE}_mean`, `{NODE}_std`, `{NODE}_slope`, `coherence` | Computed over non-NaN samples in the window; `<1` valid sample $\to$ mean `0.0`; `<2` valid samples $\to$ std/slope/corr `0.0`. |
| **2a. Direction (model, observable)** | `{NODE}_tilt_angle_rad`, `dir_pairwise_cos_mean`, `dir_pairwise_cos_min`, `dir_circular_spread` | Mean 2-D tilt change vector $(\Delta \text{tilt}_x, \Delta \text{tilt}_y)$ relative to each node's observable lead-in median (`first 30 s`). Pairwise cosine and circular spread $1 - \bar{R}$ computed over nodes with valid samples and non-zero vector norm; `<2` valid nodes $\to$ cosine `0.0`, circular spread `1.0`. |
| **2b. Direction (`diag_`, excluded)** | `diag_{NODE}_angle_to_true_centre_deg`, `diag_mean_angle_to_true_centre_deg` | Uses ground-truth `(x0, y0)` and mounting yaw to measure angle between sensor dip vector and true basin centre direction. Undefined / null-basin $\to$ `90.0°`. **Strictly excluded from `feature_columns(df)`.** |
| **3. Signed Coherence** | `coherence_signed_tx`, `coherence_signed_ty`, `coherence_signed_mean` | Mean pairwise Pearson correlation of signed $\Delta \text{tilt}_x$ and $\Delta \text{tilt}_y$ across the 4 nodes; `<2` shared finite samples on a pair $\to$ pair corr `0.0`. |
| **4. Piezo** | `{NODE}_piezo_peak_max`, `{NODE}_piezo_rms_mean`, `{NODE}_piezo_p2p_max`, `{NODE}_piezo_impulse_count`, `piezo_impulse_total`, `piezo_nodes_with_impulse`, `piezo_rms_network_mean` | Computed over finite samples; all-NaN node window $\to$ `0.0` and impulse count `0`. Impulse count = samples with `piezo_peak > 0.035`. |
| **5. Hub (LiDAR)** | `{NODE}_lidar_range_change_m`, `hub_common_mode_m`, `{NODE}_lidar_residual_m`, `{NODE}_lidar_staleness_s`, `hub_residual_abs_max_m`, `hub_staleness_max_s` | Forward-fills the latest received LiDAR reading at or before `t_end`. Before any dwell arrives for a node, range change and residual = `0.0`, staleness = `t_end`. |
| **6. Communication** | `{NODE}_missing_frac`, `comm_missing_frac_net`, `comm_max_burst_len`, `comm_nodes_unavailable`, `comm_latency_mean_s`, `comm_latency_max_s` | `missing_frac = 1 - mean(packet_received)`. Latency stats computed over received packets; if `0` packets received in window, latency mean/max fall back to `window_s` (`5.0` s). |
| **7. Long-Horizon** | `{NODE}_long_slope_deg_s`, `long_slope_max_deg_s` | Least-squares slope of `{NODE}_mean` vs `t_mid` over the trailing `N = 12` windows (`32.5` s span); `<2` finite windows $\to$ `0.0`. |

---

## 4. Datasets, Splits & `snr_vs_imu_noise` Distribution (`canary_sim/dataset.py`, `scripts/build_datasets.py`)

### Run-Level Split Counts (`manifest.csv`)
All splits are strictly by run (`simulation_id` and `seed` are globally unique across all 830 runs; no run or window is shared across splits):

| Split | `normal` | `gradual` | `rapid` | `abnormal_vibration` | `sensor_fault` | `comm_fault` | Total Runs | Total Windows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `train` | 60 | 60 | 60 | 60 | 60 | 60 | **360** | 34,200 |
| `val` | 20 | 20 | 20 | 20 | 20 | 20 | **120** | 11,400 |
| `test` | 20 | 20 | 20 | 20 | 20 | 20 | **120** | 11,400 |
| `shifted_test` | 15 | 15 | 15 | 15 | 15 | 15 | **90** | 8,550 |
| `broad_basin` | 0 | 20 | 20 | 0 | 0 | 0 | **40** | 3,800 |
| `stress` (combined) | 0 | 30 | 28 | 28 | 14 | 0 | **100** | 9,500 |
| **Total** | **115** | **165** | **163** | **143** | **129** | **115** | **830** | **78,850** |

### Window Label Counts Across Splits (`window_labels.csv`)
| Split | `normal` | `gradual` | `rapid` | `abnormal_vibration` | `sensor_fault` | `comm_fault` | `transition` (excluded) | Total Windows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `train` | 21,305 | 4,030 | 3,761 | 1,443 | 1,626 | 1,560 | 475 | **34,200** |
| `val` | 7,147 | 1,296 | 1,276 | 498 | 526 | 501 | 156 | **11,400** |
| `test` | 7,154 | 1,333 | 1,167 | 502 | 555 | 530 | 159 | **11,400** |
| `shifted_test` | 5,336 | 1,035 | 916 | 371 | 376 | 397 | 119 | **8,550** |
| `broad_basin` | 1,197 | 1,344 | 1,221 | 0 | 0 | 0 | 38 | **3,800** |
| `stress` | 3,869 | 1,991 | 1,784 | 763 | 553 | 399 | 141 | **9,500** |
| **Total** | **46,008** | **11,029** | **10,125** | **3,577** | **3,636** | **3,387** | **1,088** | **78,850** |

### Distribution of `snr_vs_imu_noise` for `gradual` and `rapid` Runs
Definition: $\text{snr\_vs\_imu\_noise} = \text{peak\_tilt\_deg} / \sigma_{\text{IMU}}$, where $\sigma_{\text{IMU}} = \text{noise\_multiplier} \times \sqrt{\tfrac{1}{2}(\sigma_{x,0}^2 + \sigma_{y,0}^2)}$ ($\sigma_{\text{IMU},0} \approx 0.188^\circ$ at multiplier $1.0$). Because `peak_tilt_deg` is log-stratified across $[0.05^\circ, 3.0^\circ]$ within every split, weak events ($\text{SNR} < 1$) are systematically represented:

| Subset (`gradual` + `rapid`) | Runs ($n$) | Min | p5 | p25 | Median | p75 | p95 | Max | Share with `snr_vs_imu_noise < 1.0` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **Headline (`train` + `val` + `test`)** | 200 | 0.150 | 0.280 | 0.736 | 2.187 | 5.437 | 12.780 | 27.203 | **64 / 200 (32.0%)** |
| **Shifted-noise test (`shifted_test`)** | 30 | 0.149 | 0.208 | 0.460 | 1.273 | 3.271 | 7.258 | 8.741 | **13 / 30 (43.3%)** |
| **Broad-basin subset (`broad_basin`)** | 40 | 0.240 | 0.262 | 0.869 | 2.085 | 6.468 | 16.729 | 19.728 | **11 / 40 (27.5%)** |
| **All single-condition basin runs** | 270 | 0.149 | 0.261 | 0.664 | 1.976 | 5.313 | 13.171 | 27.203 | **88 / 270 (32.6%)** |

---

## 5. Assumptions & Placeholders
1. **Values from `configs/noise_profile.json` (real prototype recording)**:
   - Sampling interval (`1.0 s`), IMU white-noise sd (`tilt_x = 0.233°`, `tilt_y = 0.128°` at multiplier `1.0`, pooled RMS `0.188°`), and piezo baseline median/spread/autocorrelation/crest ratios.
2. **Placeholders (`configs/dataset_default.yaml`)**:
   - `lead_in_baseline_s = 30.0` s (observable baseline window at the start of each run; guaranteed `< 0.20 * 240 = 48 s` earliest event onset).
   - `impulse_peak_thresh = 0.035` (threshold on `piezo_peak` to count an impulse sample; sits between baseline median `0.018` and real recording p99 `0.068`).
   - `abnormal_vibration` ranges: `rms_add_range = [0.09, 0.22]`, `duration_s_range = [35, 90]` s, `min_nodes = 3`, `tilt_jitter_sd_deg_range = [0.16, 0.35]°`, `impulse_peak_range = [0.22, 0.55]`.
   - `shifted_test`: `15` runs/class, IMU noise multiplier `[1.5, 2.0]`, `node_bias_sd_deg = 0.12°`, `confounder_scale = 0.85`, `confounder_n_bursts = 2`.
   - `broad_basin_subset`: `20` runs/class (`gradual`, `rapid`).
   - `stress_set`: `100` runs cycling across 7 two-condition combinations.
3. **Priority order in `stress` runs**: When two conditions overlap in the `stress` set, `label_windows` records all active classes in `label_set` (pipe-separated) and assigns primary `label` by `PRIORITY = ("rapid", "gradual", "abnormal_vibration", "sensor_fault", "comm_fault", "normal")`. Headline splits have a single condition per run, so no tie-breaking occurs there.
