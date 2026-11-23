# Manual Human Audit Sample: 30 Advisory Explanations (SYNTHETIC / SIMULATION)

> **SYNTHETIC / SIMULATION - NOT VALIDATED ON REAL DATA**
>
> **Audit Instructions for Human Reviewer:** Each of the 30 entries below (5 per stratum across 6 strata, including `missed_event` and `classifier_mistake`) displays:
> 1. **Observable Packet Values** (the exact JSON fields passed to the explainer),
> 2. **Advisory Explanation Text** (`summary` and `suggested_check`), and
> 3. **Ground-Truth Label & Stratum** shown separately at the bottom of the entry (never visible to the explainer or the judge).

---

## Entry 01 / 30 — Window `run_test_023_w46` (Run `run_test_023`)

### 1. Observable Packet Values (Input to Explainer)
- **Time Window:** `115.0 s` to `120.0 s` (midpoint `117.5 s`)
- **Classifier Output:** `predicted_class = "gradual"`, `model_score = 0.4007`
- **Persisted Alarm & Rule Baseline:** `alarm_state = "gradual"`, `threshold_rule_class = "normal"`
- **Advisory Unknown Anomaly Detector:** `unknown_anomaly = False`, `anomaly_score = 6.6169`
- **Per-Node Mean Tilt (deg):** `NODE_01 = 0.2651`, `NODE_02 = 0.4186`, `NODE_03 = 0.4317`, `NODE_04 = 0.4685`
- **Piezo RMS (V):** `network_mean = 0.00948`, `NODE_01 = 0.00612`, `NODE_02 = 0.0158`, `NODE_03 = 0.00614`, `NODE_04 = 0.00985`
- **LiDAR Range Change (m):** `hub_common_mode_m = -0.015`, `NODE_01 = 0.0`, `NODE_02 = -0.04`, `NODE_03 = 0.0`, `NODE_04 = -0.02`
- **Communications:** `missing_fraction_network = 0.0` (`NODE_01 = 0.0`, `NODE_02 = 0.0`, `NODE_03 = 0.0`, `NODE_04 = 0.0`), `latency_mean_s = 0.3857`, `latency_max_s = 0.9864`
- **Radial Convergence & SNR Bin:** `radial_convergence_residual_m = 9.092`, `radial_convergence_weighted_cos = 0.6308`, `snr_vs_imu_noise_bin = ">5"`

<details><summary>Raw Packet JSON</summary>

```json
{
  "window_id": "run_test_023_w46",
  "run_id": "run_test_023",
  "time_s": {
    "t_start": 115.0,
    "t_end": 120.0,
    "t_mid": 117.5
  },
  "predicted_class": "gradual",
  "model_score": 0.4007,
  "alarm_state": "gradual",
  "threshold_rule_class": "normal",
  "unknown_anomaly": false,
  "anomaly_score": 6.6169,
  "node_mean_tilt_deg": {
    "NODE_01": 0.2651,
    "NODE_02": 0.4186,
    "NODE_03": 0.4317,
    "NODE_04": 0.4685
  },
  "piezo_rms": {
    "network_mean": 0.00948,
    "NODE_01": 0.00612,
    "NODE_02": 0.0158,
    "NODE_03": 0.00614,
    "NODE_04": 0.00985
  },
  "lidar_range_change_m": {
    "hub_common_mode_m": -0.015,
    "NODE_01": 0.0,
    "NODE_02": -0.04,
    "NODE_03": 0.0,
    "NODE_04": -0.02
  },
  "comm": {
    "missing_fraction_network": 0.0,
    "NODE_01_missing_fraction": 0.0,
    "NODE_02_missing_fraction": 0.0,
    "NODE_03_missing_fraction": 0.0,
    "NODE_04_missing_fraction": 0.0,
    "latency_mean_s": 0.3857,
    "latency_max_s": 0.9864
  },
  "radial_convergence_residual_m": 9.092,
  "radial_convergence_weighted_cos": 0.6308,
  "snr_vs_imu_noise_bin": ">5"
}
```
</details>

### 2. Advisory Explanation Text
- **Source / Model:** `ai` (`muse-spark-1.3`)
- **Summary:** This simulated window from 115.0 to 120.0 shows predicted_class gradual consistent with model_score 0.4007 and anomaly_score 6.6169. Observed node_mean_tilt_deg values include 0.2651 and 0.4685 with lidar_range_change_m hub_common_mode_m of -0.015 and NODE_02 of -0.04.
- **Suggested Check:** Consider checking piezo_rms network_mean at 0.00948 and latency_max_s at 0.9864 for verification.

### 3. Ground-Truth Label (Withheld from Explainer & Judge)
- **Audit Stratum:** `correct_alarm` (raw stratum: `correct_alarm`, substratum: `correct_alarm_gradual`)
- **True Window Label (`true_label`):** `gradual` (Run class: `gradual`)
- **Classifier vs Truth Agreement:** `predicted_class = "gradual"` vs `true_label = "gradual"` **(MATCH)**

---

## Entry 02 / 30 — Window `run_test_043_w82` (Run `run_test_043`)

### 1. Observable Packet Values (Input to Explainer)
- **Time Window:** `205.0 s` to `210.0 s` (midpoint `207.5 s`)
- **Classifier Output:** `predicted_class = "rapid"`, `model_score = 0.7273`
- **Persisted Alarm & Rule Baseline:** `alarm_state = "rapid"`, `threshold_rule_class = "normal"`
- **Advisory Unknown Anomaly Detector:** `unknown_anomaly = False`, `anomaly_score = 7.8931`
- **Per-Node Mean Tilt (deg):** `NODE_01 = 0.3477`, `NODE_02 = 0.2952`, `NODE_03 = 0.5796`, `NODE_04 = 0.3735`
- **Piezo RMS (V):** `network_mean = 0.00832`, `NODE_01 = 0.00633`, `NODE_02 = 0.00822`, `NODE_03 = 0.00932`, `NODE_04 = 0.00943`
- **LiDAR Range Change (m):** `hub_common_mode_m = -0.0075`, `NODE_01 = -0.075`, `NODE_02 = 0.035`, `NODE_03 = 0.01`, `NODE_04 = 0.0`
- **Communications:** `missing_fraction_network = 0.0` (`NODE_01 = 0.0`, `NODE_02 = 0.0`, `NODE_03 = 0.0`, `NODE_04 = 0.0`), `latency_mean_s = 0.4344`, `latency_max_s = 0.8511`
- **Radial Convergence & SNR Bin:** `radial_convergence_residual_m = 1.455`, `radial_convergence_weighted_cos = 0.9948`, `snr_vs_imu_noise_bin = "2-5"`

<details><summary>Raw Packet JSON</summary>

```json
{
  "window_id": "run_test_043_w82",
  "run_id": "run_test_043",
  "time_s": {
    "t_start": 205.0,
    "t_end": 210.0,
    "t_mid": 207.5
  },
  "predicted_class": "rapid",
  "model_score": 0.7273,
  "alarm_state": "rapid",
  "threshold_rule_class": "normal",
  "unknown_anomaly": false,
  "anomaly_score": 7.8931,
  "node_mean_tilt_deg": {
    "NODE_01": 0.3477,
    "NODE_02": 0.2952,
    "NODE_03": 0.5796,
    "NODE_04": 0.3735
  },
  "piezo_rms": {
    "network_mean": 0.00832,
    "NODE_01": 0.00633,
    "NODE_02": 0.00822,
    "NODE_03": 0.00932,
    "NODE_04": 0.00943
  },
  "lidar_range_change_m": {
    "hub_common_mode_m": -0.0075,
    "NODE_01": -0.075,
    "NODE_02": 0.035,
    "NODE_03": 0.01,
    "NODE_04": 0.0
  },
  "comm": {
    "missing_fraction_network": 0.0,
    "NODE_01_missing_fraction": 0.0,
    "NODE_02_missing_fraction": 0.0,
    "NODE_03_missing_fraction": 0.0,
    "NODE_04_missing_fraction": 0.0,
    "latency_mean_s": 0.4344,
    "latency_max_s": 0.8511
  },
  "radial_convergence_residual_m": 1.455,
  "radial_convergence_weighted_cos": 0.9948,
  "snr_vs_imu_noise_bin": "2-5"
}
```
</details>

### 2. Advisory Explanation Text
- **Source / Model:** `ai` (`muse-spark-1.3`)
- **Summary:** This simulated packet for run_test_043_w82 from 205.0 to 210.0 shows predicted_class rapid with model_score 0.7273, consistent with rapid pattern. Reported values include anomaly_score 7.8931 and radial_convergence_residual_m 1.455 with lidar NODE_01 at -0.075.
- **Suggested Check:** Consider verifying lidar_range_change_m for NODE_01 at -0.075.

### 3. Ground-Truth Label (Withheld from Explainer & Judge)
- **Audit Stratum:** `correct_alarm` (raw stratum: `correct_alarm`, substratum: `correct_alarm_rapid`)
- **True Window Label (`true_label`):** `rapid` (Run class: `rapid`)
- **Classifier vs Truth Agreement:** `predicted_class = "rapid"` vs `true_label = "rapid"` **(MATCH)**

---

## Entry 03 / 30 — Window `run_test_078_w73` (Run `run_test_078`)

### 1. Observable Packet Values (Input to Explainer)
- **Time Window:** `182.5 s` to `187.5 s` (midpoint `185.0 s`)
- **Classifier Output:** `predicted_class = "abnormal_vibration"`, `model_score = 1.0`
- **Persisted Alarm & Rule Baseline:** `alarm_state = "abnormal_vibration"`, `threshold_rule_class = "abnormal_vibration"`
- **Advisory Unknown Anomaly Detector:** `unknown_anomaly = True`, `anomaly_score = 96.7188`
- **Per-Node Mean Tilt (deg):** `NODE_01 = 0.1027`, `NODE_02 = 0.4442`, `NODE_03 = 0.1982`, `NODE_04 = 0.2912`
- **Piezo RMS (V):** `network_mean = 0.12992`, `NODE_01 = 0.00822`, `NODE_02 = 0.19232`, `NODE_03 = 0.14732`, `NODE_04 = 0.17184`
- **LiDAR Range Change (m):** `hub_common_mode_m = 0.0175`, `NODE_01 = 0.0`, `NODE_02 = 0.06`, `NODE_03 = 0.03`, `NODE_04 = -0.02`
- **Communications:** `missing_fraction_network = 0.0` (`NODE_01 = 0.0`, `NODE_02 = 0.0`, `NODE_03 = 0.0`, `NODE_04 = 0.0`), `latency_mean_s = 0.4755`, `latency_max_s = 0.7993`
- **Radial Convergence & SNR Bin:** `radial_convergence_residual_m = 9.164`, `radial_convergence_weighted_cos = 0.7781`, `snr_vs_imu_noise_bin = "<0.5"`

<details><summary>Raw Packet JSON</summary>

```json
{
  "window_id": "run_test_078_w73",
  "run_id": "run_test_078",
  "time_s": {
    "t_start": 182.5,
    "t_end": 187.5,
    "t_mid": 185.0
  },
  "predicted_class": "abnormal_vibration",
  "model_score": 1.0,
  "alarm_state": "abnormal_vibration",
  "threshold_rule_class": "abnormal_vibration",
  "unknown_anomaly": true,
  "anomaly_score": 96.7188,
  "node_mean_tilt_deg": {
    "NODE_01": 0.1027,
    "NODE_02": 0.4442,
    "NODE_03": 0.1982,
    "NODE_04": 0.2912
  },
  "piezo_rms": {
    "network_mean": 0.12992,
    "NODE_01": 0.00822,
    "NODE_02": 0.19232,
    "NODE_03": 0.14732,
    "NODE_04": 0.17184
  },
  "lidar_range_change_m": {
    "hub_common_mode_m": 0.0175,
    "NODE_01": 0.0,
    "NODE_02": 0.06,
    "NODE_03": 0.03,
    "NODE_04": -0.02
  },
  "comm": {
    "missing_fraction_network": 0.0,
    "NODE_01_missing_fraction": 0.0,
    "NODE_02_missing_fraction": 0.0,
    "NODE_03_missing_fraction": 0.0,
    "NODE_04_missing_fraction": 0.0,
    "latency_mean_s": 0.4755,
    "latency_max_s": 0.7993
  },
  "radial_convergence_residual_m": 9.164,
  "radial_convergence_weighted_cos": 0.7781,
  "snr_vs_imu_noise_bin": "<0.5"
}
```
</details>

### 2. Advisory Explanation Text
- **Source / Model:** `ai` (`muse-spark-1.3`)
- **Summary:** This simulated window shows model_score 1.0 with predicted_class abnormal_vibration, consistent with abnormal_vibration indicators. Reported values include anomaly_score 96.7188, network piezo_rms 0.12992 and NODE_02 piezo_rms 0.19232. Tilt at NODE_02 is 0.4442 and lidar change at NODE_02 is 0.06, consistent with the simulated pattern.
- **Suggested Check:** Consider checking piezo_rms telemetry for NODE_02 at 0.19232 in the simulated record.

### 3. Ground-Truth Label (Withheld from Explainer & Judge)
- **Audit Stratum:** `correct_alarm` (raw stratum: `correct_alarm`, substratum: `correct_alarm_abnormal_vibration`)
- **True Window Label (`true_label`):** `abnormal_vibration` (Run class: `abnormal_vibration`)
- **Classifier vs Truth Agreement:** `predicted_class = "abnormal_vibration"` vs `true_label = "abnormal_vibration"` **(MATCH)**

---

## Entry 04 / 30 — Window `run_test_099_w37` (Run `run_test_099`)

### 1. Observable Packet Values (Input to Explainer)
- **Time Window:** `92.5 s` to `97.5 s` (midpoint `95.0 s`)
- **Classifier Output:** `predicted_class = "sensor_fault"`, `model_score = 0.5468`
- **Persisted Alarm & Rule Baseline:** `alarm_state = "sensor_fault"`, `threshold_rule_class = "sensor_fault"`
- **Advisory Unknown Anomaly Detector:** `unknown_anomaly = True`, `anomaly_score = 19.8371`
- **Per-Node Mean Tilt (deg):** `NODE_01 = 0.1505`, `NODE_02 = 0.1017`, `NODE_03 = 0.2073`, `NODE_04 = 0.1889`
- **Piezo RMS (V):** `network_mean = 0.0082`, `NODE_01 = 0.00989`, `NODE_02 = 0.00715`, `NODE_03 = 0.00796`, `NODE_04 = 0.00781`
- **LiDAR Range Change (m):** `hub_common_mode_m = -0.34`, `NODE_01 = -0.39`, `NODE_02 = -0.28`, `NODE_03 = -0.35`, `NODE_04 = -0.34`
- **Communications:** `missing_fraction_network = 0.0` (`NODE_01 = 0.0`, `NODE_02 = 0.0`, `NODE_03 = 0.0`, `NODE_04 = 0.0`), `latency_mean_s = 0.4335`, `latency_max_s = 0.804`
- **Radial Convergence & SNR Bin:** `radial_convergence_residual_m = 8.67`, `radial_convergence_weighted_cos = 0.9292`, `snr_vs_imu_noise_bin = "<0.5"`

<details><summary>Raw Packet JSON</summary>

```json
{
  "window_id": "run_test_099_w37",
  "run_id": "run_test_099",
  "time_s": {
    "t_start": 92.5,
    "t_end": 97.5,
    "t_mid": 95.0
  },
  "predicted_class": "sensor_fault",
  "model_score": 0.5468,
  "alarm_state": "sensor_fault",
  "threshold_rule_class": "sensor_fault",
  "unknown_anomaly": true,
  "anomaly_score": 19.8371,
  "node_mean_tilt_deg": {
    "NODE_01": 0.1505,
    "NODE_02": 0.1017,
    "NODE_03": 0.2073,
    "NODE_04": 0.1889
  },
  "piezo_rms": {
    "network_mean": 0.0082,
    "NODE_01": 0.00989,
    "NODE_02": 0.00715,
    "NODE_03": 0.00796,
    "NODE_04": 0.00781
  },
  "lidar_range_change_m": {
    "hub_common_mode_m": -0.34,
    "NODE_01": -0.39,
    "NODE_02": -0.28,
    "NODE_03": -0.35,
    "NODE_04": -0.34
  },
  "comm": {
    "missing_fraction_network": 0.0,
    "NODE_01_missing_fraction": 0.0,
    "NODE_02_missing_fraction": 0.0,
    "NODE_03_missing_fraction": 0.0,
    "NODE_04_missing_fraction": 0.0,
    "latency_mean_s": 0.4335,
    "latency_max_s": 0.804
  },
  "radial_convergence_residual_m": 8.67,
  "radial_convergence_weighted_cos": 0.9292,
  "snr_vs_imu_noise_bin": "<0.5"
}
```
</details>

### 2. Advisory Explanation Text
- **Source / Model:** `ai` (`muse-spark-1.3`)
- **Summary:** This simulated packet at t_mid 95.0 is consistent with predicted_class sensor_fault with model_score 0.5468. It shows anomaly_score 19.8371 and radial_convergence_residual_m 8.67 with hub_common_mode_m -0.34.
- **Suggested Check:** Verify lidar_range_change_m readings for NODE_01 -0.39 and NODE_02 -0.28 in the simulated window.

### 3. Ground-Truth Label (Withheld from Explainer & Judge)
- **Audit Stratum:** `correct_alarm` (raw stratum: `correct_alarm`, substratum: `correct_alarm_sensor_fault`)
- **True Window Label (`true_label`):** `sensor_fault` (Run class: `sensor_fault`)
- **Classifier vs Truth Agreement:** `predicted_class = "sensor_fault"` vs `true_label = "sensor_fault"` **(MATCH)**

---

## Entry 05 / 30 — Window `run_test_106_w34` (Run `run_test_106`)

### 1. Observable Packet Values (Input to Explainer)
- **Time Window:** `85.0 s` to `90.0 s` (midpoint `87.5 s`)
- **Classifier Output:** `predicted_class = "comm_fault"`, `model_score = 0.5855`
- **Persisted Alarm & Rule Baseline:** `alarm_state = "comm_fault"`, `threshold_rule_class = "comm_fault"`
- **Advisory Unknown Anomaly Detector:** `unknown_anomaly = True`, `anomaly_score = 17.6276`
- **Per-Node Mean Tilt (deg):** `NODE_01 = 0.2131`, `NODE_02 = 0.2022`, `NODE_03 = 0.0883`, `NODE_04 = 0.2358`
- **Piezo RMS (V):** `network_mean = 0.00634`, `NODE_01 = 0.00866`, `NODE_02 = 0.00817`, `NODE_03 = 0.00475`, `NODE_04 = 0.00377`
- **LiDAR Range Change (m):** `hub_common_mode_m = 0.0012`, `NODE_01 = -0.02`, `NODE_02 = -0.005`, `NODE_03 = 0.04`, `NODE_04 = -0.01`
- **Communications:** `missing_fraction_network = 0.15` (`NODE_01 = 0.0`, `NODE_02 = 0.6`, `NODE_03 = 0.0`, `NODE_04 = 0.0`), `latency_mean_s = 0.4891`, `latency_max_s = 0.9964`
- **Radial Convergence & SNR Bin:** `radial_convergence_residual_m = 17.894`, `radial_convergence_weighted_cos = -0.9023`, `snr_vs_imu_noise_bin = "<0.5"`

<details><summary>Raw Packet JSON</summary>

```json
{
  "window_id": "run_test_106_w34",
  "run_id": "run_test_106",
  "time_s": {
    "t_start": 85.0,
    "t_end": 90.0,
    "t_mid": 87.5
  },
  "predicted_class": "comm_fault",
  "model_score": 0.5855,
  "alarm_state": "comm_fault",
  "threshold_rule_class": "comm_fault",
  "unknown_anomaly": true,
  "anomaly_score": 17.6276,
  "node_mean_tilt_deg": {
    "NODE_01": 0.2131,
    "NODE_02": 0.2022,
    "NODE_03": 0.0883,
    "NODE_04": 0.2358
  },
  "piezo_rms": {
    "network_mean": 0.00634,
    "NODE_01": 0.00866,
    "NODE_02": 0.00817,
    "NODE_03": 0.00475,
    "NODE_04": 0.00377
  },
  "lidar_range_change_m": {
    "hub_common_mode_m": 0.0012,
    "NODE_01": -0.02,
    "NODE_02": -0.005,
    "NODE_03": 0.04,
    "NODE_04": -0.01
  },
  "comm": {
    "missing_fraction_network": 0.15,
    "NODE_01_missing_fraction": 0.0,
    "NODE_02_missing_fraction": 0.6,
    "NODE_03_missing_fraction": 0.0,
    "NODE_04_missing_fraction": 0.0,
    "latency_mean_s": 0.4891,
    "latency_max_s": 0.9964
  },
  "radial_convergence_residual_m": 17.894,
  "radial_convergence_weighted_cos": -0.9023,
  "snr_vs_imu_noise_bin": "<0.5"
}
```
</details>

### 2. Advisory Explanation Text
- **Source / Model:** `ai` (`muse-spark-1.3`)
- **Summary:** This simulated packet from 85.0 to 90.0 is consistent with comm_fault with model_score 0.5855. Telemetry shows 0.6 for NODE_02_missing_fraction and 0.15 for missing_fraction_network with latency_max_s 0.9964.
- **Suggested Check:** Verify NODE_02 telemetry stream for missing packets around 87.5.

### 3. Ground-Truth Label (Withheld from Explainer & Judge)
- **Audit Stratum:** `correct_alarm` (raw stratum: `correct_alarm`, substratum: `correct_alarm_comm_fault`)
- **True Window Label (`true_label`):** `comm_fault` (Run class: `comm_fault`)
- **Classifier vs Truth Agreement:** `predicted_class = "comm_fault"` vs `true_label = "comm_fault"` **(MATCH)**

---

## Entry 06 / 30 — Window `run_test_028_w43` (Run `run_test_028`)

### 1. Observable Packet Values (Input to Explainer)
- **Time Window:** `107.5 s` to `112.5 s` (midpoint `110.0 s`)
- **Classifier Output:** `predicted_class = "normal"`, `model_score = 0.3715`
- **Persisted Alarm & Rule Baseline:** `alarm_state = "normal"`, `threshold_rule_class = "normal"`
- **Advisory Unknown Anomaly Detector:** `unknown_anomaly = False`, `anomaly_score = 6.5803`
- **Per-Node Mean Tilt (deg):** `NODE_01 = 0.3458`, `NODE_02 = 0.224`, `NODE_03 = 0.362`, `NODE_04 = 0.2871`
- **Piezo RMS (V):** `network_mean = 0.00771`, `NODE_01 = 0.01603`, `NODE_02 = 0.00915`, `NODE_03 = 0.00378`, `NODE_04 = 0.00189`
- **LiDAR Range Change (m):** `hub_common_mode_m = 0.01`, `NODE_01 = 0.0`, `NODE_02 = -0.005`, `NODE_03 = 0.035`, `NODE_04 = 0.01`
- **Communications:** `missing_fraction_network = 0.0` (`NODE_01 = 0.0`, `NODE_02 = 0.0`, `NODE_03 = 0.0`, `NODE_04 = 0.0`), `latency_mean_s = 0.4161`, `latency_max_s = 1.148`
- **Radial Convergence & SNR Bin:** `radial_convergence_residual_m = 10.204`, `radial_convergence_weighted_cos = 0.3103`, `snr_vs_imu_noise_bin = ">5"`

<details><summary>Raw Packet JSON</summary>

```json
{
  "window_id": "run_test_028_w43",
  "run_id": "run_test_028",
  "time_s": {
    "t_start": 107.5,
    "t_end": 112.5,
    "t_mid": 110.0
  },
  "predicted_class": "normal",
  "model_score": 0.3715,
  "alarm_state": "normal",
  "threshold_rule_class": "normal",
  "unknown_anomaly": false,
  "anomaly_score": 6.5803,
  "node_mean_tilt_deg": {
    "NODE_01": 0.3458,
    "NODE_02": 0.224,
    "NODE_03": 0.362,
    "NODE_04": 0.2871
  },
  "piezo_rms": {
    "network_mean": 0.00771,
    "NODE_01": 0.01603,
    "NODE_02": 0.00915,
    "NODE_03": 0.00378,
    "NODE_04": 0.00189
  },
  "lidar_range_change_m": {
    "hub_common_mode_m": 0.01,
    "NODE_01": 0.0,
    "NODE_02": -0.005,
    "NODE_03": 0.035,
    "NODE_04": 0.01
  },
  "comm": {
    "missing_fraction_network": 0.0,
    "NODE_01_missing_fraction": 0.0,
    "NODE_02_missing_fraction": 0.0,
    "NODE_03_missing_fraction": 0.0,
    "NODE_04_missing_fraction": 0.0,
    "latency_mean_s": 0.4161,
    "latency_max_s": 1.148
  },
  "radial_convergence_residual_m": 10.204,
  "radial_convergence_weighted_cos": 0.3103,
  "snr_vs_imu_noise_bin": ">5"
}
```
</details>

### 2. Advisory Explanation Text
- **Source / Model:** `ai` (`muse-spark-1.3`)
- **Summary:** This simulated window run_test_028_w43 at 110.0 is consistent with predicted_class normal with model_score 0.3715. The anomaly_score 6.5803 and radial_convergence_residual_m 10.204 were reported with alarm_state normal. Node tilts including 0.3458 and 0.224 remain listed alongside hub_common_mode_m 0.01 consistent with normal.
- **Suggested Check:** Consider checking piezo_rms network_mean 0.00771 against NODE_01 0.01603 for verification.

### 3. Ground-Truth Label (Withheld from Explainer & Judge)
- **Audit Stratum:** `missed_event` (raw stratum: `missed_event`, substratum: `missed_gradual`)
- **True Window Label (`true_label`):** `gradual` (Run class: `gradual`)
- **Classifier vs Truth Agreement:** `predicted_class = "normal"` vs `true_label = "gradual"` **(MISMATCH — Explainer faithfully explains the classifier's output without knowing ground truth)**

---

## Entry 07 / 30 — Window `run_test_054_w85` (Run `run_test_054`)

### 1. Observable Packet Values (Input to Explainer)
- **Time Window:** `212.5 s` to `217.5 s` (midpoint `215.0 s`)
- **Classifier Output:** `predicted_class = "normal"`, `model_score = 0.2975`
- **Persisted Alarm & Rule Baseline:** `alarm_state = "normal"`, `threshold_rule_class = "normal"`
- **Advisory Unknown Anomaly Detector:** `unknown_anomaly = False`, `anomaly_score = 8.0286`
- **Per-Node Mean Tilt (deg):** `NODE_01 = 0.4967`, `NODE_02 = 0.4283`, `NODE_03 = 0.399`, `NODE_04 = 0.4499`
- **Piezo RMS (V):** `network_mean = 0.00642`, `NODE_01 = 0.00771`, `NODE_02 = 0.00409`, `NODE_03 = 0.00774`, `NODE_04 = 0.00612`
- **LiDAR Range Change (m):** `hub_common_mode_m = -0.0125`, `NODE_01 = 0.0`, `NODE_02 = 0.0`, `NODE_03 = -0.04`, `NODE_04 = -0.01`
- **Communications:** `missing_fraction_network = 0.0` (`NODE_01 = 0.0`, `NODE_02 = 0.0`, `NODE_03 = 0.0`, `NODE_04 = 0.0`), `latency_mean_s = 0.4103`, `latency_max_s = 0.7001`
- **Radial Convergence & SNR Bin:** `radial_convergence_residual_m = 3.967`, `radial_convergence_weighted_cos = -0.8453`, `snr_vs_imu_noise_bin = "1-2"`

<details><summary>Raw Packet JSON</summary>

```json
{
  "window_id": "run_test_054_w85",
  "run_id": "run_test_054",
  "time_s": {
    "t_start": 212.5,
    "t_end": 217.5,
    "t_mid": 215.0
  },
  "predicted_class": "normal",
  "model_score": 0.2975,
  "alarm_state": "normal",
  "threshold_rule_class": "normal",
  "unknown_anomaly": false,
  "anomaly_score": 8.0286,
  "node_mean_tilt_deg": {
    "NODE_01": 0.4967,
    "NODE_02": 0.4283,
    "NODE_03": 0.399,
    "NODE_04": 0.4499
  },
  "piezo_rms": {
    "network_mean": 0.00642,
    "NODE_01": 0.00771,
    "NODE_02": 0.00409,
    "NODE_03": 0.00774,
    "NODE_04": 0.00612
  },
  "lidar_range_change_m": {
    "hub_common_mode_m": -0.0125,
    "NODE_01": 0.0,
    "NODE_02": 0.0,
    "NODE_03": -0.04,
    "NODE_04": -0.01
  },
  "comm": {
    "missing_fraction_network": 0.0,
    "NODE_01_missing_fraction": 0.0,
    "NODE_02_missing_fraction": 0.0,
    "NODE_03_missing_fraction": 0.0,
    "NODE_04_missing_fraction": 0.0,
    "latency_mean_s": 0.4103,
    "latency_max_s": 0.7001
  },
  "radial_convergence_residual_m": 3.967,
  "radial_convergence_weighted_cos": -0.8453,
  "snr_vs_imu_noise_bin": "1-2"
}
```
</details>

### 2. Advisory Explanation Text
- **Source / Model:** `ai` (`muse-spark-1.3`)
- **Summary:** This simulated packet is consistent with predicted_class normal with model_score 0.2975. It reports t_mid 215.0 and anomaly_score 8.0286 with missing_fraction_network 0.0. The alarm_state normal matches threshold_rule_class normal.
- **Suggested Check:** Consider verifying piezo_rms network_mean 0.00642 against NODE_01 0.00771 in the next simulated window.

### 3. Ground-Truth Label (Withheld from Explainer & Judge)
- **Audit Stratum:** `missed_event` (raw stratum: `missed_event`, substratum: `missed_rapid`)
- **True Window Label (`true_label`):** `rapid` (Run class: `rapid`)
- **Classifier vs Truth Agreement:** `predicted_class = "normal"` vs `true_label = "rapid"` **(MISMATCH — Explainer faithfully explains the classifier's output without knowing ground truth)**

---

## Entry 08 / 30 — Window `run_test_086_w37` (Run `run_test_086`)

### 1. Observable Packet Values (Input to Explainer)
- **Time Window:** `92.5 s` to `97.5 s` (midpoint `95.0 s`)
- **Classifier Output:** `predicted_class = "normal"`, `model_score = 0.4148`
- **Persisted Alarm & Rule Baseline:** `alarm_state = "normal"`, `threshold_rule_class = "normal"`
- **Advisory Unknown Anomaly Detector:** `unknown_anomaly = False`, `anomaly_score = 6.8163`
- **Per-Node Mean Tilt (deg):** `NODE_01 = 0.2441`, `NODE_02 = 0.2735`, `NODE_03 = 0.1915`, `NODE_04 = 0.1454`
- **Piezo RMS (V):** `network_mean = 0.00676`, `NODE_01 = 0.00537`, `NODE_02 = 0.00878`, `NODE_03 = 0.00887`, `NODE_04 = 0.00401`
- **LiDAR Range Change (m):** `hub_common_mode_m = -0.0512`, `NODE_01 = -0.06`, `NODE_02 = -0.05`, `NODE_03 = -0.025`, `NODE_04 = -0.07`
- **Communications:** `missing_fraction_network = 0.0` (`NODE_01 = 0.0`, `NODE_02 = 0.0`, `NODE_03 = 0.0`, `NODE_04 = 0.0`), `latency_mean_s = 0.3631`, `latency_max_s = 0.7257`
- **Radial Convergence & SNR Bin:** `radial_convergence_residual_m = 9.517`, `radial_convergence_weighted_cos = 0.5932`, `snr_vs_imu_noise_bin = "<0.5"`

<details><summary>Raw Packet JSON</summary>

```json
{
  "window_id": "run_test_086_w37",
  "run_id": "run_test_086",
  "time_s": {
    "t_start": 92.5,
    "t_end": 97.5,
    "t_mid": 95.0
  },
  "predicted_class": "normal",
  "model_score": 0.4148,
  "alarm_state": "normal",
  "threshold_rule_class": "normal",
  "unknown_anomaly": false,
  "anomaly_score": 6.8163,
  "node_mean_tilt_deg": {
    "NODE_01": 0.2441,
    "NODE_02": 0.2735,
    "NODE_03": 0.1915,
    "NODE_04": 0.1454
  },
  "piezo_rms": {
    "network_mean": 0.00676,
    "NODE_01": 0.00537,
    "NODE_02": 0.00878,
    "NODE_03": 0.00887,
    "NODE_04": 0.00401
  },
  "lidar_range_change_m": {
    "hub_common_mode_m": -0.0512,
    "NODE_01": -0.06,
    "NODE_02": -0.05,
    "NODE_03": -0.025,
    "NODE_04": -0.07
  },
  "comm": {
    "missing_fraction_network": 0.0,
    "NODE_01_missing_fraction": 0.0,
    "NODE_02_missing_fraction": 0.0,
    "NODE_03_missing_fraction": 0.0,
    "NODE_04_missing_fraction": 0.0,
    "latency_mean_s": 0.3631,
    "latency_max_s": 0.7257
  },
  "radial_convergence_residual_m": 9.517,
  "radial_convergence_weighted_cos": 0.5932,
  "snr_vs_imu_noise_bin": "<0.5"
}
```
</details>

### 2. Advisory Explanation Text
- **Source / Model:** `ai` (`muse-spark-1.3`)
- **Summary:** This simulated window run_test_086_w37 from 92.5 to 97.5 with model_score 0.4148 is consistent with predicted_class normal. It reports alarm_state normal, anomaly_score 6.8163, and latency_mean_s 0.3631.
- **Suggested Check:** Consider verifying piezo_rms network_mean at 0.00676 for continued observation.

### 3. Ground-Truth Label (Withheld from Explainer & Judge)
- **Audit Stratum:** `missed_event` (raw stratum: `missed_event`, substratum: `missed_sensor_fault`)
- **True Window Label (`true_label`):** `sensor_fault` (Run class: `sensor_fault`)
- **Classifier vs Truth Agreement:** `predicted_class = "normal"` vs `true_label = "sensor_fault"` **(MISMATCH — Explainer faithfully explains the classifier's output without knowing ground truth)**

---

## Entry 09 / 30 — Window `run_test_100_w40` (Run `run_test_100`)

### 1. Observable Packet Values (Input to Explainer)
- **Time Window:** `100.0 s` to `105.0 s` (midpoint `102.5 s`)
- **Classifier Output:** `predicted_class = "normal"`, `model_score = 0.3353`
- **Persisted Alarm & Rule Baseline:** `alarm_state = "normal"`, `threshold_rule_class = "gradual"`
- **Advisory Unknown Anomaly Detector:** `unknown_anomaly = True`, `anomaly_score = 13.2999`
- **Per-Node Mean Tilt (deg):** `NODE_01 = 0.4349`, `NODE_02 = 0.2664`, `NODE_03 = 0.6945`, `NODE_04 = 0.2588`
- **Piezo RMS (V):** `network_mean = 0.01268`, `NODE_01 = 0.01691`, `NODE_02 = 0.01306`, `NODE_03 = 0.01086`, `NODE_04 = 0.00987`
- **LiDAR Range Change (m):** `hub_common_mode_m = 0.01`, `NODE_01 = 0.03`, `NODE_02 = -0.02`, `NODE_03 = -0.02`, `NODE_04 = 0.05`
- **Communications:** `missing_fraction_network = 0.1` (`NODE_01 = 0.4`, `NODE_02 = 0.0`, `NODE_03 = 0.0`, `NODE_04 = 0.0`), `latency_mean_s = 0.4284`, `latency_max_s = 0.7208`
- **Radial Convergence & SNR Bin:** `radial_convergence_residual_m = 15.823`, `radial_convergence_weighted_cos = 0.042`, `snr_vs_imu_noise_bin = "<0.5"`

<details><summary>Raw Packet JSON</summary>

```json
{
  "window_id": "run_test_100_w40",
  "run_id": "run_test_100",
  "time_s": {
    "t_start": 100.0,
    "t_end": 105.0,
    "t_mid": 102.5
  },
  "predicted_class": "normal",
  "model_score": 0.3353,
  "alarm_state": "normal",
  "threshold_rule_class": "gradual",
  "unknown_anomaly": true,
  "anomaly_score": 13.2999,
  "node_mean_tilt_deg": {
    "NODE_01": 0.4349,
    "NODE_02": 0.2664,
    "NODE_03": 0.6945,
    "NODE_04": 0.2588
  },
  "piezo_rms": {
    "network_mean": 0.01268,
    "NODE_01": 0.01691,
    "NODE_02": 0.01306,
    "NODE_03": 0.01086,
    "NODE_04": 0.00987
  },
  "lidar_range_change_m": {
    "hub_common_mode_m": 0.01,
    "NODE_01": 0.03,
    "NODE_02": -0.02,
    "NODE_03": -0.02,
    "NODE_04": 0.05
  },
  "comm": {
    "missing_fraction_network": 0.1,
    "NODE_01_missing_fraction": 0.4,
    "NODE_02_missing_fraction": 0.0,
    "NODE_03_missing_fraction": 0.0,
    "NODE_04_missing_fraction": 0.0,
    "latency_mean_s": 0.4284,
    "latency_max_s": 0.7208
  },
  "radial_convergence_residual_m": 15.823,
  "radial_convergence_weighted_cos": 0.042,
  "snr_vs_imu_noise_bin": "<0.5"
}
```
</details>

### 2. Advisory Explanation Text
- **Source / Model:** `ai` (`muse-spark-1.3`)
- **Summary:** This simulated packet is consistent with predicted_class normal with model_score 0.3353 and alarm_state normal. It shows t_mid 102.5 with anomaly_score 13.2999 and radial_convergence_residual_m 15.823. Telemetry includes NODE_03 tilt 0.6945 and latency_max_s 0.7208.
- **Suggested Check:** Verify NODE_01_missing_fraction 0.4 against comm logs for this simulated window.

### 3. Ground-Truth Label (Withheld from Explainer & Judge)
- **Audit Stratum:** `missed_event` (raw stratum: `missed_event`, substratum: `missed_comm_fault`)
- **True Window Label (`true_label`):** `comm_fault` (Run class: `comm_fault`)
- **Classifier vs Truth Agreement:** `predicted_class = "normal"` vs `true_label = "comm_fault"` **(MISMATCH — Explainer faithfully explains the classifier's output without knowing ground truth)**

---

## Entry 10 / 30 — Window `run_test_100_w41` (Run `run_test_100`)

### 1. Observable Packet Values (Input to Explainer)
- **Time Window:** `102.5 s` to `107.5 s` (midpoint `105.0 s`)
- **Classifier Output:** `predicted_class = "normal"`, `model_score = 0.3878`
- **Persisted Alarm & Rule Baseline:** `alarm_state = "normal"`, `threshold_rule_class = "normal"`
- **Advisory Unknown Anomaly Detector:** `unknown_anomaly = True`, `anomaly_score = 12.4995`
- **Per-Node Mean Tilt (deg):** `NODE_01 = 0.4807`, `NODE_02 = 0.2215`, `NODE_03 = 0.4225`, `NODE_04 = 0.3483`
- **Piezo RMS (V):** `network_mean = 0.01443`, `NODE_01 = 0.02567`, `NODE_02 = 0.01127`, `NODE_03 = 0.01091`, `NODE_04 = 0.00986`
- **LiDAR Range Change (m):** `hub_common_mode_m = 0.01`, `NODE_01 = 0.03`, `NODE_02 = -0.02`, `NODE_03 = -0.02`, `NODE_04 = 0.05`
- **Communications:** `missing_fraction_network = 0.1` (`NODE_01 = 0.4`, `NODE_02 = 0.0`, `NODE_03 = 0.0`, `NODE_04 = 0.0`), `latency_mean_s = 0.3789`, `latency_max_s = 0.6089`
- **Radial Convergence & SNR Bin:** `radial_convergence_residual_m = 13.197`, `radial_convergence_weighted_cos = 0.3522`, `snr_vs_imu_noise_bin = "<0.5"`

<details><summary>Raw Packet JSON</summary>

```json
{
  "window_id": "run_test_100_w41",
  "run_id": "run_test_100",
  "time_s": {
    "t_start": 102.5,
    "t_end": 107.5,
    "t_mid": 105.0
  },
  "predicted_class": "normal",
  "model_score": 0.3878,
  "alarm_state": "normal",
  "threshold_rule_class": "normal",
  "unknown_anomaly": true,
  "anomaly_score": 12.4995,
  "node_mean_tilt_deg": {
    "NODE_01": 0.4807,
    "NODE_02": 0.2215,
    "NODE_03": 0.4225,
    "NODE_04": 0.3483
  },
  "piezo_rms": {
    "network_mean": 0.01443,
    "NODE_01": 0.02567,
    "NODE_02": 0.01127,
    "NODE_03": 0.01091,
    "NODE_04": 0.00986
  },
  "lidar_range_change_m": {
    "hub_common_mode_m": 0.01,
    "NODE_01": 0.03,
    "NODE_02": -0.02,
    "NODE_03": -0.02,
    "NODE_04": 0.05
  },
  "comm": {
    "missing_fraction_network": 0.1,
    "NODE_01_missing_fraction": 0.4,
    "NODE_02_missing_fraction": 0.0,
    "NODE_03_missing_fraction": 0.0,
    "NODE_04_missing_fraction": 0.0,
    "latency_mean_s": 0.3789,
    "latency_max_s": 0.6089
  },
  "radial_convergence_residual_m": 13.197,
  "radial_convergence_weighted_cos": 0.3522,
  "snr_vs_imu_noise_bin": "<0.5"
}
```
</details>

### 2. Advisory Explanation Text
- **Source / Model:** `ai` (`muse-spark-1.3`)
- **Summary:** This simulated packet from 102.5 to 107.5 is consistent with predicted_class normal with model_score 0.3878. It lists anomaly_score 12.4995 and radial_convergence_residual_m 13.197 with NODE_01_missing_fraction 0.4.
- **Suggested Check:** Verify NODE_01 telemetry given NODE_01_missing_fraction 0.4 and latency_max_s 0.6089.

### 3. Ground-Truth Label (Withheld from Explainer & Judge)
- **Audit Stratum:** `missed_event` (raw stratum: `missed_event`, substratum: `missed_comm_fault`)
- **True Window Label (`true_label`):** `comm_fault` (Run class: `comm_fault`)
- **Classifier vs Truth Agreement:** `predicted_class = "normal"` vs `true_label = "comm_fault"` **(MISMATCH — Explainer faithfully explains the classifier's output without knowing ground truth)**

---

## Entry 11 / 30 — Window `run_test_005_w22` (Run `run_test_005`)

### 1. Observable Packet Values (Input to Explainer)
- **Time Window:** `55.0 s` to `60.0 s` (midpoint `57.5 s`)
- **Classifier Output:** `predicted_class = "rapid"`, `model_score = 0.5445`
- **Persisted Alarm & Rule Baseline:** `alarm_state = "rapid"`, `threshold_rule_class = "normal"`
- **Advisory Unknown Anomaly Detector:** `unknown_anomaly = False`, `anomaly_score = 6.0216`
- **Per-Node Mean Tilt (deg):** `NODE_01 = 0.229`, `NODE_02 = 0.2011`, `NODE_03 = 0.1547`, `NODE_04 = 0.2342`
- **Piezo RMS (V):** `network_mean = 0.00938`, `NODE_01 = 0.00712`, `NODE_02 = 0.0138`, `NODE_03 = 0.00999`, `NODE_04 = 0.00662`
- **LiDAR Range Change (m):** `hub_common_mode_m = 0.0025`, `NODE_01 = 0.04`, `NODE_02 = -0.01`, `NODE_03 = 0.0`, `NODE_04 = -0.02`
- **Communications:** `missing_fraction_network = 0.0` (`NODE_01 = 0.0`, `NODE_02 = 0.0`, `NODE_03 = 0.0`, `NODE_04 = 0.0`), `latency_mean_s = 0.424`, `latency_max_s = 0.8373`
- **Radial Convergence & SNR Bin:** `radial_convergence_residual_m = 4.08`, `radial_convergence_weighted_cos = 0.8949`, `snr_vs_imu_noise_bin = "<0.5"`

<details><summary>Raw Packet JSON</summary>

```json
{
  "window_id": "run_test_005_w22",
  "run_id": "run_test_005",
  "time_s": {
    "t_start": 55.0,
    "t_end": 60.0,
    "t_mid": 57.5
  },
  "predicted_class": "rapid",
  "model_score": 0.5445,
  "alarm_state": "rapid",
  "threshold_rule_class": "normal",
  "unknown_anomaly": false,
  "anomaly_score": 6.0216,
  "node_mean_tilt_deg": {
    "NODE_01": 0.229,
    "NODE_02": 0.2011,
    "NODE_03": 0.1547,
    "NODE_04": 0.2342
  },
  "piezo_rms": {
    "network_mean": 0.00938,
    "NODE_01": 0.00712,
    "NODE_02": 0.0138,
    "NODE_03": 0.00999,
    "NODE_04": 0.00662
  },
  "lidar_range_change_m": {
    "hub_common_mode_m": 0.0025,
    "NODE_01": 0.04,
    "NODE_02": -0.01,
    "NODE_03": 0.0,
    "NODE_04": -0.02
  },
  "comm": {
    "missing_fraction_network": 0.0,
    "NODE_01_missing_fraction": 0.0,
    "NODE_02_missing_fraction": 0.0,
    "NODE_03_missing_fraction": 0.0,
    "NODE_04_missing_fraction": 0.0,
    "latency_mean_s": 0.424,
    "latency_max_s": 0.8373
  },
  "radial_convergence_residual_m": 4.08,
  "radial_convergence_weighted_cos": 0.8949,
  "snr_vs_imu_noise_bin": "<0.5"
}
```
</details>

### 2. Advisory Explanation Text
- **Source / Model:** `ai` (`muse-spark-1.3`)
- **Summary:** This simulated window run_test_005_w22 is consistent with rapid with model_score 0.5445. It lists t_mid 57.5, anomaly_score 6.0216 and radial_convergence_residual_m 4.08. NODE_01 shows 0.229 and 0.04 while NODE_02 shows 0.2011 and -0.01.
- **Suggested Check:** Please verify lidar_range_change_m for NODE_01 at 0.04 in the next simulated telemetry review.

### 3. Ground-Truth Label (Withheld from Explainer & Judge)
- **Audit Stratum:** `false_alarm` (raw stratum: `false_alarm`, substratum: `false_alarm_persisted_on_normal_run`)
- **True Window Label (`true_label`):** `normal` (Run class: `normal`)
- **Classifier vs Truth Agreement:** `predicted_class = "rapid"` vs `true_label = "normal"` **(MISMATCH — Explainer faithfully explains the classifier's output without knowing ground truth)**

---

## Entry 12 / 30 — Window `run_test_005_w40` (Run `run_test_005`)

### 1. Observable Packet Values (Input to Explainer)
- **Time Window:** `100.0 s` to `105.0 s` (midpoint `102.5 s`)
- **Classifier Output:** `predicted_class = "normal"`, `model_score = 0.3906`
- **Persisted Alarm & Rule Baseline:** `alarm_state = "rapid"`, `threshold_rule_class = "normal"`
- **Advisory Unknown Anomaly Detector:** `unknown_anomaly = False`, `anomaly_score = 5.0757`
- **Per-Node Mean Tilt (deg):** `NODE_01 = 0.232`, `NODE_02 = 0.1796`, `NODE_03 = 0.2514`, `NODE_04 = 0.2189`
- **Piezo RMS (V):** `network_mean = 0.00502`, `NODE_01 = 0.00445`, `NODE_02 = 0.00306`, `NODE_03 = 0.00647`, `NODE_04 = 0.00608`
- **LiDAR Range Change (m):** `hub_common_mode_m = -0.0`, `NODE_01 = -0.04`, `NODE_02 = 0.0`, `NODE_03 = -0.02`, `NODE_04 = 0.06`
- **Communications:** `missing_fraction_network = 0.0` (`NODE_01 = 0.0`, `NODE_02 = 0.0`, `NODE_03 = 0.0`, `NODE_04 = 0.0`), `latency_mean_s = 0.4407`, `latency_max_s = 0.8333`
- **Radial Convergence & SNR Bin:** `radial_convergence_residual_m = 4.399`, `radial_convergence_weighted_cos = 0.914`, `snr_vs_imu_noise_bin = "<0.5"`

<details><summary>Raw Packet JSON</summary>

```json
{
  "window_id": "run_test_005_w40",
  "run_id": "run_test_005",
  "time_s": {
    "t_start": 100.0,
    "t_end": 105.0,
    "t_mid": 102.5
  },
  "predicted_class": "normal",
  "model_score": 0.3906,
  "alarm_state": "rapid",
  "threshold_rule_class": "normal",
  "unknown_anomaly": false,
  "anomaly_score": 5.0757,
  "node_mean_tilt_deg": {
    "NODE_01": 0.232,
    "NODE_02": 0.1796,
    "NODE_03": 0.2514,
    "NODE_04": 0.2189
  },
  "piezo_rms": {
    "network_mean": 0.00502,
    "NODE_01": 0.00445,
    "NODE_02": 0.00306,
    "NODE_03": 0.00647,
    "NODE_04": 0.00608
  },
  "lidar_range_change_m": {
    "hub_common_mode_m": -0.0,
    "NODE_01": -0.04,
    "NODE_02": 0.0,
    "NODE_03": -0.02,
    "NODE_04": 0.06
  },
  "comm": {
    "missing_fraction_network": 0.0,
    "NODE_01_missing_fraction": 0.0,
    "NODE_02_missing_fraction": 0.0,
    "NODE_03_missing_fraction": 0.0,
    "NODE_04_missing_fraction": 0.0,
    "latency_mean_s": 0.4407,
    "latency_max_s": 0.8333
  },
  "radial_convergence_residual_m": 4.399,
  "radial_convergence_weighted_cos": 0.914,
  "snr_vs_imu_noise_bin": "<0.5"
}
```
</details>

### 2. Advisory Explanation Text
- **Source / Model:** `ai` (`muse-spark-1.3`)
- **Summary:** This simulated window run_test_005_w40 reports predicted_class normal with model_score 0.3906 and anomaly_score 5.0757. The values including t_mid 102.5 and radial_convergence_residual_m 4.399 are consistent with normal.
- **Suggested Check:** Consider checking lidar_range_change_m for NODE_04 at 0.06 for verification.

### 3. Ground-Truth Label (Withheld from Explainer & Judge)
- **Audit Stratum:** `false_alarm` (raw stratum: `false_alarm`, substratum: `false_alarm_persisted_on_normal_run`)
- **True Window Label (`true_label`):** `normal` (Run class: `normal`)
- **Classifier vs Truth Agreement:** `predicted_class = "normal"` vs `true_label = "normal"` **(MATCH)**

---

## Entry 13 / 30 — Window `run_test_007_w43` (Run `run_test_007`)

### 1. Observable Packet Values (Input to Explainer)
- **Time Window:** `107.5 s` to `112.5 s` (midpoint `110.0 s`)
- **Classifier Output:** `predicted_class = "gradual"`, `model_score = 0.3437`
- **Persisted Alarm & Rule Baseline:** `alarm_state = "gradual"`, `threshold_rule_class = "rapid"`
- **Advisory Unknown Anomaly Detector:** `unknown_anomaly = False`, `anomaly_score = 7.7409`
- **Per-Node Mean Tilt (deg):** `NODE_01 = 0.6063`, `NODE_02 = 0.5151`, `NODE_03 = 0.3351`, `NODE_04 = 0.5491`
- **Piezo RMS (V):** `network_mean = 0.0109`, `NODE_01 = 0.01286`, `NODE_02 = 0.01023`, `NODE_03 = 0.01075`, `NODE_04 = 0.00977`
- **LiDAR Range Change (m):** `hub_common_mode_m = -0.0012`, `NODE_01 = 0.02`, `NODE_02 = -0.025`, `NODE_03 = -0.01`, `NODE_04 = 0.01`
- **Communications:** `missing_fraction_network = 0.0` (`NODE_01 = 0.0`, `NODE_02 = 0.0`, `NODE_03 = 0.0`, `NODE_04 = 0.0`), `latency_mean_s = 0.4432`, `latency_max_s = 1.1278`
- **Radial Convergence & SNR Bin:** `radial_convergence_residual_m = 9.171`, `radial_convergence_weighted_cos = 0.7168`, `snr_vs_imu_noise_bin = "<0.5"`

<details><summary>Raw Packet JSON</summary>

```json
{
  "window_id": "run_test_007_w43",
  "run_id": "run_test_007",
  "time_s": {
    "t_start": 107.5,
    "t_end": 112.5,
    "t_mid": 110.0
  },
  "predicted_class": "gradual",
  "model_score": 0.3437,
  "alarm_state": "gradual",
  "threshold_rule_class": "rapid",
  "unknown_anomaly": false,
  "anomaly_score": 7.7409,
  "node_mean_tilt_deg": {
    "NODE_01": 0.6063,
    "NODE_02": 0.5151,
    "NODE_03": 0.3351,
    "NODE_04": 0.5491
  },
  "piezo_rms": {
    "network_mean": 0.0109,
    "NODE_01": 0.01286,
    "NODE_02": 0.01023,
    "NODE_03": 0.01075,
    "NODE_04": 0.00977
  },
  "lidar_range_change_m": {
    "hub_common_mode_m": -0.0012,
    "NODE_01": 0.02,
    "NODE_02": -0.025,
    "NODE_03": -0.01,
    "NODE_04": 0.01
  },
  "comm": {
    "missing_fraction_network": 0.0,
    "NODE_01_missing_fraction": 0.0,
    "NODE_02_missing_fraction": 0.0,
    "NODE_03_missing_fraction": 0.0,
    "NODE_04_missing_fraction": 0.0,
    "latency_mean_s": 0.4432,
    "latency_max_s": 1.1278
  },
  "radial_convergence_residual_m": 9.171,
  "radial_convergence_weighted_cos": 0.7168,
  "snr_vs_imu_noise_bin": "<0.5"
}
```
</details>

### 2. Advisory Explanation Text
- **Source / Model:** `ai` (`muse-spark-1.3`)
- **Summary:** This simulated packet for window run_test_007_w43 with model_score 0.3437 is consistent with predicted_class gradual. It reports t_mid 110.0 with NODE_01 tilt 0.6063 and anomaly_score 7.7409.
- **Suggested Check:** Verify NODE_01 tilt reading of 0.6063 against raw telemetry.

### 3. Ground-Truth Label (Withheld from Explainer & Judge)
- **Audit Stratum:** `false_alarm` (raw stratum: `false_alarm`, substratum: `false_alarm_window_on_normal_run`)
- **True Window Label (`true_label`):** `normal` (Run class: `normal`)
- **Classifier vs Truth Agreement:** `predicted_class = "gradual"` vs `true_label = "normal"` **(MISMATCH — Explainer faithfully explains the classifier's output without knowing ground truth)**

---

## Entry 14 / 30 — Window `run_test_010_w73` (Run `run_test_010`)

### 1. Observable Packet Values (Input to Explainer)
- **Time Window:** `182.5 s` to `187.5 s` (midpoint `185.0 s`)
- **Classifier Output:** `predicted_class = "rapid"`, `model_score = 0.4261`
- **Persisted Alarm & Rule Baseline:** `alarm_state = "gradual"`, `threshold_rule_class = "normal"`
- **Advisory Unknown Anomaly Detector:** `unknown_anomaly = False`, `anomaly_score = 7.7904`
- **Per-Node Mean Tilt (deg):** `NODE_01 = 0.3577`, `NODE_02 = 0.3978`, `NODE_03 = 0.4402`, `NODE_04 = 0.3866`
- **Piezo RMS (V):** `network_mean = 0.00727`, `NODE_01 = 0.00714`, `NODE_02 = 0.00781`, `NODE_03 = 0.008`, `NODE_04 = 0.00614`
- **LiDAR Range Change (m):** `hub_common_mode_m = 0.0062`, `NODE_01 = 0.05`, `NODE_02 = -0.015`, `NODE_03 = 0.02`, `NODE_04 = -0.03`
- **Communications:** `missing_fraction_network = 0.0` (`NODE_01 = 0.0`, `NODE_02 = 0.0`, `NODE_03 = 0.0`, `NODE_04 = 0.0`), `latency_mean_s = 0.4529`, `latency_max_s = 0.7735`
- **Radial Convergence & SNR Bin:** `radial_convergence_residual_m = 1.003`, `radial_convergence_weighted_cos = 0.9934`, `snr_vs_imu_noise_bin = "<0.5"`

<details><summary>Raw Packet JSON</summary>

```json
{
  "window_id": "run_test_010_w73",
  "run_id": "run_test_010",
  "time_s": {
    "t_start": 182.5,
    "t_end": 187.5,
    "t_mid": 185.0
  },
  "predicted_class": "rapid",
  "model_score": 0.4261,
  "alarm_state": "gradual",
  "threshold_rule_class": "normal",
  "unknown_anomaly": false,
  "anomaly_score": 7.7904,
  "node_mean_tilt_deg": {
    "NODE_01": 0.3577,
    "NODE_02": 0.3978,
    "NODE_03": 0.4402,
    "NODE_04": 0.3866
  },
  "piezo_rms": {
    "network_mean": 0.00727,
    "NODE_01": 0.00714,
    "NODE_02": 0.00781,
    "NODE_03": 0.008,
    "NODE_04": 0.00614
  },
  "lidar_range_change_m": {
    "hub_common_mode_m": 0.0062,
    "NODE_01": 0.05,
    "NODE_02": -0.015,
    "NODE_03": 0.02,
    "NODE_04": -0.03
  },
  "comm": {
    "missing_fraction_network": 0.0,
    "NODE_01_missing_fraction": 0.0,
    "NODE_02_missing_fraction": 0.0,
    "NODE_03_missing_fraction": 0.0,
    "NODE_04_missing_fraction": 0.0,
    "latency_mean_s": 0.4529,
    "latency_max_s": 0.7735
  },
  "radial_convergence_residual_m": 1.003,
  "radial_convergence_weighted_cos": 0.9934,
  "snr_vs_imu_noise_bin": "<0.5"
}
```
</details>

### 2. Advisory Explanation Text
- **Source / Model:** `ai` (`muse-spark-1.3`)
- **Summary:** This simulated packet is consistent with predicted_class rapid with model_score 0.4261. It reports t_mid 185.0 and anomaly_score 7.7904 with radial_convergence_residual_m 1.003. Lidar values include 0.05 for NODE_01 and -0.015 for NODE_02.
- **Suggested Check:** Consider verifying lidar_range_change_m for NODE_01 at 0.05 in the next simulated sample.

### 3. Ground-Truth Label (Withheld from Explainer & Judge)
- **Audit Stratum:** `false_alarm` (raw stratum: `false_alarm`, substratum: `false_alarm_window_on_normal_run`)
- **True Window Label (`true_label`):** `normal` (Run class: `normal`)
- **Classifier vs Truth Agreement:** `predicted_class = "rapid"` vs `true_label = "normal"` **(MISMATCH — Explainer faithfully explains the classifier's output without knowing ground truth)**

---

## Entry 15 / 30 — Window `run_test_018_w78` (Run `run_test_018`)

### 1. Observable Packet Values (Input to Explainer)
- **Time Window:** `195.0 s` to `200.0 s` (midpoint `197.5 s`)
- **Classifier Output:** `predicted_class = "normal"`, `model_score = 0.293`
- **Persisted Alarm & Rule Baseline:** `alarm_state = "sensor_fault"`, `threshold_rule_class = "normal"`
- **Advisory Unknown Anomaly Detector:** `unknown_anomaly = False`, `anomaly_score = 8.2493`
- **Per-Node Mean Tilt (deg):** `NODE_01 = 0.4271`, `NODE_02 = 0.3127`, `NODE_03 = 0.2398`, `NODE_04 = 0.1501`
- **Piezo RMS (V):** `network_mean = 0.01066`, `NODE_01 = 0.01271`, `NODE_02 = 0.01624`, `NODE_03 = 0.00634`, `NODE_04 = 0.00735`
- **LiDAR Range Change (m):** `hub_common_mode_m = 0.0025`, `NODE_01 = -0.07`, `NODE_02 = -0.025`, `NODE_03 = 0.105`, `NODE_04 = 0.0`
- **Communications:** `missing_fraction_network = 0.0` (`NODE_01 = 0.0`, `NODE_02 = 0.0`, `NODE_03 = 0.0`, `NODE_04 = 0.0`), `latency_mean_s = 0.4918`, `latency_max_s = 1.004`
- **Radial Convergence & SNR Bin:** `radial_convergence_residual_m = 4.966`, `radial_convergence_weighted_cos = 0.8797`, `snr_vs_imu_noise_bin = "<0.5"`

<details><summary>Raw Packet JSON</summary>

```json
{
  "window_id": "run_test_018_w78",
  "run_id": "run_test_018",
  "time_s": {
    "t_start": 195.0,
    "t_end": 200.0,
    "t_mid": 197.5
  },
  "predicted_class": "normal",
  "model_score": 0.293,
  "alarm_state": "sensor_fault",
  "threshold_rule_class": "normal",
  "unknown_anomaly": false,
  "anomaly_score": 8.2493,
  "node_mean_tilt_deg": {
    "NODE_01": 0.4271,
    "NODE_02": 0.3127,
    "NODE_03": 0.2398,
    "NODE_04": 0.1501
  },
  "piezo_rms": {
    "network_mean": 0.01066,
    "NODE_01": 0.01271,
    "NODE_02": 0.01624,
    "NODE_03": 0.00634,
    "NODE_04": 0.00735
  },
  "lidar_range_change_m": {
    "hub_common_mode_m": 0.0025,
    "NODE_01": -0.07,
    "NODE_02": -0.025,
    "NODE_03": 0.105,
    "NODE_04": 0.0
  },
  "comm": {
    "missing_fraction_network": 0.0,
    "NODE_01_missing_fraction": 0.0,
    "NODE_02_missing_fraction": 0.0,
    "NODE_03_missing_fraction": 0.0,
    "NODE_04_missing_fraction": 0.0,
    "latency_mean_s": 0.4918,
    "latency_max_s": 1.004
  },
  "radial_convergence_residual_m": 4.966,
  "radial_convergence_weighted_cos": 0.8797,
  "snr_vs_imu_noise_bin": "<0.5"
}
```
</details>

### 2. Advisory Explanation Text
- **Source / Model:** `ai` (`muse-spark-1.3`)
- **Summary:** This simulated packet for window run_test_018_w78 from 195.0 to 200.0 is classified as normal with model_score 0.293, consistent with baseline behavior. Reported values include 0.4271, 0.1501, -0.07 and 0.105, consistent with normal simulated telemetry.
- **Suggested Check:** Please verify lidar_range_change_m values of -0.07 and 0.105 against hub_common_mode_m 0.0025.

### 3. Ground-Truth Label (Withheld from Explainer & Judge)
- **Audit Stratum:** `false_alarm` (raw stratum: `false_alarm`, substratum: `false_alarm_persisted_on_normal_run`)
- **True Window Label (`true_label`):** `normal` (Run class: `normal`)
- **Classifier vs Truth Agreement:** `predicted_class = "normal"` vs `true_label = "normal"` **(MATCH)**

---

## Entry 16 / 30 — Window `run_test_028_w57` (Run `run_test_028`)

### 1. Observable Packet Values (Input to Explainer)
- **Time Window:** `142.5 s` to `147.5 s` (midpoint `145.0 s`)
- **Classifier Output:** `predicted_class = "sensor_fault"`, `model_score = 0.3841`
- **Persisted Alarm & Rule Baseline:** `alarm_state = "gradual"`, `threshold_rule_class = "rapid"`
- **Advisory Unknown Anomaly Detector:** `unknown_anomaly = True`, `anomaly_score = 11.7985`
- **Per-Node Mean Tilt (deg):** `NODE_01 = 0.2829`, `NODE_02 = 0.445`, `NODE_03 = 1.0336`, `NODE_04 = 0.5801`
- **Piezo RMS (V):** `network_mean = 0.00527`, `NODE_01 = 0.00339`, `NODE_02 = 0.00347`, `NODE_03 = 0.0091`, `NODE_04 = 0.00513`
- **LiDAR Range Change (m):** `hub_common_mode_m = 0.0125`, `NODE_01 = -0.03`, `NODE_02 = -0.005`, `NODE_03 = 0.015`, `NODE_04 = 0.07`
- **Communications:** `missing_fraction_network = 0.0` (`NODE_01 = 0.0`, `NODE_02 = 0.0`, `NODE_03 = 0.0`, `NODE_04 = 0.0`), `latency_mean_s = 0.4556`, `latency_max_s = 0.9014`
- **Radial Convergence & SNR Bin:** `radial_convergence_residual_m = 3.746`, `radial_convergence_weighted_cos = -0.1203`, `snr_vs_imu_noise_bin = ">5"`

<details><summary>Raw Packet JSON</summary>

```json
{
  "window_id": "run_test_028_w57",
  "run_id": "run_test_028",
  "time_s": {
    "t_start": 142.5,
    "t_end": 147.5,
    "t_mid": 145.0
  },
  "predicted_class": "sensor_fault",
  "model_score": 0.3841,
  "alarm_state": "gradual",
  "threshold_rule_class": "rapid",
  "unknown_anomaly": true,
  "anomaly_score": 11.7985,
  "node_mean_tilt_deg": {
    "NODE_01": 0.2829,
    "NODE_02": 0.445,
    "NODE_03": 1.0336,
    "NODE_04": 0.5801
  },
  "piezo_rms": {
    "network_mean": 0.00527,
    "NODE_01": 0.00339,
    "NODE_02": 0.00347,
    "NODE_03": 0.0091,
    "NODE_04": 0.00513
  },
  "lidar_range_change_m": {
    "hub_common_mode_m": 0.0125,
    "NODE_01": -0.03,
    "NODE_02": -0.005,
    "NODE_03": 0.015,
    "NODE_04": 0.07
  },
  "comm": {
    "missing_fraction_network": 0.0,
    "NODE_01_missing_fraction": 0.0,
    "NODE_02_missing_fraction": 0.0,
    "NODE_03_missing_fraction": 0.0,
    "NODE_04_missing_fraction": 0.0,
    "latency_mean_s": 0.4556,
    "latency_max_s": 0.9014
  },
  "radial_convergence_residual_m": 3.746,
  "radial_convergence_weighted_cos": -0.1203,
  "snr_vs_imu_noise_bin": ">5"
}
```
</details>

### 2. Advisory Explanation Text
- **Source / Model:** `ai` (`muse-spark-1.3`)
- **Summary:** This simulated packet for window run_test_028_w57 with t_mid 145.0 is consistent with predicted_class sensor_fault with model_score 0.3841. The packet shows anomaly_score 11.7985 and radial_convergence_residual_m 3.746 with piezo_rms network_mean 0.00527.
- **Suggested Check:** Consider rechecking NODE_03 piezo_rms 0.0091 against network_mean 0.00527 in the simulated record.

### 3. Ground-Truth Label (Withheld from Explainer & Judge)
- **Audit Stratum:** `classifier_mistake` (raw stratum: `classifier_mistake`, substratum: `mistake_gradual_as_sensor_fault`)
- **True Window Label (`true_label`):** `gradual` (Run class: `gradual`)
- **Classifier vs Truth Agreement:** `predicted_class = "sensor_fault"` vs `true_label = "gradual"` **(MISMATCH — Explainer faithfully explains the classifier's output without knowing ground truth)**

---

## Entry 17 / 30 — Window `run_test_036_w90` (Run `run_test_036`)

### 1. Observable Packet Values (Input to Explainer)
- **Time Window:** `225.0 s` to `230.0 s` (midpoint `227.5 s`)
- **Classifier Output:** `predicted_class = "rapid"`, `model_score = 0.8472`
- **Persisted Alarm & Rule Baseline:** `alarm_state = "rapid"`, `threshold_rule_class = "gradual"`
- **Advisory Unknown Anomaly Detector:** `unknown_anomaly = False`, `anomaly_score = 9.991`
- **Per-Node Mean Tilt (deg):** `NODE_01 = 0.2306`, `NODE_02 = 0.4413`, `NODE_03 = 0.6364`, `NODE_04 = 0.4269`
- **Piezo RMS (V):** `network_mean = 0.00858`, `NODE_01 = 0.00539`, `NODE_02 = 0.01352`, `NODE_03 = 0.01004`, `NODE_04 = 0.00538`
- **LiDAR Range Change (m):** `hub_common_mode_m = -0.01`, `NODE_01 = -0.09`, `NODE_02 = 0.03`, `NODE_03 = 0.04`, `NODE_04 = -0.02`
- **Communications:** `missing_fraction_network = 0.0` (`NODE_01 = 0.0`, `NODE_02 = 0.0`, `NODE_03 = 0.0`, `NODE_04 = 0.0`), `latency_mean_s = 0.4446`, `latency_max_s = 0.8818`
- **Radial Convergence & SNR Bin:** `radial_convergence_residual_m = 1.467`, `radial_convergence_weighted_cos = 0.995`, `snr_vs_imu_noise_bin = ">5"`

<details><summary>Raw Packet JSON</summary>

```json
{
  "window_id": "run_test_036_w90",
  "run_id": "run_test_036",
  "time_s": {
    "t_start": 225.0,
    "t_end": 230.0,
    "t_mid": 227.5
  },
  "predicted_class": "rapid",
  "model_score": 0.8472,
  "alarm_state": "rapid",
  "threshold_rule_class": "gradual",
  "unknown_anomaly": false,
  "anomaly_score": 9.991,
  "node_mean_tilt_deg": {
    "NODE_01": 0.2306,
    "NODE_02": 0.4413,
    "NODE_03": 0.6364,
    "NODE_04": 0.4269
  },
  "piezo_rms": {
    "network_mean": 0.00858,
    "NODE_01": 0.00539,
    "NODE_02": 0.01352,
    "NODE_03": 0.01004,
    "NODE_04": 0.00538
  },
  "lidar_range_change_m": {
    "hub_common_mode_m": -0.01,
    "NODE_01": -0.09,
    "NODE_02": 0.03,
    "NODE_03": 0.04,
    "NODE_04": -0.02
  },
  "comm": {
    "missing_fraction_network": 0.0,
    "NODE_01_missing_fraction": 0.0,
    "NODE_02_missing_fraction": 0.0,
    "NODE_03_missing_fraction": 0.0,
    "NODE_04_missing_fraction": 0.0,
    "latency_mean_s": 0.4446,
    "latency_max_s": 0.8818
  },
  "radial_convergence_residual_m": 1.467,
  "radial_convergence_weighted_cos": 0.995,
  "snr_vs_imu_noise_bin": ">5"
}
```
</details>

### 2. Advisory Explanation Text
- **Source / Model:** `ai` (`muse-spark-1.3`)
- **Summary:** This simulated window run_test_036_w90 reports predicted_class rapid with model_score 0.8472, consistent with rapid behavior. Observed values include 0.6364 at NODE_03 and 9.991 for anomaly_score with t_mid at 227.5.
- **Suggested Check:** Consider rechecking piezo_rms network_mean 0.00858 in the next simulated interval.

### 3. Ground-Truth Label (Withheld from Explainer & Judge)
- **Audit Stratum:** `classifier_mistake` (raw stratum: `classifier_mistake`, substratum: `mistake_gradual_as_rapid`)
- **True Window Label (`true_label`):** `gradual` (Run class: `gradual`)
- **Classifier vs Truth Agreement:** `predicted_class = "rapid"` vs `true_label = "gradual"` **(MISMATCH — Explainer faithfully explains the classifier's output without knowing ground truth)**

---

## Entry 18 / 30 — Window `run_test_048_w64` (Run `run_test_048`)

### 1. Observable Packet Values (Input to Explainer)
- **Time Window:** `160.0 s` to `165.0 s` (midpoint `162.5 s`)
- **Classifier Output:** `predicted_class = "gradual"`, `model_score = 0.3468`
- **Persisted Alarm & Rule Baseline:** `alarm_state = "normal"`, `threshold_rule_class = "rapid"`
- **Advisory Unknown Anomaly Detector:** `unknown_anomaly = False`, `anomaly_score = 7.1233`
- **Per-Node Mean Tilt (deg):** `NODE_01 = 0.3936`, `NODE_02 = 0.5444`, `NODE_03 = 0.6653`, `NODE_04 = 0.3852`
- **Piezo RMS (V):** `network_mean = 0.00823`, `NODE_01 = 0.01012`, `NODE_02 = 0.00481`, `NODE_03 = 0.01089`, `NODE_04 = 0.0071`
- **LiDAR Range Change (m):** `hub_common_mode_m = -0.0125`, `NODE_01 = 0.02`, `NODE_02 = -0.03`, `NODE_03 = -0.04`, `NODE_04 = 0.0`
- **Communications:** `missing_fraction_network = 0.0` (`NODE_01 = 0.0`, `NODE_02 = 0.0`, `NODE_03 = 0.0`, `NODE_04 = 0.0`), `latency_mean_s = 0.4093`, `latency_max_s = 0.691`
- **Radial Convergence & SNR Bin:** `radial_convergence_residual_m = 5.513`, `radial_convergence_weighted_cos = 0.7541`, `snr_vs_imu_noise_bin = "1-2"`

<details><summary>Raw Packet JSON</summary>

```json
{
  "window_id": "run_test_048_w64",
  "run_id": "run_test_048",
  "time_s": {
    "t_start": 160.0,
    "t_end": 165.0,
    "t_mid": 162.5
  },
  "predicted_class": "gradual",
  "model_score": 0.3468,
  "alarm_state": "normal",
  "threshold_rule_class": "rapid",
  "unknown_anomaly": false,
  "anomaly_score": 7.1233,
  "node_mean_tilt_deg": {
    "NODE_01": 0.3936,
    "NODE_02": 0.5444,
    "NODE_03": 0.6653,
    "NODE_04": 0.3852
  },
  "piezo_rms": {
    "network_mean": 0.00823,
    "NODE_01": 0.01012,
    "NODE_02": 0.00481,
    "NODE_03": 0.01089,
    "NODE_04": 0.0071
  },
  "lidar_range_change_m": {
    "hub_common_mode_m": -0.0125,
    "NODE_01": 0.02,
    "NODE_02": -0.03,
    "NODE_03": -0.04,
    "NODE_04": 0.0
  },
  "comm": {
    "missing_fraction_network": 0.0,
    "NODE_01_missing_fraction": 0.0,
    "NODE_02_missing_fraction": 0.0,
    "NODE_03_missing_fraction": 0.0,
    "NODE_04_missing_fraction": 0.0,
    "latency_mean_s": 0.4093,
    "latency_max_s": 0.691
  },
  "radial_convergence_residual_m": 5.513,
  "radial_convergence_weighted_cos": 0.7541,
  "snr_vs_imu_noise_bin": "1-2"
}
```
</details>

### 2. Advisory Explanation Text
- **Source / Model:** `ai` (`muse-spark-1.3`)
- **Summary:** This simulated window run_test_048_w64 from 160.0 to 165.0 is consistent with predicted_class gradual with model_score 0.3468. It lists anomaly_score 7.1233, radial_convergence_residual_m 5.513, and hub_common_mode_m -0.0125 with NODE_01 at 0.02.
- **Suggested Check:** Verify NODE_03 tilt at 0.6653 against piezo_rms for NODE_03 at 0.01089.

### 3. Ground-Truth Label (Withheld from Explainer & Judge)
- **Audit Stratum:** `classifier_mistake` (raw stratum: `classifier_mistake`, substratum: `mistake_rapid_as_gradual`)
- **True Window Label (`true_label`):** `rapid` (Run class: `rapid`)
- **Classifier vs Truth Agreement:** `predicted_class = "gradual"` vs `true_label = "rapid"` **(MISMATCH — Explainer faithfully explains the classifier's output without knowing ground truth)**

---

## Entry 19 / 30 — Window `run_test_051_w36` (Run `run_test_051`)

### 1. Observable Packet Values (Input to Explainer)
- **Time Window:** `90.0 s` to `95.0 s` (midpoint `92.5 s`)
- **Classifier Output:** `predicted_class = "sensor_fault"`, `model_score = 0.3868`
- **Persisted Alarm & Rule Baseline:** `alarm_state = "normal"`, `threshold_rule_class = "rapid"`
- **Advisory Unknown Anomaly Detector:** `unknown_anomaly = False`, `anomaly_score = 23.9199`
- **Per-Node Mean Tilt (deg):** `NODE_01 = 1.9938`, `NODE_02 = 1.3233`, `NODE_03 = 0.3429`, `NODE_04 = 0.4474`
- **Piezo RMS (V):** `network_mean = 0.01166`, `NODE_01 = 0.00641`, `NODE_02 = 0.01037`, `NODE_03 = 0.02357`, `NODE_04 = 0.0063`
- **LiDAR Range Change (m):** `hub_common_mode_m = -0.0087`, `NODE_01 = -0.005`, `NODE_02 = 0.005`, `NODE_03 = 0.02`, `NODE_04 = -0.055`
- **Communications:** `missing_fraction_network = 0.0` (`NODE_01 = 0.0`, `NODE_02 = 0.0`, `NODE_03 = 0.0`, `NODE_04 = 0.0`), `latency_mean_s = 0.4461`, `latency_max_s = 0.786`
- **Radial Convergence & SNR Bin:** `radial_convergence_residual_m = 6.89`, `radial_convergence_weighted_cos = 0.6566`, `snr_vs_imu_noise_bin = ">5"`

<details><summary>Raw Packet JSON</summary>

```json
{
  "window_id": "run_test_051_w36",
  "run_id": "run_test_051",
  "time_s": {
    "t_start": 90.0,
    "t_end": 95.0,
    "t_mid": 92.5
  },
  "predicted_class": "sensor_fault",
  "model_score": 0.3868,
  "alarm_state": "normal",
  "threshold_rule_class": "rapid",
  "unknown_anomaly": false,
  "anomaly_score": 23.9199,
  "node_mean_tilt_deg": {
    "NODE_01": 1.9938,
    "NODE_02": 1.3233,
    "NODE_03": 0.3429,
    "NODE_04": 0.4474
  },
  "piezo_rms": {
    "network_mean": 0.01166,
    "NODE_01": 0.00641,
    "NODE_02": 0.01037,
    "NODE_03": 0.02357,
    "NODE_04": 0.0063
  },
  "lidar_range_change_m": {
    "hub_common_mode_m": -0.0087,
    "NODE_01": -0.005,
    "NODE_02": 0.005,
    "NODE_03": 0.02,
    "NODE_04": -0.055
  },
  "comm": {
    "missing_fraction_network": 0.0,
    "NODE_01_missing_fraction": 0.0,
    "NODE_02_missing_fraction": 0.0,
    "NODE_03_missing_fraction": 0.0,
    "NODE_04_missing_fraction": 0.0,
    "latency_mean_s": 0.4461,
    "latency_max_s": 0.786
  },
  "radial_convergence_residual_m": 6.89,
  "radial_convergence_weighted_cos": 0.6566,
  "snr_vs_imu_noise_bin": ">5"
}
```
</details>

### 2. Advisory Explanation Text
- **Source / Model:** `ai` (`muse-spark-1.3`)
- **Summary:** This simulated packet for run_test_051_w36 with model_score 0.3868 is consistent with predicted_class sensor_fault. It lists t_mid 92.5 and anomaly_score 23.9199 with NODE_03 piezo_rms 0.02357.
- **Suggested Check:** Consider verifying NODE_04 lidar_range_change_m of -0.055 against hub_common_mode_m of -0.0087.

### 3. Ground-Truth Label (Withheld from Explainer & Judge)
- **Audit Stratum:** `classifier_mistake` (raw stratum: `classifier_mistake`, substratum: `mistake_rapid_as_sensor_fault`)
- **True Window Label (`true_label`):** `rapid` (Run class: `rapid`)
- **Classifier vs Truth Agreement:** `predicted_class = "sensor_fault"` vs `true_label = "rapid"` **(MISMATCH — Explainer faithfully explains the classifier's output without knowing ground truth)**

---

## Entry 20 / 30 — Window `run_test_091_w59` (Run `run_test_091`)

### 1. Observable Packet Values (Input to Explainer)
- **Time Window:** `147.5 s` to `152.5 s` (midpoint `150.0 s`)
- **Classifier Output:** `predicted_class = "gradual"`, `model_score = 0.3435`
- **Persisted Alarm & Rule Baseline:** `alarm_state = "gradual"`, `threshold_rule_class = "sensor_fault"`
- **Advisory Unknown Anomaly Detector:** `unknown_anomaly = True`, `anomaly_score = 14.6626`
- **Per-Node Mean Tilt (deg):** `NODE_01 = 0.1261`, `NODE_02 = 0.1199`, `NODE_03 = 0.1794`, `NODE_04 = 1.0825`
- **Piezo RMS (V):** `network_mean = 0.00943`, `NODE_01 = 0.00543`, `NODE_02 = 0.01187`, `NODE_03 = 0.01505`, `NODE_04 = 0.00539`
- **LiDAR Range Change (m):** `hub_common_mode_m = 0.03`, `NODE_01 = 0.06`, `NODE_02 = 0.06`, `NODE_03 = -0.03`, `NODE_04 = 0.03`
- **Communications:** `missing_fraction_network = 0.0` (`NODE_01 = 0.0`, `NODE_02 = 0.0`, `NODE_03 = 0.0`, `NODE_04 = 0.0`), `latency_mean_s = 0.4164`, `latency_max_s = 0.9307`
- **Radial Convergence & SNR Bin:** `radial_convergence_residual_m = 1.802`, `radial_convergence_weighted_cos = 0.993`, `snr_vs_imu_noise_bin = "<0.5"`

<details><summary>Raw Packet JSON</summary>

```json
{
  "window_id": "run_test_091_w59",
  "run_id": "run_test_091",
  "time_s": {
    "t_start": 147.5,
    "t_end": 152.5,
    "t_mid": 150.0
  },
  "predicted_class": "gradual",
  "model_score": 0.3435,
  "alarm_state": "gradual",
  "threshold_rule_class": "sensor_fault",
  "unknown_anomaly": true,
  "anomaly_score": 14.6626,
  "node_mean_tilt_deg": {
    "NODE_01": 0.1261,
    "NODE_02": 0.1199,
    "NODE_03": 0.1794,
    "NODE_04": 1.0825
  },
  "piezo_rms": {
    "network_mean": 0.00943,
    "NODE_01": 0.00543,
    "NODE_02": 0.01187,
    "NODE_03": 0.01505,
    "NODE_04": 0.00539
  },
  "lidar_range_change_m": {
    "hub_common_mode_m": 0.03,
    "NODE_01": 0.06,
    "NODE_02": 0.06,
    "NODE_03": -0.03,
    "NODE_04": 0.03
  },
  "comm": {
    "missing_fraction_network": 0.0,
    "NODE_01_missing_fraction": 0.0,
    "NODE_02_missing_fraction": 0.0,
    "NODE_03_missing_fraction": 0.0,
    "NODE_04_missing_fraction": 0.0,
    "latency_mean_s": 0.4164,
    "latency_max_s": 0.9307
  },
  "radial_convergence_residual_m": 1.802,
  "radial_convergence_weighted_cos": 0.993,
  "snr_vs_imu_noise_bin": "<0.5"
}
```
</details>

### 2. Advisory Explanation Text
- **Source / Model:** `ai` (`muse-spark-1.3`)
- **Summary:** This simulated packet at t_mid 150.0 with model_score 0.3435 is consistent with gradual. It reports NODE_04 tilt of 1.0825 and anomaly_score of 14.6626 with radial_convergence_residual_m of 1.802.
- **Suggested Check:** Consider verifying lidar_range_change_m for NODE_03 at -0.03 for verification.

### 3. Ground-Truth Label (Withheld from Explainer & Judge)
- **Audit Stratum:** `classifier_mistake` (raw stratum: `classifier_mistake`, substratum: `mistake_sensor_fault_as_gradual`)
- **True Window Label (`true_label`):** `sensor_fault` (Run class: `sensor_fault`)
- **Classifier vs Truth Agreement:** `predicted_class = "gradual"` vs `true_label = "sensor_fault"` **(MISMATCH — Explainer faithfully explains the classifier's output without knowing ground truth)**

---

## Entry 21 / 30 — Window `run_test_020_w83` (Run `run_test_020`)

### 1. Observable Packet Values (Input to Explainer)
- **Time Window:** `207.5 s` to `212.5 s` (midpoint `210.0 s`)
- **Classifier Output:** `predicted_class = "rapid"`, `model_score = 0.6802`
- **Persisted Alarm & Rule Baseline:** `alarm_state = "rapid"`, `threshold_rule_class = "gradual"`
- **Advisory Unknown Anomaly Detector:** `unknown_anomaly = True`, `anomaly_score = 23.2631`
- **Per-Node Mean Tilt (deg):** `NODE_01 = 1.0212`, `NODE_02 = 0.7312`, `NODE_03 = 1.8628`, `NODE_04 = 1.2166`
- **Piezo RMS (V):** `network_mean = 0.00748`, `NODE_01 = 0.00508`, `NODE_02 = 0.00804`, `NODE_03 = 0.00384`, `NODE_04 = 0.01293`
- **LiDAR Range Change (m):** `hub_common_mode_m = 0.015`, `NODE_01 = 0.06`, `NODE_02 = -0.08`, `NODE_03 = 0.06`, `NODE_04 = 0.02`
- **Communications:** `missing_fraction_network = 0.0` (`NODE_01 = 0.0`, `NODE_02 = 0.0`, `NODE_03 = 0.0`, `NODE_04 = 0.0`), `latency_mean_s = 0.4323`, `latency_max_s = 0.8964`
- **Radial Convergence & SNR Bin:** `radial_convergence_residual_m = 0.971`, `radial_convergence_weighted_cos = 0.9962`, `snr_vs_imu_noise_bin = ">5"`

<details><summary>Raw Packet JSON</summary>

```json
{
  "window_id": "run_test_020_w83",
  "run_id": "run_test_020",
  "time_s": {
    "t_start": 207.5,
    "t_end": 212.5,
    "t_mid": 210.0
  },
  "predicted_class": "rapid",
  "model_score": 0.6802,
  "alarm_state": "rapid",
  "threshold_rule_class": "gradual",
  "unknown_anomaly": true,
  "anomaly_score": 23.2631,
  "node_mean_tilt_deg": {
    "NODE_01": 1.0212,
    "NODE_02": 0.7312,
    "NODE_03": 1.8628,
    "NODE_04": 1.2166
  },
  "piezo_rms": {
    "network_mean": 0.00748,
    "NODE_01": 0.00508,
    "NODE_02": 0.00804,
    "NODE_03": 0.00384,
    "NODE_04": 0.01293
  },
  "lidar_range_change_m": {
    "hub_common_mode_m": 0.015,
    "NODE_01": 0.06,
    "NODE_02": -0.08,
    "NODE_03": 0.06,
    "NODE_04": 0.02
  },
  "comm": {
    "missing_fraction_network": 0.0,
    "NODE_01_missing_fraction": 0.0,
    "NODE_02_missing_fraction": 0.0,
    "NODE_03_missing_fraction": 0.0,
    "NODE_04_missing_fraction": 0.0,
    "latency_mean_s": 0.4323,
    "latency_max_s": 0.8964
  },
  "radial_convergence_residual_m": 0.971,
  "radial_convergence_weighted_cos": 0.9962,
  "snr_vs_imu_noise_bin": ">5"
}
```
</details>

### 2. Advisory Explanation Text
- **Source / Model:** `ai` (`muse-spark-1.3`)
- **Summary:** This simulated packet for run_test_020_w83 is consistent with predicted_class rapid with model_score 0.6802. At t_mid 210.0 the node_mean_tilt_deg for NODE_03 is 1.8628 and radial_convergence_residual_m is 0.971.
- **Suggested Check:** Consider verifying lidar_range_change_m for NODE_02 at -0.08 alongside latency_max_s at 0.8964.

### 3. Ground-Truth Label (Withheld from Explainer & Judge)
- **Audit Stratum:** `unknown_anomaly_flagged` (raw stratum: `unknown_flag_event`, substratum: `unknown_anomaly_on_gradual`)
- **True Window Label (`true_label`):** `gradual` (Run class: `gradual`)
- **Classifier vs Truth Agreement:** `predicted_class = "rapid"` vs `true_label = "gradual"` **(MISMATCH — Explainer faithfully explains the classifier's output without knowing ground truth)**

---

## Entry 22 / 30 — Window `run_test_057_w87` (Run `run_test_057`)

### 1. Observable Packet Values (Input to Explainer)
- **Time Window:** `217.5 s` to `222.5 s` (midpoint `220.0 s`)
- **Classifier Output:** `predicted_class = "rapid"`, `model_score = 0.8074`
- **Persisted Alarm & Rule Baseline:** `alarm_state = "rapid"`, `threshold_rule_class = "rapid"`
- **Advisory Unknown Anomaly Detector:** `unknown_anomaly = True`, `anomaly_score = 28.1018`
- **Per-Node Mean Tilt (deg):** `NODE_01 = 0.2087`, `NODE_02 = 0.8477`, `NODE_03 = 2.2632`, `NODE_04 = 0.999`
- **Piezo RMS (V):** `network_mean = 0.00937`, `NODE_01 = 0.00634`, `NODE_02 = 0.00738`, `NODE_03 = 0.01005`, `NODE_04 = 0.01373`
- **LiDAR Range Change (m):** `hub_common_mode_m = 0.0025`, `NODE_01 = 0.0`, `NODE_02 = 0.02`, `NODE_03 = -0.01`, `NODE_04 = 0.0`
- **Communications:** `missing_fraction_network = 0.0` (`NODE_01 = 0.0`, `NODE_02 = 0.0`, `NODE_03 = 0.0`, `NODE_04 = 0.0`), `latency_mean_s = 0.475`, `latency_max_s = 0.9728`
- **Radial Convergence & SNR Bin:** `radial_convergence_residual_m = 2.186`, `radial_convergence_weighted_cos = 0.9984`, `snr_vs_imu_noise_bin = ">5"`

<details><summary>Raw Packet JSON</summary>

```json
{
  "window_id": "run_test_057_w87",
  "run_id": "run_test_057",
  "time_s": {
    "t_start": 217.5,
    "t_end": 222.5,
    "t_mid": 220.0
  },
  "predicted_class": "rapid",
  "model_score": 0.8074,
  "alarm_state": "rapid",
  "threshold_rule_class": "rapid",
  "unknown_anomaly": true,
  "anomaly_score": 28.1018,
  "node_mean_tilt_deg": {
    "NODE_01": 0.2087,
    "NODE_02": 0.8477,
    "NODE_03": 2.2632,
    "NODE_04": 0.999
  },
  "piezo_rms": {
    "network_mean": 0.00937,
    "NODE_01": 0.00634,
    "NODE_02": 0.00738,
    "NODE_03": 0.01005,
    "NODE_04": 0.01373
  },
  "lidar_range_change_m": {
    "hub_common_mode_m": 0.0025,
    "NODE_01": 0.0,
    "NODE_02": 0.02,
    "NODE_03": -0.01,
    "NODE_04": 0.0
  },
  "comm": {
    "missing_fraction_network": 0.0,
    "NODE_01_missing_fraction": 0.0,
    "NODE_02_missing_fraction": 0.0,
    "NODE_03_missing_fraction": 0.0,
    "NODE_04_missing_fraction": 0.0,
    "latency_mean_s": 0.475,
    "latency_max_s": 0.9728
  },
  "radial_convergence_residual_m": 2.186,
  "radial_convergence_weighted_cos": 0.9984,
  "snr_vs_imu_noise_bin": ">5"
}
```
</details>

### 2. Advisory Explanation Text
- **Source / Model:** `ai` (`muse-spark-1.3`)
- **Summary:** This simulated packet for window run_test_057_w87 shows predicted_class rapid with model_score 0.8074. The pattern is consistent with rapid, with NODE_03 at 2.2632 and radial_convergence_residual_m 2.186. Telemetry includes t_mid 220.0 and anomaly_score 28.1018 with unknown_anomaly true.
- **Suggested Check:** Consider verifying the NODE_03 tilt reading of 2.2632 against raw telemetry.

### 3. Ground-Truth Label (Withheld from Explainer & Judge)
- **Audit Stratum:** `unknown_anomaly_flagged` (raw stratum: `unknown_flag_event`, substratum: `unknown_anomaly_on_rapid`)
- **True Window Label (`true_label`):** `rapid` (Run class: `rapid`)
- **Classifier vs Truth Agreement:** `predicted_class = "rapid"` vs `true_label = "rapid"` **(MATCH)**

---

## Entry 23 / 30 — Window `run_test_077_w44` (Run `run_test_077`)

### 1. Observable Packet Values (Input to Explainer)
- **Time Window:** `110.0 s` to `115.0 s` (midpoint `112.5 s`)
- **Classifier Output:** `predicted_class = "abnormal_vibration"`, `model_score = 1.0`
- **Persisted Alarm & Rule Baseline:** `alarm_state = "abnormal_vibration"`, `threshold_rule_class = "abnormal_vibration"`
- **Advisory Unknown Anomaly Detector:** `unknown_anomaly = True`, `anomaly_score = 93.3664`
- **Per-Node Mean Tilt (deg):** `NODE_01 = 0.3476`, `NODE_02 = 0.5091`, `NODE_03 = 0.2934`, `NODE_04 = 0.2701`
- **Piezo RMS (V):** `network_mean = 0.10578`, `NODE_01 = 0.13167`, `NODE_02 = 0.10216`, `NODE_03 = 0.08944`, `NODE_04 = 0.09985`
- **LiDAR Range Change (m):** `hub_common_mode_m = -0.0125`, `NODE_01 = 0.01`, `NODE_02 = -0.015`, `NODE_03 = -0.035`, `NODE_04 = -0.01`
- **Communications:** `missing_fraction_network = 0.1` (`NODE_01 = 0.0`, `NODE_02 = 0.2`, `NODE_03 = 0.0`, `NODE_04 = 0.2`), `latency_mean_s = 0.4242`, `latency_max_s = 0.7319`
- **Radial Convergence & SNR Bin:** `radial_convergence_residual_m = 8.145`, `radial_convergence_weighted_cos = -0.7553`, `snr_vs_imu_noise_bin = "<0.5"`

<details><summary>Raw Packet JSON</summary>

```json
{
  "window_id": "run_test_077_w44",
  "run_id": "run_test_077",
  "time_s": {
    "t_start": 110.0,
    "t_end": 115.0,
    "t_mid": 112.5
  },
  "predicted_class": "abnormal_vibration",
  "model_score": 1.0,
  "alarm_state": "abnormal_vibration",
  "threshold_rule_class": "abnormal_vibration",
  "unknown_anomaly": true,
  "anomaly_score": 93.3664,
  "node_mean_tilt_deg": {
    "NODE_01": 0.3476,
    "NODE_02": 0.5091,
    "NODE_03": 0.2934,
    "NODE_04": 0.2701
  },
  "piezo_rms": {
    "network_mean": 0.10578,
    "NODE_01": 0.13167,
    "NODE_02": 0.10216,
    "NODE_03": 0.08944,
    "NODE_04": 0.09985
  },
  "lidar_range_change_m": {
    "hub_common_mode_m": -0.0125,
    "NODE_01": 0.01,
    "NODE_02": -0.015,
    "NODE_03": -0.035,
    "NODE_04": -0.01
  },
  "comm": {
    "missing_fraction_network": 0.1,
    "NODE_01_missing_fraction": 0.0,
    "NODE_02_missing_fraction": 0.2,
    "NODE_03_missing_fraction": 0.0,
    "NODE_04_missing_fraction": 0.2,
    "latency_mean_s": 0.4242,
    "latency_max_s": 0.7319
  },
  "radial_convergence_residual_m": 8.145,
  "radial_convergence_weighted_cos": -0.7553,
  "snr_vs_imu_noise_bin": "<0.5"
}
```
</details>

### 2. Advisory Explanation Text
- **Source / Model:** `ai` (`muse-spark-1.3`)
- **Summary:** This simulated packet for run_test_077_w44 is consistent with abnormal_vibration with model_score 1.0. It lists anomaly_score 93.3664 and piezo_rms network_mean 0.10578 with radial_convergence_residual_m 8.145. The window 110.0 to 115.0 includes hub_common_mode_m -0.0125.
- **Suggested Check:** Consider verifying piezo_rms for NODE_01 at 0.13167.

### 3. Ground-Truth Label (Withheld from Explainer & Judge)
- **Audit Stratum:** `unknown_anomaly_flagged` (raw stratum: `unknown_flag_event`, substratum: `unknown_anomaly_on_abnormal_vibration`)
- **True Window Label (`true_label`):** `abnormal_vibration` (Run class: `abnormal_vibration`)
- **Classifier vs Truth Agreement:** `predicted_class = "abnormal_vibration"` vs `true_label = "abnormal_vibration"` **(MATCH)**

---

## Entry 24 / 30 — Window `run_test_101_w34` (Run `run_test_101`)

### 1. Observable Packet Values (Input to Explainer)
- **Time Window:** `85.0 s` to `90.0 s` (midpoint `87.5 s`)
- **Classifier Output:** `predicted_class = "comm_fault"`, `model_score = 0.9128`
- **Persisted Alarm & Rule Baseline:** `alarm_state = "comm_fault"`, `threshold_rule_class = "comm_fault"`
- **Advisory Unknown Anomaly Detector:** `unknown_anomaly = True`, `anomaly_score = 19.8139`
- **Per-Node Mean Tilt (deg):** `NODE_01 = 0.2633`, `NODE_02 = 0.44`, `NODE_03 = 0.1936`, `NODE_04 = 0.2874`
- **Piezo RMS (V):** `network_mean = 0.00672`, `NODE_01 = 0.00782`, `NODE_02 = 0.00691`, `NODE_03 = 0.00564`, `NODE_04 = 0.00649`
- **LiDAR Range Change (m):** `hub_common_mode_m = -0.0125`, `NODE_01 = 0.05`, `NODE_02 = -0.005`, `NODE_03 = -0.065`, `NODE_04 = -0.03`
- **Communications:** `missing_fraction_network = 0.2` (`NODE_01 = 0.8`, `NODE_02 = 0.0`, `NODE_03 = 0.0`, `NODE_04 = 0.0`), `latency_mean_s = 0.464`, `latency_max_s = 1.0086`
- **Radial Convergence & SNR Bin:** `radial_convergence_residual_m = 25.272`, `radial_convergence_weighted_cos = -0.943`, `snr_vs_imu_noise_bin = "<0.5"`

<details><summary>Raw Packet JSON</summary>

```json
{
  "window_id": "run_test_101_w34",
  "run_id": "run_test_101",
  "time_s": {
    "t_start": 85.0,
    "t_end": 90.0,
    "t_mid": 87.5
  },
  "predicted_class": "comm_fault",
  "model_score": 0.9128,
  "alarm_state": "comm_fault",
  "threshold_rule_class": "comm_fault",
  "unknown_anomaly": true,
  "anomaly_score": 19.8139,
  "node_mean_tilt_deg": {
    "NODE_01": 0.2633,
    "NODE_02": 0.44,
    "NODE_03": 0.1936,
    "NODE_04": 0.2874
  },
  "piezo_rms": {
    "network_mean": 0.00672,
    "NODE_01": 0.00782,
    "NODE_02": 0.00691,
    "NODE_03": 0.00564,
    "NODE_04": 0.00649
  },
  "lidar_range_change_m": {
    "hub_common_mode_m": -0.0125,
    "NODE_01": 0.05,
    "NODE_02": -0.005,
    "NODE_03": -0.065,
    "NODE_04": -0.03
  },
  "comm": {
    "missing_fraction_network": 0.2,
    "NODE_01_missing_fraction": 0.8,
    "NODE_02_missing_fraction": 0.0,
    "NODE_03_missing_fraction": 0.0,
    "NODE_04_missing_fraction": 0.0,
    "latency_mean_s": 0.464,
    "latency_max_s": 1.0086
  },
  "radial_convergence_residual_m": 25.272,
  "radial_convergence_weighted_cos": -0.943,
  "snr_vs_imu_noise_bin": "<0.5"
}
```
</details>

### 2. Advisory Explanation Text
- **Source / Model:** `ai` (`muse-spark-1.3`)
- **Summary:** This simulated packet for run_test_101_w34 is consistent with predicted_class comm_fault with model_score 0.9128. It shows NODE_01_missing_fraction 0.8 and missing_fraction_network 0.2 with latency_max_s 1.0086. The anomaly_score is 19.8139 and lidar_range_change_m for NODE_01 is 0.05.
- **Suggested Check:** Consider verifying NODE_01 telemetry continuity given NODE_01_missing_fraction 0.8 and latency_mean_s 0.464.

### 3. Ground-Truth Label (Withheld from Explainer & Judge)
- **Audit Stratum:** `unknown_anomaly_flagged` (raw stratum: `unknown_flag_event`, substratum: `unknown_anomaly_on_comm_fault`)
- **True Window Label (`true_label`):** `comm_fault` (Run class: `comm_fault`)
- **Classifier vs Truth Agreement:** `predicted_class = "comm_fault"` vs `true_label = "comm_fault"` **(MATCH)**

---

## Entry 25 / 30 — Window `run_test_107_w67` (Run `run_test_107`)

### 1. Observable Packet Values (Input to Explainer)
- **Time Window:** `167.5 s` to `172.5 s` (midpoint `170.0 s`)
- **Classifier Output:** `predicted_class = "normal"`, `model_score = 0.3378`
- **Persisted Alarm & Rule Baseline:** `alarm_state = "normal"`, `threshold_rule_class = "normal"`
- **Advisory Unknown Anomaly Detector:** `unknown_anomaly = True`, `anomaly_score = 8.1908`
- **Per-Node Mean Tilt (deg):** `NODE_01 = 0.3974`, `NODE_02 = 0.2699`, `NODE_03 = 0.4298`, `NODE_04 = 0.3452`
- **Piezo RMS (V):** `network_mean = 0.00641`, `NODE_01 = 0.01254`, `NODE_02 = 0.00367`, `NODE_03 = 0.00545`, `NODE_04 = 0.00401`
- **LiDAR Range Change (m):** `hub_common_mode_m = 0.01`, `NODE_01 = 0.03`, `NODE_02 = 0.03`, `NODE_03 = -0.05`, `NODE_04 = 0.03`
- **Communications:** `missing_fraction_network = 0.05` (`NODE_01 = 0.2`, `NODE_02 = 0.0`, `NODE_03 = 0.0`, `NODE_04 = 0.0`), `latency_mean_s = 0.4407`, `latency_max_s = 0.8578`
- **Radial Convergence & SNR Bin:** `radial_convergence_residual_m = 10.711`, `radial_convergence_weighted_cos = 0.2537`, `snr_vs_imu_noise_bin = "<0.5"`

<details><summary>Raw Packet JSON</summary>

```json
{
  "window_id": "run_test_107_w67",
  "run_id": "run_test_107",
  "time_s": {
    "t_start": 167.5,
    "t_end": 172.5,
    "t_mid": 170.0
  },
  "predicted_class": "normal",
  "model_score": 0.3378,
  "alarm_state": "normal",
  "threshold_rule_class": "normal",
  "unknown_anomaly": true,
  "anomaly_score": 8.1908,
  "node_mean_tilt_deg": {
    "NODE_01": 0.3974,
    "NODE_02": 0.2699,
    "NODE_03": 0.4298,
    "NODE_04": 0.3452
  },
  "piezo_rms": {
    "network_mean": 0.00641,
    "NODE_01": 0.01254,
    "NODE_02": 0.00367,
    "NODE_03": 0.00545,
    "NODE_04": 0.00401
  },
  "lidar_range_change_m": {
    "hub_common_mode_m": 0.01,
    "NODE_01": 0.03,
    "NODE_02": 0.03,
    "NODE_03": -0.05,
    "NODE_04": 0.03
  },
  "comm": {
    "missing_fraction_network": 0.05,
    "NODE_01_missing_fraction": 0.2,
    "NODE_02_missing_fraction": 0.0,
    "NODE_03_missing_fraction": 0.0,
    "NODE_04_missing_fraction": 0.0,
    "latency_mean_s": 0.4407,
    "latency_max_s": 0.8578
  },
  "radial_convergence_residual_m": 10.711,
  "radial_convergence_weighted_cos": 0.2537,
  "snr_vs_imu_noise_bin": "<0.5"
}
```
</details>

### 2. Advisory Explanation Text
- **Source / Model:** `ai` (`muse-spark-1.3`)
- **Summary:** This simulated packet is consistent with predicted_class normal with model_score 0.3378. It reports t_mid 170.0 and anomaly_score 8.1908 with missing_fraction_network 0.05.
- **Suggested Check:** Consider verifying piezo_rms network_mean 0.00641 against node telemetry.

### 3. Ground-Truth Label (Withheld from Explainer & Judge)
- **Audit Stratum:** `unknown_anomaly_flagged` (raw stratum: `unknown_flag_normal`, substratum: `unknown_anomaly_on_normal`)
- **True Window Label (`true_label`):** `normal` (Run class: `comm_fault`)
- **Classifier vs Truth Agreement:** `predicted_class = "normal"` vs `true_label = "normal"` **(MATCH)**

---

## Entry 26 / 30 — Window `run_test_086_w34` (Run `run_test_086`)

### 1. Observable Packet Values (Input to Explainer)
- **Time Window:** `85.0 s` to `90.0 s` (midpoint `87.5 s`)
- **Classifier Output:** `predicted_class = "normal"`, `model_score = 0.4289`
- **Persisted Alarm & Rule Baseline:** `alarm_state = "normal"`, `threshold_rule_class = "normal"`
- **Advisory Unknown Anomaly Detector:** `unknown_anomaly = False`, `anomaly_score = 7.288`
- **Per-Node Mean Tilt (deg):** `NODE_01 = 0.1739`, `NODE_02 = 0.2024`, `NODE_03 = 0.2763`, `NODE_04 = 0.2514`
- **Piezo RMS (V):** `network_mean = 0.0073`, `NODE_01 = 0.00517`, `NODE_02 = 0.0064`, `NODE_03 = 0.01271`, `NODE_04 = 0.00491`
- **LiDAR Range Change (m):** `hub_common_mode_m = -0.0537`, `NODE_01 = -0.06`, `NODE_02 = -0.05`, `NODE_03 = -0.035`, `NODE_04 = -0.07`
- **Communications:** `missing_fraction_network = 0.0` (`NODE_01 = 0.0`, `NODE_02 = 0.0`, `NODE_03 = 0.0`, `NODE_04 = 0.0`), `latency_mean_s = 0.3653`, `latency_max_s = 0.6541`
- **Radial Convergence & SNR Bin:** `radial_convergence_residual_m = 11.376`, `radial_convergence_weighted_cos = 0.1996`, `snr_vs_imu_noise_bin = "<0.5"`

<details><summary>Raw Packet JSON</summary>

```json
{
  "window_id": "run_test_086_w34",
  "run_id": "run_test_086",
  "time_s": {
    "t_start": 85.0,
    "t_end": 90.0,
    "t_mid": 87.5
  },
  "predicted_class": "normal",
  "model_score": 0.4289,
  "alarm_state": "normal",
  "threshold_rule_class": "normal",
  "unknown_anomaly": false,
  "anomaly_score": 7.288,
  "node_mean_tilt_deg": {
    "NODE_01": 0.1739,
    "NODE_02": 0.2024,
    "NODE_03": 0.2763,
    "NODE_04": 0.2514
  },
  "piezo_rms": {
    "network_mean": 0.0073,
    "NODE_01": 0.00517,
    "NODE_02": 0.0064,
    "NODE_03": 0.01271,
    "NODE_04": 0.00491
  },
  "lidar_range_change_m": {
    "hub_common_mode_m": -0.0537,
    "NODE_01": -0.06,
    "NODE_02": -0.05,
    "NODE_03": -0.035,
    "NODE_04": -0.07
  },
  "comm": {
    "missing_fraction_network": 0.0,
    "NODE_01_missing_fraction": 0.0,
    "NODE_02_missing_fraction": 0.0,
    "NODE_03_missing_fraction": 0.0,
    "NODE_04_missing_fraction": 0.0,
    "latency_mean_s": 0.3653,
    "latency_max_s": 0.6541
  },
  "radial_convergence_residual_m": 11.376,
  "radial_convergence_weighted_cos": 0.1996,
  "snr_vs_imu_noise_bin": "<0.5"
}
```
</details>

### 2. Advisory Explanation Text
- **Source / Model:** `ai` (`muse-spark-1.3`)
- **Summary:** This simulated packet for run_test_086_w34 is consistent with predicted_class normal with model_score 0.4289 and alarm_state normal. It shows t_mid 87.5, anomaly_score 7.288, latency_mean_s 0.3653 and missing_fraction_network 0.0.
- **Suggested Check:** Verify latency_max_s reading of 0.6541 against recent telemetry.

### 3. Ground-Truth Label (Withheld from Explainer & Judge)
- **Audit Stratum:** `fault_window` (raw stratum: `fault_window`, substratum: `fault_window_sensor_fault`)
- **True Window Label (`true_label`):** `sensor_fault` (Run class: `sensor_fault`)
- **Classifier vs Truth Agreement:** `predicted_class = "normal"` vs `true_label = "sensor_fault"` **(MISMATCH — Explainer faithfully explains the classifier's output without knowing ground truth)**

---

## Entry 27 / 30 — Window `run_test_091_w50` (Run `run_test_091`)

### 1. Observable Packet Values (Input to Explainer)
- **Time Window:** `125.0 s` to `130.0 s` (midpoint `127.5 s`)
- **Classifier Output:** `predicted_class = "normal"`, `model_score = 0.4035`
- **Persisted Alarm & Rule Baseline:** `alarm_state = "normal"`, `threshold_rule_class = "sensor_fault"`
- **Advisory Unknown Anomaly Detector:** `unknown_anomaly = False`, `anomaly_score = 10.2088`
- **Per-Node Mean Tilt (deg):** `NODE_01 = 0.1274`, `NODE_02 = 0.2037`, `NODE_03 = 0.2286`, `NODE_04 = 0.7375`
- **Piezo RMS (V):** `network_mean = 0.01092`, `NODE_01 = 0.00839`, `NODE_02 = 0.01397`, `NODE_03 = 0.0114`, `NODE_04 = 0.00991`
- **LiDAR Range Change (m):** `hub_common_mode_m = 0.005`, `NODE_01 = 0.06`, `NODE_02 = -0.03`, `NODE_03 = -0.04`, `NODE_04 = 0.03`
- **Communications:** `missing_fraction_network = 0.0` (`NODE_01 = 0.0`, `NODE_02 = 0.0`, `NODE_03 = 0.0`, `NODE_04 = 0.0`), `latency_mean_s = 0.4282`, `latency_max_s = 0.9006`
- **Radial Convergence & SNR Bin:** `radial_convergence_residual_m = 1.404`, `radial_convergence_weighted_cos = 0.9875`, `snr_vs_imu_noise_bin = "<0.5"`

<details><summary>Raw Packet JSON</summary>

```json
{
  "window_id": "run_test_091_w50",
  "run_id": "run_test_091",
  "time_s": {
    "t_start": 125.0,
    "t_end": 130.0,
    "t_mid": 127.5
  },
  "predicted_class": "normal",
  "model_score": 0.4035,
  "alarm_state": "normal",
  "threshold_rule_class": "sensor_fault",
  "unknown_anomaly": false,
  "anomaly_score": 10.2088,
  "node_mean_tilt_deg": {
    "NODE_01": 0.1274,
    "NODE_02": 0.2037,
    "NODE_03": 0.2286,
    "NODE_04": 0.7375
  },
  "piezo_rms": {
    "network_mean": 0.01092,
    "NODE_01": 0.00839,
    "NODE_02": 0.01397,
    "NODE_03": 0.0114,
    "NODE_04": 0.00991
  },
  "lidar_range_change_m": {
    "hub_common_mode_m": 0.005,
    "NODE_01": 0.06,
    "NODE_02": -0.03,
    "NODE_03": -0.04,
    "NODE_04": 0.03
  },
  "comm": {
    "missing_fraction_network": 0.0,
    "NODE_01_missing_fraction": 0.0,
    "NODE_02_missing_fraction": 0.0,
    "NODE_03_missing_fraction": 0.0,
    "NODE_04_missing_fraction": 0.0,
    "latency_mean_s": 0.4282,
    "latency_max_s": 0.9006
  },
  "radial_convergence_residual_m": 1.404,
  "radial_convergence_weighted_cos": 0.9875,
  "snr_vs_imu_noise_bin": "<0.5"
}
```
</details>

### 2. Advisory Explanation Text
- **Source / Model:** `ai` (`muse-spark-1.3`)
- **Summary:** This simulated window shows predicted_class normal with model_score 0.4035, consistent with normal behavior. Values include 0.1274 and 0.005 with anomaly_score 10.2088 and latency_mean_s 0.4282.
- **Suggested Check:** Consider verifying latency_mean_s telemetry for continued consistency with the simulated packet.

### 3. Ground-Truth Label (Withheld from Explainer & Judge)
- **Audit Stratum:** `fault_window` (raw stratum: `fault_window`, substratum: `fault_window_sensor_fault`)
- **True Window Label (`true_label`):** `sensor_fault` (Run class: `sensor_fault`)
- **Classifier vs Truth Agreement:** `predicted_class = "normal"` vs `true_label = "sensor_fault"` **(MISMATCH — Explainer faithfully explains the classifier's output without knowing ground truth)**

---

## Entry 28 / 30 — Window `run_test_112_w33` (Run `run_test_112`)

### 1. Observable Packet Values (Input to Explainer)
- **Time Window:** `82.5 s` to `87.5 s` (midpoint `85.0 s`)
- **Classifier Output:** `predicted_class = "comm_fault"`, `model_score = 0.9906`
- **Persisted Alarm & Rule Baseline:** `alarm_state = "comm_fault"`, `threshold_rule_class = "comm_fault"`
- **Advisory Unknown Anomaly Detector:** `unknown_anomaly = True`, `anomaly_score = 35.5154`
- **Per-Node Mean Tilt (deg):** `NODE_01 = 0.2959`, `NODE_02 = 0.226`, `NODE_03 = 0.2021`, `NODE_04 = 0.2424`
- **Piezo RMS (V):** `network_mean = 0.0065`, `NODE_01 = 0.00607`, `NODE_02 = 0.00435`, `NODE_03 = 0.00599`, `NODE_04 = 0.00957`
- **LiDAR Range Change (m):** `hub_common_mode_m = 0.0388`, `NODE_01 = 0.02`, `NODE_02 = 0.015`, `NODE_03 = 0.06`, `NODE_04 = 0.06`
- **Communications:** `missing_fraction_network = 0.05` (`NODE_01 = 0.0`, `NODE_02 = 0.0`, `NODE_03 = 0.2`, `NODE_04 = 0.0`), `latency_mean_s = 1.7529`, `latency_max_s = 5.9585`
- **Radial Convergence & SNR Bin:** `radial_convergence_residual_m = 16.579`, `radial_convergence_weighted_cos = -0.9479`, `snr_vs_imu_noise_bin = "<0.5"`

<details><summary>Raw Packet JSON</summary>

```json
{
  "window_id": "run_test_112_w33",
  "run_id": "run_test_112",
  "time_s": {
    "t_start": 82.5,
    "t_end": 87.5,
    "t_mid": 85.0
  },
  "predicted_class": "comm_fault",
  "model_score": 0.9906,
  "alarm_state": "comm_fault",
  "threshold_rule_class": "comm_fault",
  "unknown_anomaly": true,
  "anomaly_score": 35.5154,
  "node_mean_tilt_deg": {
    "NODE_01": 0.2959,
    "NODE_02": 0.226,
    "NODE_03": 0.2021,
    "NODE_04": 0.2424
  },
  "piezo_rms": {
    "network_mean": 0.0065,
    "NODE_01": 0.00607,
    "NODE_02": 0.00435,
    "NODE_03": 0.00599,
    "NODE_04": 0.00957
  },
  "lidar_range_change_m": {
    "hub_common_mode_m": 0.0388,
    "NODE_01": 0.02,
    "NODE_02": 0.015,
    "NODE_03": 0.06,
    "NODE_04": 0.06
  },
  "comm": {
    "missing_fraction_network": 0.05,
    "NODE_01_missing_fraction": 0.0,
    "NODE_02_missing_fraction": 0.0,
    "NODE_03_missing_fraction": 0.2,
    "NODE_04_missing_fraction": 0.0,
    "latency_mean_s": 1.7529,
    "latency_max_s": 5.9585
  },
  "radial_convergence_residual_m": 16.579,
  "radial_convergence_weighted_cos": -0.9479,
  "snr_vs_imu_noise_bin": "<0.5"
}
```
</details>

### 2. Advisory Explanation Text
- **Source / Model:** `ai` (`muse-spark-1.3`)
- **Summary:** This simulated packet at t_mid 85.0 shows predicted_class comm_fault with model_score 0.9906, consistent with observed telemetry. NODE_03 shows 0.2 missing fraction and latency_max_s of 5.9585 was recorded.
- **Suggested Check:** Verify NODE_03 communication link and review latency telemetry around 85.0.

### 3. Ground-Truth Label (Withheld from Explainer & Judge)
- **Audit Stratum:** `fault_window` (raw stratum: `fault_window`, substratum: `fault_window_comm_fault`)
- **True Window Label (`true_label`):** `comm_fault` (Run class: `comm_fault`)
- **Classifier vs Truth Agreement:** `predicted_class = "comm_fault"` vs `true_label = "comm_fault"` **(MATCH)**

---

## Entry 29 / 30 — Window `run_test_117_w51` (Run `run_test_117`)

### 1. Observable Packet Values (Input to Explainer)
- **Time Window:** `127.5 s` to `132.5 s` (midpoint `130.0 s`)
- **Classifier Output:** `predicted_class = "comm_fault"`, `model_score = 0.9764`
- **Persisted Alarm & Rule Baseline:** `alarm_state = "comm_fault"`, `threshold_rule_class = "comm_fault"`
- **Advisory Unknown Anomaly Detector:** `unknown_anomaly = True`, `anomaly_score = 34.0049`
- **Per-Node Mean Tilt (deg):** `NODE_01 = 0.1747`, `NODE_02 = 0.343`, `NODE_03 = 0.1846`, `NODE_04 = 0.1923`
- **Piezo RMS (V):** `network_mean = 0.00923`, `NODE_01 = 0.01019`, `NODE_02 = 0.01235`, `NODE_03 = 0.00402`, `NODE_04 = 0.01035`
- **LiDAR Range Change (m):** `hub_common_mode_m = 0.0062`, `NODE_01 = 0.03`, `NODE_02 = -0.015`, `NODE_03 = 0.04`, `NODE_04 = -0.03`
- **Communications:** `missing_fraction_network = 0.0` (`NODE_01 = 0.0`, `NODE_02 = 0.0`, `NODE_03 = 0.0`, `NODE_04 = 0.0`), `latency_mean_s = 1.7089`, `latency_max_s = 5.6553`
- **Radial Convergence & SNR Bin:** `radial_convergence_residual_m = 7.237`, `radial_convergence_weighted_cos = 0.2948`, `snr_vs_imu_noise_bin = "<0.5"`

<details><summary>Raw Packet JSON</summary>

```json
{
  "window_id": "run_test_117_w51",
  "run_id": "run_test_117",
  "time_s": {
    "t_start": 127.5,
    "t_end": 132.5,
    "t_mid": 130.0
  },
  "predicted_class": "comm_fault",
  "model_score": 0.9764,
  "alarm_state": "comm_fault",
  "threshold_rule_class": "comm_fault",
  "unknown_anomaly": true,
  "anomaly_score": 34.0049,
  "node_mean_tilt_deg": {
    "NODE_01": 0.1747,
    "NODE_02": 0.343,
    "NODE_03": 0.1846,
    "NODE_04": 0.1923
  },
  "piezo_rms": {
    "network_mean": 0.00923,
    "NODE_01": 0.01019,
    "NODE_02": 0.01235,
    "NODE_03": 0.00402,
    "NODE_04": 0.01035
  },
  "lidar_range_change_m": {
    "hub_common_mode_m": 0.0062,
    "NODE_01": 0.03,
    "NODE_02": -0.015,
    "NODE_03": 0.04,
    "NODE_04": -0.03
  },
  "comm": {
    "missing_fraction_network": 0.0,
    "NODE_01_missing_fraction": 0.0,
    "NODE_02_missing_fraction": 0.0,
    "NODE_03_missing_fraction": 0.0,
    "NODE_04_missing_fraction": 0.0,
    "latency_mean_s": 1.7089,
    "latency_max_s": 5.6553
  },
  "radial_convergence_residual_m": 7.237,
  "radial_convergence_weighted_cos": 0.2948,
  "snr_vs_imu_noise_bin": "<0.5"
}
```
</details>

### 2. Advisory Explanation Text
- **Source / Model:** `ai` (`muse-spark-1.3`)
- **Summary:** This simulated window run_test_117_w51 with t_mid 130.0 is consistent with predicted_class comm_fault with model_score 0.9764. Latency values of 1.7089 and 5.6553 were recorded with anomaly_score 34.0049.
- **Suggested Check:** Verify latency_max_s telemetry for this simulated window.

### 3. Ground-Truth Label (Withheld from Explainer & Judge)
- **Audit Stratum:** `fault_window` (raw stratum: `fault_window`, substratum: `fault_window_comm_fault`)
- **True Window Label (`true_label`):** `comm_fault` (Run class: `comm_fault`)
- **Classifier vs Truth Agreement:** `predicted_class = "comm_fault"` vs `true_label = "comm_fault"` **(MATCH)**

---

## Entry 30 / 30 — Window `run_test_118_w43` (Run `run_test_118`)

### 1. Observable Packet Values (Input to Explainer)
- **Time Window:** `107.5 s` to `112.5 s` (midpoint `110.0 s`)
- **Classifier Output:** `predicted_class = "comm_fault"`, `model_score = 0.9631`
- **Persisted Alarm & Rule Baseline:** `alarm_state = "comm_fault"`, `threshold_rule_class = "comm_fault"`
- **Advisory Unknown Anomaly Detector:** `unknown_anomaly = True`, `anomaly_score = 5000.1544`
- **Per-Node Mean Tilt (deg):** `NODE_01 = 0.0`, `NODE_02 = 0.2009`, `NODE_03 = 0.4633`, `NODE_04 = 0.0`
- **Piezo RMS (V):** `network_mean = 0.00241`, `NODE_01 = 0.0`, `NODE_02 = 0.00488`, `NODE_03 = 0.00476`, `NODE_04 = 0.0`
- **LiDAR Range Change (m):** `hub_common_mode_m = 0.0375`, `NODE_01 = 0.07`, `NODE_02 = 0.065`, `NODE_03 = 0.045`, `NODE_04 = -0.03`
- **Communications:** `missing_fraction_network = 0.5` (`NODE_01 = 1.0`, `NODE_02 = 0.0`, `NODE_03 = 0.0`, `NODE_04 = 1.0`), `latency_mean_s = 0.4627`, `latency_max_s = 1.0563`
- **Radial Convergence & SNR Bin:** `radial_convergence_residual_m = 17.143`, `radial_convergence_weighted_cos = -1.0`, `snr_vs_imu_noise_bin = "<0.5"`

<details><summary>Raw Packet JSON</summary>

```json
{
  "window_id": "run_test_118_w43",
  "run_id": "run_test_118",
  "time_s": {
    "t_start": 107.5,
    "t_end": 112.5,
    "t_mid": 110.0
  },
  "predicted_class": "comm_fault",
  "model_score": 0.9631,
  "alarm_state": "comm_fault",
  "threshold_rule_class": "comm_fault",
  "unknown_anomaly": true,
  "anomaly_score": 5000.1544,
  "node_mean_tilt_deg": {
    "NODE_01": 0.0,
    "NODE_02": 0.2009,
    "NODE_03": 0.4633,
    "NODE_04": 0.0
  },
  "piezo_rms": {
    "network_mean": 0.00241,
    "NODE_01": 0.0,
    "NODE_02": 0.00488,
    "NODE_03": 0.00476,
    "NODE_04": 0.0
  },
  "lidar_range_change_m": {
    "hub_common_mode_m": 0.0375,
    "NODE_01": 0.07,
    "NODE_02": 0.065,
    "NODE_03": 0.045,
    "NODE_04": -0.03
  },
  "comm": {
    "missing_fraction_network": 0.5,
    "NODE_01_missing_fraction": 1.0,
    "NODE_02_missing_fraction": 0.0,
    "NODE_03_missing_fraction": 0.0,
    "NODE_04_missing_fraction": 1.0,
    "latency_mean_s": 0.4627,
    "latency_max_s": 1.0563
  },
  "radial_convergence_residual_m": 17.143,
  "radial_convergence_weighted_cos": -1.0,
  "snr_vs_imu_noise_bin": "<0.5"
}
```
</details>

### 2. Advisory Explanation Text
- **Source / Model:** `ai` (`muse-spark-1.3`)
- **Summary:** This simulated window run_test_118_w43 at 110.0 reports predicted_class comm_fault with model_score 0.9631, consistent with telemetry patterns. Comm fields show 0.5 missing_fraction_network with 1.0 for NODE_01_missing_fraction and 1.0 for NODE_04_missing_fraction alongside anomaly_score 5000.1544.
- **Suggested Check:** Verify NODE_01_missing_fraction value of 1.0 against latency_max_s value of 1.0563 in the next simulated check.

### 3. Ground-Truth Label (Withheld from Explainer & Judge)
- **Audit Stratum:** `fault_window` (raw stratum: `fault_window`, substratum: `fault_window_comm_fault`)
- **True Window Label (`true_label`):** `comm_fault` (Run class: `comm_fault`)
- **Classifier vs Truth Agreement:** `predicted_class = "comm_fault"` vs `true_label = "comm_fault"` **(MATCH)**

---
