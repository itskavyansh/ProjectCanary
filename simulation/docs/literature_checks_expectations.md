# Pre-Registered Expectations: Literature-Basis Sensitivity Checks (SYNTHETIC / SIMULATION)

> [!CAUTION]
> **SYNTHETIC / SIMULATION — PRE-REGISTERED HYPOTHESES (WRITTEN & COMMITTED BEFORE SCORING).**
> This document records our predictions for the two Part B sensitivity checks (**B1: High-Tilt Sensitivity** and **B2: Realistic-Depth $\sigma$ Sensitivity**) **before** generating or scoring any runs in `outputs/eval_literature/`. All models (`threshold_rule`, `logistic_regression`, `random_forest`) and both v3 operating points (`episode_fa<=1`, `strict_window_fa<=1`) remain **100% frozen** (`v1.0-frozen`). Nothing here is validated on real field data.

---

## 1. Summary of the Two Sensitivity Experiments

1. **B1 — High-Tilt Sensitivity (`high_tilt_3_6` and `high_tilt_6_12`):**
   - Same elliptical Gaussian generator as the control (`sigma_radii` $\in [0.4, 1.5]$, centre within $\pm 1.5\,R_{\text{net}}$), but with `peak_tilt_deg` log-stratified in **$[3.0^\circ, 6.0^\circ]$** (Arm 1: covering the upper half of the Ashton longwall measured range $0.7^\circ\text{--}5.9^\circ$) and **$[6.0^\circ, 12.0^\circ]$** (Arm 2: extreme post-failure / steep-trough range).
   - 30 `gradual` and 30 `rapid` runs per arm (new seeds disjoint from all previous runs), scored alongside the 60-run Gaussian control ($[0.05^\circ, 3.0^\circ]$) and the 40 shared `normal` runs from `eval_shape_mismatch`.
2. **B2 — Realistic-Depth $\sigma$ Sensitivity (`sigma = H / (tanB * sqrt(2*pi))`):**
   - Placeholder grid $H \in \{50, 100, 200, 300\}\text{ m} \times \tan\beta \in \{1.5, 2.0, 2.5\}$ (since $\tan\beta$ for the target Indian coalfield is not yet sourced).
   - Five cells ($H=50\text{ m}$ for all $\tan\beta$, and $H=100\text{ m}$ for $\tan\beta \in \{2.0, 2.5\}$) have $\sigma \in [0.564, 1.410]\,R_{\text{net}} \le 1.5\,R_{\text{net}}$ and already lie inside the main training cap.
   - Seven cells exceed the $1.5\,R_{\text{net}}$ ($21.21\text{ m}$) cap:
     - $(100\text{ m}, 1.5) \to 1.881\,R_{\text{net}}$ ($26.60\text{ m}$)
     - $(200\text{ m}, 2.5) \to 2.257\,R_{\text{net}}$ ($31.92\text{ m}$)
     - $(200\text{ m}, 2.0) \to 2.821\,R_{\text{net}}$ ($39.89\text{ m}$)
     - $(300\text{ m}, 2.5) \to 3.385\,R_{\text{net}}$ ($47.87\text{ m}$)
     - $(200\text{ m}, 1.5) \to 3.761\,R_{\text{net}}$ ($53.19\text{ m}$)
     - $(300\text{ m}, 2.0) \to 4.231\,R_{\text{net}}$ ($59.84\text{ m}$)
     - $(300\text{ m}, 1.5) \to 5.642\,R_{\text{net}}$ ($79.79\text{ m}$)
   - For each of these 7 cells: 10 `gradual` and 10 `rapid` runs (140 runs total), with centre in $\pm 1.5\,R_{\text{net}}$ (`x0, y0` $\in [10 \pm 21.21]\text{ m}$) and `peak_tilt_deg` log-stratified in $[0.05^\circ, 3.0^\circ]$.

---

## 2. Pre-Registered Predictions and Mechanistic Reasons

### Prediction 1 (B2 — Realistic-Depth $\sigma > 1.5\,R_{\text{net}}$): Convergence Degradation & `planar_slope`-Like Behavior
- **Geometric mechanism:**
  - The 4 corner nodes sit on a $20\times 20\text{ m}$ square ($R_{\text{net}} = 14.14\text{ m}$). When $\sigma$ is $30\text{--}80\text{ m}$ ($2.3\text{--}5.6\,R_{\text{net}}$) and the basin centre $(x_0, y_0)$ is sampled within $\pm 21.2\text{ m}$ of $(10, 10)$, the entire $20\times 20\text{ m}$ array subtends a small angle as seen from $(x_0, y_0)$ whenever the basin is off-centre, and the curvature across $20\text{ m}$ is small ($L / \sigma \approx 0.25\text{--}0.65$).
  - Consequently, the 4 corner downslope vectors become **nearly parallel** across the $20\text{ m}$ aperture (high `dir_pairwise_cos_mean`), so their ray intersection point $\hat{c}$ is poorly conditioned and regularized toward $(10, 10)$, just like `planar_slope` in the shape-mismatch stress test.
  - Furthermore, because $s_{\max} = \tan(\text{peak\_tilt\_deg})\,\sigma_{\min} / e^{-1/2}$ scales linearly with $\sigma_{\min}$, a wide basin ($\sigma \approx 40\text{--}80\text{ m}$) with a moderate peak tilt ($1^\circ\text{--}3^\circ$) requires a large vertical subsidence $s_{\max} \approx 1.5\text{--}7\text{ m}$. This large vertical drop changes the 3-D slant distance from the central LiDAR hub $(10, 10, 0)$ to the corners by $\Delta r \approx w^2 / (2 r_0) > 0.10\text{ m}$, which can trip the `hub_common_mode_abs_m_th` ($0.09\text{ m}$) and `hub_residual_abs_max_m_th` ($0.11\text{ m}$) thresholds that were calibrated on compact basins ($s_{\max} < 1.5\text{ m}$)!
- **Specific predictions for B2:**
  1. **Radial convergence features will degrade monotonically as $\sigma / R_{\text{net}}$ grows from $1.88$ to $5.64$:** `dir_convergence_residual_m` will increase and `dir_convergence_weighted_cos` will decrease relative to the control, while `dir_pairwise_cos_mean` will become strongly positive (approaching `planar_slope`).
  2. **`gradual -> sensor_fault` confusions will rise substantially on the ML models (`random_forest`, `logistic_regression`)** as $\sigma$ increases, cutting `gradual` window recall and `gradual` run-level detection rate at the v3 operating points.
  3. **What happens to `threshold_rule` at large $\sigma$:** At moderate tilts where $|w|$ is small enough that LiDAR slant-range change stays below $0.09\text{ m}$, `threshold_rule` will hold up well (as on `planar_slope`, because all 4 nodes tilt together with ratio $\approx 1.0$). However, on high-tilt wide-$\sigma$ runs where $s_{\max}$ is several meters, the non-linear 3-D LiDAR slant-range change $\Delta r \approx w^2/(2r_0) \ge 0.09\text{ m}$ will trip `hub_fault` in `ThresholdRuleClassifier` (`out[hub_fault | imu_fault] = "sensor_fault"`), causing `sensor_fault` confusions on those high-$s_{\max}$ windows too.

### Prediction 2 (B1 — High-Tilt Arms $3^\circ\text{--}6^\circ$ and $6^\circ\text{--}12^\circ$): Out-of-Range Saturation & `sensor_fault` / `rapid` Hijacking
- **Algorithmic mechanism in the frozen models:**
  - The frozen training set (`outputs/datasets_v3`) had `peak_tilt_deg` strictly capped at **$3.0^\circ$** for all `gradual` and `rapid` runs. The only training runs that ever exhibited node tilts above $3.5^\circ$ were **`sensor_fault`** runs (`implausible_reading` injecting $120^\circ\text{--}175^\circ$, `bias_step`, or `slow_drift`).
  - In `ThresholdRuleClassifier` (`canary_sim/models.py` line 180 & 300), there is a hard-coded guard `imu_implausible_tilt_deg_th = 3.5` deg: **whenever `max_node_mean >= 3.5` deg, `imu_fault` is set to `True` and overwrites the prediction to `"sensor_fault"`!** And even before a window reaches $3.5^\circ$, line 269 classifies any window with `tilt_max >= 1.25 * basin_tilt_max_deg_th` and small 12-window slope as `"rapid"` rather than `"gradual"`.
  - In `RandomForestClassifier` and `LogisticRegression`, large tilt magnitudes ($3^\circ\text{--}12^\circ$), large within-window standard deviations during the ramp, and larger 3-D LiDAR range changes push the feature vector outside the training support of `gradual`/`rapid` and into the half-space/leaves dominated by `sensor_fault` and `rapid`.
- **Specific predictions for B1:**
  1. **Integrity check (`TILT_LIMIT_NORMAL_DEG = 15.0` deg in `validation_checks.py`):**
     - Since `peak_tilt_deg` is at most $6.0^\circ$ in Arm 1 and $12.0^\circ$ in Arm 2 (plus small mounting offset $\sim 0.02^\circ$ and white noise $\sim 0.2^\circ$), the 5-sample window mean tilt `{node}_mean` will stay **below $15.0^\circ$**, so the $15^\circ$ physical tilt check will **NOT** be triggered in either arm. However, the manifest range check `basin peak_tilt_deg <= 3.0 deg` in `validation_checks.py` (line 232) **WILL** be triggered by design because `peak_tilt_deg` is in $[3, 12]^\circ$.
  2. **`ThresholdRuleClassifier` on $3^\circ\text{--}6^\circ$ and $6^\circ\text{--}12^\circ$:**
     - As the S-curve ramp crosses $3.5^\circ$ (and whenever $s_{\max}$ causes LiDAR slant-range change $\ge 0.09\text{ m}$), `ThresholdRuleClassifier` will switch its window predictions from `gradual`/`rapid` to **`sensor_fault`**! Thus **`gradual -> sensor_fault` confusion counts will surge dramatically** in `high_tilt_3_6` and even more in `high_tilt_6_12`, and `gradual` window recall will **drop** as tilt increases from $3^\circ\text{--}6^\circ$ to $6^\circ\text{--}12^\circ$ (an inverted-U / saturation effect where stronger signal actually degrades recall!). Run-level detection may still occur during the early part of the ramp while tilt is passing through $0.3^\circ\text{--}3.4^\circ$, before `sensor_fault` takes over.
  3. **`RandomForest` and `LogisticRegression` on $3^\circ\text{--}6^\circ$ and $6^\circ\text{--}12^\circ$:**
     - Radial convergence features (`dir_convergence_residual_m` and `dir_convergence_weighted_cos`) will actually be **cleaner (more convergent)** at $3^\circ\text{--}12^\circ$ than in the control because the SNR is huge ($\text{SNR} \sim 15\text{--}100$).
     - However, because tree splits and linear weights on magnitude, std, slope, and LiDAR range were fit on $\le 3^\circ$ basins, **`gradual` window recall will degrade in `high_tilt_6_12`** as `gradual` windows get misclassified as **`rapid`** (due to steeper ramp slope $\propto \text{peak\_tilt\_deg}$) or **`sensor_fault`** (due to out-of-range magnitude/LiDAR shift). Consequently, **`gradual` run-level detection rate on `high_tilt_6_12` will drop** compared to the $\text{SNR}>5$ bin of the control, whereas `rapid` run detection will remain high (`~0.85–1.00`).

### Prediction 3 (Normal-Run False Alarms): Unchanged
- Because the 40 `normal` runs use the untouched null-field path (`s_max = 0`) and are evaluated with the exact same frozen models and operating points, **false-alarm rates on the normal runs will be 100% unchanged** (`0.450` episodes/run for RF and Threshold Rule, `0.350` for LR at `episode_fa<=1`).

---

## 3. Summary Checklist of Pre-Registered Predictions

| ID | Pre-Registered Prediction | Direction & Mechanism |
|---|---|---|
| **L1** | In B2 (realistic depth $\sigma > 1.5\,R_{\text{net}}$), `dir_convergence_residual_m` increases, `dir_convergence_weighted_cos` decreases, and `dir_pairwise_cos_mean` becomes positive as $\sigma/R_{\text{net}}$ grows from $1.88$ to $5.64$. | Tilt across the $20\text{ m}$ array becomes nearly uniform/parallel (`planar_slope` limit). |
| **L2** | In B2, `gradual -> sensor_fault` confusions increase and `gradual` window recall / run detection drop for the ML models (`random_forest`, `logistic_regression`) compared to the control. | ML models treat large convergence residual + LiDAR range shift as `sensor_fault`. |
| **L3** | In B1 (`high_tilt_3_6` and `high_tilt_6_12`), `dir_convergence_residual_m` is low (high SNR), but `ThresholdRuleClassifier` suffers a massive surge in `gradual -> sensor_fault` confusions once node tilt exceeds `imu_implausible_tilt_deg_th = 3.5` deg. | Hard-coded $3.5^\circ$ implausible-tilt cutoff and $0.09\text{ m}$ LiDAR common-mode cutoff in `ThresholdRuleClassifier`. |
| **L4** | In B1 (`high_tilt_6_12`), `gradual` window recall and `gradual` run detection degrade (especially at `strict_window_fa<=1`) because steepened ramp slopes ($\propto \text{peak\_tilt}$) and out-of-range magnitudes push `gradual` windows into `rapid` and `sensor_fault`. | Training range truncation ($\le 3^\circ$) and slope scaling with amplitude. |
| **L5** | In B1, the $15^\circ$ physical tilt limit (`TILT_LIMIT_NORMAL_DEG = 15.0` deg) is **not** triggered on $3^\circ\text{--}6^\circ$ or $6^\circ\text{--}12^\circ$, whereas the manifest check `peak_tilt_deg <= 3.0` deg **is** triggered. | Max window mean tilt stays $\le 12.1^\circ < 15.0^\circ$. |
| **L6** | Normal-run false alarms are bitwise unchanged from the 40 shape-mismatch normal runs. | Identical normal runs and frozen models. |
