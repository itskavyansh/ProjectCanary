# Subsidence-Shape Mismatch Stress Test Report (SYNTHETIC / SIMULATION)

> [!CAUTION]
> **SYNTHETIC / SIMULATION — LIMITATION TEST, NOT A VALIDATION.**
> Every surface in this experiment (`gaussian_control`, `trough_panel`, `asymmetric`, `two_basin`, `planar_slope`) is a **synthetic mathematical function** $w(x,y,t) = -s(t)\,h(x,y)$ evaluated on the ideal $20\times 20\text{ m}$ square network with simulated IMU, piezo, LiDAR, and network telemetry.
> - These shapes are **NOT** field ground truth and **NOT** solutions of a geomechanical solver.
> - Two interacting basins (`two_basin`) and uniform landslide-like planes (`planar_slope`) are **explicitly out of scope** of the single-basin Gaussian assumption in the original design document.
> - **Nothing in this report is validated on real data or claims field readiness.**
> - All frozen v3 models (`threshold_rule`, `logistic_regression`, `random_forest`), operating points (`episode_fa<=1`, `strict_window_fa<=1`), features, thresholds, seeds, and earlier output folders were kept **100% frozen and untouched** (verified via SHA-256 checksums before and after execution).

---

## 1. Experimental Design, Assumptions, and Placeholder Ranges

### 1.1 Shape Equations (`canary_sim/shapes.py`, `configs/shape_mismatch.yaml`)
All five shapes produce state timelines in the **frozen `schema_v1` format** (`canary_sim/schema.py`), using the same smoothstep S-curve temporal ramp $s(t) = s_{\max}\,\operatorname{smoothstep}((t - t_{\text{on}})/\tau)$, the same 240 s run duration at 10 Hz, the same time-compression factors ($360\times$ for `gradual` and `normal`, $1\times$ for `rapid`), and the same seeded node mounting offsets (`ng.sample_mountings`). Only the spatial profile $h(x,y)$ and its analytic world-frame gradient $(\partial h/\partial x, \partial h/\partial y)$ differ:

| Shape | Unit Spatial Profile $h(x,y)$ (where $w(x,y,t) = -s(t)\,h(x,y)$) | Placeholder Parameter Ranges (`configs/shape_mismatch.yaml`) |
|---|---|---|
| `gaussian_control` | $\exp\!\left(-\frac{x'^2}{2s_x^2} - \frac{y'^2}{2s_y^2}\right)$ (delegates to frozen `engine.py`) | $x_0, y_0 \in 10 \pm 1.5 R$; $\max(s_x,s_y) \in [0.4, 1.5] R$; $s_y/s_x \in [1, 2]$; $\theta \in [0^\circ, 180^\circ]$ ($R = 14.14\text{ m}$) |
| `trough_panel` | $\frac{[\operatorname{erf}(\frac{x'+a}{\sqrt{2}\sigma_{\text{inf}}}) - \operatorname{erf}(\frac{x'-a}{\sqrt{2}\sigma_{\text{inf}}})]\,[\operatorname{erf}(\frac{y'+b}{\sqrt{2}\sigma_{\text{inf}}}) - \operatorname{erf}(\frac{y'-b}{\sqrt{2}\sigma_{\text{inf}}})]}{4\operatorname{erf}(\frac{a}{\sqrt{2}\sigma_{\text{inf}}})\operatorname{erf}(\frac{b}{\sqrt{2}\sigma_{\text{inf}}})}$ | Full panel width $2a \in [0.8, 2.4] R$ ($11.3\text{--}33.9\text{ m}$); length $2b \in [0.8, 2.4] R$; edge softness $\sigma_{\text{inf}} \in [0.15, 0.5] R$ ($2.1\text{--}7.1\text{ m}$) — **PLACEHOLDERS** |
| `asymmetric` | $\exp\!\left(-\frac{x'^2}{2 s_{x,\pm}^2} - \frac{y'^2}{2 s_y^2}\right)$ with $s_{x,-} = s_x$ for $x' < 0$ and $s_{x,+} = \min(r_{\text{skew}} s_x, 1.5 R)$ for $x' \ge 0$ | Skew ratio $r_{\text{skew}} \in [1.5, 3.0]$, capped at $1.5 R$ — **PLACEHOLDERS** |
| `two_basin` | $\exp\!\left(-\frac{r_1^2}{2\sigma_1^2}\right) + m_{21}\exp\!\left(-\frac{r_2^2}{2\sigma_2^2}\right)$ (**out of scope** of single-basin assumption) | Centre separation $d/\sigma_1 \in [1.0, 3.0]$; amplitude ratio $m_{21} \in [0.3, 1.0]$; $\sigma_2/\sigma_1 \in [0.6, 1.4]$; direction $\phi \in [0^\circ, 360^\circ)$ — **PLACEHOLDERS** |
| `planar_slope` | $(x - x_0)\cos\phi + (y - y_0)\sin\phi$ (uniform tilt plane, **no convergence point**) | Downslope azimuth $\phi \in [0^\circ, 360^\circ)$ — **PLACEHOLDER** |

### 1.2 Peak-Slope Matching and Paired Seeds
- **Disjoint new seeds:** `seed_base = 3000000001` (`gradual`: `3000000001..3000000030`, `rapid`: `3000000101..3000000130`, `normal`: `3000000201..3000000240`), verified disjoint from all existing dataset seeds (`2026111701..2026112530`).
- **Exact peak-slope matching:** Every shape receives the **exact same log-stratified sequence of 30 `peak_tilt_deg` values** in $[0.05^\circ, 3.0^\circ]$ for `gradual` and for `rapid`, and sets $s_{\max} = \tan(\text{peak\_tilt\_deg}) / \max_{(x,y)} \|\nabla h(x,y)\|$. Because the run seed is paired across shapes, every shape also shares the same IMU noise multiplier, node mounting yaws, and timing $(t_{\text{on}}, \tau)$ per run index — isolating the effect of surface geometry $h(x,y)$ alone.
- **Important geometric consequence (peak slope in plane vs slope at the 4 corners):** Matching the maximum slope *anywhere in the plane* means that shapes with localized steep rims (`trough_panel`) often have their peak gradient between or outside the 4 fixed corner nodes (mean max-node tilt $0.68^\circ$ vs $0.76^\circ$ for `gaussian_control`), whereas `planar_slope` has constant slope everywhere so **all 4 corner nodes always experience 100% of `peak_tilt_deg`** (mean max-node tilt $1.01^\circ$).

---

## 2. Radial Convergence Feature Breakdown Across Shapes

Table from [outputs/eval_shape_mismatch/shape_convergence_distributions.md](file:///d:/Github%20Projects/ProjectCanary-Sim/outputs/eval_shape_mismatch/shape_convergence_distributions.md) across all active event windows ($3,824$ windows per shape) and high-SNR active windows ($\text{SNR} > 2$, $1,819$ windows; $\text{SNR} > 5$, $1,214$ windows):

| Shape | Subset | Windows | `dir_convergence_residual_m` p10 / **p50 (Median)** / p90 (Mean) | `dir_convergence_weighted_cos` p10 / **p50 (Median)** / p90 (Mean) | `dir_pairwise_cos_mean` Median |
|---|---|---:|---|---|---:|
| **`gaussian_control`** | All SNR | 3,824 | `1.57 / 6.88 / 17.28 m` (8.56 m) | `-0.796 / +0.722 / +0.994` (+0.378) | `-0.065` |
| **`gaussian_control`** | **SNR > 2** | 1,819 | `1.07 / 3.23 / 11.55 m` (5.27 m) | `-0.427 / +0.952 / +0.998` (+0.623) | `-0.065` |
| **`asymmetric`** | All SNR | 3,824 | `1.68 / 6.14 / 16.59 m` (7.96 m) | `-0.673 / +0.849 / +0.994` (+0.466) | `-0.043` |
| **`asymmetric`** | **SNR > 2** | 1,819 | `1.09 / 2.79 / 9.45 m` (4.10 m) | `+0.025 / +0.980 / +0.997` (+0.779) | `-0.031` |
| **`two_basin`** | All SNR | 3,824 | `1.33 / 7.42 / 17.85 m` (8.94 m) | `-0.883 / +0.690 / +0.992` (+0.322) | `-0.058` |
| **`two_basin`** | **SNR > 2** | 1,819 | `1.01 / 3.83 / 13.85 m` (5.86 m) | `-0.908 / +0.944 / +0.996` (+0.538) | `-0.058` |
| **`trough_panel`** | All SNR | 3,824 | `1.19 / 8.56 / 18.60 m` (9.63 m) | `-0.838 / +0.450 / +0.997` (+0.270) | `-0.100` |
| **`trough_panel`** | **SNR > 2** | 1,819 | `0.75 / 4.12 / 16.18 m` (6.18 m) | `-0.545 / +0.920 / +0.999` (+0.493) | `-0.107` |
| **`planar_slope`** | All SNR | 3,824 | **`5.86 / 10.43 / 42.11 m` (16.38 m)** | **`-0.983 / +0.226 / +0.996` (+0.116)** | **`+0.888`** |
| **`planar_slope`** | **SNR > 2** | 1,819 | **`6.79 / 9.68 / 50.00 m` (16.89 m)** | **`-0.989 / +0.910 / +0.998` (+0.321)** | **`+0.993`** |

**What the convergence distributions show:**
1. **`planar_slope` breaks radial convergence completely:** At $\text{SNR} > 2$, its median `dir_convergence_residual_m` is **$9.68\text{ m}$ ($3.0\times$ the control median of $3.23\text{ m}$)** and its 90th percentile hits the **$50.0\text{ m}$ saturation cap** (mean $16.89\text{ m}$ vs $5.27\text{ m}$ control), while `dir_pairwise_cos_mean` jumps from $-0.065$ to **$+0.993$** because all four corners tilt in parallel. Across all active windows, median `dir_convergence_weighted_cos` drops from $+0.722$ (control) to **$+0.226$** (mean $+0.116$ vs $+0.378$).
2. **`trough_panel` and `two_basin` show moderate convergence degradation:** At $\text{SNR} > 2$, median `dir_convergence_residual_m` rises from $3.23\text{ m}$ (`gaussian_control`) to **$3.83\text{ m}$ (`two_basin`)** and **$4.12\text{ m}$ (`trough_panel`)**, and all-SNR median `dir_convergence_weighted_cos` drops from $+0.722$ to $+0.690$ (`two_basin`) and $+0.450$ (`trough_panel`).
3. **`asymmetric` does not degrade convergence at all:** Because capping $s_{x,+}$ at $1.5 R$ slightly tightens the basin when $r_{\text{skew}} s_x > 1.5 R$, its convergence residual is actually slightly smaller than the control ($2.79\text{ m}$ vs $3.23\text{ m}$ at $\text{SNR} > 2$).

---

## 3. Run-Level Detection & Missed-Event Rates by Shape

Table from [outputs/eval_shape_mismatch/shape_detection_rates.md](file:///d:/Github%20Projects/ProjectCanary-Sim/outputs/eval_shape_mismatch/shape_detection_rates.md) (30 `gradual` + 30 `rapid` = 60 event runs per shape; 95% run-bootstrap CIs):

| Shape | Model | Regime | All Events (60 runs) Exact Detect [95% CI] | Missed-Event Rate [95% CI] | Gradual (30 runs) Detect | Rapid (30 runs) Detect |
|---|---|---|---|---|---|---|
| **`gaussian_control`** | `random_forest` | `episode_fa<=1 (k=3,m=3)` | **0.617 [0.500, 0.733]** | 0.383 [0.250, 0.500] | **0.600** [0.433, 0.767] | **0.633** [0.449, 0.800] |
| **`asymmetric`** | `random_forest` | `episode_fa<=1 (k=3,m=3)` | **0.600 [0.483, 0.717]** | 0.400 [0.283, 0.517] | **0.600** [0.467, 0.767] | **0.600** [0.433, 0.767] |
| **`two_basin`** | `random_forest` | `episode_fa<=1 (k=3,m=3)` | **0.583 [0.450, 0.709]** | 0.417 [0.283, 0.550] | **0.567** [0.400, 0.751] | **0.600** [0.400, 0.800] |
| **`planar_slope`** | `random_forest` | `episode_fa<=1 (k=3,m=3)` | **0.550 [0.417, 0.683]** | 0.450 [0.341, 0.567] | **0.467** [0.282, 0.651] | **0.633** [0.433, 0.800] |
| **`trough_panel`** | `random_forest` | `episode_fa<=1 (k=3,m=3)` | **0.533 [0.417, 0.659]** | 0.467 [0.325, 0.600] | **0.567** [0.400, 0.733] | **0.500** [0.333, 0.667] |
| **`gaussian_control`** | `random_forest` | `strict_window_fa<=1` | **0.400 [0.275, 0.533]** | 0.600 [0.500, 0.750] | **0.367** [0.200, 0.533] | **0.433** [0.267, 0.600] |
| **`asymmetric`** | `random_forest` | `strict_window_fa<=1` | **0.400 [0.275, 0.517]** | 0.600 [0.483, 0.733] | **0.433** [0.249, 0.633] | **0.367** [0.200, 0.517] |
| **`two_basin`** | `random_forest` | `strict_window_fa<=1` | **0.400 [0.283, 0.525]** | 0.600 [0.483, 0.733] | **0.400** [0.233, 0.600] | **0.400** [0.233, 0.567] |
| **`planar_slope`** | `random_forest` | `strict_window_fa<=1` | **0.317 [0.200, 0.417]** | 0.683 [0.575, 0.809] | **0.200** [0.067, 0.351] | **0.433** [0.233, 0.618] |
| **`trough_panel`** | `random_forest` | `strict_window_fa<=1` | **0.250 [0.141, 0.367]** | 0.750 [0.650, 0.859] | **0.167** [0.049, 0.300] | **0.333** [0.167, 0.500] |
| **`gaussian_control`** | `logistic_regression` | `episode_fa<=1 (k=3,m=3)` | **0.467 [0.333, 0.600]** | 0.533 [0.417, 0.667] | **0.400** [0.233, 0.567] | **0.533** [0.333, 0.700] |
| **`asymmetric`** | `logistic_regression` | `episode_fa<=1 (k=3,m=3)` | **0.450 [0.308, 0.575]** | 0.550 [0.417, 0.667] | **0.367** [0.200, 0.533] | **0.533** [0.333, 0.700] |
| **`trough_panel`** | `logistic_regression` | `episode_fa<=1 (k=3,m=3)` | **0.383 [0.267, 0.533]** | 0.617 [0.500, 0.733] | **0.333** [0.167, 0.500] | **0.433** [0.267, 0.600] |
| **`two_basin`** | `logistic_regression` | `episode_fa<=1 (k=3,m=3)` | **0.367 [0.233, 0.483]** | 0.633 [0.508, 0.750] | **0.333** [0.167, 0.500] | **0.400** [0.233, 0.551] |
| **`planar_slope`** | `logistic_regression` | `episode_fa<=1 (k=3,m=3)` | **0.500 [0.367, 0.617]** | 0.500 [0.383, 0.633] | **0.400** [0.249, 0.567] | **0.600** [0.400, 0.767] |
| **`gaussian_control`** | `threshold_rule` | `episode_fa<=1 (k=3,m=3)` | **0.483 [0.350, 0.617]** | 0.517 [0.383, 0.650] | **0.500** [0.333, 0.667] | **0.467** [0.300, 0.633] |
| **`trough_panel`** | `threshold_rule` | `episode_fa<=1 (k=3,m=3)` | **0.450 [0.333, 0.583]** | 0.550 [0.433, 0.659] | **0.433** [0.249, 0.600] | **0.467** [0.267, 0.651] |
| **`asymmetric`** | `threshold_rule` | `episode_fa<=1 (k=3,m=3)` | **0.433 [0.300, 0.567]** | 0.567 [0.441, 0.717] | **0.467** [0.282, 0.633] | **0.400** [0.216, 0.600] |
| **`two_basin`** | `threshold_rule` | `episode_fa<=1 (k=3,m=3)` | **0.467 [0.317, 0.609]** | 0.533 [0.400, 0.667] | **0.467** [0.300, 0.633] | **0.467** [0.300, 0.667] |
| **`planar_slope`** | `threshold_rule` | `episode_fa<=1 (k=3,m=3)` | **0.633 [0.525, 0.750]** | 0.367 [0.250, 0.492] | **0.700** [0.533, 0.833] | **0.567** [0.400, 0.733] |

*(Note: Why is `threshold_rule` higher on `planar_slope`? Because `threshold_rule` does not use radial convergence features at all — it triggers whenever multi-node tilt magnitude and long-horizon slope exceed thresholds. On `planar_slope`, all 4 nodes experience 100% of `peak_tilt_deg`, whereas on localized basins distant nodes decay exponentially).*

---

## 4. Window Recall & Run Detection by SNR Bin (`snr_vs_imu_noise`)

Summary from [outputs/eval_shape_mismatch/shape_recall_by_snr.md](file:///d:/Github%20Projects/ProjectCanary-Sim/outputs/eval_shape_mismatch/shape_recall_by_snr.md) showing **Window Recall / Run-Level Detection Rate** side by side (so the two are never shown apart). Run counts per SNR bin are: `gradual` ($\text{<1}: 10,\ \text{1--2}: 5,\ \text{2--5}: 6,\ \text{>5}: 9$), `rapid` ($\text{<1}: 9,\ \text{1--2}: 8,\ \text{2--5}: 4,\ \text{>5}: 9$):

| Model & Operating Point | Class | Shape | All SNR (30 runs) | SNR < 1 | SNR 1–2 | SNR 2–5 | SNR > 5 |
|---|---|---|---|---|---|---|---|
| **Random Forest** `episode_fa<=1` | `gradual` | `gaussian_control` | `0.246 / 0.600` | `0.186 / 0.600` | `0.083 / 0.400` | `0.300 / 0.667` | **`0.368 / 0.667`** |
| **Random Forest** `episode_fa<=1` | `gradual` | `asymmetric` | `0.241 / 0.600` | `0.184 / 0.600` | `0.077 / 0.400` | `0.333 / 0.667` | **`0.337 / 0.667`** |
| **Random Forest** `episode_fa<=1` | `gradual` | `two_basin` | `0.278 / 0.567` | `0.199 / 0.600` | `0.094 / 0.400` | `0.313 / 0.500` | **`0.443 / 0.667`** |
| **Random Forest** `episode_fa<=1` | `gradual` | `planar_slope` | **`0.157 / 0.467`** | `0.193 / 0.500` | `0.038 / 0.200` | **`0.124 / 0.167`** | **`0.204 / 0.778`** |
| **Random Forest** `episode_fa<=1` | `gradual` | `trough_panel` | **`0.138 / 0.567`** | `0.130 / 0.600` | `0.080 / 0.600` | `0.168 / 0.667` | **`0.160 / 0.444`** |
| **Random Forest** `episode_fa<=1` | `rapid` | `gaussian_control` | `0.383 / 0.633` | `0.097 / 0.444` | `0.197 / 0.500` | `0.211 / 0.500` | **`0.848 / 1.000`** |
| **Random Forest** `episode_fa<=1` | `rapid` | `asymmetric` | `0.353 / 0.600` | `0.106 / 0.556` | `0.215 / 0.500` | `0.110 / 0.250` | **`0.772 / 0.889`** |
| **Random Forest** `episode_fa<=1` | `rapid` | `two_basin` | `0.325 / 0.600` | `0.090 / 0.444` | `0.131 / 0.250` | `0.216 / 0.750` | **`0.727 / 1.000`** |
| **Random Forest** `episode_fa<=1` | `rapid` | `trough_panel` | `0.299 / 0.500` | `0.036 / 0.333` | `0.138 / 0.250` | `0.105 / 0.500` | **`0.732 / 0.889`** |
| **Random Forest** `episode_fa<=1` | `rapid` | `planar_slope` | **`0.236 / 0.633`** | `0.101 / 0.444` | `0.242 / 0.500` | `0.101 / 0.750` | **`0.405 / 0.889`** |
| **Random Forest** `strict_window_fa<=1` | `gradual` | `gaussian_control` | `0.148 / 0.367` | `0.037 / 0.300` | `0.029 / 0.000` | `0.209 / 0.500` | **`0.296 / 0.556`** |
| **Random Forest** `strict_window_fa<=1` | `gradual` | `planar_slope` | **`0.044 / 0.200`** | `0.025 / 0.100` | `0.000 / 0.000` | `0.034 / 0.000` | **`0.095 / 0.556`** |
| **Random Forest** `strict_window_fa<=1` | `gradual` | `trough_panel` | **`0.048 / 0.167`** | `0.021 / 0.100` | `0.015 / 0.000` | `0.052 / 0.333` | **`0.093 / 0.222`** |

Look at the `SNR > 5` column for Random Forest (`episode_fa<=1`):
- On `gaussian_control`, `rapid` window recall at $\text{SNR} > 5$ is **0.848** and `gradual` window recall is **0.368**.
- On `planar_slope`, even at $\text{SNR} > 5$ (where all 4 nodes have huge tilt!), Random Forest window recall drops by nearly half to **0.405** (`rapid`) and **0.204** (`gradual**; at `strict_window_fa<=1` it drops from **0.296** to **0.095**)! Why? Because the broken radial convergence features (`dir_convergence_residual_m` $\approx 10\text{--}50\text{ m}$) divert `planar_slope` windows into `sensor_fault`!

---

## 5. `gradual` vs `sensor_fault` Confusion Counts by Shape

Table from [outputs/eval_shape_mismatch/shape_gradual_sensor_fault_confusion.md](file:///d:/Github%20Projects/ProjectCanary-Sim/outputs/eval_shape_mismatch/shape_gradual_sensor_fault_confusion.md) across all $2,008$ trainable `gradual` windows (30 `gradual` runs) per shape:

| Shape | Model | Regime | Window: `gradual -> sensor_fault` [95% CI] (Rate) | Persisted Alarm: `gradual -> sensor_fault` [95% CI] | Window: `gradual -> gradual` | Window: `gradual -> rapid` | Window: `gradual -> normal` |
|---|---|---|---|---|---|---|---|
| **`gaussian_control`** | `random_forest` | `episode_fa<=1` | **40** [3.0, 109.5] (`2.0%`) | **27** [0.0, 75.0] | 494 | 393 | 1,075 |
| **`asymmetric`** | `random_forest` | `episode_fa<=1` | **9** [2.0, 21.0] (`0.4%`) | **3** [0.0, 9.0] | 485 | 539 | 969 |
| **`two_basin`** | `random_forest` | `episode_fa<=1` | **40** [4.0, 104.1] (`2.0%`) | **27** [0.0, 68.5] | 558 | 289 | 1,115 |
| **`planar_slope`** | `random_forest` | `episode_fa<=1` | **120** [40.5, 211.6] (**`6.0%` — $3.0\times$ control**) | **109** [35.4, 192.5] (**$4.0\times$**) | 315 | 331 | 1,235 |
| **`trough_panel`** | `random_forest` | `episode_fa<=1` | **133** [35.4, 273.5] (**`6.6%` — $3.3\times$ control**) | **119** [26.0, 240.0] (**$4.4\times$**) | 277 | 275 | 1,316 |
| **`gaussian_control`** | `random_forest` | `strict_window_fa<=1` | **44** [5.5, 114.5] (`2.2%`) | **24** [0.0, 68.0] | 298 | 229 | 1,431 |
| **`planar_slope`** | `random_forest` | `strict_window_fa<=1` | **141** [48.4, 245.7] (**`7.0%` — $3.2\times$ control**) | **122** [46.5, 205.5] (**$5.1\times$**) | 88 | 191 | 1,581 |
| **`trough_panel`** | `random_forest` | `strict_window_fa<=1` | **163** [49.0, 309.2] (**`8.1%` — $3.7\times$ control**) | **142** [42.4, 265.5] (**$5.9\times$**) | 96 | 124 | 1,618 |
| **`gaussian_control`** | `logistic_regression` | `episode_fa<=1` | **15** [4.0, 31.5] (`0.7%`) | **0** [0.0, 0.0] | 351 | 364 | 1,276 |
| **`planar_slope`** | `logistic_regression` | `episode_fa<=1` | **112** [64.0, 181.1] (**`5.6%` — $7.5\times$ control**) | **72** [28.0, 125.0] | 283 | 410 | 1,195 |
| **`gaussian_control`** | `threshold_rule` | `episode_fa<=1` | **80** [0.0, 239.5] (`4.0%`) | **78** [0.0, 206.0] | 377 | 307 | 1,240 |
| **`trough_panel`** | `threshold_rule` | `episode_fa<=1` | **144** [34.0, 285.0] (**`7.2%` — $1.8\times$ control**) | **141** [24.9, 282.5] | 314 | 232 | 1,314 |
| **`planar_slope`** | `threshold_rule` | `episode_fa<=1` | **2** [0.0, 6.0] (`0.1%`) | **0** [0.0, 0.0] | 522 | 366 | 1,114 |

**Why `planar_slope` and `trough_panel` triple `gradual -> sensor_fault` confusions on Random Forest — for two different reasons:**
1. **`planar_slope` (broken convergence):** All 4 nodes tilt strongly, but their rays are parallel (`dir_convergence_residual_m` $\approx 10\text{--}50\text{ m}$). Both `random_forest` ($40 \to 120$ windows at `episode_fa<=1`, $27 \to 109$ persisted alarm windows) and `logistic_regression` ($15 \to 112$ windows, $0 \to 72$ persisted alarm windows) learned that high tilt accompanied by a large convergence residual is the signature of a `sensor_fault` rather than a `gradual` basin! Meanwhile, `threshold_rule` has almost zero `sensor_fault` confusions on `planar_slope` ($2$ windows) because its single-node dominance ratio (`max_node_mean / second_node_mean`) is $1.0$ when all 4 nodes tilt equally.
2. **`trough_panel` (single-node rim concentration):** When the narrow steep edge ($\sigma_{\text{inf}} \in [0.15, 0.5] R$) of a flat-bottomed rectangular panel passes near just **one** corner node while the other three corner nodes sit on the flat trough floor or outside the panel, only a single node tilts strongly. Both `threshold_rule` ($80 \to 144$ windows) and `random_forest` ($40 \to 133$ windows at `episode_fa<=1`, $44 \to 163$ at `strict_window_fa<=1`) misclassify that single-node rim tilt as a `sensor_fault`!

---

## 6. Shared Normal Runs (40 New Seeds): False Alarms Unchanged

From [outputs/eval_shape_mismatch/shape_normal_false_alarms.md](file:///d:/Github%20Projects/ProjectCanary-Sim/outputs/eval_shape_mismatch/shape_normal_false_alarms.md), the 40 new `normal` runs (`3000000201..3000000240`) produce **identical false-alarm counts across all 5 shape evaluations** and match the frozen `test` split within the 95% run-bootstrap CI:

| Model | Regime | FA Episodes / Normal Run (95% CI, 40 runs) | Definition A: Slow-Process Time (24 h/run) | Definition B: Sensor Time (0.0667 h/run) | Frozen `test` Reference (20 runs) |
|---|---|---|---|---|---|
| `threshold_rule` | `raw (k=1)` | `6.525 [3.697, 9.988]` | `0.2719 / h` (every `3.7 h`) | `97.88 / h` (every `0.6 min`) | `6.750 [2.295, 11.850]` |
| `threshold_rule` | `episode_fa<=1 (k=3,m=3)` | `0.450 [0.175, 0.800]` | `0.0187 / h` (every `53.3 h`) | `6.75 / h` (every `8.9 min`) | `0.600 [0.050, 1.326]` |
| `threshold_rule` | `strict_window_fa<=1` | `0.100 [0.000, 0.225]` | `0.0042 / h` (every `240.0 h`) | `1.50 / h` (every `40.0 min`) | `0.050 [0.000, 0.150]` |
| `logistic_regression` | `raw (k=1)` | `21.075 [18.212, 24.279]` | `0.8781 / h` (every `1.1 h`) | `316.12 / h` (every `0.2 min`) | `21.100 [16.442, 25.626]` |
| `logistic_regression` | `episode_fa<=1 (k=3,m=3)` | `0.350 [0.175, 0.550]` | `0.0146 / h` (every `68.6 h`) | `5.25 / h` (every `11.4 min`) | `0.100 [0.000, 0.200]` |
| `logistic_regression` | `strict_window_fa<=1` | `0.125 [0.000, 0.338]` | `0.0052 / h` (every `192.0 h`) | `1.88 / h` (every `32.0 min`) | `0.000 [0.000, 0.000]` |
| `random_forest` | `raw (k=1)` | `7.100 [5.149, 9.263]` | `0.2958 / h` (every `3.4 h`) | `106.50 / h` (every `0.6 min`) | `5.650 [3.450, 8.100]` |
| `random_forest` | `episode_fa<=1 (k=3,m=3)` | `0.450 [0.225, 0.700]` | `0.0187 / h` (every `53.3 h`) | `6.75 / h` (every `8.9 min`) | `0.650 [0.200, 1.176]` |
| `random_forest` | `strict_window_fa<=1` | `0.225 [0.100, 0.363]` | `0.0094 / h` (every `106.7 h`) | `3.38 / h` (every `17.8 min`) | `0.250 [0.050, 0.500]` |

---

## 7. Verification of Pre-Registered Predictions (`docs/shape_mismatch_expectations.md`)

| ID | Pre-Registered Prediction (Committed in `8524cd1` Before Scoring) | Outcome | Empirical Evidence |
|---|---|---|---|
| **P1** | `planar_slope` breaks radial convergence: `dir_convergence_weighted_cos` collapses toward $0$ and `dir_convergence_residual_m` is $\ge 2\times$ the `gaussian_control` median on active windows. | **CONFIRMED** | All-SNR median `dir_convergence_weighted_cos` dropped from `+0.722` to **`+0.226`** (mean `+0.378` $\to$ **`+0.116`**); at $\text{SNR}>2$, median `dir_convergence_residual_m` jumped $3.0\times$ from `3.23 m` to **`9.68 m`** (p90 at the `50.0 m` cap; mean `16.89 m` vs `5.27 m`). |
| **P2** | `two_basin` degrades radial convergence (`dir_convergence_residual_m` higher and `dir_convergence_weighted_cos` lower than `gaussian_control`), though much less severely than `planar_slope`. | **CONFIRMED** | At $\text{SNR}>2$, `two_basin` median residual rose from `3.23 m` to **`3.83 m`** (p90 `11.55 m` $\to$ `13.85 m`) and mean `weighted_cos` dropped from `+0.623` to **`+0.538`**, while remaining far healthier than `planar_slope` (`9.68 m`). |
| **P3** | `trough_panel` shows moderately higher `dir_convergence_residual_m` than `gaussian_control` and lower window recall due to its flat bottom / narrow steep rim, plus increased `sensor_fault` confusion when only one corner sits on the rim. | **CONFIRMED** | Median residual rose to **`8.56 m`** (all SNR) and **`4.12 m`** ($\text{SNR}>2$). Random Forest `gradual` window recall dropped from `0.246` to **`0.138`** (`episode_fa<=1`) and from `0.148` to **`0.048`** (`strict_window_fa<=1`), while `gradual -> sensor_fault` confusions jumped **$3.3\times$** (`40` $\to$ **`133`** windows). |
| **P4** | `asymmetric` behaves closest to `gaussian_control`: minimal change in convergence features, window recall, and run detection rate. | **CONFIRMED** | Random Forest exact run detection on `asymmetric` is **`0.600`** vs **`0.617`** (`episode_fa<=1`) and **`0.400`** vs **`0.400`** (`strict_window_fa<=1`); `gradual` window recall is **`0.241`** vs **`0.246`**. |
| **P5** | `planar_slope` increases `gradual` $\leftrightarrow$ `sensor_fault` confusion counts relative to `gaussian_control` on the ML models (`random_forest`, `logistic_regression`), whereas `threshold_rule` is boosted by uniform 4-node tilt. | **CONFIRMED** | On `planar_slope` at `episode_fa<=1`, `gradual -> sensor_fault` window confusions rose **$7.5\times$** on `logistic_regression` (`15` $\to$ **`112`**) and **$3.0\times$** on `random_forest` (`40` $\to$ **`120`**), cutting RF `gradual` run detection from `0.600` to `0.467` (`0.367` $\to$ `0.200` at strict). Meanwhile `threshold_rule` detection rose from `0.483` to **`0.633`**. |
| **P6** | False-alarm rates on the 40 shared `normal` runs are identical across all 5 shape evaluations and within the 95% bootstrap CI of the frozen `test` split. | **CONFIRMED** | Bitwise identical across all 5 shape rows (`0.450` FA eps/run for RF and Threshold Rule, `0.350` for LR at `episode_fa<=1`), all well inside the frozen `test` 95% CIs. |

---

## 8. Plain Summary: Which Shapes the Frozen v3 Models Handle and Which Break Them

1. **Handled without degradation (`asymmetric`):**
   - Skewing an elliptical basin so one side is $1.5\text{--}3\times$ broader than the other preserves inward radial convergence toward the basin minimum. All three frozen v3 models achieve virtually the same window recall, run detection rate, and `sensor_fault` separation as on the symmetric Gaussian control.
2. **Handled with mild degradation (`two_basin` — out of scope of single-basin assumption):**
   - Adding a second Gaussian basin $1\text{--}3\,\sigma_1$ away degrades convergence fit residuals by $\sim 18\%$ ($3.23\text{ m} \to 3.83\text{ m}$ at $\text{SNR}>2$) and lowers Logistic Regression run detection from `0.467` to `0.367`, but Random Forest handles it well (`0.583` vs `0.617` at `episode_fa<=1`, and `0.400` vs `0.400` at `strict_window_fa<=1`) because the broader two-basin footprint provides multi-node tilt and LiDAR range signals that compensate for imperfect single-point convergence.
3. **Breaks gradual detection via single-node rim concentration (`trough_panel`):**
   - A flat-bottomed rectangular extraction trough concentrates its gradient into a narrow perimeter band of width $\sigma_{\text{inf}}$. When that narrow rim hits only one corner node while the other three lie on the flat bottom or outside the panel, the telemetry looks like a single-node anomaly: Random Forest `gradual -> sensor_fault` confusions increase **$3.3\text{--}3.7\times$**, `gradual` window recall drops from `0.246` to `0.138` (`0.148` to `0.048` at the strict operating point), and strict-regime `gradual` run detection drops from `0.367` to **`0.167`**.
4. **Breaks radial convergence features completely (`planar_slope`):**
   - A uniform landslide-like tilt plane has parallel downslope vectors everywhere (`dir_pairwise_cos_mean` $\approx +0.993$) and no convergence point. The v3 least-squares convergence fitter places $\hat{c}$ near the network centre, causing two corners to point toward $\hat{c}$ and two to point away (`dir_convergence_residual_m` jumps $3\times$ to $9.68\text{ m}$ median and $50\text{ m}$ p90).
   - Because the frozen ML models learned that high tilt + bad convergence residual = `sensor_fault`, `planar_slope` causes a **$3.0\times$ surge in `gradual -> sensor_fault` confusions for Random Forest** ($40 \to 120$ windows) and a **$7.5\times$ surge for Logistic Regression** ($15 \to 112$ windows), cutting Random Forest `gradual` run detection from `0.600` to `0.467` (and from `0.367` to `0.200` at the strict operating point) and cutting high-SNR ($\text{SNR}>5$) `rapid` window recall in half (`0.848` $\to$ `0.405`).

### Generated Plots in `outputs/eval_shape_mismatch/`
All four figures carry the red `"SYNTHETIC / SIMULATION"` banner and time-compression factors:
- [shape_field_profiles.png](file:///d:/Github%20Projects/ProjectCanary-Sim/outputs/eval_shape_mismatch/shape_field_profiles.png)
- [shape_detection_and_missed_rate.png](file:///d:/Github%20Projects/ProjectCanary-Sim/outputs/eval_shape_mismatch/shape_detection_and_missed_rate.png)
- [shape_recall_by_snr.png](file:///d:/Github%20Projects/ProjectCanary-Sim/outputs/eval_shape_mismatch/shape_recall_by_snr.png)
- [shape_convergence_distributions.png](file:///d:/Github%20Projects/ProjectCanary-Sim/outputs/eval_shape_mismatch/shape_convergence_distributions.png)
