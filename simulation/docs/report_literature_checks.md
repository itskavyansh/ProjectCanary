# Literature-Basis Sensitivity Checks & Timescale Audit Report (SYNTHETIC / SIMULATION)

> [!CAUTION]
> **SYNTHETIC / SIMULATION — NOT VALIDATED ON REAL DATA.**
> Every run in this report is generated from synthetic mathematical equations and simulated sensor/network noise.
> - **Nothing in this report is validated on real field ground truth or claims field readiness.**
> - The Knothe influence tangent $\tan(\beta)$ and time coefficient $c$ remain **unsourced for the target Indian coalfield** (literature leads only: Singh & Yadav 1995, Raniganj; *Sadhana* 2021 Adriyala longwall paper); the $(H, \tan\beta)$ grid evaluated here is a **PLACEHOLDER** grid.
> - All frozen v3 models (`threshold_rule`, `logistic_regression`, `random_forest`), operating points (`episode_fa<=1`, `strict_window_fa<=1`), features, thresholds, seeds, datasets, and earlier output folders (`v1.0-frozen`) were kept **100% frozen and untouched** (verified via SHA-256 checksums before and after execution).

---

## 1. Summary of Scope and Pre-Registration

1. **Part A — Literature Provenance ([docs/provenance_note.md](file:///d:/Github%20Projects/ProjectCanary-Sim/docs/provenance_note.md), Section 5):** Added the exact literature basis table (Ashton Underground longwall end-of-panel reports 2008–2012, Jeran & Trevits USBM RI 9552, Colorado DRMS rule document, *Sustainability* 2020 12(4):1528, Zhang et al. *Scientific Reports* 12:18433 2022, Hejmanowski 2015, OSMRE 2007, Agnico Meadowbank 2019, and unsourced Indian coalfield leads) with all unsourced values marked **PLACEHOLDER**, plus the **Timescale** paragraph.
2. **Pre-Registration ([docs/literature_checks_expectations.md](file:///d:/Github%20Projects/ProjectCanary-Sim/docs/literature_checks_expectations.md)):** Written and committed in `ad1a7ad` (containing only that file) **before** any literature-check scoring outputs existed (verified in `tests/test_literature_checks.py::test_preregistration_committed_before_scoring_outputs`).
3. **Part B1 — High-Tilt Sensitivity (`high_tilt_3_6` and `high_tilt_6_12`):** 30 `gradual` + 30 `rapid` Gaussian runs with log-stratified peak tilt in $[3.0^\circ, 6.0^\circ]$ (upper half of Ashton measured longwall tilts $0.7^\circ\text{--}5.9^\circ$) and $[6.0^\circ, 12.0^\circ]$ (post-failure / extreme tilt), using disjoint new seeds (`4000000001..4000000329`).
4. **Part B2 — Realistic-Depth Knothe Spread (`sigma = H / (tanB * sqrt(2*pi))`):** Evaluated across the 12-cell **PLACEHOLDER** grid $H \in \{50, 100, 200, 300\}\text{ m} \times \tan\beta \in \{1.5, 2.0, 2.5\}$, generating 10 `gradual` + 10 `rapid` runs for each of the 7 cells where $\sigma > 1.5\,R_{\text{net}}$ (140 runs total, seeds `4000001001..4000001659`).
5. **Part C — Timescale Audit:** Audited all 165 `gradual` runs in the frozen manifest (`outputs/datasets_v3/manifest.csv`) against the rough real-world order-of-magnitude reference of **`0.1 deg/day`** (**DERIVED, ORDER OF MAGNITUDE**, from ~`60 mm/m` [`3.4 deg`] over ~`30 days`), and tabulated Definition A false-alarm rates under alternative 30-day compression factors (`10,000x`, `10,800x`, `30,000x`).

---

## 2. Part B1: High-Tilt Sensitivity ($3^\circ\text{--}6^\circ$ and $6^\circ\text{--}12^\circ$)

### 2.1 Impossible-Value Check Audit ([impossible_value_check_report.md](file:///d:/Github%20Projects/ProjectCanary-Sim/outputs/eval_literature/impossible_value_check_report.md))

| Arm | Runs | Windows | Max Window Node Tilt (`{node}_mean`, deg) | Runs > `15.0 deg` (`TILT_LIMIT_NORMAL_DEG`) | Runs $\ge$ `3.5 deg` (`ThresholdRule` `imu_implausible_th`) | `validation_checks.check_impossible` Result |
|---|---:|---:|---:|---:|---:|---|
| `gaussian_control` (`0.05-3 deg`) | 60 | 5,700 | **`3.085 deg`** | **0** (0 win) | **0** (0 win) | `NONE` (PASS) |
| `high_tilt_3_6` (`3-6 deg`) | 60 | 5,700 | **`6.025 deg`** | **0** (0 win) | **30 / 60** (1,420 win) | `basin peak_tilt_deg outside configured [0.05, 3.0] deg` |
| `high_tilt_6_12` (`6-12 deg`) | 60 | 5,700 | **`10.696 deg`** | **0** (0 win) | **60 / 60** (3,034 win) | `basin peak_tilt_deg outside configured [0.05, 3.0] deg` |

- **Physical tilt limit (`15.0 deg`) in `validation_checks.py`:** **Not triggered** on either arm (max window node mean tilt is `6.025 deg` in `3-6 deg` and `10.696 deg` in `6-12 deg`).
- **Configured range check (`[0.05, 3.0] deg`) in `validation_checks.py`:** **Triggered** on both high-tilt arms because `peak_tilt_deg > 3.0 deg`.
- **`ThresholdRuleClassifier` internal implausible-tilt cutoff (`3.5 deg` in `models.py` line 180):** **Triggered on 30/60 runs (1,420 windows) in `3-6 deg` and 60/60 runs (3,034 windows) in `6-12 deg`**, directly overwriting basin predictions to `"sensor_fault"` once a node's window mean tilt crosses $3.5^\circ$!

### 2.2 High-Tilt Detection, Window Recall & `gradual -> sensor_fault` Confusions

From [literature_detection_rates.md](file:///d:/Github%20Projects/ProjectCanary-Sim/outputs/eval_literature/literature_detection_rates.md), [literature_recall_by_snr.csv](file:///d:/Github%20Projects/ProjectCanary-Sim/outputs/eval_literature/literature_recall_by_snr.csv), and [literature_gradual_sensor_fault_confusion.md](file:///d:/Github%20Projects/ProjectCanary-Sim/outputs/eval_literature/literature_gradual_sensor_fault_confusion.md) (all high-tilt runs lie in the $\text{SNR} > 5$ bin):

| Arm | Model | Regime | All Events (60) Exact Detect [95% CI] | Gradual: Win Recall / Run Detect | Rapid: Win Recall / Run Detect | Win `gradual -> sensor_fault` (Rate) | Win `gradual -> rapid` |
|---|---|---|---|---|---|---|---:|
| **`gaussian_control`** | `threshold_rule` | `episode_fa<=1` | **0.483** [0.350, 0.592] | `0.188 / 0.500` | `0.276 / 0.467` | **80** (`4.0%`) | 307 |
| **`high_tilt_3_6`** | `threshold_rule` | `episode_fa<=1` | **0.650** [0.533, 0.783] | `0.244 / 0.633` | `0.233 / 0.667` | **786** (**`39.0%`**) | 335 |
| **`high_tilt_6_12`** | `threshold_rule` | `episode_fa<=1` | **0.367** [0.250, 0.483] | `0.119 / 0.667` | **`0.014 / 0.067`** | **1,382** (**`67.8%`**) | 183 |
| **`high_tilt_6_12`** | `threshold_rule` | `strict_window_fa<=1` | **0.100** [0.033, 0.183] | `0.119 / 0.200` | **`0.014 / 0.000`** | **1,382** (**`67.8%`**) | 183 |
| **`gaussian_control`** | `logistic_regression` | `episode_fa<=1` | **0.467** [0.325, 0.583] | `0.175 / 0.400` | `0.346 / 0.533` | **15** (`0.7%`) | 364 |
| **`high_tilt_3_6`** | `logistic_regression` | `episode_fa<=1` | **0.583** [0.467, 0.700] | `0.307 / 0.600` | `0.407 / 0.567` | **77** (`3.8%`) | 597 |
| **`high_tilt_6_12`** | `logistic_regression` | `episode_fa<=1` | **0.500** [0.383, 0.633] | `0.500 / 0.700` | **`0.130 / 0.300`** | **186** (`9.1%`) | 433 |
| **`gaussian_control`** | `random_forest` | `episode_fa<=1` | **0.617** [0.491, 0.742] | `0.246 / 0.600` | `0.383 / 0.633` | **40** (`2.0%`) | 393 |
| **`high_tilt_3_6`** | `random_forest` | `episode_fa<=1` | **0.767** [0.667, 0.867] | `0.286 / 0.633` | `0.629 / 0.900` | **207** (**`10.3%`**) | **748** |
| **`high_tilt_6_12`** | `random_forest` | `episode_fa<=1` | **0.767** [0.667, 0.875] | **`0.221 / 0.633`** | `0.659 / 0.900` | **149** (`7.3%`) | **1,131** |
| **`gaussian_control`** | `random_forest` | `strict_window_fa<=1` | **0.400** [0.267, 0.517] | `0.148 / 0.367` | `0.300 / 0.433` | **44** (`2.2%`) | 229 |
| **`high_tilt_3_6`** | `random_forest` | `strict_window_fa<=1` | **0.733** [0.633, 0.850] | `0.227 / 0.633` | `0.546 / 0.833` | **241** (**`12.0%`**) | 570 |
| **`high_tilt_6_12`** | `random_forest` | `strict_window_fa<=1` | **0.667** [0.550, 0.783] | **`0.171 / 0.467`** | `0.588 / 0.867` | **164** (`8.1%`) | **1,004** |

**What happens in B1 (`3-6 deg` and `6-12 deg`):**
1. **Radial convergence is extremely clean at high tilt** (`dir_convergence_residual_m` median drops from `6.88 m` on control to **`2.80 m`** on `3-6 deg` and **`1.74 m`** on `6-12 deg`; `weighted_cos` median rises from `0.722` to **`0.976`** and **`0.993`**).
2. **`ThresholdRuleClassifier` breaks above $3.5^\circ$:** Because `imu_implausible_tilt_deg_th = 3.5` deg was tuned on a training set where basins never exceeded $3.0^\circ$, `ThresholdRuleClassifier` flags **39.0%** of `gradual` windows in `3-6 deg` and **67.8%** in `6-12 deg` as `sensor_fault`. For `rapid` runs in `6-12 deg`, the ramp crosses $3.5^\circ$ in 1–2 windows (before $k=3$ or $k=5$ persistence can trigger a `rapid` alarm), causing `rapid` run detection to collapse to **`0.067`** (`episode_fa<=1`) and **`0.000`** (`strict_window_fa<=1`).
3. **`RandomForest` and `LogisticRegression` show out-of-range class-boundary crossover:**
   - Because `gradual` ramp slope ($\text{deg/s}$) scales linearly with `peak_tilt_deg`, a `gradual` basin with $6^\circ\text{--}12^\circ$ peak tilt has the same window slope as a $1^\circ\text{--}3^\circ$ `rapid` basin! As a result, `RandomForest` misclassifies **`1,131 / 2,038` (`55.5%`) of `gradual` windows in `6-12 deg` as `rapid`** (and `149` windows as `sensor_fault`), so `gradual` window recall drops from `0.286` (`3-6 deg`) to `0.221` (`6-12 deg`; and at strict from `0.227` to `0.171`, with strict `gradual` run detection dropping from `0.633` to `0.467`).
   - Meanwhile, `LogisticRegression` `rapid` run detection drops from `0.567` (`3-6 deg`) to **`0.300`** (`6-12 deg`, and **`0.167`** at strict) because linear decision boundaries push extreme-tilt/LiDAR windows into `sensor_fault`.

---

## 3. Part B2: Realistic-Depth Knothe Spread ($\sigma = H / (\tan\beta\sqrt{2\pi})$)

### 3.1 Knothe Spread Grid ([realistic_depth_sigma_grid.md](file:///d:/Github%20Projects/ProjectCanary-Sim/outputs/eval_literature/realistic_depth_sigma_grid.md))
All values of $(H, \tan\beta)$ are a **PLACEHOLDER** grid until cited Indian coalfield values are filled in:

| Cell ID | Depth $H$ (m) | $\tan\beta$ | Influence Radius $R = H/\tan\beta$ (m) | Gaussian $\sigma$ (m) | $\sigma$ in Network Radii ($R_{\text{net}} = 14.14\text{ m}$) | Status vs Main Cap ($1.5\,R_{\text{net}}$) |
|---|---:|---:|---:|---:|---:|---|
| `H50_tanB1p5` | 50 | 1.5 | 33.33 | **13.30** | **0.940** | $\le 1.5$ (already in main training range) |
| `H50_tanB2p0` | 50 | 2.0 | 25.00 | **9.97** | **0.705** | $\le 1.5$ (already in main training range) |
| `H50_tanB2p5` | 50 | 2.5 | 20.00 | **7.98** | **0.564** | $\le 1.5$ (already in main training range) |
| `H100_tanB2p0` | 100 | 2.0 | 50.00 | **19.95** | **1.411** | $\le 1.5$ (already in main training range) |
| `H100_tanB2p5` | 100 | 2.5 | 40.00 | **15.96** | **1.128** | $\le 1.5$ (already in main training range) |
| **`H100_tanB1p5`** | 100 | 1.5 | 66.67 | **26.60** | **1.881** | **$> 1.5$ (20 runs simulated)** |
| **`H200_tanB2p5`** | 200 | 2.5 | 80.00 | **31.92** | **2.257** | **$> 1.5$ (20 runs simulated)** |
| **`H200_tanB2p0`** | 200 | 2.0 | 100.00 | **39.89** | **2.821** | **$> 1.5$ (20 runs simulated)** |
| **`H300_tanB2p5`** | 300 | 2.5 | 120.00 | **47.87** | **3.385** | **$> 1.5$ (20 runs simulated)** |
| **`H200_tanB1p5`** | 200 | 1.5 | 133.33 | **53.19** | **3.761** | **$> 1.5$ (20 runs simulated)** |
| **`H300_tanB2p0`** | 300 | 2.0 | 150.00 | **59.84** | **4.231** | **$> 1.5$ (20 runs simulated)** |
| **`H300_tanB1p5`** | 300 | 1.5 | 200.00 | **79.79** | **5.642** | **$> 1.5$ (20 runs simulated)** |

### 3.2 Realistic-Depth Results Across the 7 Wide-$\sigma$ Cells (140 Runs)

| Arm ($\sigma$ in Radii) | RF `episode_fa<=1` All / Grad / Rapid Detect | LR `episode_fa<=1` All / Grad / Rapid Detect | Thr `episode_fa<=1` All / Grad / Rapid Detect | RF Win Recall Grad / Rapid | Win `gradual -> sensor_fault` (RF / LR / Thr) | SNR>2 Convergence: `residual_m` p50 / `weighted_cos` p50 / `pairwise_cos` p50 |
|---|---|---|---|---|---|---|
| **`gaussian_control`** (`0.4-1.5R`) | **0.617** (`0.60 / 0.63`) | **0.467** (`0.40 / 0.53`) | **0.483** (`0.50 / 0.47`) | `0.246 / 0.383` | `40 / 15 / 80` | `3.23 m / +0.952 / -0.065` |
| **`depth_all_gt_1p5`** (140 runs) | **0.629** (`0.57 / 0.69`) | **0.464** (**`0.26` / `0.67`**) | **0.486** (`0.51 / 0.46`) | `0.280 / 0.422` | `15 / 19 / 1` | `1.20 m / +0.996 / +0.286` |
| `H100_tanB1p5` (`1.881R`) | **0.700** (`0.60 / 0.80`) | **0.500** (`0.30 / 0.70`) | **0.500** (`0.60 / 0.40`) | `0.359 / 0.421` | `2 / 0 / 0` | `1.15 m / +0.995 / -0.007` |
| `H200_tanB2p5` (`2.257R`) | **0.700** (`0.70 / 0.70`) | **0.500** (`0.30 / 0.70`) | **0.550** (`0.60 / 0.50`) | `0.242 / 0.453` | `0 / 3 / 0` | `1.09 m / +0.996 / +0.445` |
| `H200_tanB2p0` (`2.821R`) | **0.600** (`0.60 / 0.60`) | **0.450** (`0.40 / 0.50`) | **0.500** (`0.50 / 0.50`) | `0.307 / 0.427` | `1 / 1 / 0` | `1.26 m / +0.995 / +0.236` |
| `H300_tanB2p5` (`3.385R`) | **0.600** (`0.60 / 0.60`) | **0.450** (`0.20 / 0.70`) | **0.450** (`0.50 / 0.40`) | `0.265 / 0.415` | `0 / 1 / 0` | `1.07 m / +0.998 / +0.209` |
| `H200_tanB1p5` (`3.761R`) | **0.650** (`0.60 / 0.70`) | **0.500** (`0.40 / 0.60`) | **0.650** (`0.70 / 0.60`) | `0.325 / 0.371` | `2 / 8 / 0` | `1.18 m / +0.993 / +0.497` |
| `H300_tanB2p0` (`4.231R`) | **0.500** (`0.40 / 0.60`) | **0.400** (`0.20 / 0.60`) | **0.350** (`0.30 / 0.40`) | `0.230 / 0.418` | `10 / 6 / 1` | `1.36 m / +0.995 / +0.139` |
| `H300_tanB1p5` (`5.642R`) | **0.650** (`0.50 / 0.80`) | **0.450** (**`0.00` / `0.90`**) | **0.400** (`0.40 / 0.40`) | `0.235 / 0.449` | `0 / 0 / 0` | `1.08 m / +0.997 / +0.373` |

**Why B2 did NOT collapse into `planar_slope`, and what DID degrade at large $\sigma$:**
1. **Why radial convergence stayed sharp (`1.20 m` vs `3.23 m` at $\text{SNR}>2$) even though `dir_pairwise_cos_mean` turned positive (`+0.286` vs `-0.065`):**
   - Because the B2 instruction specifies *"centre within +-1.5 radii as before"* ($x_0, y_0 \in 10 \pm 21.2\text{ m}$) and the Knothe influence function is **isotropic** ($s_x = s_y = \sigma$), every node's true gradient vector $-\nabla w \propto (x_0 - x, y_0 - y)$ points **100% exactly at $(x_0, y_0)$** (with zero elliptical anisotropy distortion!), and $(x_0, y_0)$ is only $\le 21.2\text{ m}$ away from the network centre (well inside the $[-100, 120]\text{ m}$ convergence solver box).
   - Moreover, when $\sigma$ is $26.6\text{--}79.8\text{ m}$, the Gaussian envelope $\exp(-r^2/(2\sigma^2))$ no longer decays to zero at the far corners of the $20\times 20\text{ m}$ square. Thus **all 4 corners receive strong, high-SNR tilt vectors** pointing at $(x_0, y_0)$! When $(x_0, y_0)$ lies outside the $20\times 20\text{ m}$ square (between $10\text{ m}$ and $21.2\text{ m}$ from the centre), all 4 corners lie on the same side of $(x_0, y_0)$, so their vectors have positive pairwise cosine (`+0.286` to `+0.497`), yet their lines still intersect cleanly at $(x_0, y_0)$!
   - *(Caveat for the submission: if the basin centre at depth $H=300\text{ m}$ were instead located $1\text{--}2\,\sigma = 80\text{--}160\text{ m}$ away from the $20\times 20\text{ m}$ array, the intersection point would lie outside the $[-100, 120]\text{ m}$ clip box and the array would enter the true `planar_slope` regime).*
2. **What DID degrade in B2 (`gradual -> rapid` confusion on `LogisticRegression` and at $\sigma \ge 4.2\,R_{\text{net}}$):**
   - Because all 4 nodes experience stronger tilt and larger LiDAR vertical settlement ($s_{\max} \propto \sigma$, up to $7.3\text{ m}$ at $H=300\text{ m}$), `LogisticRegression` misclassified **`1,453 / 4,685` (`31.0%`) of `gradual` windows as `rapid`** (compared to only `515` predicted as `gradual`), causing `LogisticRegression` `gradual` run detection to drop from **`0.400` (control) to `0.257` pooled** — and to **`0.000` (0/10 runs!) at `H300_tanB1p5` ($\sigma = 5.64\,R_{\text{net}}$)**!
   - At the two widest cells (`H300_tanB2p0`, $\sigma = 4.23\,R_{\text{net}}$, and `H300_tanB1p5`, $\sigma = 5.64\,R_{\text{net}}$), `RandomForest` `gradual` run detection also began to dip (`0.400` and `0.500` vs `0.600` control), and `ThresholdRule` dropped from `0.500` to `0.300` and `0.400`.

---

## 4. Verification of Pre-Registered Predictions ([docs/literature_checks_expectations.md](file:///d:/Github%20Projects/ProjectCanary-Sim/docs/literature_checks_expectations.md))

| ID | Pre-Registered Prediction (Committed in `ad1a7ad` Before Scoring) | Outcome | Empirical Explanation |
|---|---|---|---|
| **L1** | In B2 ($\sigma > 1.5\,R_{\text{net}}$), `dir_pairwise_cos_mean` becomes positive as $\sigma/R_{\text{net}}$ grows, and `dir_convergence_residual_m` increases (`planar_slope` limit). | **PARTIALLY CONFIRMED (Refuted on Residual)** | `dir_pairwise_cos_mean` **did** turn positive (`-0.065` $\to$ **`+0.286`**, up to `+0.497`), confirming more parallel tilts. However, `dir_convergence_residual_m` **decreased** (`3.23 m` $\to$ `1.20 m` at $\text{SNR}>2$) because Knothe basins are isotropic ($s_x=s_y$, zero elliptical ray distortion), $(x_0, y_0)$ stayed within $\pm 21.2\text{ m}$ (inside the solver clip box), and wide $\sigma$ gave all 4 corners high SNR. |
| **L2** | In B2, `gradual` detection degrades on ML models while `threshold_rule` holds up. | **PARTIALLY CONFIRMED** | `LogisticRegression` `gradual` run detection degraded from `0.400` to **`0.257`** (and **`0.000`** at `H300_tanB1p5`), but via `gradual -> rapid` confusions (`1,453` windows) rather than `sensor_fault`. `RandomForest` (`0.629` vs `0.617`) and `ThresholdRule` (`0.486` vs `0.483`) held up well until $\sigma \ge 4.2\,R_{\text{net}}$. |
| **L3** | In B1 (`3-6 deg` and `6-12 deg`), `dir_convergence_residual_m` is low, but `ThresholdRuleClassifier` suffers a massive surge in `gradual -> sensor_fault` confusions above `3.5 deg`. | **CONFIRMED** | Median residual dropped to `2.80 m` and `1.74 m`, while `ThresholdRule` `gradual -> sensor_fault` confusions surged **$9.8\times$** (`80` $\to$ **`786`**, `39.0%`) in `3-6 deg` and **$17.3\times$** (`80` $\to$ **`1,382`**, `67.8%`) in `6-12 deg`, collapsing `ThresholdRule` `rapid` detection in `6-12 deg` to **`0.067`** (`0.000` at strict). |
| **L4** | In B1 (`high_tilt_6_12`), `gradual` window recall degrades on `RandomForest` as steeper ramp slopes push `gradual` windows into `rapid` and `sensor_fault`. | **CONFIRMED** | In `6-12 deg`, `RandomForest` misclassified **`1,131 / 2,038` (`55.5%`)** of `gradual` windows as `rapid` and `149` (`7.3%`) as `sensor_fault`, dropping `gradual` window recall from `0.286` (`3-6 deg`) to `0.221` (`0.227` $\to$ `0.171` at strict, and strict `gradual` run detection from `0.633` to `0.467`). |
| **L5** | In B1, the `15.0 deg` physical limit (`TILT_LIMIT_NORMAL_DEG`) is **not** triggered, whereas the manifest check `peak_tilt_deg <= 3.0` deg **is** triggered. | **CONFIRMED** | Max window node tilt was `6.025 deg` (`3-6 deg`) and `10.696 deg` (`6-12 deg`) — `0` windows exceeded `15.0 deg`, while the `[0.05, 3.0] deg` manifest check fired on both arms. |
| **L6** | Normal-run false alarms are bitwise unchanged from the 40 shape-mismatch normal runs. | **CONFIRMED** | Identical across all models and regimes (`0.450` FA eps/run for RF and Threshold Rule, `0.350` for LR at `episode_fa<=1`). |

---

## 5. Part C: Timescale Audit of Frozen Gradual Runs

### 5.1 Implied Physical Tilt Rates at $360\times$ vs. Real-World Order of Magnitude ([timescale_audit_summary.md](file:///d:/Github%20Projects/ProjectCanary-Sim/outputs/eval_literature/timescale_audit_summary.md))

In the frozen simulator configuration, each 240 s run at compression factor `360` represents $24\text{ h} = 1.0\text{ day}$, and the gradual S-curve ramp duration $\tau \in [96, 192]\text{ s}$ represents $\tau_{\text{phys}} \in [0.40, 0.80]\text{ days}$ ($9.6\text{ to }19.2\text{ hours}$).
Comparing against the rough real-world order-of-magnitude reference **`0.1 deg/day`** (**DERIVED, ORDER OF MAGNITUDE** — from ~`60 mm/m` [`3.4 deg`] developing over ~`30 days`, not a cited value):

| Frozen Manifest Subset | Gradual Runs | Runs Faster Than `0.1 deg/day` (Share) | Implied Rate at `360x` (`deg/day`): min / p10 / **p50 (Median)** / Mean / p90 / max | Speedup Factor vs `0.1 deg/day`: min / **p50 (Median)** / Mean / p90 / max |
|---|---:|---|---|---|
| **`ALL_GRADUAL_RUNS`** | **165** | **161 / 165 (`97.6%`)** | `0.087 / 0.150 / 0.763 / 1.547 / 4.495 / 6.597` | `0.87x / 7.63x / 15.47x / 44.95x / 65.97x` |
| `split:train` | 60 | **59 / 60 (`98.3%`)** | `0.087 / 0.173 / 0.760 / 1.474 / 4.270 / 5.955` | `0.87x / 7.60x / 14.74x / 42.70x / 59.55x` |
| `split:val` | 20 | **20 / 20 (`100.0%`)** | `0.118 / 0.148 / 0.870 / 1.459 / 3.501 / 6.002` | `1.18x / 8.70x / 14.59x / 35.01x / 60.02x` |
| `split:test` | 20 | **19 / 20 (`95.0%`)** | `0.100 / 0.175 / 0.686 / 1.422 / 3.544 / 5.636` | `1.00x / 6.86x / 14.22x / 35.44x / 56.37x` |
| `split:shifted_test` | 15 | **14 / 15 (`93.3%`)** | `0.088 / 0.160 / 0.714 / 1.555 / 3.880 / 6.597` | `0.88x / 7.14x / 15.55x / 38.80x / 65.97x` |
| `split:broad_basin` | 20 | **20 / 20 (`100.0%`)** | `0.106 / 0.134 / 0.680 / 1.586 / 4.189 / 6.457` | `1.06x / 6.80x / 15.86x / 41.89x / 64.57x` |
| `split:stress` | 30 | **29 / 30 (`96.7%`)** | `0.093 / 0.126 / 1.033 / 1.803 / 4.736 / 5.482` | `0.93x / 10.33x / 18.03x / 47.36x / 54.82x` |

### 5.2 Should the $360\times$ ("24 h per run") Claim Be Replaced, and What Would It Change?

- **Plain Recommendation:** **Yes — in prose and documentation, the $360\times$ ("24 h per run") label should be treated as an arbitrary time-compression placeholder rather than a realistic longwall subsidence duration.** At $360\times$, **97.6% (161/165)** of gradual runs rise faster than the `0.1 deg/day` order-of-magnitude reference (median **`7.63x` faster**, mean **`15.47x` faster**, max **`65.97x` faster**). To represent a ~30-day active longwall subsidence phase inside a 240 s simulation run, the physical compression factor would need to be **`10,800x` (`30 days` per 240 s run)** or **`10,000x` to `30,000x`**.
- **What changing the compression factor changes (and what it does NOT change):**
  1. **It changes ONLY the labelled physical duration of a run and the Definition A ("slow-process time") false-alarm-per-hour column** (see table below from [normal_false_alarms_with_alt_compression.md](file:///d:/Github%20Projects/ProjectCanary-Sim/outputs/eval_literature/normal_false_alarms_with_alt_compression.md)).
  2. **It does NOT change the simulated sensor telemetry, window features, model scores, recall, or the stricter Definition B ("sensor time") false-alarm rate**, because the IMU noise, piezo vibration, LiDAR sweeps, and 5 s windows run in uncompressed sensor time ($240\text{ s} = 0.0667\text{ h}$) and the classifiers rely primarily on multi-node magnitude, direction convergence, and long-horizon window slopes computed in simulation seconds.

| Model | Regime | FA Episodes / Run (40 Normal Runs) | Definition B: Sensor Time (`1x`, `0.0667 h/run`) | Definition A (Frozen `360x`, `24 h/run`) | Alternative `A_alt` (`10,000x`, `666.7 h/run`) | Alternative `A_alt` (`10,800x` = `30 d/run`, `720 h/run`) | Alternative `A_alt` (`30,000x`, `2,000 h/run`) |
|---|---|---|---|---|---|---|---|
| `threshold_rule` | `episode_fa<=1` | `0.450 [0.175, 0.800]` | `6.75/h` (`8.9 min`) | `0.0187/h` (`53.3 h`) | `0.000675/h` (`1,481.5 h`) | `0.000625/h` (`1,600.0 h`) | `0.000225/h` (`4,444.4 h`) |
| `threshold_rule` | `strict_window_fa<=1` | `0.100 [0.000, 0.225]` | `1.50/h` (`40.0 min`) | `0.0042/h` (`240.0 h`) | `0.000150/h` (`6,666.7 h`) | `0.000139/h` (`7,200.0 h`) | `0.000050/h` (`20,000.0 h`) |
| `logistic_regression` | `episode_fa<=1` | `0.350 [0.175, 0.550]` | `5.25/h` (`11.4 min`) | `0.0146/h` (`68.6 h`) | `0.000525/h` (`1,904.8 h`) | `0.000486/h` (`2,057.1 h`) | `0.000175/h` (`5,714.3 h`) |
| `logistic_regression` | `strict_window_fa<=1` | `0.125 [0.000, 0.338]` | `1.88/h` (`32.0 min`) | `0.0052/h` (`192.0 h`) | `0.000188/h` (`5,333.3 h`) | `0.000174/h` (`5,760.0 h`) | `0.000063/h` (`16,000.0 h`) |
| `random_forest` | `episode_fa<=1` | `0.450 [0.225, 0.700]` | `6.75/h` (`8.9 min`) | `0.0187/h` (`53.3 h`) | `0.000675/h` (`1,481.5 h`) | `0.000625/h` (`1,600.0 h`) | `0.000225/h` (`4,444.4 h`) |
| `random_forest` | `strict_window_fa<=1` | `0.225 [0.100, 0.363]` | `3.38/h` (`17.8 min`) | `0.0094/h` (`106.7 h`) | `0.000338/h` (`2,963.0 h`) | `0.000313/h` (`3,200.0 h`) | `0.000112/h` (`8,888.9 h`) |

---

## 6. Plain Summary: What Holds, What Breaks, and What to State in the Submission

1. **What holds at realistic tilt and depth:**
   - **Random Forest run-level detection holds up across both the upper longwall tilt range (`3-6 deg`: `0.767` vs `0.617` control at `episode_fa<=1`) and the realistic-depth Knothe spread grid (`depth_all_gt_1p5`: `0.629` vs `0.617` control)**, as long as the subsidence centre lies within $\pm 1.5\,R_{\text{net}}$ of the array.
   - **Shallow-to-moderate depths ($H = 50\text{--}100\text{ m}$ at $\tan\beta \in \{2.0, 2.5\}$) have $\sigma \in [0.56, 1.41]\,R_{\text{net}} \le 1.5\,R_{\text{net}}$**, so they are already directly inside the main training range.
2. **What breaks:**
   - **Hard-coded `3.5 deg` implausible-tilt threshold in `ThresholdRuleClassifier`:** Above $3.5^\circ$ (which overlaps the upper half of measured Ashton longwall tilts, $0.7^\circ\text{--}5.9^\circ$), `ThresholdRuleClassifier` misclassifies $39\%\text{--}68\%$ of `gradual` windows as `sensor_fault` and misses $93\%\text{--}100\%$ of `rapid` runs in `6-12 deg`. In any future model revision, the implausible-tilt fault threshold should be set near the physical sensor saturation/mount limit ($\ge 15^\circ\text{--}30^\circ$), not $3.5^\circ$.
   - **`gradual` vs `rapid` slope crossover at high tilt (`6-12 deg`) and deep wide basins ($H = 300\text{ m}$):** Because ramp slope ($\text{deg/s}$) scales with `peak_tilt_deg`, `RandomForest` misclassifies $55.5\%$ of `6-12 deg` `gradual` windows as `rapid`, and `LogisticRegression` loses `gradual` detection at $H = 300\text{ m}, \tan\beta = 1.5$ (`0.000` detection).
   - **Off-array deep basins:** As shown by `planar_slope` in the shape-mismatch report, if a deep basin ($\sigma \gg 20\text{ m}$) is also centred far outside the $20\times 20\text{ m}$ network, the 4 corner tilts become parallel without a local intersection point and ML `gradual -> sensor_fault` confusions triple.
3. **What to state plainly in the submission:**
   - Frame the main `0.05-3 deg` range as an **early-warning tilt regime** (covering the lower half of measured longwall maximum tilts, `0.7-5.9 deg`).
   - State that $\tan(\beta)$ and Knothe $c$ are **not yet sourced for the target Indian coalfield** and that all data and sensitivity grids are **synthetic simulations, not validated on real field data**.
   - Report false alarms primarily in **Definition B (sensor time)** and state that the `360x` ("24 h/run") compression factor is a label placeholder whose implied gradual tilt rate (`0.76 deg/day` median) is ~`7.6x` faster than a rough 30-day longwall order of magnitude (`~0.1 deg/day`, derived).

### Generated Plots in `outputs/eval_literature/`
All three plots carry the red `"SYNTHETIC / SIMULATION"` banner and compression factor:
- [high_tilt_sensitivity.png](file:///d:/Github%20Projects/ProjectCanary-Sim/outputs/eval_literature/high_tilt_sensitivity.png)
- [realistic_depth_sensitivity.png](file:///d:/Github%20Projects/ProjectCanary-Sim/outputs/eval_literature/realistic_depth_sensitivity.png)
- [timescale_audit_gradual_rates.png](file:///d:/Github%20Projects/ProjectCanary-Sim/outputs/eval_literature/timescale_audit_gradual_rates.png)
