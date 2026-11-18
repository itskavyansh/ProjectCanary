# Pre-Registered Expectations: Subsidence-Shape Mismatch Stress Test (SYNTHETIC / SIMULATION)

> [!CAUTION]
> **SYNTHETIC / SIMULATION — PRE-REGISTERED HYPOTHESES (WRITTEN & COMMITTED BEFORE SCORING).**
> This document records our predictions for the shape-mismatch stress test **before** generating or scoring any runs in `outputs/eval_shape_mismatch/`. The frozen v3 models (`threshold_rule`, `logistic_regression`, `random_forest`) and frozen v3 operating points were trained and tuned exclusively on single elliptical Gaussian basins (`engine.py`). All alternative shapes below are still synthetic mathematical surfaces, **not** field ground truth and **not** a geomechanical validation.

---

## 1. Experimental Setup Summary

- **Frozen models under test:** v3 `threshold_rule`, `logistic_regression`, and `random_forest` (`outputs/eval_v3/trained_models_v3.pkl` and `operating_points_v3.pkl`), with zero retraining, retuning, or threshold adjustments.
- **Five synthetic shapes (`canary_sim/shapes.py`):**
  1. `gaussian_control`: Elliptical Gaussian bowl (exact training distribution control).
  2. `trough_panel`: Product of $\operatorname{erf}$ differences along $x'$ and $y'$ (flat-bottomed rectangular extraction panel with steep perimeter flanks).
  3. `asymmetric`: Split elliptical Gaussian with different spread on the two sides of the principal axis ($\sigma_{+x'} / \sigma_{-x'} \in [1.5, 3.0]$).
  4. `two_basin`: Superposition of two isotropic Gaussians separated by $1\text{--}3\,\sigma_1$ with amplitude ratio $[0.3, 1.0]$ (explicitly out of scope in the original single-basin specification).
  5. `planar_slope`: Uniform tilt plane $w(x,y,t) = -s(t)\,[(x-x_0)\cos\phi + (y-y_0)\sin\phi]$ with constant slope direction $\phi$ everywhere and no convergence point.
- **Peak-slope matching:** Every shape is scaled so that $\max_{(x,y)} \|\nabla w(x,y,t_{\text{final}})\| = \tan(\text{peak\_tilt\_deg})$ using the identical log-stratified `peak_tilt_deg` sequence and paired seeds ($30$ `gradual` + $30$ `rapid` runs per shape, plus $40$ shared `normal` runs).

---

## 2. Mechanism Analysis & Predictions per Shape

### 2.1 `gaussian_control` (Control Group)
- **Radial convergence geometry:** All 4 corner downslope vectors point toward the single basin centre $(x_0, y_0)$ up to elliptical anisotropy ($s_y / s_x \in [1, 2]$) and IMU noise.
- **Expected metrics:**
  - `dir_convergence_weighted_cos`: High positive alignment ($\sim 0.7\text{--}0.9$ on active windows with $\text{SNR} > 2$).
  - `dir_convergence_residual_m`: Low orthogonal residual ($\sim 2\text{--}5\text{ m}$ at moderate-to-high SNR).
  - Detection & recall: Baseline reference matching the frozen `test` split (~65–70% exact run detection at `episode_fa<=1`, low overall window recall $\sim 0.15\text{--}0.30$ rising strongly in the $\text{SNR} > 5$ bin).

### 2.2 `asymmetric` (Skewed Single Basin)
- **Physical / geometric effect:** Introducing different decay scales $\sigma_{-x'}$ and $\sigma_{+x'}$ on the two halves of the $x'$ axis changes the relative tilt magnitudes on opposite sides of $(x_0, y_0)$ and bends off-axis gradient vectors toward the steeper axis, but **every point still dips inward toward the single minimum at $(x_0, y_0)$**.
- **Prediction (Should NOT degrade materially):**
  1. **Radial convergence features (`dir_convergence_weighted_cos`, `dir_convergence_residual_m`):** Will remain close to `gaussian_control` (at most a slight increase of $\sim 0.5\text{--}1.5\text{ m}$ in residual, well within the spread already caused by elliptical aspect ratios $s_y/s_x \in [1, 2]$).
  2. **Detection rate & window recall:** Within bootstrap noise ($\pm 5\text{--}8$ percentage points) of `gaussian_control` across all three models and SNR bins.
  3. **`gradual` vs `sensor_fault` confusions:** No meaningful increase relative to `gaussian_control`.

### 2.3 `trough_panel` (Flat-Bottomed Rectangular Erf Trough)
- **Physical / geometric effect:**
  - Downslope vectors on the flanks still point **inward** toward the panel interior, but normals to the straight edges of a rectangle intersect along a central ridge segment rather than a single point.
  - Because the peak gradient is concentrated along a narrow perimeter band of width $\sim \sigma_{\text{inf}} \in [0.15, 0.5]\,R_{\text{net}}$, nodes sitting inside the flat panel bottom or far outside the steep rim experience near-zero tilt even when `peak_tilt_deg` on the rim is large.
- **Prediction (Mild-to-moderate degradation):**
  1. **Radial convergence features:** `dir_convergence_residual_m` should be **moderately higher** than `gaussian_control` (rays normal to box edges do not meet at a point), while `dir_convergence_weighted_cos` should remain positive (inward dipping) but slightly lower than `gaussian_control`.
  2. **Detection rate & window recall:** Moderate reduction in `gradual` and `rapid` window recall (and a modest drop in run-level detection at intermediate SNR) when the steep rim misses the corner nodes, partially offset when the rim passes near a corner node.
  3. **`gradual` vs `sensor_fault` confusions:** Slight increase compared to `gaussian_control` when only one corner node lies on the narrow steep edge of the panel (mimicking a single-node anomaly).

### 2.4 `two_basin` (Two Interacting Gaussians, $1\text{--}3\,\sigma_1$ Separation)
- **Physical / geometric effect:**
  - Explicitly violates the single-centre assumption: nodes near $(x_0, y_0)$ dip toward Basin 1 while nodes near $(x_2, y_2)$ dip toward Basin 2, and the saddle between them has competing gradients.
  - However, having two subsidence bowls in the $20\times 20\text{ m}$ domain increases the spatial footprint of ground movement, so more corner nodes are likely to see non-trivial tilt and LiDAR range change at a given peak slope.
- **Prediction (Degraded convergence residual; mixed/competing effect on detection):**
  1. **Radial convergence features:** `dir_convergence_residual_m` will **increase (degrade)** relative to `gaussian_control`, and `dir_convergence_weighted_cos` will be **lower** than `gaussian_control` because 4 rays pointing toward two separate centres cannot be satisfied by a single focal point $\hat{c}$.
  2. **Detection rate & window recall:** Two competing effects — poorer convergence alignment pushes against `gradual` classification, while broader multi-node tilt and LiDAR displacement energy pulls in favour of detection. We expect **direction alignment to degrade**, `gradual` vs `sensor_fault` confusions to **increase** relative to `gaussian_control`, and overall run-level detection rate to stay within $\sim 10$ percentage points of control (potentially slightly lower if convergence penalty dominates, or slightly higher at low SNR if the wider footprint dominates).

### 2.5 `planar_slope` (Uniform Tilt Plane — No Convergence Point)
- **Physical / geometric effect:**
  - All 4 corner nodes tilt with the **exact same world-frame downslope direction** $u = (\cos\phi, \sin\phi)$ and the **exact same tilt magnitude** equal to `peak_tilt_deg` (100% of the peak slope is applied to every node, unlike any localized basin where distant nodes decay exponentially).
  - In `_fit_convergence_point` (`canary_sim/features.py`), when all 4 rays are parallel ($n_1 = n_2 = n_3 = n_4$), the $2\times 2$ normal matrix is rank-1 plus $\lambda I$, placing $\hat{c}$ near the regularizer centre $(10, 10)$. Relative to $(10, 10)$, two corners of the square lie upslope ($\cos_k > 0$, pointing toward $\hat{c}$) and two corners lie downslope ($\cos_k < 0$, pointing **away** from $\hat{c}$).
- **Prediction (Severe breakdown of radial convergence; strong 4-node magnitude signal):**
  1. **Radial convergence features WILL BREAK completely:**
     - `dir_convergence_weighted_cos` will collapse from $\sim +0.7\text{--}0.9$ (control) down to **$\approx 0.0$** (since two nodes have $+\cos_k$ and two have $-\cos_k$ of equal weight).
     - `dir_convergence_residual_m` will jump to **$\sim 8\text{--}14\text{ m}$** (roughly $2\text{--}3\times$ the control residual) because the two downslope rays with $\cos_k < 0$ are penalized by their full Euclidean distance to $\hat{c}$.
     - Conversely, raw pairwise direction cosine (`dir_pairwise_cos_mean`) will be near **$+1.0$** (parallel vectors).
  2. **Impact on models & `gradual` vs `sensor_fault` confusion:**
     - Because the v3 Random Forest and Logistic Regression use `dir_convergence_residual_m` and `dir_convergence_align_cos` to separate `gradual` basins from `sensor_fault`, the broken convergence features on `planar_slope` should cause a **sharp increase in `gradual` $\to$ `sensor_fault` (or `rapid`) misclassifications** for `logistic_regression` (and increased `sensor_fault` confusion for `random_forest`).
     - At the same time, because `planar_slope` gives **every** node the full `peak_tilt_deg` (so mean node tilt is $2\text{--}3\times$ higher than an off-centre Gaussian of the same peak tilt!), `threshold_rule` and magnitude/slope-driven splits will see much stronger per-node tilt and LiDAR range changes, which may boost raw event triggering despite the broken convergence features.

### 2.6 Shared `normal` Runs
- **Prediction:** Because the 40 new `normal` runs use `null_field=True` (`s_max = 0`, zero settlement everywhere) and are shared identically across all shape evaluations, **false-alarm rates on the normal runs must be 100% identical across all 5 shapes** and statistically consistent (within the 95% run-bootstrap CI) with the frozen `test` normal-run false-alarm rate ($\sim 0.65$ episodes/run for RF at `episode_fa<=1`).

---

## 3. Summary Checklist of Pre-Registered Predictions to Verify After Scoring

| ID | Pre-Registered Prediction | Metric / Comparison |
|---|---|---|
| **P1** | `planar_slope` breaks radial convergence: `dir_convergence_weighted_cos` collapses near $0$ and `dir_convergence_residual_m` is $\ge 2\times$ the `gaussian_control` median on active windows. | Median & IQR of `dir_convergence_weighted_cos` and `dir_convergence_residual_m` vs `gaussian_control` |
| **P2** | `two_basin` degrades radial convergence (`dir_convergence_residual_m` higher and `dir_convergence_weighted_cos` lower than `gaussian_control`), though much less severely than `planar_slope`. | `two_basin` vs `gaussian_control` convergence distributions |
| **P3** | `trough_panel` shows moderately higher `dir_convergence_residual_m` than `gaussian_control` and lower window recall at intermediate SNR due to its flat bottom / narrow steep rim. | `trough_panel` vs `gaussian_control` convergence & recall |
| **P4** | `asymmetric` behaves closest to `gaussian_control`: minimal change in convergence features, window recall, and run detection rate. | `asymmetric` vs `gaussian_control` across all metrics |
| **P5** | `planar_slope` increases `gradual` $\leftrightarrow$ `sensor_fault` confusion counts relative to `gaussian_control` on the ML models that weight convergence residuals. | `gradual` vs `sensor_fault` confusion count table |
| **P6** | False-alarm rates on the 40 shared `normal` runs are identical across all 5 shape evaluations and within the 95% bootstrap CI of the frozen `test` split. | Normal-run FA episodes/run & FA/hour |
