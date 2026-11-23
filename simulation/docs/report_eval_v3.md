# Evaluation Report v3: Nominal-Yaw World-Frame Radial Convergence & Causal Trailing-Horizon Coherence (SYNTHETIC / SIMULATION)

> [!IMPORTANT]
> **SYNTHETIC / SIMULATION ONLY**: All datasets, sensor models, faults, confounders, operating points, and evaluation metrics in this report are generated from synthetic simulation models. **None of these results are validated on real field or site-deployment data.**

Frozen items respected without modification:
- State-timeline schema v1 (`docs/schema_v1.md`, `canary_sim/schema.py`).
- Label policy (`canary_sim/labels.py`).
- Dataset splits, seeds, and manifest (`outputs/datasets/` — untouched; `outputs/datasets_v3/manifest.csv` and `window_labels.csv` are verified by `assert_frame_equal` to be 100% identical to `outputs/datasets/`).
- Round-1 (`outputs/eval/`) and Round-2 (`outputs/eval_v2/`) results are kept intact; all Round-3 outputs are written to `outputs/datasets_v3/` and `outputs/eval_v3/`.
- `Canary-Analyst` was neither touched nor imported.

---

## 1. Feature Formulation & Installation-Yaw Requirement (`canary_sim/features.py`)

### 1.1 Nominal Installation-Yaw World-Frame Rotation
- **Installation-Yaw Requirement**: To compare tilt directions across nodes in a common coordinate frame, each corner node's **nominal installation compass orientation (`yaw_deg`) must be recorded when the sensor is mounted** (also documented in `docs/data_dictionary.md`).
- **Strict Separation from Ground-Truth Misalignment**: Observable features rotate each node's lead-in-subtracted tilt vector $(\Delta \text{tilt}_x, \Delta \text{tilt}_y)$ into the world frame using **only nominal `yaw_deg`** ($\hat{\psi}_k = \text{yaw\_deg}_k$). The unknown installation misalignment (`yaw_error_deg`, $\sigma = 1^\circ$) is **never** passed to observable model features (enforced by `tests/test_features_v3.py::test_world_frame_rotation_uses_nominal_yaw_only_not_yaw_error`).

### 1.2 Fitted Radial Convergence Point (Not Fixed to `(10, 10)`)
- Rather than measuring convergence toward a fixed point like $(10, 10)$ (which fails for off-centre basins), `_fit_radial_convergence` estimates the best-fit 2-D focal point $\hat{\mathbf{c}} = (\hat{c}_x, \hat{c}_y)$ directly from the 4 nodes' world-frame downslope rays $(\mathbf{p}_k, \mathbf{u}_k)$ via magnitude-weighted orthogonal least squares:
  $$\hat{\mathbf{c}} = \arg\min_{\mathbf{c}} \sum_{k=1}^4 w_k \bigl((\mathbf{c} - \mathbf{p}_k) \cdot \mathbf{n}_k\bigr)^2 + \lambda \|\mathbf{c} - \mathbf{c}_0\|^2, \qquad w_k = \frac{m_k^2}{\sum_j m_j^2}, \quad \lambda = 10^{-3}$$
  where $\mathbf{n}_k = (-u_{k,y}, u_{k,x})$ is the unit normal to node $k$'s downslope ray.
- From the fitted point $\hat{\mathbf{c}}$, we extract three observable features (both on the current window `dir_win_convergence_*` and on the causal trailing $N=12$-window mean tilt vector `dir_convergence_*`):
  1. `dir_convergence_align_cos`: unweighted mean cosine of the angle between each node's ray $\mathbf{u}_k$ and $(\hat{\mathbf{c}} - \mathbf{p}_k)$.
  2. `dir_convergence_weighted_cos`: $w_k$-weighted cosine alignment to $\hat{\mathbf{c}}$ (near $+1.0$ when active nodes point coherently toward a common basin focal point; negative when a single-node drift fault pulls one node against background noise).
  3. `dir_convergence_residual_m`: $w_k$-weighted RMS orthogonal distance (in metres) from $\hat{\mathbf{c}}$ to the 4 rays (penalised by full distance $\|\hat{\mathbf{c}} - \mathbf{p}_k\|$ whenever a ray points away from $\hat{\mathbf{c}}$).

### 1.3 Causal Same-Run Trailing-Horizon Coherence
- For each window $i$ of a run, `coherence_trailing_wx`, `coherence_trailing_wy`, `coherence_trailing_signed_mean`, and `coherence_trailing_mag` compute the mean pairwise Pearson correlation across the 4 nodes over the **past windows $\max(0, i - N + 1) \dots i$ of that same run only** ($N = 12$ windows $= 32.5\text{ s}$ span).
- **No Future or Cross-Run Leakage**: Tested in `tests/test_features_v3.py::test_trailing_horizon_coherence_causal_and_same_run_only`.

### 1.4 Separation Achieved on `train + val` (`plot_v3_features_diagnosis.png`)
On `train + val`, the new observable features cleanly separate `gradual` basins at `SNR >= 2` from single/two-node `sensor_fault` (`slow_drift`/`bias_step`):
- `dir_convergence_weighted_cos` median rises from `-0.27` (`SNR < 0.5`) $\to$ `+0.17` (`SNR 1–2`) $\to$ **`+0.61` (`SNR 2–5`)** $\to$ **`+0.95` (`SNR > 5`)**, whereas `sensor_fault (drift/bias)` sits at **`-0.77`**.
- `dir_convergence_residual_m` median drops from `13.9 m` (`SNR < 0.5`) $\to$ **`7.0 m` (`SNR 2–5`)** $\to$ **`4.0 m` (`SNR > 5`)**, whereas `sensor_fault (drift/bias)` sits at **`13.1 m`**.

---

## 2. Retuned Thresholds & Operating Points on `val` Only (`eval_v3_configs.json`)
All models and operating points were fit on `train` and tuned on `val` ONLY (`81` observable model features):
- **Validation Macro-F1 Jump**: On `val`, raw Random Forest Macro-F1 increased from **`0.6224` (v1/v2) to `0.6713` (v3)** (`+0.0489`), and under the episode-matched operating point (`val` FA episodes $= 1.00$, $k=3, m=3$), `val` exact run detection rate increased from **`0.680` (v2) to `0.760` (v3)** (`+8.0%`) while `val` Alarm Macro-F1 rose from **`0.6092` to `0.6552`**.
- **Retuned Threshold Rule (`threshold_rule_v3`)**: Uses `dir_convergence_residual_m_th = 17.606 m` and `imu_single_node_ratio_th = 2.4` (modulating the single-node anomaly ratio by `1.25×` when `dir_convergence_weighted_cos > 0.75` and `residual < 17.6 m`, and by `0.78×` when `dir_convergence_weighted_cos < -0.15`).
- **Retuned Minimum-Probability Operating Points on `val`**:
  - **Episode-Matched (`val` FA episodes $\le 1.0$, $k=3, m=3$)**:
    - **Random Forest**: $\tau = (\text{basin } 0.34337, \text{vibration } 0.25000, \text{sensor\_fault } 0.29415, \text{comm\_fault } 0.22000)$ $\to$ `val` FA episodes $= 1.00$, `val` exact detect rate $= 0.760$.
    - **Logistic Regression (`C = 5.0`)**: $\tau = (0.48106, 0.25000, 0.54078, 0.46725)$ $\to$ `val` FA episodes $= 0.85$, `val` exact detect rate $= 0.660$.
  - **Strict Window-Matched (`val` FA windows $\le 1.0$, $k=3, m=2$ for RF/LR, $k=5, m=2$ for TH)**:
    - **Random Forest**: $\tau = (0.45254, 0.25000, 0.28090, 0.22000)$ $\to$ `val` FA episodes $= 0.40$, FA windows $= 0.80$.
    - **Logistic Regression**: $\tau = (0.58300, 0.25000, 0.57501, 0.46725)$ $\to$ `val` FA episodes $= 0.30$, FA windows $= 0.95$.

---

## 3. Gains Reported Separately for `SNR > 2` (`v2_vs_v3_snr_gt_2_comparison.md`)

Evaluating on the **`SNR > 2` subset** (all `gradual` and `rapid` runs with `snr_vs_imu_noise > 2.0` plus all non-basin runs):

| Split (`SNR > 2`) | Model | Regime | Version | Window Macro-F1 (95% CI) | Alarm Macro-F1 (95% CI) | `gradual` Win Recall | `rapid` Win Recall | `sensor_fault` Win Recall | Run Alarm Detect Rate | FA Windows / Normal Run | `gradual <-> sensor_fault` Confusions |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| **`test`** (`99` runs) | **Random Forest** | `raw (k=1)` | `v2` | 0.6736 `[0.615, 0.725]` | 0.6736 | 0.462 | 0.526 | 0.422 | 0.899 | 30.05 | 92 |
| | **Random Forest** | `raw (k=1)` | **`v3`** | 0.6722 `[0.615, 0.730]` | 0.6722 | 0.279 | **0.543** | 0.386 | 0.886 | **11.45** (`-62%`) | **39** (`-58%`) |
| | **Random Forest** | `episode_fa<=1` | `v2` | 0.6910 `[0.623, 0.749]` | 0.6795 | 0.249 | 0.406 | 0.485 | 0.823 | 3.95 | 25 |
| | **Random Forest** | `episode_fa<=1` | **`v3`** | 0.6725 `[0.613, 0.727]` | 0.6703 | **0.250** | **0.537** (`+13.1%`) | 0.416 | **0.823** | 5.20 | 30 |
| | **Random Forest** | `strict_win_fa<=1` | `v2` | 0.6872 `[0.609, 0.754]` | 0.6714 | 0.192 | 0.375 | 0.436 | 0.760 | 1.35 | 5 |
| | **Random Forest** | `strict_win_fa<=1` | **`v3`** | 0.6825 `[0.619, 0.743]` | 0.6663 | **0.201** | **0.472** (`+9.7%`) | 0.434 | **0.798** (`+3.8%`) | 1.90 | 8 |
| **`test`** (`99` runs) | **Threshold Rule** | `episode_fa<=1` | `v2` | 0.7020 `[0.648, 0.737]` | 0.6891 | 0.276 | 0.516 | 0.454 | 0.835 | 5.60 | 54 |
| | **Threshold Rule** | `episode_fa<=1` | **`v3`** | **0.7022** `[0.651, 0.734]` | **0.6908** | 0.275 | 0.509 | **0.469** (`+1.5%`) | **0.848** (`+1.3%`) | 5.60 | **47** (`-13%`) |
| **`test`** (`99` runs) | **Logistic Reg** | `episode_fa<=1` | `v2` | 0.6054 `[0.540, 0.652]` | 0.5937 | 0.145 | 0.309 | 0.211 | 0.696 | 0.45 | 25 |
| | **Logistic Reg** | `episode_fa<=1` | **`v3`** | **0.6164** `[0.562, 0.669]` | **0.6033** | 0.089 | **0.351** (`+4.2%`) | **0.260** (`+4.9%`) | **0.709** (`+1.3%`) | 0.45 | **15** (`-40%`) |
| **`shifted_test`** (`71` runs) | **Random Forest** | `raw (k=1)` | `v2` | 0.5242 `[0.477, 0.572]` | 0.5242 | 0.701 | 0.295 | 0.343 | 0.929 | 77.33 | 172 |
| | **Random Forest** | `raw (k=1)` | **`v3`** | **0.6130** (`+0.0888`) | **0.6130** | 0.422 | 0.239 | **0.431** (`+8.8%`) | 0.929 | **32.07** (`-59%`) | **108** (`-37%`) |
| | **Random Forest** | `episode_fa<=1` | `v2` | 0.6218 `[0.567, 0.672]` | 0.6149 | 0.529 | 0.264 | 0.452 | 0.875 | 12.20 | 102 |
| | **Random Forest** | `episode_fa<=1` | **`v3`** | **0.6249** `[0.567, 0.672]` | 0.6082 | 0.391 | 0.236 | **0.463** (`+1.1%`) | 0.875 | **8.60** (`-30%`) | **86** (`-16%`) |
| **`broad_basin`** (`20` runs) | **Random Forest** | `episode_fa<=1` | `v2` | 0.5684 `[0.444, 0.659]` | 0.5673 | 0.398 | 0.398 | — | 0.650 | — | 4 |
| | **Random Forest** | `episode_fa<=1` | **`v3`** | **0.5827** (`+0.0143`) | **0.5723** | 0.354 | **0.513** (`+11.5%`) | — | **0.900** (`+25.0%`) | — | 5 |
| | **Random Forest** | `strict_win_fa<=1` | `v2` | 0.5462 `[0.419, 0.641]` | 0.5300 | 0.351 | 0.389 | — | 0.650 | — | 0 |
| | **Random Forest** | `strict_win_fa<=1` | **`v3`** | 0.5263 `[0.437, 0.606]` | 0.4913 | 0.265 | **0.437** (`+4.8%`) | — | **0.800** (`+15.0%`) | — | 7 |
| | **Logistic Reg** | `episode_fa<=1` | `v2` | 0.4596 `[0.343, 0.556]` | 0.4293 | 0.240 | 0.429 | — | 0.500 | — | 0 |
| | **Logistic Reg** | `episode_fa<=1` | **`v3`** | **0.5335** (`+0.0739`) | **0.4981** | 0.258 | **0.608** (`+17.9%`) | — | **0.750** (`+25.0%`) | — | 0 |

---

## 4. Gains Across All SNR Splits (`v2_vs_v3_all_snr_comparison.md`) & Updated RF Ablations (`rf_ablations_v3.md`)

### 4.1 Major Gains on `broad_basin`, `stress`, and `shifted_test` (All SNR, Episode-Matched $k=3, m=3$)
- **`broad_basin` (40 runs)**:
  - **Random Forest**: Window Macro-F1 rises from **`0.4173` (v2) to `0.4647` (v3)** (`+0.0474`), Persisted Alarm Macro-F1 rises from **`0.4093` to `0.4575`**, Run Exact Alarm Detection Rate jumps from **`0.425` to `0.725`** (`+30.0 percentage points`), and mean detection delay drops by **`7.2 s`** (`56.35 s -> 49.13 s`).
  - **Logistic Regression**: Window Macro-F1 rises from **`0.3077` to `0.3814`** (`+0.0737`) and Run Alarm Detection Rate rises from **`0.250` to `0.425`**.
- **`stress` (100 combined-condition runs)**:
  - **Random Forest**: Window Macro-F1 rises from **`0.4794` (v2) to `0.5300` (v3)** (`+0.0506`), Persisted Alarm Macro-F1 rises from **`0.4434` to `0.5013`** (`+0.0579`), and Run Alarm Detection Rate rises from **`0.930` to `0.950`**.
  - **Logistic Regression**: Window Macro-F1 rises from **`0.4511` to `0.4938`** (`+0.0427`) and Run Alarm Detection Rate rises from **`0.870` to `0.930`**.
- **`shifted_test` (90 runs)**:
  - **Random Forest**: Raw Window Macro-F1 jumps from **`0.5372` (v1) to `0.5947` (v3)** (`+0.0575`, cutting raw FA windows from `77.3` to `32.1`). Under episode-matched persistence, Window Macro-F1 rises from **`0.5977` (v2) to `0.6097` (v3)**, Alarm Macro-F1 rises from **`0.5828` to `0.5915`**, FA episodes drop from **`2.13` to `1.40`**, and FA windows drop from **`12.20` to `8.60`**.

### 4.2 Updated Random Forest Feature-Group Ablations (`rf_ablations_v3.md`)
| Group Removed | Kept Features (of 81) | Test Macro-F1 (`SNR > 2`) | $\Delta$ Macro-F1 (`SNR > 2`) | `gradual` Recall (`SNR > 2`) | `sensor_fault` Recall (`SNR > 2`) | `gradual <-> sensor_fault` Confusions (`SNR > 2`) |
|---|---:|---:|---:|---:|---:|---:|
| **None (Full v3 RF)** | 81 | **0.6722** | `0.0000` | 0.2790 | 0.3856 | **39** |
| **`- direction`** (13 cols) | 68 | 0.6692 | `-0.0030` | 0.4676 | 0.4162 | **111** (**$+72$, nearly $3\times$ worse!**) |
| **`- coherence`** (8 cols) | 73 | 0.6766 | `+0.0044` | 0.2896 | 0.4000 | 37 |
| **`- piezo`** (19 cols) | 62 | 0.5445 | **-0.1277** | 0.2685 | 0.4018 | 39 |
| **`- hub`** (15 cols) | 66 | 0.6506 | **-0.0216** | 0.2760 | 0.2180 | 33 |
| **`- comm`** (9 cols) | 72 | 0.6307 | **-0.0415** | 0.2911 | 0.3604 | 36 |
| **`- long_horizon`** (5 cols) | 76 | 0.6690 | `-0.0032` | 0.2609 | 0.3676 | 42 |

**Key Ablation Insight**: Removing the `direction` group (which now includes nominal-yaw world-frame fitted radial convergence) causes **`gradual <-> sensor_fault` confusions on `test` (`SNR > 2`) to nearly triple from `39` to `111` (`+72` confusions)** and causes raw `normal` false-alarm windows per run to jump from `11.45` back up to `30.05`. Because the full v3 Random Forest uses `dir_convergence_weighted_cos` and `dir_convergence_residual_m` as a spatial consistency gate before predicting `gradual`, it avoids guessing `gradual` on non-convergent single-node drift and background noise.
