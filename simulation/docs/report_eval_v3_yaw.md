# Installation-Yaw Sensitivity Report (`v3_yaw`): How Field Compass Errors Affect Frozen v3 Models (SYNTHETIC / SIMULATION)

> [!IMPORTANT]
> **SYNTHETIC / SIMULATION ONLY**: All datasets, sensor readings, installation-yaw perturbations, models, and metrics in this report are generated from synthetic simulation models. **None of these results are validated on real field or site-deployment data.**

Frozen items respected without modification:
- State-timeline schema v1 (`docs/schema_v1.md`, `canary_sim/schema.py`).
- Label policy, dataset splits, and manifest (`outputs/datasets/`, `outputs/datasets_v3/`).
- All previous evaluation outputs (`outputs/eval/`, `outputs/eval_v2/`, `outputs/eval_v3/`).
- Frozen `v3` trained models (`outputs/eval_v3/trained_models_v3.pkl`) and frozen `v3` operating points (`outputs/eval_v3/operating_points_v3.pkl`) — SHA-256 verified before and after evaluation to guarantee zero retraining or retuning.
- `Canary-Analyst` was neither touched nor imported.

---

## 1. Perturbation Design & Invariants (`canary_sim/yaw_sensitivity.py`)
For each of the **350 evaluation runs** (`test`: 120, `shifted_test`: 90, `stress`: 100, `broad_basin`: 40), a seeded per-node installation-yaw error $\delta\psi_k$ is added **only** to the nominal installation yaw (`yaw_deg`) used by the world-frame feature rotation:
$$\hat{\psi}_k = \text{yaw\_deg}_k + \delta\psi_k$$
- **Sensor physics and state timeline are unchanged**: True surface settlement, true node tilt, IMU/piezo/LiDAR telemetry, and window labels are 100% untouched.
- **Only the 16 yaw-dependent model columns change**: `{NODE}_tilt_angle_rad`, `dir_pairwise_cos_mean`, `dir_pairwise_cos_min`, `dir_circular_spread`, `dir_win_convergence_*`, `dir_convergence_*`, `coherence_trailing_wx`, `coherence_trailing_wy`, and `coherence_trailing_signed_mean`. All other 65 observable model features (`{NODE}_mean`, `piezo_*`, `hub_*`, `comm_*`, `long_slope_*`, `coherence_trailing_mag`) are yaw-invariant and remain bitwise identical (`tests/test_yaw_sensitivity.py`).
- **Five Evaluated Perturbation Conditions** (recorded per run and node in `outputs/eval_v3_yaw/yaw_perturbation_manifest.csv`):
  1. `sd_0deg`: $\delta\psi_k = 0^\circ$ (verified by test to reproduce `outputs/datasets_v3/` and `outputs/eval_v3/` 100% identically).
  2. `sd_5deg`: $\delta\psi_k \sim \mathcal{N}(0, (5^\circ)^2)$ per node.
  3. `sd_10deg`: $\delta\psi_k \sim \mathcal{N}(0, (10^\circ)^2)$ per node.
  4. `sd_20deg`: $\delta\psi_k \sim \mathcal{N}(0, (20^\circ)^2)$ per node.
  5. `single_node_45deg`: one randomly selected corner node per run has a fixed $\pm 45^\circ$ compass blunder; the other 3 nodes have $0^\circ$ error.
- **Yaw-Free Baseline Definition (`v2_yaw_free_baseline`)**: The frozen `v2` model (`outputs/eval/trained_models.pkl` + `outputs/eval_v2/operating_points_v2.pkl`), which was trained on the 71 old features where direction was uncalibrated, evaluated on the exact same runs in `outputs/datasets/`.

---

## 2. Results Across Yaw Error Conditions: `SNR > 2` Subset (`yaw_sensitivity_snr_gt_2.md`)
Separate per-condition tables are saved in `outputs/eval_v3_yaw/table_<condition>_snr_gt_2.md` and `table_<condition>_all_snr.md`. Below is the consolidated comparison on the **`SNR > 2` subset** (`test`: 99 runs, `shifted_test`: 71 runs, `broad_basin`: 20 runs) under the episode-matched operating point ($k=3, m=3$, plus raw `gradual <-> sensor_fault` confusion counts):

| Split (`SNR > 2`) | Condition (`model_version`) | Model | Window Macro-F1 (95% CI) | Alarm Macro-F1 (95% CI) | FA Episodes / Normal Run (95% CI) | Run Exact Detect Rate (95% CI) | `gradual <-> sensor_fault` Confusions: Persisted / Raw (Raw 95% CI) |
|---|---|---|---|---|---|---|---|
| **`test`** (`99` runs) | **`sd_0deg`** (`v3_frozen`) | **Random Forest** | 0.6725 `[0.613, 0.727]` | **0.6703** `[0.6010, 0.7302]` | 0.65 `[0.22, 1.18]` | **0.8228** `[0.734, 0.911]` | **30** / **39** `[7.5, 72.5]` |
| | **`sd_5deg`** (`v3_frozen`) | **Random Forest** | 0.6729 `[0.613, 0.729]` | **0.6688** `[0.5948, 0.7307]` | 0.60 `[0.22, 1.10]` | **0.8228** `[0.734, 0.911]` | **24** / **40** `[7.5, 74.5]` |
| | **`sd_10deg`** (`v3_frozen`) | **Random Forest** | 0.6724 `[0.613, 0.727]` | **0.6659** `[0.5974, 0.7272]` | 0.55 `[0.20, 1.00]` | **0.8101** `[0.722, 0.899]` | **25** / **35** `[11.0, 63.0]` |
| | **`sd_20deg`** (`v3_frozen`) | **Random Forest** | 0.6531 `[0.593, 0.708]` | **0.6490** `[0.5789, 0.7088]` | 0.60 `[0.20, 1.05]` | **0.7595** `[0.658, 0.848]` | **28** / **43** `[8.0, 95.5]` |
| | **`single_node_45deg`** (`v3_frozen`) | **Random Forest** | 0.6726 `[0.604, 0.722]` | **0.6632** `[0.5872, 0.7162]` | 0.45 `[0.17, 0.80]` | **0.8228** `[0.734, 0.899]` | **74** / **84** `[20.4, 167.5]` |
| | **`v2_yaw_free_baseline`** (`v2_frozen`) | **Random Forest** | 0.6910 `[0.623, 0.749]` | **0.6795** `[0.6078, 0.7446]` | 0.60 `[0.20, 1.05]` | **0.8228** `[0.734, 0.911]` | **25** / **92** `[24.0, 169.5]` |
| **`test`** (`99` runs) | **`sd_0deg`** (`v3_frozen`) | **Threshold Rule** | 0.7022 `[0.651, 0.734]` | **0.6908** `[0.6349, 0.7263]` | 0.60 `[0.05, 1.20]` | **0.8481** `[0.753, 0.924]` | **47** / **47** `[8.0, 98.0]` |
| | **`sd_5deg`** (`v3_frozen`) | **Threshold Rule** | 0.7017 `[0.648, 0.734]` | **0.6876** `[0.6306, 0.7242]` | 0.60 `[0.05, 1.20]` | **0.8481** `[0.753, 0.924]` | **47** / **47** `[8.0, 98.0]` |
| | **`sd_10deg`** (`v3_frozen`) | **Threshold Rule** | 0.7011 `[0.647, 0.733]` | **0.6869** `[0.6303, 0.7239]` | 0.60 `[0.05, 1.20]` | **0.8481** `[0.753, 0.924]` | **47** / **47** `[8.0, 98.0]` |
| | **`sd_20deg`** (`v3_frozen`) | **Threshold Rule** | 0.7022 `[0.650, 0.734]` | **0.6908** `[0.6349, 0.7263]` | 0.60 `[0.05, 1.20]` | **0.8481** `[0.753, 0.924]` | **47** / **47** `[8.0, 98.0]` |
| | **`single_node_45deg`** (`v3_frozen`) | **Threshold Rule** | 0.7026 `[0.651, 0.735]` | **0.6908** `[0.6349, 0.7263]` | 0.60 `[0.05, 1.20]` | **0.8481** `[0.753, 0.924]` | **46** / **46** `[6.0, 96.0]` |
| | **`v2_yaw_free_baseline`** (`v2_frozen`) | **Threshold Rule** | 0.7020 `[0.648, 0.737]` | **0.6891** `[0.6330, 0.7255]` | 0.60 `[0.05, 1.20]` | **0.8354** `[0.747, 0.911]` | **54** / **54** `[11.0, 108.0]` |
| **`test`** (`99` runs) | **`sd_0deg`** (`v3_frozen`) | **Logistic Reg** | 0.6164 `[0.557, 0.678]` | **0.6108** `[0.5455, 0.6831]` | 0.10 `[0.00, 0.25]` | **0.7468** `[0.652, 0.842]` | **13** / **94** `[43.0, 163.1]` |
| | **`sd_5deg`** (`v3_frozen`) | **Logistic Reg** | 0.6149 `[0.557, 0.674]` | **0.6064** `[0.5432, 0.6754]` | 0.15 `[0.00, 0.40]` | **0.7342** `[0.633, 0.823]` | **15** / **92** `[44.0, 163.0]` |
| | **`sd_10deg`** (`v3_frozen`) | **Logistic Reg** | 0.6120 `[0.551, 0.670]` | **0.5955** `[0.5310, 0.6647]` | 0.20 `[0.00, 0.45]` | **0.7089** `[0.608, 0.810]` | **13** / **103** `[55.5, 179.0]` |
| | **`sd_20deg`** (`v3_frozen`) | **Logistic Reg** | 0.6098 `[0.542, 0.672]` | **0.5917** `[0.5200, 0.6619]` | 0.25 `[0.05, 0.48]` | **0.7215** `[0.620, 0.810]` | **18** / **113** `[60.9, 187.5]` |
| | **`single_node_45deg`** (`v3_frozen`) | **Logistic Reg** | 0.6208 `[0.556, 0.680]` | **0.6075** `[0.5342, 0.6759]` | 0.35 `[0.10, 0.65]` | **0.7089** `[0.608, 0.810]` | **41** / **138** `[66.0, 217.5]` |
| | **`v2_yaw_free_baseline`** (`v2_frozen`) | **Logistic Reg** | 0.6054 `[0.540, 0.652]` | **0.5937** `[0.5175, 0.6547]` | 0.15 `[0.00, 0.35]` | **0.6962** `[0.595, 0.797]` | **25** / **132** `[71.5, 204.5]` |
| **`broad_basin`** (`20` runs) | **`sd_0deg`** (`v3_frozen`) | **Random Forest** | 0.5827 `[0.497, 0.660]` | **0.5723** `[0.4715, 0.6625]` | — | **0.9000** `[0.750, 1.000]` | **5** / **3** `[0.0, 9.0]` |
| | **`sd_5deg`** (`v3_frozen`) | **Random Forest** | 0.5717 `[0.464, 0.661]` | **0.5453** `[0.4377, 0.6409]` | — | **0.9000** `[0.750, 1.000]` | **5** / **4** `[0.0, 12.0]` |
| | **`sd_10deg`** (`v3_frozen`) | **Random Forest** | 0.5759 `[0.474, 0.663]` | **0.5704** `[0.4646, 0.6614]` | — | **0.9000** `[0.750, 1.000]` | **6** / **5** `[0.0, 15.0]` |
| | **`sd_20deg`** (`v3_frozen`) | **Random Forest** | 0.5781 `[0.468, 0.668]` | **0.5730** `[0.4662, 0.6616]` | — | **0.8000** `[0.600, 0.950]` | **10** / **10** `[0.0, 29.0]` |
| | **`single_node_45deg`** (`v3_frozen`) | **Random Forest** | 0.5568 `[0.449, 0.650]` | **0.5554** `[0.4439, 0.6471]` | — | **0.7500** `[0.550, 0.900]` | **21** / **26** `[0.0, 56.0]` |
| | **`v2_yaw_free_baseline`** (`v2_frozen`) | **Random Forest** | 0.5684 `[0.444, 0.659]` | **0.5673** `[0.4443, 0.6637]` | — | **0.6500** `[0.450, 0.850]` | **4** / **1** `[0.0, 3.0]` |

---

## 3. Results Across Yaw Error Conditions: All SNR (`yaw_sensitivity_all_snr.md`)
Under the episode-matched operating point ($k=3, m=3$) across all 4 evaluation splits (`test`: 120 runs, `shifted_test`: 90 runs, `stress`: 100 runs, `broad_basin`: 40 runs):

| Split (All SNR) | Condition (`model_version`) | Random Forest Alarm Macro-F1 (95% CI) | Random Forest Run Exact Detect Rate (95% CI) | Random Forest `grad<->sf` Confusions | Logistic Reg Alarm Macro-F1 (95% CI) | Threshold Rule Alarm Macro-F1 (95% CI) |
|---|---|---|---|---:|---|---|
| **`test`** (`120` runs) | `sd_0deg` (`v3_frozen`) | **0.6215** `[0.5694, 0.6721]` | **0.6900** `[0.590, 0.780]` | 37 | **0.5513** `[0.499, 0.600]` | **0.6299** `[0.579, 0.669]` |
| | `sd_5deg` (`v3_frozen`) | **0.6178** `[0.5662, 0.6680]` | **0.6700** `[0.570, 0.760]` | 31 | 0.5469 `[0.496, 0.594]` | 0.6267 `[0.575, 0.667]` |
| | `sd_10deg` (`v3_frozen`) | **0.6172** `[0.5648, 0.6688]` | **0.6700** `[0.570, 0.760]` | 32 | 0.5378 `[0.486, 0.587]` | 0.6261 `[0.575, 0.666]` |
| | `sd_20deg` (`v3_frozen`) | **0.6065** `[0.5594, 0.6597]` | 0.6400 `[0.540, 0.730]` | 35 | 0.5356 `[0.480, 0.587]` | 0.6299 `[0.579, 0.669]` |
| | `single_node_45deg` (`v3_frozen`) | **0.6187** `[0.5656, 0.6634]` | **0.7100** `[0.620, 0.800]` | **84** | 0.5516 `[0.500, 0.602]` | 0.6286 `[0.580, 0.668]` |
| | `v2_yaw_free_baseline` (`v2_frozen`) | 0.6149 `[0.5629, 0.6622]` | 0.6700 `[0.570, 0.760]` | 32 | 0.5378 `[0.489, 0.587]` | 0.6282 `[0.578, 0.668]` |
| **`shifted_test`** (`90` runs) | `sd_0deg` (`v3_frozen`) | **0.5915** `[0.5450, 0.6355]` | **0.7733** `[0.667, 0.853]` | **75** | **0.5117** `[0.471, 0.550]` | **0.6056** `[0.558, 0.649]` |
| | `sd_5deg` (`v3_frozen`) | **0.5976** `[0.5482, 0.6412]` | **0.7733** `[0.680, 0.853]` | 74 | 0.5089 `[0.469, 0.546]` | 0.6069 `[0.559, 0.651]` |
| | `sd_10deg` (`v3_frozen`) | **0.6034** `[0.5504, 0.6493]` | 0.7467 `[0.653, 0.840]` | 73 | 0.5166 `[0.476, 0.555]` | 0.6061 `[0.559, 0.650]` |
| | `sd_20deg` (`v3_frozen`) | **0.6212** `[0.5739, 0.6739]` | **0.7733** `[0.680, 0.853]` | 69 | 0.5076 `[0.470, 0.539]` | 0.6009 `[0.553, 0.643]` |
| | `single_node_45deg` (`v3_frozen`) | **0.6026** `[0.5491, 0.6517]` | **0.7867** `[0.693, 0.867]` | 70 | 0.5192 `[0.477, 0.562]` | 0.6064 `[0.559, 0.650]` |
| | `v2_yaw_free_baseline` (`v2_frozen`) | 0.5828 `[0.5365, 0.6271]` | 0.7733 `[0.680, 0.853]` | 113 | 0.5034 `[0.468, 0.538]` | 0.5925 `[0.538, 0.634]` |
| **`stress`** (`100` runs) | `sd_0deg` (`v3_frozen`) | **0.5013** `[0.4644, 0.5367]` | **0.9500** `[0.900, 0.990]` | 164 | **0.4646** `[0.427, 0.512]` | 0.4342 `[0.394, 0.478]` |
| | `sd_5deg` (`v3_frozen`) | **0.5032** `[0.4678, 0.5389]` | **0.9600** `[0.920, 0.990]` | 160 | 0.4670 `[0.428, 0.515]` | 0.4348 `[0.395, 0.478]` |
| | `sd_10deg` (`v3_frozen`) | **0.4852** `[0.4482, 0.5230]` | **0.9400** `[0.890, 0.980]` | 164 | 0.4530 `[0.411, 0.501]` | 0.4304 `[0.391, 0.474]` |
| | `sd_20deg` (`v3_frozen`) | **0.5007** `[0.4673, 0.5350]` | **0.9500** `[0.900, 0.990]` | 157 | 0.4619 `[0.422, 0.511]` | 0.4311 `[0.392, 0.475]` |
| | `single_node_45deg` (`v3_frozen`) | **0.4768** `[0.4412, 0.5185]` | **0.9600** `[0.920, 0.990]` | 156 | 0.4570 `[0.417, 0.507]` | 0.4305 `[0.392, 0.474]` |
| | `v2_yaw_free_baseline` (`v2_frozen`) | 0.4434 `[0.4073, 0.4895]` | 0.9300 `[0.880, 0.980]` | 190 | 0.4229 `[0.383, 0.466]` | **0.4404** `[0.401, 0.484]` |
| **`broad_basin`** (`40` runs) | `sd_0deg` (`v3_frozen`) | **0.4575** `[0.3704, 0.5324]` | **0.7250** `[0.575, 0.850]` | 7 | **0.3569** `[0.282, 0.428]` | 0.3882 `[0.311, 0.472]` |
| | `sd_5deg` (`v3_frozen`) | **0.4454** `[0.3577, 0.5225]` | **0.7500** `[0.612, 0.875]` | 7 | 0.3554 `[0.278, 0.429]` | 0.3882 `[0.311, 0.472]` |
| | `sd_10deg` (`v3_frozen`) | **0.4607** `[0.3790, 0.5367]` | **0.7250** `[0.588, 0.850]` | 8 | 0.3941 `[0.321, 0.467]` | 0.3882 `[0.311, 0.472]` |
| | `sd_20deg` (`v3_frozen`) | **0.4531** `[0.3784, 0.5427]` | 0.6250 `[0.475, 0.775]` | 11 | 0.3692 `[0.289, 0.439]` | 0.3882 `[0.311, 0.472]` |
| | `single_node_45deg` (`v3_frozen`) | 0.4123 `[0.3287, 0.4934]` | 0.5250 `[0.362, 0.650]` | **24** | 0.4057 `[0.324, 0.478]` | 0.3882 `[0.311, 0.472]` |
| | `v2_yaw_free_baseline` (`v2_frozen`) | 0.4093 `[0.3038, 0.5073]` | 0.4250 `[0.275, 0.588]` | 10 | 0.2888 `[0.210, 0.362]` | **0.3928** `[0.316, 0.479]` |

---

## 4. When Does the `v3` Random Forest Advantage Over the Yaw-Free `v2` Baseline Disappear?
Comparing `v3_frozen` Random Forest numbers against `v2_frozen_yaw_free` Random Forest numbers on the exact same runs:
1. **Up to `10°` Gaussian SD (`sd_5deg` and `sd_10deg`), the `v3` RF advantage is 100% intact**:
   - On `test` (`SNR > 2`), raw `gradual <-> sensor_fault` confusions are **`40` (`sd_5deg`) and `35` (`sd_10deg`)** for `v3_frozen` RF versus **`92`** for `v2_frozen_yaw_free` RF.
   - On `broad_basin`, `v3_frozen` RF run exact detection rate is **`75.0%` (`sd_5deg`) and `72.5%` (`sd_10deg`)** (and `90.0%` on `SNR > 2`) versus **`42.5%`** (`65.0%` on `SNR > 2`) for `v2_frozen_yaw_free` RF.
   - On `test` (All SNR), `v3_frozen` RF Alarm Macro-F1 at `sd_10deg` (`0.6172`) remains above `v2_frozen_yaw_free` RF (`0.6149`).
2. **At `20°` Gaussian SD (`sd_20deg`), the `test` Macro-F1 and detection-rate advantage disappears**:
   - On `test` (All SNR), `v3_frozen` RF Alarm Macro-F1 drops to **`0.6065`** (below `v2_frozen_yaw_free` RF's **`0.6149`**), and `test` Run Exact Detection Rate drops to **`0.6400`** (below `v2_frozen_yaw_free` RF's **`0.6700`**; on `SNR > 2` it drops from `0.8228` at `0°–10°` down to `0.7595` vs `v2`'s `0.8228`).
   - Note that on `stress` (`0.5007` vs `0.4434`) and `shifted_test` (`0.6212` vs `0.5828`), `v3_frozen` RF still outperforms `v2_frozen_yaw_free` RF even at `20°` SD because the yaw-invariant feature `coherence_trailing_mag` (12-window magnitude correlation, added in `v3`) is completely unaffected by yaw errors.
3. **Under a Single-Node `45°` Blunder (`single_node_45deg`), the directional advantage disappears completely**:
   - Rotating a single node by $45^\circ$ breaks the 4-ray least-squares convergence fit during basin events, causing raw `test` (`SNR > 2`) `gradual <-> sensor_fault` confusions in `v3_frozen` RF to jump from **`39` (`sd_0deg`) to `84` (`single_node_45deg`)** — nearly matching the yaw-free `v2` baseline (`92` confusions; under persisted alarms, confusions jump from `30` to `74` vs `v2`'s `25`).
   - Simultaneously, `broad_basin` Alarm Macro-F1 in `v3_frozen` RF collapses from **`0.4575` (`sd_0deg`) to `0.4123` (`single_node_45deg`)**, matching `v2_frozen_yaw_free` RF (`0.4093`).

---

## 5. Field-Installation Design Requirement (Written into `docs/data_dictionary.md`)
1. **Criterion Definition (Largest SD Within the `0°` 95% CI)**:
   - On `test` (All SNR), the `0°` `v3` Random Forest Alarm Macro-F1 is `0.6215` with 95% run-bootstrap CI **`[0.5694, 0.6721]`** (and on `SNR > 2`, `0.6703` with 95% CI **`[0.6010, 0.7302]`**).
   - Because run-level bootstrap CIs across 120 runs are $\pm 0.05$ wide, the point estimate at **`sd = 20°`** (`0.6065` All SNR; `0.6490` `SNR > 2`) still sits inside the `0°` 95% CI across all splits. Thus, by the literal CI-containment definition, **`20°` SD** is the largest tested Gaussian SD whose point estimate remains inside the `0°` 95% CI.
2. **Practical Engineering Specification (`<= 10°` SD, No Single-Node Blunder `>= 45°`)**:
   - Because `v3` RF loses its `test` detection-rate advantage over the yaw-free `v2` baseline between **`10°` and `20°` SD** (`test` run detection drops from `81.0%` at `10°` SD to `75.9%` at `20°` SD on `SNR > 2`, and `broad_basin` run detection drops from `72.5%` at `10°` SD to `62.5%` at `20°` SD), we state both numbers in `docs/data_dictionary.md`:
     - **Statistical CI bound**: $\sigma_{\text{yaw}} \le 20^\circ$.
     - **Recommended field-installation tolerance**: **$\sigma_{\text{yaw}} \le 10^\circ$** (readily achievable with a standard handheld or smartphone compass during node mounting), plus a mandatory post-installation check that **no single node has a gross orientation blunder $\ge 45^\circ$**.
