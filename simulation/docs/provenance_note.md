# Provenance and Model Specification Note

> [!CAUTION]
> **SYNTHETIC SIMULATION DISCLAIMER:**
> All state timelines, sensor telemetry, fault injections, and evaluation splits in ProjectCanary-Sim are **entirely synthetic simulations**. This system is **NOT** a geomechanical solver (such as a finite-element continuum model or geotechnical settlement code like FLAC/PLAXIS). It has **NOT** been validated against real field ground truth (e.g. active mine highwalls, real tailings dams, or railway subsidence). It is an edge algorithmic benchmarking harness for distributed sensor networks under synthetic failure modes.

---

## 1. Ground Settlement Physics Model

### 1.1 Spatial Profile
Ground displacement $w(x, y, t)$ (vertical subsidence, in meters) at any coordinate $(x, y)$ is generated from an analytic Gaussian or elliptical subsidence bowl centered at $(x_c, y_c)$:

For the **elliptical basin**:
Let rotated coordinates relative to the center $(x_c, y_c)$ with strike/azimuth $\psi$ be:
$$x' = (x - x_c)\cos\psi + (y - y_c)\sin\psi$$
$$y' = -(x - x_c)\sin\psi + (y - y_c)\cos\psi$$

The spatial attenuation factor is:
$$G(x, y) = \exp\left( -\frac{1}{2}\left[ \left(\frac{x'}{s_x}\right)^2 + \left(\frac{y'}{s_y}\right)^2 \right] \right)$$

When $s_x = s_y = \sigma$, this simplifies to a circular Gaussian bowl:
$$G(x, y) = \exp\left( -\frac{(x - x_c)^2 + (y - y_c)^2}{2\sigma^2} \right)$$

- **Main sampling range:** $\max(s_x, s_y) \le 1.5 R_{\text{network}}$ ($1.5 \times 14.14\text{ m} \approx 21.2\text{ m}$) in `configs/scenario_default.yaml`.
- **Broad basin subset:** When `broad_basin_subset: true`, $\max(s_x, s_y) \le 2.5 R_{\text{network}}$.

### 1.2 Temporal Progression
The settlement depth over time is governed by:
$$w(x, y, t) = W_{\max} \cdot S(t) \cdot G(x, y)$$

where $S(t) \in [0, 1]$ represents the temporal development:
- **Gradual settlement:** Smooth S-curve (cubic Hermite or logistic-like sigmoid) developing over a duration $\Delta t_{\text{event}} \in [20, 60]\text{ s}$ starting at $t_{\text{on}}$.
- **Rapid settlement:** Steep linear or rapid sigmoidal onset developing over $\Delta t_{\text{event}} \in [3, 8]\text{ s}$.

### 1.3 Tilt from Spatial Gradient
The physical surface tilt vector $\mathbf{\theta} = (\theta_x, \theta_y)$ represents the gradient of vertical settlement:
$$\theta_x(x, y, t) = -\frac{\partial w(x, y, t)}{\partial x}, \quad \theta_y(x, y, t) = -\frac{\partial w(x, y, t)}{\partial y}$$
Values are converted from radians to degrees ($\times \frac{180}{\pi}$). Because the spatial profile $G(x, y)$ is analytic, the derivatives are evaluated directly without numerical finite-difference discretization noise.

---

## 2. Sensor and Telemetry Models

### 2.1 IMU Node Model (MPU6050)
Each corner node (`NODE_01` through `NODE_04`) measures two-axis tilt $(\theta_x, \theta_y)$:
$$\tilde{\theta}_i(t) = \theta_i(t) + b_{\text{fixed}, i} + b_{\text{drift}, i}(t) + \epsilon_{\text{white}, i}(t)$$

- **Fixed Bias ($b_{\text{fixed}}$):** Drawn per node per run from $\mathcal{N}(0, \sigma_{\text{bias}}^2)$.
- **Slow Drift ($b_{\text{drift}}(t)$):** Discrete-time Wiener process / random walk:
  $$b_{\text{drift}}(t + \Delta t) = b_{\text{drift}}(t) + \mathcal{N}(0, \sigma_{\text{drift}}^2 \Delta t)$$
- **White Noise ($\epsilon_{\text{white}}$):** Derived from physical bench recordings of an MPU6050 sensor (`configs/noise_profile.json`). To account for continuous-time integration with lag-1 autocorrelation $\rho_1$:
  $$\sigma_{\text{white}} = \frac{\sigma_{\text{step}}}{\sqrt{2(1 - \rho_1)}}$$
  If $\rho_1$ is absent, falls back to $\sigma_{\text{step}} / \sqrt{2}$.
- **Noise Multiplier:** Sampled uniformly in $[0.5, 2.0]\times$ nominal per run.
- **NODE_04 Handling:** The empirical recording in `noise_profile.json` contained bench recordings for three physical units (`NODE_01`..`NODE_03`). `NODE_04` noise parameters are proxied using `NODE_02` parameters, as documented in `docs/report_sensors_faults.md`.

### 2.2 Piezoelectric Vibration Sensor (ADS1115 ADC)
Mounted at each corner node to measure high-frequency ground vibrations. Outputs three aggregate metrics per telemetry window:
- `piezo_peak`: Maximum peak amplitude.
- `piezo_rms`: Root-mean-square vibration energy.
- `piezo_peak_to_peak`: Envelope span.
Modelled as a baseline Gaussian electronic noise floor plus exponentially decaying impulse wavepackets triggered during seismic or mechanical excitation.

### 2.3 LiDAR Hub Model
A central scanning distance sensor at $(10, 10)$:
- Sweeps clockwise and anticlockwise across the 4 corner nodes with a motor dwell time at each node.
- Yields a range update once per complete sweep ($\approx 4\text{ s}$ period), not every 1 s telemetry sample.
- Output range:
  $$\tilde{r}_i = \|\mathbf{x}_{\text{hub}} - \mathbf{x}_{\text{node}, i}(t)\|_2 + \epsilon_{\text{lidar}} + q_{\text{step}}$$
  where $\epsilon_{\text{lidar}} \sim \mathcal{N}(0, \sigma_{\text{range}}^2)$ and $q_{\text{step}}$ is quantization truncation.
- Noise and quantization parameters are marked as placeholders in `configs/sensor_default.yaml`.

---

## 3. Direction and Convergence Features (v3 Feature Set)

### 3.1 World-Frame Rotation & Nominal Installation Yaw
IMUs measure tilt in their local body frame $(x_{\text{local}}, y_{\text{local}})$. To aggregate multi-node direction vectors:
$$\begin{pmatrix} \theta_{\text{world}, x} \\ \theta_{\text{world}, y} \end{pmatrix} = \begin{pmatrix} \cos\psi_{\text{nom}} & -\sin\psi_{\text{nom}} \\ \sin\psi_{\text{nom}} & \cos\psi_{\text{nom}} \end{pmatrix} \begin{pmatrix} \theta_{\text{local}, x} \\ \theta_{\text{local}, y} \end{pmatrix}$$
where $\psi_{\text{nom}}$ is the **nominal installation yaw** configured per node.

> [!WARNING]
> **Yaw Sensitivity Requirement:**
> The feature extraction layer assumes nominal installation yaw is known. Under experimental yaw perturbation (`docs/report_eval_v3_yaw.md`):
> - Gaussian yaw errors with $\text{sd} \le 5^\circ$: negligible impact on macro-F1 ($<0.01$ drop).
> - $\text{sd} = 10^\circ$: moderate degradation ($\approx 0.04$ macro-F1 drop).
> - $\text{sd} = 20^\circ$ or $45^\circ$ single-node error: severe degradation ($\ge 0.12$ drop) due to corrupted radial convergence and false confusion with sensor faults.
> Field deployment requires mechanical alignment fixtures ($\le 5^\circ$) or solar/compass in-situ calibration.

### 3.2 Radial Convergence Point Fitting
In v3, the subsidence convergence point is **not** assumed to be at a known coordinate. Instead:
- For each window with $\text{SNR} > 2$, the 4 world-frame tilt vectors define 4 lines in the $(x, y)$ plane.
- An unconstrained least-squares intersection point $(\hat{x}_c, \hat{y}_c)$ is fitted.
- `radial_alignment_deg`: Mean angular difference between the observed tilt vectors and vectors pointing toward $(\hat{x}_c, \hat{y}_c)$.
- `radial_fit_residual_deg`: Root-mean-square residual of the fitted lines.
- `trailing_coherence_mean`: Cosine similarity of tilt direction vectors across a trailing 3-window history of the same run (strictly causal; no future leakage).

---

## 4. Summary of Model Limitations

1. **Not a Continuum Solver:** Displacements are kinematic geometric profiles, not solutions to stress-strain partial differential equations with constitutive soil/rock models.
2. **Homogeneous Idealization:** Does not model geological layering, soil moisture effects, groundwater pore pressure, or tectonic fault lines.
3. **Synthetic Fault Distributions:** Hardware failure rates and comm packet loss patterns are synthetically configured rather than empirically mined from multi-year field telemetry logs.
4. **Benchmarking Purpose:** Results represent upper bounds under controlled synthetic conditions; field performance will depend heavily on physical installation tolerances and site-specific geomorphology.

---

## 5. Parameter ranges and literature basis

All values without a cited source are marked **PLACEHOLDER**.

| Parameter / Topic | Simulator Setting & Status | Literature Basis & Cited Sources | Assessment / Scope Note |
|---|---|---|---|
| **Peak tilt** | **PLACEHOLDER:** `0.05` to `3` deg (about `1` to `52` mm/m). | Ashton Underground longwall end-of-panel reports (Yancoal, SCT Operations, `2008-2012`): measured maximum tilts of about `13` to `103` mm/m (about `0.7` to `5.9` deg; `1` mm/m = `0.0573` deg), with some predictions higher. | The main range covers the lower half of real longwall maximum tilts; treat it as an early-warning range. |
| **Max settlement** | Derived from peak tilt and spread (**PLACEHOLDER** range). | Ashton reports (Yancoal, SCT Operations, `2008-2012`): maximum subsidence of `51` to `59%` of seam thickness measured; Ashton and a NSW appendix: `62` to `68%` predicted; one US case reports a subsidence factor of `0.75`. | Subsidence factor relates maximum vertical settlement to extracted seam thickness. |
| **Time profile** | Smoothstep S-curve (**PLACEHOLDER** simplification). | Jeran and Trevits, US Bureau of Mines RI `9552`: most movement within about `30` days of undermining in one study, total duration about a year in the northern Appalachian basin. Colorado DRMS rule document: about `15%` of $S_{\max}$ at the face, `5` to `9%` residual afterwards. *Sustainability* `2020`, `12(4)`, `1528`: active period defined as above `1.7` mm/day. Standard form is the Knothe time function with coefficient $c$ (Zhang et al., *Scientific Reports* `12:18433`, `2022`; Hejmanowski `2015`, *Int J Coal Sci Technol*). | The smoothstep S-curve is a simplification. |
| **Spread** | Gaussian spread $\sigma$ (**PLACEHOLDER** cap in `configs/scenario_default.yaml`). | Knothe influence function $\exp(-\pi r^2 / R^2)$ with $R = H / \tan(\beta)$ is a Gaussian with $\sigma = H / (\tan(\beta)\sqrt{2\pi})$. Angle of draw `12` deg at one Ashton panel (Ashton Underground longwall reports, `2008-2012`). | $\tan(\beta)$ is not set here; it must be taken from a cited source. |
| **Rapid events (collapse type)** | **PLACEHOLDER** | No source found; **PLACEHOLDER**. | Uncalibrated synthetic collapse regime (**PLACEHOLDER**). |
| **Vibration** | **PLACEHOLDER** relative scaling in piezo model. | Surface geophone peak particle velocities of about `0.2` to `30` mm/s near blasts (OSMRE `2007` acoustic response report); example site limit of `13` mm/s (Agnico Meadowbank `2019` blast monitoring report). | Relative scaling only; the piezo model is not calibrated to these. |
| **Not sourced** | **PLACEHOLDER** | Unsourced items and literature leads: $\tan(\beta)$ and Knothe $c$ for Indian coalfields (leads: Singh and Yadav `1995`, Raniganj; *Sadhana* `2021` Adriyala longwall paper), any rapid-event rate, any downloadable raw vibration dataset. | Must be sourced before claiming site-specific physical calibration. |

### Timescale
In the frozen simulator configuration, a gradual or normal run is `240` s of sensor time standing for `24` h of the slow ground process (**PLACEHOLDER** compression factor `360`), whereas cited gradual longwall subsidence develops over weeks to months (most movement within about `30` days of undermining in Jeran and Trevits, USBM RI `9552`). Using a rough real-world order-of-magnitude reference (**DERIVED, ORDER OF MAGNITUDE**, not a cited value): about `60` mm/m (`3.4` deg) developing over about `30` days gives an average tilt rate of about `0.1` deg/day, whereas compressing the full rise into a fraction of a single `24` h (`1`-day) run produces much higher implied deg/day rates (audited across all frozen gradual runs in `docs/report_literature_checks.md`). Replacing the `360`x compression label ("`24` h per run") with a `30`-day compression factor (for example `10,000`x to `30,000`x) changes **only** the labelled physical duration and the Definition A (slow-process time) false-alarm-per-hour figure; it does **not** change the simulated sensor data, window features, or any model score.

