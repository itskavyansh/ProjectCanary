# Evaluation Extras v2 (SYNTHETIC / SIMULATION)

> [!CAUTION]
> **SYNTHETIC / SIMULATION.** Every number below comes from simulated data (analytic settlement field + modelled IMU,
> piezo, LiDAR and network). The settlement engine is not a geomechanical solver. **Nothing here is validated on real data,
> and nothing here shows field readiness.** No model, feature, threshold, operating point, seed or dataset was changed or
> refit; the frozen v3 models are only used to predict. `outputs/eval_extras/` is untouched (byte-identical, hash-checked
> before and after by the runner and by a test).

Tables (CSV + Markdown) are in `outputs/eval_extras_v2/`; code in `canary_sim/eval_extras_v2.py`, `scripts/run_eval_extras_v2.py`.

## 1. False alarms per hour: two definitions

**Which process runs on which timescale.** Only the settlement field is time-compressed (compression factor 360 in the
normal, gradual, vibration and fault runs; 1 in rapid runs): 240 s of simulation stands for 24 h of ground movement.
IMU noise, piezo vibration, LiDAR sweeps, packet loss and the 5 s windows are **not** compressed: they run in real sensor
time, so a 240 s run contains 240 s of sensor noise and 95 windows, no more.

| | divisor per normal run | label |
|---|---|---|
| Definition A (as in `eval_extras`) | 240 s x 360 = 86 400 s = **24 h** | slow-process time (ground-movement timescale) |
| Definition B (new) | 240 s = **0.0667 h** | sensor time (noise, vibration and window timescale) |

B is 360x larger than A. **B is the stricter and more defensible figure for false alarms caused by sensor noise and benign
vibration**, because those processes are not compressed; A only makes sense for alarms driven by the slow ground process.
**Neither is a validated field rate.** All normal runs of the listed splits share one compression factor (360), so factors
are not mixed (the code raises if a split ever mixes them). Point estimates and the per-run column are identical to
`eval_extras` (checked, max difference 0.0); the A-column CI endpoints differ slightly, see section 6.

Episodes after persistence, 95% run-bootstrap CIs (20 normal runs in `test`, 15 in `shifted_test`); "every" = one false-alarm
episode every N hours (or minutes) under that definition; missed-event = event runs with no correct alarm.
Full table with raw regime, `val`/`train` and false-alarm windows per hour: `false_alarms_per_hour_both_definitions.md/.csv`.

| split | model | regime | FA eps / run | A: FA/h (slow) | A: every | B: FA/h (sensor) | B: every | missed-event |
|---|---|---|---|---|---|---|---|---|
| test | threshold | episode k3m3 | 0.600 [0.050, 1.326] | 0.0250 [0.0021, 0.0553] | 40.0 h | 9.00 [0.75, 19.89] | 6.7 min | 0.300 [0.210, 0.390] |
| test | threshold | strict | 0.050 [0.000, 0.150] | 0.0021 [0.0000, 0.0063] | 480 h | 0.75 [0.00, 2.25] | 80 min | 0.350 [0.265, 0.445] |
| test | logistic | episode k3m3 | 0.100 [0.000, 0.200] | 0.0042 [0.0000, 0.0083] | 240 h | 1.50 [0.00, 3.00] | 40 min | 0.390 [0.290, 0.480] |
| test | logistic | strict | 0.000 [0.000, 0.000] | 0.0000 | none observed | 0.00 | none observed | 0.450 [0.360, 0.545] |
| test | random forest | episode k3m3 | 0.650 [0.200, 1.176] | 0.0271 [0.0083, 0.0490] | 36.9 h | 9.75 [3.00, 17.64] | 6.2 min | 0.310 [0.230, 0.410] |
| test | random forest | strict | 0.250 [0.050, 0.500] | 0.0104 [0.0021, 0.0208] | 96.0 h | 3.75 [0.75, 7.50] | 16.0 min | 0.360 [0.280, 0.460] |
| shifted_test | threshold | episode k3m3 | 2.467 [1.463, 3.502] | 0.1028 [0.0610, 0.1459] | 9.7 h | 37.00 [21.95, 52.52] | 1.6 min | 0.173 [0.093, 0.267] |
| shifted_test | threshold | strict | 0.200 [0.000, 0.400] | 0.0083 [0.0000, 0.0167] | 120 h | 3.00 [0.00, 6.00] | 20 min | 0.333 [0.240, 0.427] |
| shifted_test | logistic | episode k3m3 | 1.267 [0.667, 1.968] | 0.0528 [0.0278, 0.0820] | 18.9 h | 19.00 [10.00, 29.52] | 3.2 min | 0.307 [0.213, 0.413] |
| shifted_test | logistic | strict | 0.600 [0.200, 1.067] | 0.0250 [0.0083, 0.0444] | 40.0 h | 9.00 [3.00, 16.00] | 6.7 min | 0.467 [0.347, 0.573] |
| shifted_test | random forest | episode k3m3 | 1.400 [0.698, 2.333] | 0.0583 [0.0291, 0.0972] | 17.1 h | 21.00 [10.47, 35.00] | 2.9 min | 0.227 [0.147, 0.320] |
| shifted_test | random forest | strict | 0.200 [0.000, 0.400] | 0.0083 [0.0000, 0.0167] | 120 h | 3.00 [0.00, 6.00] | 20 min | 0.280 [0.187, 0.400] |

What this does and does not show:
* Under definition B the "at most about one false alarm per normal run" target that was used for tuning means roughly one
  false-alarm episode per 4 minutes of sensor time (best strict cases: one per 16-80 minutes). Under A the same counts look
  like one per day or several days. Only B reflects how often sensor noise and benign vibration trip the alarm logic.
* "None observed" (0 false alarms in 20 runs = 1.33 h of sensor time, 480 h under A) is a zero count, not a zero rate; by the
  rule of three the 95% upper bound is about 2.3 per sensor hour (B) or 0.006 per slow-process hour (A).
* Stricter operating points trade false alarms for missed events; confidence intervals are wide (15-20 normal runs per split).

## 2. Accuracy table fixes

`window_accuracy_and_f1_v2.md/.csv` (frozen v3 models; test, shifted_test, stress, broad_basin; raw and both operating points).
* Classes with zero true windows in a split show **N/A** (recall and precision undefined), e.g. vibration, sensor_fault and
  comm_fault on `broad_basin`. Macro-F1 averages only the classes present.
* Caption corrected. Share of scored windows that are `normal`: **test 63.6%, shifted_test 63.3%, stress 41.3%,
  broad_basin 31.8%** (`class_share_of_windows_per_split.csv`). `normal` is the largest class in test, shifted_test and
  stress but **not** in broad_basin (gradual 35.7%), so "normal dominates every split" would be wrong and is not claimed;
  the word "balanced" is removed. The class mix is set by the dataset design and is not the real-world mix.
* Every gradual/rapid row shows window recall and run-level detection side by side (last two columns).

## 3. Gradual and rapid: window recall and run-level detection, together

`gradual_rapid_recall_and_detection.md/.csv`. Window recall = share of true gradual (rapid) windows classified as that class
(before persistence). Run-level detection = share of runs with at least one alarm of that class after persistence.

**Plainly: gradual and rapid window recall at the operating points is low** (test, episode point: gradual 0.16 / 0.09 / 0.16
and rapid 0.27 / 0.21 / 0.28 for threshold / logistic / random forest; strict point lower still). **The run-level detection rate
counts any correct alarm anywhere in a run**, so it is much higher (test, episode point: gradual 0.45 / 0.30 / 0.40, rapid
0.45 / 0.35 / 0.40) and must never be quoted without the window recall next to it. In the stress set only an alarm of the run's
own class counts, which is stricter than the headline "exact detect" rate used elsewhere.

### Recall by `snr_vs_imu_noise` bin, v3 operating points (`gradual_rapid_recall_by_snr.md/.csv`)

Pooled over test, shifted_test, stress and broad_basin (per-split rows are in the CSV; many cells hold few runs):
window recall / run-level detection.

| model | regime | class | SNR < 1 | 1 - 2 | 2 - 5 | > 5 |
|---|---|---|---|---|---|---|
| threshold | episode | gradual | 0.12 / 0.33 | 0.10 / 0.26 | 0.20 / 0.54 | 0.34 / 0.92 |
| threshold | episode | rapid | 0.12 / 0.27 | 0.10 / 0.20 | 0.30 / 0.50 | 0.55 / 0.90 |
| random forest | episode | gradual | 0.10 / 0.41 | 0.22 / 0.47 | 0.15 / 0.46 | 0.28 / 0.77 |
| random forest | episode | rapid | 0.12 / 0.38 | 0.21 / 0.60 | 0.36 / 0.68 | 0.59 / 0.90 |
| random forest | strict | gradual | 0.02 / 0.07 | 0.11 / 0.32 | 0.07 / 0.38 | 0.24 / 0.58 |
| random forest | strict | rapid | 0.06 / 0.23 | 0.11 / 0.40 | 0.28 / 0.64 | 0.52 / 0.85 |
| logistic | episode | gradual | 0.06 / 0.22 | 0.07 / 0.21 | 0.08 / 0.15 | 0.18 / 0.46 |
| logistic | episode | rapid | 0.08 / 0.27 | 0.16 / 0.53 | 0.34 / 0.68 | 0.50 / 0.80 |

Run counts per bin (gradual / rapid): 27 / 26, 19 / 15, 13 / 22, 26 / 20.

**Reliable detection needs a tilt of about 5 times the IMU noise.** At the episode operating point, run-level detection only reaches roughly 0.8-0.9 in the
SNR > 5 bin (0.77-0.92 for the threshold rule and random forest; logistic regression stays at 0.46 for gradual; the strict
point is lower, e.g. 0.58 for random forest gradual), and it is 0.2-0.6 below SNR 2. Even above SNR 5 gradual window recall stays at 0.18-0.34, so most individual windows of a gradual event are still
missed; detection comes from a few correct windows late in a run.

## 4. Feature importance wording (from `eval_extras`, VAL only, unchanged)

Piezo and comm lead the permutation ranking (macro-F1 drop about 0.18 each) **because abnormal_vibration and comm_fault
are close to separable by construction**: those classes were generated with signatures (impulse bursts across several
nodes; packet loss, bursts, latency) that no other class has, and the models separate them almost perfectly on the single-condition test
sets (vibration precision/recall about 1.0, comm_fault about 0.9). In the stress set, where conditions are combined,
comm_fault precision drops to about 0.28, so separability by construction does not carry over to mixed conditions. That ranking says the model uses them; it does not show they are hard
or valuable detection problems. **The discriminating finding is direction versus magnitude**: with nominal-yaw
world-frame rotation and a fitted convergence point, the direction group (drop 0.072) outranks magnitude (0.052), i.e.
multi-node radial convergence is what separates basin events from single-node faults. Coherence is last (0.003).

## 5. Tests and checks

`tests/test_eval_extras_v2.py`: per-hour conversion for both definitions against hand-computed values (24 h, 1/15 h,
40 h / 9 per h); mixed compression factors rejected; N/A for empty classes (toy case and the generated `broad_basin` rows);
no `.fit(` in the v2 sources and the frozen model file hash unchanged; same seed gives identical frames; hand-computed
recall-versus-detection example; SNR bin edges; `outputs/eval_extras` byte-identical to the snapshot hashes in
`outputs/eval_extras_v2/eval_extras_original_sha256.json` (the runner also asserts this before and after).

## 6. Changes versus `eval_extras`

* New: definition B and "one false alarm every N hours" under A and B (previous report gave only A).
* A-column CI endpoints are now the per-run CI divided by 24 h. The previous run drew a separate bootstrap for the
  per-hour column, so its endpoints differ slightly (e.g. test random forest episode: [0.0083, 0.0479] before,
  [0.0083, 0.0490] now); point estimates are identical.
* N/A instead of 0.00 for classes with no true windows; caption corrected (actual shares, "balanced" removed).
* Added window recall with run-level detection, and recall by SNR bin.
* Corrected an earlier statement: normal windows are 63.6% of scored test windows, not "over 65%".
* Report wording on gradual/rapid recall, run-level detection, the 5x SNR requirement and why piezo/comm lead the ranking.
* Not changed: models, operating points, features, thresholds, seeds, datasets, and every file in `outputs/eval_extras/`
  (including its figures and the old report `docs/report_eval_extras.md`, whose wording is superseded by this document).
