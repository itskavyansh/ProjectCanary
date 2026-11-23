# Advisory Alarm Explainer Evaluation & Negative-Control Patch Report (SYNTHETIC / SIMULATION)

> **SYNTHETIC / SIMULATION - NOT VALIDATED ON REAL DATA**
>
> **Scope:** Advisory explainer evaluation patch only (`muse-spark-1.3`). All classifiers, features, thresholds, operating points, `packets.json`, cached explanations, replay JSONs, and existing classifier benchmark results remain frozen.

---

## 1. Headline Metrics: First-Run vs. After Cache Fill ($N = 150$ Stratified Packets)

On the initial live run at `max_completion_tokens = 2048`, **148 / 150 (`98.67%`)** packets succeeded on the first try with **0 deterministic guard rejections** (`0.0%` unfaithful numbers, `0.0%` wrong class mentions, `0.0%` forbidden claims). The remaining **2 / 150 (`1.33%`)** packets (`run_test_005_w43` in `false_alarm` and `run_test_106_w29` in `fault_window`) hit the `2048` completion-token limit during internal reasoning tokens (`json_parse_error` on truncated output) and deterministically fell back to the template on the first run before being filled in the cache on the second pass.

**Statistical Note on Zero-Width Bootstrap Intervals & Rule of Three:** Whenever an observed sample proportion is $0/n$ (`0.0000`) or $n/n$ (`1.0000`), nonparametric resampling produces a zero-width bootstrap interval (`[0.0000, 0.0000]` or `[1.0000, 1.0000]`). Next to every zero-width bootstrap interval in this report and in the evaluation CSV files (`overall_metrics_ci95.csv`, `metrics_by_stratum.csv`, `negative_control_summary.csv`, `heldout_negative_control_summary.csv`), we report the **exact (Clopper-Pearson) 95% binomial confidence interval** (`Exact CP: [lo, hi]`). For zero observed counts ($k = 0$ in $n$ trials), we also state the **rule-of-three 95% upper bound** ($3/n$): for $N = 150$, $0/150$ has exact Clopper-Pearson 95% CI `[0.0000, 0.0243]` and rule-of-three upper bound $3/150 = 0.0200$ (`2.00%`); for $N = 42$, $0/42$ has exact 95% CI `[0.0000, 0.0841]` and rule-of-three upper bound $3/42 = 0.0714$ (`7.14%`); for $N = 36$, $0/36$ has exact 95% CI `[0.0000, 0.0974]` and rule-of-three upper bound $3/36 = 0.0833$ (`8.33%`); for $N = 6$, $0/6$ has exact 95% CI `[0.0000, 0.4593]` and rule-of-three upper bound $3/6 = 0.5000$ (`50.0%`).

In accordance with Requirement 2, **first-run numbers are reported as the primary headline estimate**, and post-cache numbers are reported in the separate **After Cache Fill** column (`outputs/eval_explainer/overall_metrics_ci95.csv`):

| Metric | N | First-Run Headline Estimate [95% Bootstrap CI] (Exact CP / Rule-of-3) | After Cache Fill Estimate [95% Bootstrap CI] (Exact CP / Rule-of-3) | Notes |
|---|---:|---|---|---|
| `first_try_guard_pass_rate` | 150 | **0.9867** [0.9667, 1.0000] | `1.0000` [1.0000, 1.0000] (Exact CP: [0.9757, 1.0000]) | 148/150 first-try pass on first run (2 API token cutoffs at 2048 limit, 0 guard rejections); 150/150 after cache fill |
| `api_cutoff_rate` | 150 | **0.0133** [0.0000, 0.0333] | `0.0000` [0.0000, 0.0000] (Exact CP: [0.0000, 0.0243]; rule-of-3 <= 0.0200) | 2/150 hit 2048 completion_tokens limit on first run (run_test_005_w43, run_test_106_w29); 0/150 after cache fill & 4096 token limit fix |
| `guard_rejection_rate_pre_guard` | 150 | **0.0000** [0.0000, 0.0000] (Exact CP: [0.0000, 0.0243]; rule-of-3 <= 0.0200) | `0.0000` [0.0000, 0.0000] (Exact CP: [0.0000, 0.0243]; rule-of-3 <= 0.0200) | 0/150 guard rejections on complete JSON responses across both runs |
| `retry_rate` | 150 | **0.0000** [0.0000, 0.0000] (Exact CP: [0.0000, 0.0243]; rule-of-3 <= 0.0200) | `0.0000` [0.0000, 0.0000] (Exact CP: [0.0000, 0.0243]; rule-of-3 <= 0.0200) | 0/150 guard-triggered retries (API cutoffs fell back directly) |
| `fallback_rate` | 150 | **0.0133** [0.0000, 0.0333] | `0.0000` [0.0000, 0.0000] (Exact CP: [0.0000, 0.0243]; rule-of-3 <= 0.0200) | 2/150 template fallbacks on first run due to 2048 token cutoff; 0/150 after cache fill |
| `unfaithful_number_rate_pre_guard` | 150 | **0.0000** [0.0000, 0.0000] (Exact CP: [0.0000, 0.0243]; rule-of-3 <= 0.0200) | `0.0000` [0.0000, 0.0000] (Exact CP: [0.0000, 0.0243]; rule-of-3 <= 0.0200) | 0/150 unfaithful numbers before guard |
| `unfaithful_number_rate_post_guard` | 150 | **0.0000** [0.0000, 0.0000] (Exact CP: [0.0000, 0.0243]; rule-of-3 <= 0.0200) | `0.0000` [0.0000, 0.0000] (Exact CP: [0.0000, 0.0243]; rule-of-3 <= 0.0200) | 0/150 unfaithful numbers after guard |
| `class_contradiction_rate_pre_guard` | 150 | **0.0000** [0.0000, 0.0000] (Exact CP: [0.0000, 0.0243]; rule-of-3 <= 0.0200) | `0.0000` [0.0000, 0.0000] (Exact CP: [0.0000, 0.0243]; rule-of-3 <= 0.0200) | 0/150 class contradictions before guard |
| `class_contradiction_rate_post_guard` | 150 | **0.0000** [0.0000, 0.0000] (Exact CP: [0.0000, 0.0243]; rule-of-3 <= 0.0200) | `0.0000` [0.0000, 0.0000] (Exact CP: [0.0000, 0.0243]; rule-of-3 <= 0.0200) | 0/150 class contradictions after guard |
| `forbidden_claim_rate_pre_guard` | 150 | **0.0000** [0.0000, 0.0000] (Exact CP: [0.0000, 0.0243]; rule-of-3 <= 0.0200) | `0.0000` [0.0000, 0.0000] (Exact CP: [0.0000, 0.0243]; rule-of-3 <= 0.0200) | 0/150 forbidden claims before guard |
| `forbidden_claim_rate_post_guard` | 150 | **0.0000** [0.0000, 0.0000] (Exact CP: [0.0000, 0.0243]; rule-of-3 <= 0.0200) | `0.0000` [0.0000, 0.0000] (Exact CP: [0.0000, 0.0243]; rule-of-3 <= 0.0200) | 0/150 forbidden claims after guard |
| `final_guard_pass_rate` | 150 | **1.0000** [1.0000, 1.0000] (Exact CP: [0.9757, 1.0000]) | `1.0000` [1.0000, 1.0000] (Exact CP: [0.9757, 1.0000]) | 150/150 final explanations pass deterministic guard |
| `mean_char_length` | 150 | **319.0133** [310.8445, 327.2405] | `319.0133` [310.6700, 326.9895] | Mean explanation character length (summary + suggested_check) |
| `mean_tokens_per_explanation` | 150 | **2431.8667** [2389.8125, 2478.7640] | `2431.8667` [2392.0177, 2474.0980] | Mean total tokens per explanation (including reasoning tokens) |
| `mean_cost_usd_per_explanation` | 150 | **0.0063** [0.0061, 0.0065] | `0.0063` [0.0061, 0.0065] | Mean API cost per explanation in USD |
| `mean_latency_s` | 150 | **8.6640** [8.3945, 8.9564] | `8.6640` [8.3853, 8.9808] | Mean API wall latency per explanation in seconds |

---

## 2. Pre-Registration Comparison (with H8 Split into Spend Cap vs. Point Estimate)

| ID | Prediction | Pre-Registered Target | Observed (First-Run Headline) | Confirmed? | Notes |
|---|---|---|---|---|---|
| **H1_first_try_guard_pass_rate** | First-Try Guard Pass Rate (Headline First-Run vs After Cache Fill) | `0.70 to 0.92` | `0.9867 (148/150 first-run; 1.0000 after cache fill)` | **No** | First-run headline pass rate was 0.9867 (148/150 first-try, 2 API token cutoffs at 2048 limit, 0 guard rejections; 1.0000 after cache fill), exceeding the upper bound of 0.92. |
| **H2_retry_rate** | Guard-Triggered Retry Rate | `0.08 to 0.30` | `0.0000` | **No** | Observed guard-triggered retry rate was 0.0000 (0/150 guard rejections; the 2 API token cutoffs fell back directly). |
| **H3_fallback_rate** | Template Fallback Rate (First-Run vs After Cache Fill) | `0.00 to 0.12` | `0.0133 (2/150 first-run; 0.0000 after cache fill)` | **Yes** | Confirmed within 0% to 12% range both on first run (0.0133, 2/150 from API token cutoffs) and after cache fill (0.0000, 0/150). |
| **H4_unfaithful_numbers_pre_and_post_guard** | Unfaithful Number Rate (Pre-Guard vs Post-Guard) | `Pre: 0.04 to 0.22; Post: 0.00` | `Pre: 0.0000; Post: 0.0000` | **No** | Post-guard unfaithful number rate is strictly 0.0000 (confirmed); pre-guard rate on valid responses was 0.0000 (below 0.04). |
| **H6_class_contradiction_pre_and_post_guard** | Class-Contradiction Rate (Pre-Guard vs Post-Guard) | `Pre: 0.02 to 0.18; Post: 0.00` | `Pre: 0.0000; Post: 0.0000` | **No** | Post-guard class contradiction rate is strictly 0.0000 (confirmed); pre-guard rate was 0.0000 (below 0.02). |
| **H7_classifier_mistakes_faithfulness_and_no_extra_confidence** | Classifier Mistakes Faithfulness & Zero Confidence Claims | `1.0000 reflects wrong predicted_class; 0.0000 forbidden confidence/prob claims` | `reflects_predicted_class=1.0000, forbidden_confidence=0.0000` | **Yes** | Confirmed: 100% of classifier-mistake explanations faithfully report the classifier's predicted class with hedged language and zero confidence claims. |
| **H8a_spend_cap_respected** | H8a. Hard Budget Cap Held ($5.00 Initial / $2.00 Patch) | `Initial <= $5.00 USD; Patch <= $2.00 USD` | `Initial: $1.1002 / $5.00 USD; Patch: $0.1830 / $1.00 USD` | **Yes** | Confirmed: initial evaluation spend ($1.1002 USD) stayed well under the $5.00 hard cap, and patch spend ($0.1830 USD) stayed well under the $1.00 patch spend cap. |
| **H8b_spend_point_estimate** | H8b. Pre-Registered Point Estimate (< $0.50 USD) | `< $0.50 USD` | `$1.1002 USD (initial eval); $1.6996 USD (cumulative with patch)` | **No** | Not confirmed: initial 150-packet + 40-judge spend was $1.1002 USD (above the <$0.50 point estimate) because muse-spark-1.3 billed ~1,100 internal reasoning tokens per call alongside ~120 visible JSON output tokens. |
| **H9_judge_agreement_rate** | Secondary Judge Agreement on Final Explanations (N=40) | `>= 0.90 rating >= 4 and >= 0.90 agreement with guard` | `rating>=4: 1.0000, agreement: 1.0000` | **Yes** | Confirmed on 40-packet random sample (subject to same-model judge limitation; see Section 4 for negative-control sensitivity). |

---

## 3. Completion Token Limit Fix (`2048 -> 4096`) & 20-Packet Uncached Spot Check

- **Configuration Change:** Raised `max_completion_tokens` from `2048` to `4096` in `simulation/canary_sim/explainer/explainer_config.json` and `ChatClient.complete` (`simulation/canary_sim/explainer/client.py`).
- **20-Packet Uncached Spot Check (Seed `20261124`):** Evaluated `20` packets fresh against the live API without reading or mutating the primary packet cache (explicitly including both previously cut-off packets `run_test_005_w43` and `run_test_106_w29` plus 18 stratified packets).
- **Spot-Check Results:**
  - **API Token Cutoffs at `max_completion_tokens = 4096`:** **`0 / 20` (`0.0%`, Exact CP 95% CI `[0.0000, 0.1684]`, rule-of-three upper bound `0.1500`)** — raising the completion token limit completely eliminated reasoning-token cutoffs.
  - **Valid JSON Responses:** `20 / 20` (`100.0%`, Exact CP 95% CI `[0.8316, 1.0000]`).
  - **First-Try Guard v2 Pass Rate on Spot Check:** `20 / 20` (`100.0%`, Exact CP 95% CI `[0.8316, 1.0000]`).
  - **Observed Completion Tokens (Reasoning + Output):** mean `1228.8` tokens, max `1919` tokens (well below the new `4096` ceiling).

---

## 4. Negative Control (Seed `20261124`): 42 Corrupted + 42 Untouched Explanations

Pre-registered expectations were committed alone in `docs/explainer_negative_control_expectations.md` (`commit 714a4c8`) before running any corruptions. We took 42 cached faithful explanations (7 per corruption type, 1 from each of the 7 evaluation strata) and programmatically created 6 corruption types alongside 42 untouched faithful explanations ($N = 84$ total).

> [!IMPORTANT]
> **Why Guard v2's `42/42` on the Original Corruptions Is Not Independent Evidence:**
> Guard v2's `42/42` (`100.0%`) catch rate on the original 6 corruption types (`a_number_perturbed_10pct` through `f_unsupported_trend_claim`) is **not independent out-of-sample evidence**, because Guard v2's label-value binding rules (`check_label_value_bindings`) and forbidden trend-word regexes (`forbidden_trend_regexes`) were written *after* inspecting Guard v1's failures on those exact corruption templates (`b_node_label_swap` and `f_unsupported_trend_claim`). Independent out-of-sample evaluation requires held-out corruption mechanisms not used to design Guard v2 (reported in Section 6 below).

Both the deterministic guard (**Guard v1** before patch and **Guard v2** after adding label-value binding and trend-word rules) and the **Secondary LLM Judge** (`muse-spark-1.3`, separate instruction, never given `truth`) were evaluated on all 84 explanations:

| Corruption / Control Type | N | Metric | Guard v1 (Pre-Patch) [95% Boot CI] (Exact CP / Rule-of-3) | Guard v2 (Patched, In-Sample) [95% Boot CI] (Exact CP / Rule-of-3) | Secondary Judge (`muse-spark-1.3`) [95% Boot CI] (Exact CP / Rule-of-3) | Pre-Registered Guard v1 Expectation |
|---|---:|---|---|---|---|---|
| `a_number_perturbed_10pct` | 7 | `catch_rate` | **7/7** (`1.0000` [1.0000, 1.0000] (Exact CP: [0.5904, 1.0000])) | **7/7** (`1.0000` [1.0000, 1.0000] (Exact CP: [0.5904, 1.0000])) | **7/7** (`1.0000` [1.0000, 1.0000] (Exact CP: [0.5904, 1.0000])) | CATCH (100% — bag-of-numbers mismatch) |
| `b_node_label_swap` | 7 | `catch_rate` | **0/7** (`0.0000` [0.0000, 0.0000] (Exact CP: [0.0000, 0.4096]; rule-of-3 <= 0.4286)) | **7/7** (`1.0000` [1.0000, 1.0000] (Exact CP: [0.5904, 1.0000])) | **7/7** (`1.0000` [1.0000, 1.0000] (Exact CP: [0.5904, 1.0000])) | MISS (0% — valid packet number on wrong node) |
| `c_wrong_class_named` | 7 | `catch_rate` | **7/7** (`1.0000` [1.0000, 1.0000] (Exact CP: [0.5904, 1.0000])) | **7/7** (`1.0000` [1.0000, 1.0000] (Exact CP: [0.5904, 1.0000])) | **7/7** (`1.0000` [1.0000, 1.0000] (Exact CP: [0.5904, 1.0000])) | CATCH (100% — wrong/missing class token) |
| `d_forbidden_claim_added` | 7 | `catch_rate` | **7/7** (`1.0000` [1.0000, 1.0000] (Exact CP: [0.5904, 1.0000])) | **7/7** (`1.0000` [1.0000, 1.0000] (Exact CP: [0.5904, 1.0000])) | **7/7** (`1.0000` [1.0000, 1.0000] (Exact CP: [0.5904, 1.0000])) | CATCH (100% — matches `\bthe\s+ground\s+is\b`) |
| `e_deleted_simulated_keyword` | 7 | `catch_rate` | **7/7** (`1.0000` [1.0000, 1.0000] (Exact CP: [0.5904, 1.0000])) | **7/7** (`1.0000` [1.0000, 1.0000] (Exact CP: [0.5904, 1.0000])) | **7/7** (`1.0000` [1.0000, 1.0000] (Exact CP: [0.5904, 1.0000])) | CATCH (100% — missing `simulated`/`synthetic`) |
| `f_unsupported_trend_claim` | 7 | `catch_rate` | **0/7** (`0.0000` [0.0000, 0.0000] (Exact CP: [0.0000, 0.4096]; rule-of-3 <= 0.4286)) | **7/7** (`1.0000` [1.0000, 1.0000] (Exact CP: [0.5904, 1.0000])) | **7/7** (`1.0000` [1.0000, 1.0000] (Exact CP: [0.5904, 1.0000])) | MISS (0% — verbal trend words had no rule) |
| `ALL_CORRUPTED_COMBINED` | 42 | `catch_rate` | **28/42** (`0.6667` [0.5000, 0.7863]) | **42/42** (`1.0000` [1.0000, 1.0000] (Exact CP: [0.9159, 1.0000])) | **42/42** (`1.0000` [1.0000, 1.0000] (Exact CP: [0.9159, 1.0000])) | 4/6 types caught (28/42 = 66.7%) |
| `UNTOUCHED_FALSE_POSITIVE_RATE` | 42 | `false_positive_rate` | **0/42** (`0.0000` [0.0000, 0.0000] (Exact CP: [0.0000, 0.0841]; rule-of-3 <= 0.0714)) | **0/42** (`0.0000` [0.0000, 0.0000] (Exact CP: [0.0000, 0.0841]; rule-of-3 <= 0.0714)) | **0/42** (`0.0000` [0.0000, 0.0000] (Exact CP: [0.0000, 0.0841]; rule-of-3 <= 0.0714)) | 0% false positive (0/42) |

---

## 5. Guard v2 Fixes & Offline Re-Evaluation Over All Cached Explanations

Without weakening any existing check in `simulation/canary_sim/explainer/guard.py`, we added two deterministic rules addressing the initial negative-control blind spots:
1. **Label-Value Binding (`check_label_value_bindings`):** Whenever a node identifier (`NODE_01`..`NODE_04`), node metric (`NODE_0X_missing_fraction`), scalar metric (`model_score`, `anomaly_score`, `hub_common_mode_m`, `radial_convergence_residual_m`, `radial_convergence_weighted_cos`, `missing_fraction_network`, `latency_mean_s`, `latency_max_s`, `piezo_rms.network_mean`), or sensor family (`node_mean_tilt_deg`, `piezo_rms`, `lidar_range_change_m`, `missing_fraction`) appears bound to a numeric value, that number must match that specific field of the packet within rounding tolerance.
2. **Unsupported Trend / Derived-Word Rejection (`forbidden_trend_regexes`):** Rejects `rising`, `falling`, `increasing`, `decreasing`, `doubled`, `spiking`, `worsening`, `accelerating`, and `steady`/`steadily` unless the packet explicitly contains a supporting trend/rate field (`trend`, `rate_of_change`, `slope`, `derivative`, `tilt_rate`, `velocity`, `acceleration`).

### Offline Re-Evaluation of Cached Explanations Under Guard v2
- **150 Stratified Evaluation Sample Explanations (`outputs/eval_explainer/cache/`):** **`0 / 150` failed** (`150 / 150` passed Guard v2 offline; Exact CP 95% CI `[0.9757, 1.0000]`; `0` required regeneration).
- **40 Dashboard Replay Explanations (`public/sim/explanations_<class>.json`):** **`0 / 40` failed** (`40 / 40` passed Guard v2 offline; Exact CP 95% CI `[0.9119, 1.0000]`; `0` required regeneration).

---

## 6. Held-Out Negative Control (Seed `20261125`): 36 New Corruptions + 42 Untouched Controls

To test frozen Guard v2 and the secondary LLM judge on **out-of-sample corruption types NOT used when designing Guard v2**, we pre-registered and committed `docs/explainer_heldout_expectations.md` alone (`commit e23aed9`) before building or running 36 new corrupted explanations (`6 per type`, seed `20261125`) alongside the same 42 untouched controls ($N = 78$ total). **Guard v2 was NOT modified in this step.**

| Held-Out Corruption / Control Type | N | Metric | Frozen Guard v2 [95% Boot CI] (Exact CP / Rule-of-3) | Secondary Judge (`muse-spark-1.3`) [95% Boot CI] (Exact CP / Rule-of-3) | Pre-Registered Guard v2 Prediction | What Slips Through Guard v2 |
|---|---:|---|---|---|---|---|
| `i_trend_metaphor_unlisted` | 6 | `catch_rate` | **0/6** (`0.0000` [0.0000, 0.0000] (Exact CP: [0.0000, 0.4593]; rule-of-3 <= 0.5000)) | **6/6** (`1.0000` [1.0000, 1.0000] (Exact CP: [0.5407, 1.0000])) | MISS (0/6) | Multi-word trend metaphors (`creeping upward`, `climbing progressively`, `gaining momentum`) without digits or listed forbidden trend stems. |
| `ii_wrong_sensor_family` | 6 | `catch_rate` | **0/6** (`0.0000` [0.0000, 0.0000] (Exact CP: [0.0000, 0.4593]; rule-of-3 <= 0.5000)) | **6/6** (`1.0000` [1.0000, 1.0000] (Exact CP: [0.5407, 1.0000])) | MISS / PARTIAL (0/6 to 3/6) | Valid node tilt value attached to `piezo energy at <val> V` without literal `piezo_rms` or an adjacent `NODE_0X` prefix. |
| `iii_cross_sentence_node_number` | 6 | `catch_rate` | **0/6** (`0.0000` [0.0000, 0.0000] (Exact CP: [0.0000, 0.4593]; rule-of-3 <= 0.5000)) | **6/6** (`1.0000` [1.0000, 1.0000] (Exact CP: [0.5407, 1.0000])) | MISS (0/6) | Cross-sentence anaphora (`Simulated NODE_01 ... That sensor recorded a mean tilt of <NODE_02_tilt> deg`) across a period boundary. |
| `iv_hedge_removed_certainty` | 6 | `catch_rate` | **0/6** (`0.0000` [0.0000, 0.0000] (Exact CP: [0.0000, 0.4593]; rule-of-3 <= 0.5000)) | **6/6** (`1.0000` [1.0000, 1.0000] (Exact CP: [0.5407, 1.0000])) | MISS (0/6) | Removing `consistent with` and stating present-perfect physical certainty (`proving the ground has moved`), which avoids `\bthe\s+ground\s+is\b`. |
| `v_invented_number_as_words` | 6 | `catch_rate` | **0/6** (`0.0000` [0.0000, 0.0000] (Exact CP: [0.0000, 0.4593]; rule-of-3 <= 0.5000)) | **6/6** (`1.0000` [1.0000, 1.0000] (Exact CP: [0.5407, 1.0000])) | MISS (0/6) | Invented plausible magnitudes written as English words (`about half a degree`) rather than Arabic digits. |
| `vi_contradictory_suggested_check` | 6 | `catch_rate` | **0/6** (`0.0000` [0.0000, 0.0000] (Exact CP: [0.0000, 0.4593]; rule-of-3 <= 0.5000)) | **5/6** (`0.8333` [0.5000, 1.0000]) | MISS (0/6) | A `suggested_check` that cites a quiet node's real tilt value while falsely calling it the primary anomaly source over the active node. |
| `ALL_HELDOUT_CORRUPTED_COMBINED` | 36 | `catch_rate` | **0/36** (`0.0000` [0.0000, 0.0000] (Exact CP: [0.0000, 0.0974]; rule-of-3 <= 0.0833)) | **35/36** (`0.9722` [0.9167, 1.0000]) | LOW CATCH (0/36 to 6/36) | All 36 out-of-vocabulary / cross-sentence / semantic corruptions slip past the lexical/digit regex guard (`0/36` caught by Guard v2). |
| `UNTOUCHED_FALSE_POSITIVE_RATE` | 42 | `false_positive_rate` | **0/42** (`0.0000` [0.0000, 0.0000] (Exact CP: [0.0000, 0.0841]; rule-of-3 <= 0.0714)) | **0/42** (`0.0000` [0.0000, 0.0000] (Exact CP: [0.0000, 0.0841]; rule-of-3 <= 0.0714)) | PASS (0/42 false positive) | None (0/42 false positives on faithful explanations). |

---

## 7. What the Deterministic Guard Still Cannot Catch (Confirmed by Held-Out Control)

The held-out negative control (`0/36` caught by frozen Guard v2 vs. high catch rate by the semantic judge) empirically confirms the exact structural boundaries of a deterministic lexical/numerical guard:
1. **Unlisted Trend Metaphors (`i_trend_metaphor_unlisted`, `0/6` caught by Guard v2):** Phrases like *"creeping upward"*, *"climbing progressively"*, and *"gaining momentum"* convey unsupported temporal trajectories without using digits or any stem in `forbidden_trend_regexes`.
2. **Sensor-Family Misattribution Outside Literal Regex Labels (`ii_wrong_sensor_family`, `0/6` caught by Guard v2):** Attaching a valid tilt reading from the packet to *"piezo energy at `<tilt>` V"* slips past `check_label_value_bindings` when no `NODE_0X` token precedes the number in the same clause and the sensor is referred to as `piezo energy` rather than `piezo_rms` / `piezo rms`.
3. **Cross-Sentence Anaphora (`iii_cross_sentence_node_number`, `0/6` caught by Guard v2):** Naming `NODE_01` in one sentence and attributing `NODE_02`'s tilt to *"That sensor"* in the next sentence crosses the period boundary where `check_label_value_bindings` stops scanning.
4. **Removed Hedges & Unlisted Certainty Verbs (`iv_hedge_removed_certainty`, `0/6` caught by Guard v2):** Replacing *"is consistent with"* with *"confirms ... proving the ground has moved"* slips through because Guard v2 blocks `the ground is` and `actually moving`, not `the ground has moved` or the absence of a hedge.
5. **Invented Numbers Written as Words (`v_invented_number_as_words`, `0/6` caught by Guard v2):** Because `NUMBER_TOKEN_RE` extracts only digit tokens, written-out magnitudes like *"about half a degree"* (`~0.5 deg`) are invisible to the numeric check.
6. **Contradictory `suggested_check` Targeting a Quiet Node (`vi_contradictory_suggested_check`, `0/6` caught by Guard v2):** Asking the operator to inspect the quietest node first as the primary anomaly source (while citing its true low tilt reading) satisfies every lexical, length, and number-binding rule despite being diagnostically backwards.
7. **Classifier Mistakes & Missed Events (Ground-Truth Blindness):** Because the explainer only receives observable window features and the classifier's output (`predicted_class`, `model_score`, `alarm_state`), it never sees `true_label`. The Replay UI displays the one-line warning under any explanation where `true_label != predicted_class`: *"Explains the classifier's output only; it does not know the true state."*

---

## 8. Spend & Artifacts Summary

- **Current Prompt Spend (36 Held-Out Negative-Control Judge Calls):** `$0.1830 USD` (strictly below the **$1.00 USD** spend cap).
- **Previous Patch Spend (20-Packet Uncached Spot Check + 84-Item Negative Control Judge Calls):** `$0.4164 USD`.
- **Cumulative Total Spend Across Initial Eval + All Negative Controls:** `$1.6996 USD`.
- **Pre-Registered Held-Out Expectations:** `docs/explainer_heldout_expectations.md` and `simulation/docs/explainer_heldout_expectations.md` (`commit e23aed9`).
- **Held-Out Negative Control Tables:** `simulation/outputs/eval_explainer/heldout_negative_control_summary.csv` and `simulation/outputs/eval_explainer/heldout_negative_control_records.csv`.
- **Original Negative Control Tables:** `simulation/outputs/eval_explainer/negative_control_summary.csv` and `simulation/outputs/eval_explainer/negative_control_records.csv`.
- **Manual Audit File:** `simulation/outputs/eval_explainer/manual_audit_sample.md`.
- **Spot-Check Table:** `simulation/outputs/eval_explainer/spot_check_20_uncached_4096.csv`.
