# Advisory Alarm Explainer Evaluation & Negative-Control Patch Report (SYNTHETIC / SIMULATION)

> **SYNTHETIC / SIMULATION - NOT VALIDATED ON REAL DATA**
>
> **Scope:** Advisory explainer evaluation patch only (`muse-spark-1.3`). All classifiers, features, thresholds, operating points, `packets.json`, cached explanations, replay JSONs, and existing classifier benchmark results remain frozen.

---

## 1. Headline Metrics: First-Run vs. After Cache Fill ($N = 150$ Stratified Packets)

On the initial live run at `max_completion_tokens = 2048`, **148 / 150 (`98.67%`)** packets succeeded on the first try with **0 deterministic guard rejections** (`0.0%` unfaithful numbers, `0.0%` wrong class mentions, `0.0%` forbidden claims). The remaining **2 / 150 (`1.33%`)** packets (`run_test_005_w43` in `false_alarm` and `run_test_106_w29` in `fault_window`) hit the `2048` completion-token limit during internal reasoning tokens (`json_parse_error` on truncated output) and deterministically fell back to the template on the first run before being filled in the cache on the second pass.

In accordance with Requirement 2, **first-run numbers are reported as the primary headline estimate**, and post-cache numbers are reported in the separate **After Cache Fill** column (`outputs/eval_explainer/overall_metrics_ci95.csv`):

| Metric | N | First-Run Headline Estimate [95% Bootstrap CI] | After Cache Fill Estimate [95% Bootstrap CI] | Notes |
|---|---:|---|---|---|
| `first_try_guard_pass_rate` | 150 | **0.9867** [0.9667, 1.0000] | `1.0000` [1.0000, 1.0000] | 148/150 first-try pass on first run (2 API token cutoffs at 2048 limit, 0 guard rejections); 150/150 after cache fill |
| `api_cutoff_rate` | 150 | **0.0133** [0.0000, 0.0333] | `0.0000` [0.0000, 0.0000] | 2/150 hit 2048 completion_tokens limit on first run (run_test_005_w43, run_test_106_w29); 0/150 after cache fill & 4096 token limit fix |
| `guard_rejection_rate_pre_guard` | 150 | **0.0000** [0.0000, 0.0000] | `0.0000` [0.0000, 0.0000] | 0/150 guard rejections on complete JSON responses across both runs |
| `retry_rate` | 150 | **0.0000** [0.0000, 0.0000] | `0.0000` [0.0000, 0.0000] | 0/150 guard-triggered retries (API cutoffs fell back directly) |
| `fallback_rate` | 150 | **0.0133** [0.0000, 0.0333] | `0.0000` [0.0000, 0.0000] | 2/150 template fallbacks on first run due to 2048 token cutoff; 0/150 after cache fill |
| `unfaithful_number_rate_pre_guard` | 150 | **0.0000** [0.0000, 0.0000] | `0.0000` [0.0000, 0.0000] | 0/150 unfaithful numbers before guard |
| `unfaithful_number_rate_post_guard` | 150 | **0.0000** [0.0000, 0.0000] | `0.0000` [0.0000, 0.0000] | 0/150 unfaithful numbers after guard |
| `class_contradiction_rate_pre_guard` | 150 | **0.0000** [0.0000, 0.0000] | `0.0000` [0.0000, 0.0000] | 0/150 class contradictions before guard |
| `class_contradiction_rate_post_guard` | 150 | **0.0000** [0.0000, 0.0000] | `0.0000` [0.0000, 0.0000] | 0/150 class contradictions after guard |
| `forbidden_claim_rate_pre_guard` | 150 | **0.0000** [0.0000, 0.0000] | `0.0000` [0.0000, 0.0000] | 0/150 forbidden claims before guard |
| `forbidden_claim_rate_post_guard` | 150 | **0.0000** [0.0000, 0.0000] | `0.0000` [0.0000, 0.0000] | 0/150 forbidden claims after guard |
| `final_guard_pass_rate` | 150 | **1.0000** [1.0000, 1.0000] | `1.0000` [1.0000, 1.0000] | 150/150 final explanations pass deterministic guard |
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
| **H8a_spend_cap_respected** | H8a. Hard Budget Cap Held ($5.00 Initial / $2.00 Patch) | `Initial <= $5.00 USD; Patch <= $2.00 USD` | `Initial: $1.1002 / $5.00 USD; Patch: $0.4164 / $2.00 USD` | **Yes** | Confirmed: initial evaluation spend ($1.1002 USD) stayed well under the $5.00 hard cap, and patch spend ($0.4164 USD) stayed well under the $2.00 patch spend cap. |
| **H8b_spend_point_estimate** | H8b. Pre-Registered Point Estimate (< $0.50 USD) | `< $0.50 USD` | `$1.1002 USD (initial eval); $1.5166 USD (cumulative with patch)` | **No** | Not confirmed: initial 150-packet + 40-judge spend was $1.1002 USD (above the <$0.50 point estimate) because muse-spark-1.3 billed ~1,100 internal reasoning tokens per call alongside ~120 visible JSON output tokens. |
| **H9_judge_agreement_rate** | Secondary Judge Agreement on Final Explanations (N=40) | `>= 0.90 rating >= 4 and >= 0.90 agreement with guard` | `rating>=4: 1.0000, agreement: 1.0000` | **Yes** | Confirmed on 40-packet random sample (subject to same-model judge limitation; see Section 4 for negative-control sensitivity). |

---

## 3. Completion Token Limit Fix (`2048 -> 4096`) & 20-Packet Uncached Spot Check

- **Configuration Change:** Raised `max_completion_tokens` from `2048` to `4096` in `simulation/canary_sim/explainer/explainer_config.json` and `ChatClient.complete` (`simulation/canary_sim/explainer/client.py`).
- **20-Packet Uncached Spot Check (Seed `20261124`):** Evaluated `20` packets fresh against the live API without reading or mutating the primary packet cache (explicitly including both previously cut-off packets `run_test_005_w43` and `run_test_106_w29` plus 18 stratified packets).
- **Spot-Check Results:**
  - **API Token Cutoffs at `max_completion_tokens = 4096`:** **`0 / 20` (`0.0%`)** — raising the completion token limit completely eliminated reasoning-token cutoffs.
  - **Valid JSON Responses:** `20 / 20` (`100.0%`).
  - **First-Try Guard v2 Pass Rate on Spot Check:** `20 / 20` (`100.0%`).
  - **Observed Completion Tokens (Reasoning + Output):** mean `1228.8` tokens, max `1919` tokens (well below the new `4096` ceiling).

---

## 4. Negative Control (Seed `20261124`): 42 Corrupted + 42 Untouched Explanations

Pre-registered expectations were committed alone in `docs/explainer_negative_control_expectations.md` (`commit 714a4c8`) before running any corruptions. We took 42 cached faithful explanations (7 per corruption type, 1 from each of the 7 evaluation strata) and programmatically created 6 corruption types alongside 42 untouched faithful explanations ($N = 84$ total).

Both the deterministic guard (**Guard v1** before patch and **Guard v2** after adding label-value binding and trend-word rules) and the **Secondary LLM Judge** (`muse-spark-1.3`, separate instruction, never given `truth`) were evaluated on all 84 explanations:

| Corruption / Control Type | N | Metric | Guard v1 (Pre-Patch) [95% CI] | Guard v2 (Patched) [95% CI] | Secondary Judge (`muse-spark-1.3`) [95% CI] | Pre-Registered Guard v1 Expectation |
|---|---:|---|---|---|---|---|
| `a_number_perturbed_10pct` | 7 | `catch_rate` | **7/7** (`1.0000` [1.0000, 1.0000]) | **7/7** (`1.0000` [1.0000, 1.0000]) | **7/7** (`1.0000` [1.0000, 1.0000]) | CATCH (100% — bag-of-numbers mismatch) |
| `b_node_label_swap` | 7 | `catch_rate` | **0/7** (`0.0000` [0.0000, 0.0000]) | **7/7** (`1.0000` [1.0000, 1.0000]) | **7/7** (`1.0000` [1.0000, 1.0000]) | MISS (0% — valid packet number on wrong node) |
| `c_wrong_class_named` | 7 | `catch_rate` | **7/7** (`1.0000` [1.0000, 1.0000]) | **7/7** (`1.0000` [1.0000, 1.0000]) | **7/7** (`1.0000` [1.0000, 1.0000]) | CATCH (100% — wrong/missing class token) |
| `d_forbidden_claim_added` | 7 | `catch_rate` | **7/7** (`1.0000` [1.0000, 1.0000]) | **7/7** (`1.0000` [1.0000, 1.0000]) | **7/7** (`1.0000` [1.0000, 1.0000]) | CATCH (100% — matches `\bthe\s+ground\s+is\b`) |
| `e_deleted_simulated_keyword` | 7 | `catch_rate` | **7/7** (`1.0000` [1.0000, 1.0000]) | **7/7** (`1.0000` [1.0000, 1.0000]) | **7/7** (`1.0000` [1.0000, 1.0000]) | CATCH (100% — missing `simulated`/`synthetic`) |
| `f_unsupported_trend_claim` | 7 | `catch_rate` | **0/7** (`0.0000` [0.0000, 0.0000]) | **7/7** (`1.0000` [1.0000, 1.0000]) | **7/7** (`1.0000` [1.0000, 1.0000]) | MISS (0% — verbal trend words had no rule) |
| `ALL_CORRUPTED_COMBINED` | 42 | `catch_rate` | **28/42** (`0.6667` [0.5000, 0.7863]) | **42/42** (`1.0000` [1.0000, 1.0000]) | **42/42** (`1.0000` [1.0000, 1.0000]) | 4/6 types caught (28/42 = 66.7%) |
| `UNTOUCHED_FALSE_POSITIVE_RATE` | 42 | `false_positive_rate` | **0/42** (`0.0000` [0.0000, 0.0000]) | **0/42** (`0.0000` [0.0000, 0.0000]) | **0/42** (`0.0000` [0.0000, 0.0000]) | 0% false positive (0/42) |

---

## 5. Guard v2 Fixes & Offline Re-Evaluation Over All Cached Explanations

Without weakening any existing check in `simulation/canary_sim/explainer/guard.py`, we added two deterministic rules addressing the negative-control blind spots:
1. **Label-Value Binding (`check_label_value_bindings`):** Whenever a node identifier (`NODE_01`..`NODE_04`), node metric (`NODE_0X_missing_fraction`), scalar metric (`model_score`, `anomaly_score`, `hub_common_mode_m`, `radial_convergence_residual_m`, `radial_convergence_weighted_cos`, `missing_fraction_network`, `latency_mean_s`, `latency_max_s`, `piezo_rms.network_mean`), or sensor family (`node_mean_tilt_deg`, `piezo_rms`, `lidar_range_change_m`, `missing_fraction`) appears bound to a numeric value, that number must match that specific field of the packet within rounding tolerance.
2. **Unsupported Trend / Derived-Word Rejection (`forbidden_trend_regexes`):** Rejects `rising`, `falling`, `increasing`, `decreasing`, `doubled`, `spiking`, `worsening`, `accelerating`, and `steady`/`steadily` unless the packet explicitly contains a supporting trend/rate field (`trend`, `rate_of_change`, `slope`, `derivative`, `tilt_rate`, `velocity`, `acceleration`).

### Offline Re-Evaluation of Cached Explanations Under Guard v2
- **150 Stratified Evaluation Sample Explanations (`outputs/eval_explainer/cache/`):** **`0 / 150` failed** (`150 / 150` passed Guard v2 offline; `0` required regeneration).
- **40 Dashboard Replay Explanations (`public/sim/explanations_<class>.json`):** **`0 / 40` failed** (`40 / 40` passed Guard v2 offline; `0` required regeneration).

---

## 6. What the Deterministic Guard Still Cannot Catch (Honest Limitations)

Even after the Guard v2 patch (`42/42` catch rate on the 6 negative-control corruption types and `0/42` false positives), a deterministic lexical/numerical guard has fundamental limits that must be stated transparently:
1. **Classifier Mistakes & Missed Events (Ground-Truth Blindness):** Because the explainer only receives observable window features and the classifier's output (`predicted_class`, `model_score`, `alarm_state`), it never sees `true_label`. When the frozen classifier misses a real event (`predicted_class == 'normal'` during an active `gradual` or `rapid` event) or confuses two classes, the explanation faithfully explains the classifier's wrong prediction. To make this explicit to operators, the Replay UI now displays the one-line warning under any explanation where `predicted_class == 'normal'` while `true_label != 'normal'` (and whenever `true_label != predicted_class`): *"Explains the classifier's output only; it does not know the true state."*
2. **Exact-Collision Node Swaps (Equal Values Across Two Nodes):** If two nodes happen to have the exact same numeric value in the packet (e.g., `NODE_01_missing_fraction = 0.0` and `NODE_02_missing_fraction = 0.0`, or identical rounded LiDAR range changes), swapping `NODE_01` and `NODE_02` for that identical number remains numerically true for both nodes and cannot be distinguished from the packet alone.
3. **Unlisted Synonymous Trend Phrasing:** While Guard v2 blocks `rising`, `falling`, `increasing`, `decreasing`, `doubled`, `spiking`, `worsening`, `accelerating`, and `steady`/`steadily`, an LLM could phrase an unsupported temporal trajectory using unlisted synonyms or multi-word metaphors (e.g., *"creeping upward"*, *"climbing progressively"*, *"gaining momentum"*, *"halved"*, *"tripled"*) without using forbidden tokens or digits.
4. **Cross-Clause Anaphora / Indirect Attribution:** `check_label_value_bindings` verifies direct local bindings between a node/sensor label and its adjacent number, but cannot parse arbitrary multi-sentence coreference (e.g., *"Consider NODE_01 and NODE_02. The former recorded a tilt of 0.3891 deg"* where `0.3891` actually belongs to `NODE_02`).
5. **Domain Quality of the Suggested Check:** The guard verifies that `suggested_check` is a single concise sentence with valid packet numbers and no safety/evacuation overclaims, but it cannot verify whether the suggested engineering check is the *most diagnostically informative* sensor to inspect first.

---

## 7. Patch Spend & Artifacts Summary

- **Patch Spend (20-Packet Uncached Spot Check + 84-Item Negative Control Judge Calls):** `$0.4164 USD` (strictly below the **$2.00 USD** patch spend cap; `$2.00 USD` cap respected).
- **Cumulative Total Spend Across Initial Eval + Patch:** `$1.5166 USD`.
- **Manual Audit File:** `simulation/outputs/eval_explainer/manual_audit_sample.md` (30 explanations, 5 per stratum including `missed_event` and `classifier_mistake`, with packet values and text shown first and `true_label` shown separately at the bottom).
- **Negative Control Tables:** `simulation/outputs/eval_explainer/negative_control_summary.csv` and `simulation/outputs/eval_explainer/negative_control_records.csv`.
- **Spot-Check Table:** `simulation/outputs/eval_explainer/spot_check_20_uncached_4096.csv`.
