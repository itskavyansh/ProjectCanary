# Pre-Registered Negative Control Expectations for the Advisory Alarm Explainer (SYNTHETIC / SIMULATION)

**Status:** Pre-registered and committed **before** running the negative-control corruption suite (seed `20261124`, 42 corrupted explanations across 6 corruption types + 42 untouched faithful explanations) or implementing any new guard patches.

---

## 1. Objective & Design

To test the sensitivity and specificity of our verification pipeline, we take 42 cached faithful explanations ($7$ per corruption type, stratified across evaluation strata) and apply six programmatic corruptions, alongside 42 untouched cached faithful explanations ($N = 84$ total):

- **(a) `number_perturbed_10pct`**: Change one cited numeric value by approximately $\pm 10\%$.
- **(b) `node_label_swap`**: Attribute a cited valid packet number to the wrong node (swap `NODE_0X` labels, e.g., `NODE_01` $\leftrightarrow$ `NODE_02`).
- **(c) `wrong_class_named`**: Replace or add a different class name from `{normal, gradual, rapid, abnormal_vibration, sensor_fault, comm_fault}`.
- **(d) `forbidden_claim_added`**: Inject a forbidden real-world collapse claim (`"the ground is collapsing"`).
- **(e) `deleted_simulated_keyword`**: Delete all occurrences of the words `"simulated"` and `"synthetic"`.
- **(f) `unsupported_trend_claim`**: Inject an unsupported temporal trend or rate-of-change claim (e.g., `"tilt has doubled"` or `"rising steadily"`).

Both the **current deterministic guard (`ExplanationGuard` v1)** and the **secondary LLM judge (`muse-spark-1.3`, separate verification instruction, never given `truth`)** will evaluate all 84 explanations.

---

## 2. Pre-Registered Predictions for Current Guard (`ExplanationGuard` v1)

| Corruption Type ($n = 7$ each) | Current Guard Prediction | Predicted Catch Rate | Architectural Reason |
|---|---|---|---|
| **(a) `number_perturbed_10pct`** (change one cited number by ~10%) | **CATCH** | **100% (`7/7`)** (unless a $\pm 10\%$ perturbation of a non-zero number accidentally collides within rounding tolerance with another number present in the 23-number packet pool; we predict `7/7`, or $\ge 85\%$) | `ExplanationGuard.check` extracts every standalone numeric token via `extract_numbers_from_text` and requires each to match at least one value in `packet.all_numeric_values()` within rounding (`rel_tol=1e-4, abs_tol=1e-4` or 1–5 decimal rounding). Perturbing a cited number by ~10% produces an ungrounded value outside tolerance (`unfaithful_numbers`). |
| **(b) `node_label_swap`** (attribute a correct number to the wrong node by swapping `NODE_0X` labels) | **MISS** | **0% (`0/7`)** | **Unbound Bag-of-Numbers Check:** `ExplanationGuard` v1 strips `NODE_01..NODE_04` identifiers before extracting numbers and validates extracted numbers against a flat list `packet.all_numeric_values()`. Because the swapped number is still a valid number from the packet (belonging to the other node), the bag-of-numbers check passes with zero errors. There is currently no label-value binding rule tying `NODE_0X` or a sensor name to its specific packet field. |
| **(c) `wrong_class_named`** (name a different class) | **CATCH** | **100% (`7/7`)** | `ExplanationGuard` v1 explicitly enforces that `packet.predicted_class` is present (`missing_predicted_class`) AND checks all other classes in `VALID_CLASSES` (in both snake_case and space-separated forms), rejecting any match (`wrong_class_mentioned`). Whether the corruption replaces `predicted_class` or adds a second class name, v1 catches it deterministically. |
| **(d) `forbidden_claim_added`** (add `"the ground is collapsing"`) | **CATCH** | **100% (`7/7`)** | `explainer_config.json` includes `"\\bthe\\s+ground\\s+is\\b"` in `forbidden_claim_regexes`. Adding `"the ground is collapsing"` deterministically triggers `forbidden_claim: matched 'the ground is'`. (If appended as an extra sentence, it may also trigger sentence-count limits.) |
| **(e) `deleted_simulated_keyword`** (delete `"simulated"` / `"synthetic"`) | **CATCH** | **100% (`7/7`)** | `ExplanationGuard` v1 checks `any(w in lower_combined for w in self.required_sim_words)` where `required_simulation_words = ["simulated", "synthetic"]`. Removing `"simulated"` (when `"synthetic"` is not present) deterministically triggers `missing_simulated_or_synthetic_keyword`. |
| **(f) `unsupported_trend_claim`** (add `"tilt has doubled"` or `"rising steadily"`) | **MISS** | **0% (`0/7`)** | **No Trend / Derived-Word Lexicon Check:** As long as the injected trend phrase uses words rather than digits (e.g., `"tilt has doubled"`, `"rising steadily"`), `extract_numbers_from_text` extracts no new numeric token, and none of the existing `forbidden_claim_regexes` prohibit temporal trend or rate-of-change words (`rising`, `falling`, `doubled`, `steadily`, `accelerating`, `worsening`, `spiking`). Thus `ExplanationGuard` v1 misses purely verbal trend hallucinations. |
| **Untouched Faithful Controls ($n = 42$)** | **PASS (0% False Positive)** | **0% False Positive (`0/42` flagged)** | Every cached explanation already passed `ExplanationGuard` v1 at cache generation time, and the guard is deterministic. |

---

## 3. Pre-Registered Predictions for Secondary LLM Judge (`muse-spark-1.3`)

- **Expected Strengths Relative to Current Guard v1:** Because the judge prompt asks the LLM to verify whether every number and node/sensor attribution matches the packet JSON and whether any unsupported claims are made, the LLM judge should catch a high proportion of **(a) `number_perturbed_10pct`**, **(b) `node_label_swap`** (which Guard v1 misses!), **(c) `wrong_class_named`**, and **(d) `forbidden_claim_added`**.
- **Expected Weaknesses Relative to Current Guard v1:**
  1. **(e) `deleted_simulated_keyword`**: The LLM judge focuses on factual faithfulness to the packet JSON and real-world safety overclaims; unless its rubric explicitly penalises omission of the word `"simulated"`, it is likely to miss (pass) explanations where `"simulated"` was deleted (`0%–30%` catch rate).
  2. **(f) `unsupported_trend_claim`**: Because `"rising steadily"` or `"tilt has doubled"` sounds plausible in prose next to static window features (or may be confused with `lidar.range_change_m`), the LLM judge may only partially catch verbal trend claims (`40%–85%` catch rate).
  3. **Non-Zero False Positive / Self-Bias Variance:** On the 42 untouched explanations, the LLM judge is predicted to have a low false-positive rate (`0%–10%`), whereas the deterministic guard has strictly `0%`.

---

## 4. Planned Guard v2 Remediation (After Running Negative Control)

Based on the two predicted structural blind spots in `ExplanationGuard` v1—**(b) `node_label_swap`** and **(f) `unsupported_trend_claim`**—we will patch `guard.py` (without weakening any existing check) to add:
1. **Label-Value Binding (`unfaithful_node_or_sensor_binding`):** When a node identifier (`NODE_01`..`NODE_04`) or specific sensor/metric label appears bound to a numeric value in a clause, verify that the number matches that specific node's or sensor's field in the packet, not merely *some* number anywhere in the packet.
2. **Unsupported Trend / Derived-Word Rejection (`unsupported_trend_term`):** Reject trend or rate-of-change terms (`rising`, `falling`, `increasing`, `decreasing`, `doubled`, `spiking`, `worsening`, `accelerating`, `steady`/`steadily`) unless the packet explicitly contains a supporting field.
