# Pre-Registered Faithfulness Expectations for the Advisory Alarm Explainer (SYNTHETIC / SIMULATION)

**Status:** Pre-registered and committed **before** any real LLM API call (`muse-spark-1.3`) on the 150-packet stratified evaluation sample (`outputs/explainer_sample/packets.json`, fixed seed `20261123`).

---

## 1. Scope & Ground-Rules

1. **Advisory Only:** The explainer reads a frozen v3 `ExplainerPacket` (`predicted_class`, `model_score`, `alarm_state`, `threshold_rule_class`, `unknown_anomaly`, `anomaly_score`, per-node mean tilt, per-node and network piezo RMS, per-node LiDAR range change and hub common-mode, communication missing fractions and latency, radial convergence residual and weighted cosine, and `snr_vs_imu_noise_bin`) and outputs `{ "summary": "...", "suggested_check": "..." }`.
2. **Zero Authority:** The explainer never alters `predicted_class`, `alarm_state`, `model_score`, or any number, and never sees the ground-truth `truth` field.
3. **Deterministic Guard (`ExplanationGuard`):** Rejects any output that (a) includes a number not matching a packet value within rounding, (b) fails to name `predicted_class` or mentions any conflicting class name, (c) uses forbidden real-world claims, evacuation/safety advice, or calls `model_score` a `"probability"` or `"confidence"`, (d) violates sentence/character length limits (`summary` 2–3 sentences, `suggested_check` 1 sentence, total 40–750 chars), (e) omits `"simulated"`/`"synthetic"`, or (f) implies a changed alarm state.
4. **Single Retry & Deterministic Fallback:** On first guard failure, the explainer retries at most once with the guard's error list appended to the prompt. On second failure (or API error / budget cap), it falls back to `generate_template_explanation` (`source: "template"`).

---

## 2. Pre-Registered Quantitative Predictions ($N = 150$ Stratified Packets)

| Metric | Pre-Registered Prediction (Range) | Rationale |
|---|---|---|
| **H1. First-Try Guard Pass Rate** | **70% – 92%** (`0.70` to `0.92`) | With explicit prompt constraints ("copy every number character-for-character", "name only `predicted_class`", "include `simulated`"), `muse-spark-1.3` should pass on the first try on the majority of packets, with failures concentrated in derived-number arithmetic or secondary class mentions. |
| **H2. Retry Rate (Attempt 2 Triggered)** | **8% – 30%** (`0.08` to `0.30`) | Equal to $1 - \text{first-try pass rate}$ (assuming zero HTTP errors). Most first-try failures will trigger a single retry with explicit guard feedback. |
| **H3. Final Template Fallback Rate** | **0% – 12%** (`0.00` to `0.12`) | Giving the model the exact list of unmatched numbers or forbidden class tokens on retry should fix over two-thirds of first-try failures, leaving at most 12% requiring the deterministic template fallback. |
| **H4. Unfaithful Number Rate (Pre-Guard vs Post-Guard)** | **Pre-guard: 4% – 22%** (`0.04` to `0.22`)<br>**Post-guard: 0.0%** (`0.00`) | Before the guard, LLMs occasionally convert fractions into percentages (e.g., `0.0125` missing fraction written as `1.25%`), compute differences or ratios across nodes, or cite external domain thresholds (`0.607 deg`, `3.5 deg`, `4 nodes`). After the deterministic guard + template fallback, the unfaithful number rate is strictly `0.0%` by construction. |
| **H5. Where Hallucinated / Unfaithful Numbers Are Most Likely** | **1) Percentages from fractions (`comm` missing fraction or `model_score` × 100)**<br>**2) Derived node differences / counts (e.g., "across 4 nodes", "3 out of 4 nodes", or `NODE_01 - NODE_02`)**<br>**3) Memorised or external thresholds not in the packet** | Numbers that are mathematically valid derivations of packet fields (`100 * missing_fraction_network`, node counts like `4`, or differences between maximum and minimum node tilts) are the most common source of pre-guard number rejections because they are not verbatim packet values. |
| **H6. Class-Contradiction / Wrong-Class Rate (Pre-Guard vs Post-Guard)** | **Pre-guard: 2% – 18%** (`0.02` to `0.18`), highest in `false_alarm`, `missed_event`, and `classifier_mistake` strata where `threshold_rule_class` or `alarm_state` differs from `predicted_class`<br>**Post-guard: 0.0%** (`0.00`) | When `predicted_class` is `sensor_fault` or `rapid` while `alarm_state` or `threshold_rule_class` is `normal` or `gradual`, the LLM may be tempted to contrast the two classes in prose, triggering `wrong_class_mentioned`. |
| **H7. Behaviour on Classifier Mistakes (`classifier_mistake` Stratum, $n = 26$)** | **100% of post-guard explanations faithfully reflect the classifier's (wrong) `predicted_class` and add 0% forbidden confidence/probability claims**; pre-guard raw outputs will also reflect `predicted_class` rather than ground truth (since `truth` is withheld), though without the guard an LLM would risk sounding unwarrantedly confident if not constrained to hedged `"consistent with"` wording and `model_score`. | Because the explainer never sees `truth` (`true_label`), it cannot know the classifier made a mistake. Its job is to faithfully report what the frozen classifier predicted (`predicted_class` and `model_score`) using hedged language (`"consistent with"`, `"simulated"`) and zero forbidden confidence terms (`"probability"`, `"confidence"`, `"certain"`). |
| **H8. Total Spend vs Budget Cap** | **Total spend < \$0.50 USD** (well below the **\$5.00 USD** hard cap) | 150 packets plus up to ~35 retries and 40 judge calls at ~600 input tokens and ~120 output tokens per call totals roughly 150k input + 30k output tokens, costing ~$0.30–$0.35 USD at $1.25/M input and $4.25/M output. |

---

## 3. Secondary Judge Pre-Registration ($N = 40$ Random Sample)

- A secondary judge call (using a separate system instruction that rates faithfulness to the packet JSON from `1` to `5` without seeing `truth`) will be run on 40 randomly sampled final explanations (fixed seed `20261123`).
- **Prediction (H9):** Because every final explanation has passed `ExplanationGuard` (or been replaced by the deterministic template), we predict **$\ge 90\%$ of final explanations will receive a judge faithfulness rating of $\ge 4$ out of 5**, with high agreement ($\ge 90\%$) between the deterministic guard and the judge (`rating >= 4`).
- **Explicit Limitation:** Because only one model endpoint (`muse-spark-1.3`) is configured, the secondary judge uses the **same underlying model** with a different verification instruction and is therefore **not an independent external judge**. The deterministic guard (`guard.py`), not the LLM judge, holds primary verification authority.
