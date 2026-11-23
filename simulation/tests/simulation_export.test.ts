import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import {
  EXPECTED_SIM_BANNER,
  validateSimEnvelope,
  type ReplayClassKey,
} from "../../src/types/simulation.ts";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const SIM_DIR = path.resolve(__dirname, "..", "..", "public", "sim");

const EXPECTED_FILES = [
  "summary.json",
  "results_by_split.json",
  "results_false_alarms.json",
  "results_recall_vs_snr.json",
  "results_model_comparison.json",
  "robustness.json",
  "parameters.json",
  "replay_normal.json",
  "replay_gradual.json",
  "replay_rapid.json",
  "replay_abnormal_vibration.json",
  "replay_sensor_fault.json",
  "replay_comm_fault.json",
] as const;

const EXPECTED_EXPLANATION_FILES = [
  "explanations_normal.json",
  "explanations_gradual.json",
  "explanations_rapid.json",
  "explanations_abnormal_vibration.json",
  "explanations_sensor_fault.json",
  "explanations_comm_fault.json",
] as const;

const REPLAY_CLASSES: ReplayClassKey[] = [
  "normal",
  "gradual",
  "rapid",
  "abnormal_vibration",
  "sensor_fault",
  "comm_fault",
];

const SECRET_PATTERNS: RegExp[] = [
  /eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}/, // JWT token
  /sk_(live|test)_[A-Za-z0-9]{16,}/,
  /sbp_[A-Za-z0-9]{20,}/,
  /SUPABASE_SERVICE_ROLE_KEY/i,
  /FAST2SMS_API_KEY/i,
  /META_API_KEY/i,
  /PRIVATE_KEY/i,
];

function readSimJson(fileName: string): Record<string, unknown> {
  const fullPath = path.join(SIM_DIR, fileName);
  assert.ok(fs.existsSync(fullPath), `Expected public/sim/${fileName} to exist`);
  const raw = fs.readFileSync(fullPath, "utf-8");
  return JSON.parse(raw) as Record<string, unknown>;
}

// -----------------------------------------------------------------------------
// SCENARIO 1: All exported JSON files exist, respect size limits, and carry banner
// -----------------------------------------------------------------------------
test("Scenario 1: All public/sim JSON files load, stay within size limits, and carry the mandatory banner", () => {
  let totalBytes = 0;

  for (const fileName of EXPECTED_FILES) {
    const fullPath = path.join(SIM_DIR, fileName);
    assert.ok(fs.existsSync(fullPath), `Missing file: public/sim/${fileName}`);

    const stat = fs.statSync(fullPath);
    assert.ok(stat.size > 0, `${fileName} must not be empty`);
    assert.ok(
      stat.size <= 300 * 1024,
      `${fileName} (${stat.size} bytes) exceeds 300 KB limit`
    );
    totalBytes += stat.size;

    const parsed = readSimJson(fileName);
    validateSimEnvelope(parsed, fileName, ["file_id"]);
    assert.equal(parsed.synthetic, true);
    assert.equal(parsed.banner, EXPECTED_SIM_BANNER);
  }

  assert.ok(
    totalBytes <= 2 * 1024 * 1024,
    `Total public/sim export size (${totalBytes} bytes) exceeds 2 MB limit`
  );
});

// -----------------------------------------------------------------------------
// SCENARIO 2: summary.json, results_*.json, robustness.json, parameters.json schemas
// -----------------------------------------------------------------------------
test("Scenario 2: Core study JSON files match expected keys and structure", () => {
  const summary = readSimJson("summary.json");
  validateSimEnvelope(summary, "summary.json", [
    "study_title",
    "study_description",
    "headline_claims",
    "limitations",
    "dataset_overview",
  ]);
  const claims = summary.headline_claims as Record<string, unknown>[];
  assert.ok(Array.isArray(claims) && claims.length >= 7);
  for (const claim of claims) {
    assert.ok(typeof claim.id === "string" && claim.id.length > 0);
    assert.ok(typeof claim.title === "string" && claim.title.length > 0);
    assert.ok(typeof claim.source_file === "string" && claim.source_file.length > 0);
    assert.ok(typeof claim.caveat === "string" && claim.caveat.length > 0);
    assert.ok(claim.number && typeof claim.number === "object");
  }

  const bySplit = readSimJson("results_by_split.json");
  validateSimEnvelope(bySplit, "results_by_split.json", [
    "window_metrics_by_split",
    "v2_vs_v3_all_snr",
    "v2_vs_v3_snr_gt_2",
    "class_share_of_windows_per_split",
    "stress_set_scoring",
  ]);

  const falseAlarms = readSimJson("results_false_alarms.json");
  validateSimEnvelope(falseAlarms, "results_false_alarms.json", [
    "definition_notes",
    "false_alarms_table",
  ]);
  const faTable = falseAlarms.false_alarms_table as Record<string, unknown>[];
  assert.ok(Array.isArray(faTable) && faTable.length > 0);
  assert.ok("B_sensor_fa_episodes_per_hour" in faTable[0]);
  assert.ok("A_slow_fa_episodes_per_hour" in faTable[0]);

  const recallSnr = readSimJson("results_recall_vs_snr.json");
  validateSimEnvelope(recallSnr, "results_recall_vs_snr.json", [
    "gradual_rapid_recall_by_snr_v3",
    "gradual_rapid_recall_and_detection_v3",
    "recall_vs_snr_5bins_test",
  ]);

  const modelComp = readSimJson("results_model_comparison.json");
  validateSimEnvelope(modelComp, "results_model_comparison.json", [
    "rf_vs_threshold_comparison",
    "permutation_importance_groups",
    "permutation_importance_top20",
    "rf_ablations_v3",
  ]);

  const robustness = readSimJson("robustness.json");
  validateSimEnvelope(robustness, "robustness.json", [
    "yaw_sensitivity",
    "shape_mismatch",
    "literature_checks",
    "timescale_audit",
    "unknown_detector_complement",
  ]);
  const unkComp = robustness.unknown_detector_complement as Record<string, unknown>;
  assert.equal(unkComp.advisory, true);

  const parameters = readSimJson("parameters.json");
  validateSimEnvelope(parameters, "parameters.json", [
    "literature_table",
    "unsourced_items",
    "timescale_note",
  ]);
  const litTable = parameters.literature_table as Record<string, unknown>[];
  assert.ok(Array.isArray(litTable) && litTable.length >= 6);
  assert.ok(litTable.some((r) => r.status_tag === "PLACEHOLDER"));
});

// -----------------------------------------------------------------------------
// SCENARIO 3: All 6 replay_<class>.json files match expected window keys & model_score
// -----------------------------------------------------------------------------
test("Scenario 3: All 6 replay_<class>.json files contain valid 95-window timelines and model_score", () => {
  for (const cls of REPLAY_CLASSES) {
    const fileName = `replay_${cls}.json`;
    const rep = readSimJson(fileName);
    validateSimEnvelope(rep, fileName, [
      "run_class",
      "simulation_id",
      "selection_rule",
      "run_metadata",
      "operating_point",
      "windows",
    ]);
    assert.equal(rep.run_class, cls);

    const windows = rep.windows as Record<string, unknown>[];
    assert.ok(Array.isArray(windows));
    assert.equal(windows.length, 95);

    const w0 = windows[0];
    for (const reqKey of [
      "window_idx",
      "t_start",
      "t_end",
      "t_mid",
      "true_label",
      "model_score",
      "model_scores_by_class",
      "predicted_class",
      "alarm_state",
      "node_mean_tilt_deg",
      "piezo_rms",
      "lidar_range_change_m",
      "comm_missing_fraction",
      "unknown_anomaly",
    ]) {
      assert.ok(reqKey in w0, `${fileName} window[0] missing key ${reqKey}`);
    }
    assert.equal(typeof w0.model_score, "number");
    assert.equal(typeof w0.unknown_anomaly, "boolean");
  }
});

// -----------------------------------------------------------------------------
// SCENARIO 4: No secret-looking strings and no probability/confidence field names
// -----------------------------------------------------------------------------
test("Scenario 4: No exported JSON file contains secret-looking strings or probability/confidence fields", () => {
  for (const fileName of [...EXPECTED_FILES, ...EXPECTED_EXPLANATION_FILES]) {
    const fullPath = path.join(SIM_DIR, fileName);
    const rawText = fs.readFileSync(fullPath, "utf-8");
    const lower = rawText.toLowerCase();

    assert.ok(
      !lower.includes("probability"),
      `${fileName} must use model_score, never probability`
    );
    assert.ok(
      !lower.includes("confidence"),
      `${fileName} must use model_score, never confidence`
    );

    for (const pattern of SECRET_PATTERNS) {
      assert.ok(
        !pattern.test(rawText),
        `Secret-looking pattern ${pattern} matched in ${fileName}`
      );
    }
  }
});

// -----------------------------------------------------------------------------
// SCENARIO 6: All 6 explanations_<class>.json files respect schema, <100 KB size, no secrets
// -----------------------------------------------------------------------------
test("Scenario 6: All 6 explanations_<class>.json files stay under 100 KB, match schema, and carry advisory notice", () => {
  for (const cls of REPLAY_CLASSES) {
    const fileName = `explanations_${cls}.json`;
    const fullPath = path.join(SIM_DIR, fileName);
    assert.ok(fs.existsSync(fullPath), `Missing file: public/sim/${fileName}`);

    const stat = fs.statSync(fullPath);
    assert.ok(stat.size > 0, `${fileName} must not be empty`);
    assert.ok(
      stat.size <= 100 * 1024,
      `${fileName} (${stat.size} bytes) exceeds 100 KB limit`
    );

    const parsed = readSimJson(fileName);
    validateSimEnvelope(parsed, fileName, [
      "file_id",
      "run_class",
      "simulation_id",
      "advisory_notice",
      "explanations",
    ]);
    assert.equal(parsed.run_class, cls);
    assert.equal(
      parsed.advisory_notice,
      "Advisory text only. It does not change the alarm."
    );

    const explanations = parsed.explanations as Record<string, unknown>[];
    assert.ok(Array.isArray(explanations));
    assert.ok(
      explanations.length >= 1 && explanations.length <= 12,
      `${fileName} must have 1..12 explanations, got ${explanations.length}`
    );

    for (const item of explanations) {
      assert.equal(typeof item.window_idx, "number");
      assert.ok(typeof item.summary === "string" && item.summary.length > 0);
      assert.ok(
        typeof item.suggested_check === "string" && item.suggested_check.length > 0
      );
      assert.ok(
        item.source === "ai" || item.source === "template",
        `Invalid source ${String(item.source)} in ${fileName}`
      );
      assert.ok(typeof item.model_id === "string" && item.model_id.length > 0);
      const gr = item.guard_result as Record<string, unknown>;
      assert.ok(gr && typeof gr === "object");
      assert.equal(gr.passed, true);
      const pkt = item.packet as Record<string, unknown>;
      assert.ok(pkt && typeof pkt === "object");
      assert.equal(typeof pkt.model_score, "number");
      assert.ok(!("truth" in pkt) && !("true_label" in pkt));
    }
  }
});

// -----------------------------------------------------------------------------
// SCENARIO 5: Malformed or incomplete JSON is rejected by validateSimEnvelope
// -----------------------------------------------------------------------------
test("Scenario 5: validateSimEnvelope throws clear error on missing banner or missing keys", () => {
  assert.throws(
    () => validateSimEnvelope(null, "broken.json", ["headline_claims"]),
    /malformed/i
  );
  assert.throws(
    () =>
      validateSimEnvelope(
        { synthetic: true, banner: "WRONG BANNER" },
        "broken.json",
        []
      ),
    /banner/i
  );
  assert.throws(
    () =>
      validateSimEnvelope(
        {
          synthetic: true,
          banner: EXPECTED_SIM_BANNER,
          compression_factor: { gradual: 360, rapid: 1 },
          git_commit: "8e3cdde347",
        },
        "broken.json",
        ["missing_key"]
      ),
    /missing expected key "missing_key"/i
  );
});
