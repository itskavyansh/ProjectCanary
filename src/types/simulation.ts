export const EXPECTED_SIM_BANNER =
  "SYNTHETIC / SIMULATION - NOT VALIDATED ON REAL DATA";

export type SimTabId =
  | "overview"
  | "results"
  | "robustness"
  | "replay"
  | "parameters";

export type ReplayClassKey =
  | "normal"
  | "gradual"
  | "rapid"
  | "abnormal_vibration"
  | "sensor_fault"
  | "comm_fault";

export interface SimCommonEnvelope {
  synthetic: boolean;
  banner: string;
  compression_factor: {
    gradual: number;
    normal: number;
    abnormal_vibration?: number;
    sensor_fault?: number;
    comm_fault?: number;
    rapid: number;
    sensor_duration_s?: number;
    slow_process_hours_per_run?: number;
    sensor_hours_per_run?: number;
    note: string;
  };
  git_commit: string;
  seed: number;
  file_id: string;
}

export interface HeadlineClaim {
  id: string;
  title: string;
  number: Record<string, number | number[] | string | null>;
  display_value: string;
  source_file: string;
  secondary_source_file?: string;
  caveat: string;
}

export interface StudyLimitation {
  id: string;
  title: string;
  detail: string;
}

export interface SimSummaryData extends SimCommonEnvelope {
  study_title: string;
  study_description: string;
  headline_claims: HeadlineClaim[];
  limitations: StudyLimitation[];
  dataset_overview: {
    splits: Record<string, { runs: number; description: string }>;
    window_length_s: number;
    window_step_s: number;
    run_duration_s: number;
    windows_per_run: number;
    n_observable_features_v3: number;
  };
}

export interface WindowMetricRow {
  split: string;
  model: string;
  regime: string;
  n_windows: number;
  n_runs: number;
  window_accuracy: number | null;
  window_accuracy_ci_lo: number | null;
  window_accuracy_ci_hi: number | null;
  window_macro_f1: number | null;
  window_macro_f1_ci_lo: number | null;
  window_macro_f1_ci_hi: number | null;
  prec_normal: number | null;
  rec_normal: number | null;
  supp_normal: number;
  prec_gradual: number | null;
  rec_gradual: number | null;
  supp_gradual: number;
  prec_rapid: number | null;
  rec_rapid: number | null;
  supp_rapid: number;
  prec_abnormal_vibration: number | null;
  rec_abnormal_vibration: number | null;
  supp_abnormal_vibration: number;
  prec_sensor_fault: number | null;
  rec_sensor_fault: number | null;
  supp_sensor_fault: number;
  prec_comm_fault: number | null;
  rec_comm_fault: number | null;
  supp_comm_fault: number;
  gradual_run_detect_rate: number | null;
  gradual_run_detect_ci_lo: number | null;
  gradual_run_detect_ci_hi: number | null;
  rapid_run_detect_rate: number | null;
  rapid_run_detect_ci_lo: number | null;
  rapid_run_detect_ci_hi: number | null;
}

export interface V2VsV3AllSnrRow {
  split: string;
  version: string;
  regime: string;
  model: string;
  k: number;
  m: number;
  window_macro_f1: number;
  window_macro_f1_ci_lo: number;
  window_macro_f1_ci_hi: number;
  alarm_macro_f1: number;
  alarm_macro_f1_ci_lo: number;
  alarm_macro_f1_ci_hi: number;
  fa_episodes_per_normal_run: number | null;
  fa_windows_per_normal_run: number | null;
  alarm_exact_detect_rate: number;
  alarm_exact_detect_ci_lo: number;
  alarm_exact_detect_ci_hi: number;
  mean_delay_exact_s: number | null;
}

export interface V2VsV3SnrGt2Row {
  split: string;
  version: string;
  regime: string;
  model: string;
  n_runs: number;
  window_macro_f1: number;
  window_macro_f1_ci_lo: number;
  window_macro_f1_ci_hi: number;
  alarm_macro_f1: number;
  alarm_macro_f1_ci_lo: number;
  alarm_macro_f1_ci_hi: number;
  gradual_window_recall: number | null;
  rapid_window_recall: number | null;
  sensor_fault_window_recall: number | null;
  gradual_run_alarm_detect_rate: number | null;
  all_events_run_alarm_detect_rate: number | null;
  fa_episodes_per_normal_run: number | null;
  fa_windows_per_normal_run: number | null;
  total_gradual_sf_confusions: number;
}

export interface ClassShareRow {
  split: string;
  normal: number;
  gradual: number;
  rapid: number;
  sensor_fault: number;
  comm_fault: number;
  abnormal_vibration: number;
}

export interface StressScoringRow {
  model: string;
  primary_macro_f1: number;
  primary_macro_f1_ci_lo: number;
  primary_macro_f1_ci_hi: number;
  set_match_accuracy_all: number;
  n_multi_condition_windows: number;
  primary_accuracy_on_multi_windows: number;
  set_match_accuracy_on_multi_windows: number;
}

export interface SimResultsBySplitData extends SimCommonEnvelope {
  source_files: string[];
  caption_note: string;
  window_metrics_by_split: WindowMetricRow[];
  v2_vs_v3_all_snr: V2VsV3AllSnrRow[];
  v2_vs_v3_snr_gt_2: V2VsV3SnrGt2Row[];
  class_share_of_windows_per_split: ClassShareRow[];
  stress_set_scoring: StressScoringRow[];
}

export interface FalseAlarmRow {
  split: string;
  model: string;
  regime: string;
  k: number;
  m: number;
  n_normal_runs: number;
  n_event_runs: number;
  sensor_duration_s: number;
  compression_factor: number;
  hours_per_run_A_slow: number;
  hours_per_run_B_sensor: number;
  fa_episodes_per_run: number;
  fa_episodes_per_run_ci_lo: number;
  fa_episodes_per_run_ci_hi: number;
  A_slow_fa_episodes_per_hour: number;
  A_slow_ci_lo: number;
  A_slow_ci_hi: number;
  A_slow_one_fa_every_hours: number | null;
  B_sensor_fa_episodes_per_hour: number;
  B_sensor_ci_lo: number;
  B_sensor_ci_hi: number;
  B_sensor_one_fa_every_hours: number | null;
  fa_windows_per_run: number;
  A_slow_fa_windows_per_hour: number;
  B_sensor_fa_windows_per_hour: number;
  exact_detect_rate: number;
  exact_detect_ci_lo: number;
  exact_detect_ci_hi: number;
  missed_event_rate: number;
  missed_event_ci_lo: number;
  missed_event_ci_hi: number;
}

export interface SimResultsFalseAlarmsData extends SimCommonEnvelope {
  source_files: string[];
  definition_notes: {
    primary_display_order: string[];
    definition_B_sensor_time: string;
    definition_A_slow_process_time: string;
    zero_count_caveat: string;
  };
  false_alarms_table: FalseAlarmRow[];
}

export interface GradualRapidSnrRow {
  split: string;
  model: string;
  regime: string;
  class: string;
  snr_bin?: string;
  n_runs: number;
  n_windows: number;
  window_recall: number | null;
  window_recall_ci_lo: number | null;
  window_recall_ci_hi: number | null;
  run_detect_rate: number | null;
  run_detect_ci_lo: number | null;
  run_detect_ci_hi: number | null;
}

export interface Recall5BinRow {
  split: string;
  model: string;
  class: string;
  snr_bin: string;
  n_runs: number;
  n_windows: number;
  exact_recall: number;
  exact_recall_ci_lo: number;
  exact_recall_ci_hi: number;
  basin_recall: number;
  basin_recall_ci_lo: number;
  basin_recall_ci_hi: number;
}

export interface SimResultsRecallVsSnrData extends SimCommonEnvelope {
  source_files: string[];
  detection_limit_note: string;
  gradual_rapid_recall_by_snr_v3: GradualRapidSnrRow[];
  gradual_rapid_recall_and_detection_v3: GradualRapidSnrRow[];
  recall_vs_snr_5bins_test: Recall5BinRow[];
}

export interface RfVsThresholdRow {
  split: string;
  metric: string;
  rf_val: number;
  th_val: number;
  rf_minus_th: number;
  diff_ci_lo: number;
  diff_ci_hi: number;
  rf_fails_to_beat_th: boolean;
}

export interface PermGroupRow {
  group: string;
  feature_count: number;
  mean_importance: number;
  std_importance: number;
  min_importance: number;
  max_importance: number;
  repeats: number;
  rank: number;
}

export interface PermFeatureRow {
  feature: string;
  group: string;
  mean_importance: number;
  std_importance: number;
  min_importance: number;
  max_importance: number;
  repeats: number;
  rank: number;
}

export interface RfAblationV3Row {
  group_removed: string;
  n_features_kept: number;
  macro_f1_all_snr: number;
  delta_macro_f1_all_snr: number;
  macro_f1_snr_gt_2: number;
  delta_macro_f1_snr_gt_2: number;
  gradual_recall_snr_gt_2: number;
  sensor_fault_recall_snr_gt_2: number;
  gradual_sf_confusions_snr_gt_2: number;
}

export interface SimResultsModelComparisonData extends SimCommonEnvelope {
  source_files: string[];
  importance_caveat: string;
  rf_vs_threshold_comparison: RfVsThresholdRow[];
  permutation_importance_groups: PermGroupRow[];
  permutation_importance_top20: PermFeatureRow[];
  rf_ablations_v3: RfAblationV3Row[];
  rf_ablations_v1: Record<string, unknown>[];
}

export interface YawRow {
  split: string;
  condition: string;
  yaw_sd_deg: number | null;
  regime: string;
  model: string;
  model_version: string;
  n_runs: number;
  n_windows: number;
  window_macro_f1: number;
  window_macro_f1_ci_lo: number;
  window_macro_f1_ci_hi: number;
  alarm_macro_f1: number;
  alarm_macro_f1_ci_lo: number;
  alarm_macro_f1_ci_hi: number;
  fa_episodes_per_normal_run: number | null;
  fa_episodes_ci_lo: number | null;
  fa_episodes_ci_hi: number | null;
  fa_windows_per_normal_run: number | null;
  run_exact_detect_rate: number;
  run_exact_detect_ci_lo: number;
  run_exact_detect_ci_hi: number;
  gradual_to_sf_confusions: number;
  sf_to_gradual_confusions: number;
  gradual_sf_confusions: number;
  gradual_sf_confusions_ci_lo: number;
  gradual_sf_confusions_ci_hi: number;
}

export interface ShapeDetectionRow {
  shape: string;
  subset: string;
  model: string;
  regime: string;
  k: number;
  m: number;
  n_runs: number;
  exact_detect_rate: number;
  exact_detect_ci_lo: number;
  exact_detect_ci_hi: number;
  missed_event_rate: number;
  missed_event_ci_lo: number;
  missed_event_ci_hi: number;
}

export interface ShapeConfusionRow {
  shape: string;
  model: string;
  regime: string;
  n_gradual_windows: number;
  win_gradual_to_sensor_fault: number;
  win_gradual_to_sensor_fault_ci_lo: number;
  win_gradual_to_sensor_fault_ci_hi: number;
  win_gradual_to_sensor_fault_rate: number;
  alarm_gradual_to_sensor_fault: number;
  win_gradual_to_gradual: number;
  win_gradual_to_rapid: number;
  win_gradual_to_normal: number;
}

export interface ShapeConvergenceRow {
  shape: string;
  subset: string;
  n_windows: number;
  n_runs: number;
  dir_convergence_residual_m_p10: number;
  dir_convergence_residual_m_p50: number;
  dir_convergence_residual_m_p90: number;
  dir_convergence_residual_m_mean: number;
  dir_convergence_weighted_cos_p10: number;
  dir_convergence_weighted_cos_p50: number;
  dir_convergence_weighted_cos_p90: number;
  dir_convergence_weighted_cos_mean: number;
  dir_pairwise_cos_mean_p50: number;
}

export interface LitDetectionRow {
  arm: string;
  subset: string;
  model: string;
  regime: string;
  k: number;
  m: number;
  n_runs: number;
  exact_detect_rate: number;
  exact_detect_ci_lo: number;
  exact_detect_ci_hi: number;
  missed_event_rate: number;
  missed_event_ci_lo: number;
  missed_event_ci_hi: number;
}

export interface LitConfusionRow {
  arm: string;
  model: string;
  regime: string;
  n_gradual_runs: number;
  n_gradual_windows: number;
  win_gradual_to_sensor_fault: number;
  win_gradual_to_sensor_fault_rate: number;
  alarm_gradual_to_sensor_fault: number;
  win_gradual_to_gradual: number;
  win_gradual_to_rapid: number;
  win_gradual_to_normal: number;
}

export interface LitImpossibleRow {
  arm: string;
  n_runs: number;
  n_windows: number;
  max_window_node_mean_deg: number;
  median_run_max_node_mean_deg: number;
  runs_exceeding_15deg_normal_limit: number;
  windows_exceeding_15deg_normal_limit: number;
  runs_exceeding_3p5deg_threshold_rule_implausible: number;
  windows_exceeding_3p5deg_threshold_rule_implausible: number;
  validation_check_passed: boolean;
  validation_check_messages: string;
}

export interface LitDepthGridRow {
  cell_id: string;
  H_m: number;
  tan_beta: number;
  R_influence_m: number;
  sigma_m: number;
  sigma_radii: number;
  above_main_cap_1p5: boolean;
  placeholder_grid: boolean;
}

export interface TimescaleAuditRow {
  subset: string;
  n_gradual_runs: number;
  n_faster_than_0p1_deg_per_day: number;
  frac_faster_than_0p1_deg_per_day: number;
  rate_deg_per_day_min: number;
  rate_deg_per_day_p10: number;
  rate_deg_per_day_p50_median: number;
  rate_deg_per_day_mean: number;
  rate_deg_per_day_p90: number;
  rate_deg_per_day_max: number;
  factor_vs_0p1_min: number;
  factor_vs_0p1_p50_median: number;
  factor_vs_0p1_mean: number;
  factor_vs_0p1_p90: number;
  factor_vs_0p1_max: number;
}

export interface AltCompressionFaRow {
  model: string;
  regime: string;
  k: number;
  m: number;
  n_normal_runs: number;
  fa_episodes_per_run: number;
  fa_episodes_per_run_ci_lo: number;
  fa_episodes_per_run_ci_hi: number;
  B_sensor_fa_per_hour_1x: number;
  B_sensor_one_fa_every: string;
  A_slow_hours_per_run_360x: number;
  A_slow_fa_per_hour_360x: number;
  A_slow_one_fa_every_360x: string;
  A_slow_fa_per_hour_10000x: number;
  A_slow_one_fa_every_10000x: string;
  A_slow_fa_per_hour_10800x: number;
  A_slow_one_fa_every_10800x: string;
  A_slow_fa_per_hour_30000x: number;
  A_slow_one_fa_every_30000x: string;
}

export interface UnknownComplementRow {
  split_or_arm: string;
  classifier: string;
  detector: string;
  n_event_runs: number;
  clf_exact_detect_rate: number;
  det_only_flag_rate: number;
  n_clf_exact_missed: number;
  n_missed_flagged_by_det: number;
  frac_of_clf_exact_missed_flagged_by_det: number | null;
  combined_or_detect_rate: number;
  combined_or_detect_ci_lo: number;
  combined_or_detect_ci_hi: number;
  gated_unknown_detect_rate: number;
  gated_unknown_detect_ci_lo: number;
  gated_unknown_detect_ci_hi: number;
  clf_only_fa_per_normal_run: number | null;
  det_only_fa_per_normal_run: number | null;
  combined_or_fa_per_normal_run: number | null;
  extra_fa_per_normal_run_or: number | null;
  extra_B_sensor_fa_per_hour_or: number | null;
  extra_A_slow_fa_per_hour_or: number | null;
  extra_fa_per_normal_run_gated: number | null;
  extra_B_sensor_fa_per_hour_gated: number | null;
  total_fa_per_normal_run_with_gated: number | null;
}

export interface UnknownAurocRow {
  detector: string;
  class_vs_normal: string;
  n_class_windows: number;
  n_normal_windows: number;
  auroc: number;
  auroc_ci_lo: number;
  auroc_ci_hi: number;
  window_flag_recall_raw: number;
  window_flag_recall_persisted: number;
}

export interface SimRobustnessData extends SimCommonEnvelope {
  source_files: string[];
  yaw_sensitivity: {
    summary_note: string;
    all_snr: YawRow[];
    snr_gt_2: YawRow[];
  };
  shape_mismatch: {
    summary_note: string;
    detection_rates: ShapeDetectionRow[];
    gradual_sensor_fault_confusion: ShapeConfusionRow[];
    recall_by_snr: GradualRapidSnrRow[];
    convergence_distributions: ShapeConvergenceRow[];
    normal_false_alarms: Record<string, unknown>[];
  };
  literature_checks: {
    summary_note: string;
    detection_rates: LitDetectionRow[];
    gradual_sensor_fault_confusion: LitConfusionRow[];
    impossible_value_check: LitImpossibleRow[];
    realistic_depth_sigma_grid: LitDepthGridRow[];
  };
  timescale_audit: {
    summary_note: string;
    gradual_rate_summary: TimescaleAuditRow[];
    false_alarms_under_alt_compression: AltCompressionFaRow[];
  };
  unknown_detector_complement: {
    advisory: boolean;
    advisory_banner: string;
    thresholds: Record<
      string,
      {
        fitted_on_n_windows: number;
        threshold: number;
        threshold_quantile_val: number;
        val_fa_episodes_per_normal_run: number;
        val_fa_windows_per_normal_run: number;
      }
    >;
    complement_table: UnknownComplementRow[];
    run_level_detection_and_fa: Record<string, unknown>[];
    test_per_class_auroc: UnknownAurocRow[];
  };
}

export interface LiteratureParameterRow {
  parameter_or_topic: string;
  simulator_setting_and_status: string;
  literature_basis_and_sources: string;
  assessment_or_scope_note: string;
  is_placeholder: boolean;
  status_tag: string;
}

export interface UnsourcedParameterItem {
  item: string;
  status: string;
  literature_leads: string;
  impact: string;
}

export interface SimParametersData extends SimCommonEnvelope {
  source_file: string;
  literature_table: LiteratureParameterRow[];
  unsourced_items: UnsourcedParameterItem[];
  timescale_note: string;
}

export interface ReplayWindowRecord {
  window_idx: number;
  t_start: number;
  t_end: number;
  t_mid: number;
  t_rel_event_start_s: number | null;
  true_label: string;
  label_set: string;
  model_score: number;
  predicted_class_model_score: number;
  model_scores_by_class: Record<string, number>;
  predicted_class: string;
  predicted_class_raw: string;
  alarm_state: string;
  node_mean_tilt_deg: {
    NODE_01: number;
    NODE_02: number;
    NODE_03: number;
    NODE_04: number;
  };
  piezo_rms: {
    network_mean: number;
    NODE_01: number;
    NODE_02: number;
    NODE_03: number;
    NODE_04: number;
  };
  lidar_range_change_m: {
    hub_common_mode_m: number;
    hub_residual_abs_max_m: number;
    NODE_01: number;
    NODE_02: number;
    NODE_03: number;
    NODE_04: number;
  };
  comm_missing_fraction: {
    network: number;
    max_burst_len: number;
    NODE_01: number;
    NODE_02: number;
    NODE_03: number;
    NODE_04: number;
  };
  dir_convergence_residual_m: number;
  dir_convergence_weighted_cos: number;
  anomaly_score: number;
  unknown_anomaly_raw: boolean;
  unknown_anomaly: boolean;
}

export interface SimReplayData extends SimCommonEnvelope {
  run_class: ReplayClassKey;
  simulation_id: string;
  selection_rule: string;
  selected_rank_in_class_test_split: number;
  class_test_split_run_count: number;
  downsample_step: number;
  run_metadata: {
    seed: number;
    split: string;
    run_class: string;
    peak_tilt_deg: number;
    imu_noise_sd_deg: number;
    noise_multiplier: number;
    snr_vs_imu_noise: number;
    run_compression_factor: number;
    duration_s: number;
    n_windows: number;
    fault_types: string | null;
  };
  operating_point: {
    model: string;
    regime: string;
    k: number;
    m: number;
    group_thresholds: Record<string, number>;
    n_alarm_episodes_in_run: number;
    n_alarm_windows_in_run: number;
    unknown_detector_advisory: {
      detector: string;
      threshold: number;
      k: number;
      m: number;
      n_flagged_windows_in_run: number;
    };
  };
  windows: ReplayWindowRecord[];
}

export interface SimBundle {
  summary: SimSummaryData;
  resultsBySplit: SimResultsBySplitData;
  resultsFalseAlarms: SimResultsFalseAlarmsData;
  resultsRecallVsSnr: SimResultsRecallVsSnrData;
  resultsModelComparison: SimResultsModelComparisonData;
  robustness: SimRobustnessData;
  parameters: SimParametersData;
  replays: Record<ReplayClassKey, SimReplayData>;
}

export function validateSimEnvelope(
  raw: unknown,
  fileName: string,
  requiredKeys: string[]
): asserts raw is SimCommonEnvelope & Record<string, unknown> {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
    throw new Error(
      `Simulation export file "${fileName}" is malformed (expected a JSON object).`
    );
  }
  const obj = raw as Record<string, unknown>;
  if (obj.synthetic !== true) {
    throw new Error(
      `Simulation export file "${fileName}" is missing required "synthetic": true flag.`
    );
  }
  if (obj.banner !== EXPECTED_SIM_BANNER) {
    throw new Error(
      `Simulation export file "${fileName}" has missing or invalid safety banner.`
    );
  }
  if (!obj.compression_factor || typeof obj.compression_factor !== "object") {
    throw new Error(
      `Simulation export file "${fileName}" is missing "compression_factor" metadata.`
    );
  }
  if (typeof obj.git_commit !== "string" || obj.git_commit.trim().length < 7) {
    throw new Error(
      `Simulation export file "${fileName}" is missing valid "git_commit" hash.`
    );
  }
  for (const key of requiredKeys) {
    if (!(key in obj) || obj[key] === undefined || obj[key] === null) {
      throw new Error(
        `Simulation export file "${fileName}" is missing expected key "${key}".`
      );
    }
  }
}
