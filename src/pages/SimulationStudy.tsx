import { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate, useParams, useSearchParams } from "react-router-dom";
import {
  AlertTriangle,
  BarChart3,
  BookOpen,
  Compass,
  FileCode2,
  FlaskConical,
  Info,
  Layers,
  Pause,
  Play,
  RotateCcw,
  ShieldAlert,
  SkipBack,
  SkipForward,
  Sliders,
} from "lucide-react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { DataStatus } from "../components/DataStatus";
import { StatusBadge } from "../components/StatusBadge";
import type { SensorReading } from "../types/sensor";
import { deepSanitizeSim } from "./simulationText";
import {
  EXPECTED_SIM_BANNER,
  validateSimEnvelope,
  type ReplayClassKey,
  type SimBundle,
  type SimParametersData,
  type SimReplayData,
  type SimReplayExplanationsData,
  type SimResultsBySplitData,
  type SimResultsFalseAlarmsData,
  type SimResultsModelComparisonData,
  type SimResultsRecallVsSnrData,
  type SimRobustnessData,
  type SimSummaryData,
  type SimTabId,
} from "../types/simulation";

/* Simulation-local metric tile: no live-dashboard thresholds or severity colours. */
function MetricTile({
  label,
  value,
  unit,
  size = "md",
  decimals = 2,
}: {
  label: string;
  value?: number | null;
  unit: string;
  size?: "md" | "lg";
  decimals?: number;
}) {
  return (
    <div
      className={`flex h-full min-w-0 flex-col rounded-lg border border-[var(--border)] bg-[var(--surface-alt)] ${
        size === "lg" ? "px-4 py-3.5 sm:px-5 sm:py-4" : "px-3 py-2 sm:py-2.5"
      }`}
    >
      <p className="min-w-0 truncate text-[0.64rem] font-semibold uppercase leading-tight tracking-[0.035em] text-[var(--text-muted)]">
        {label}
      </p>
      <div className="mt-1 flex min-w-0 flex-wrap items-baseline gap-x-1 font-mono text-xl font-bold tracking-tight text-[var(--text-primary)] sm:text-2xl">
        {value != null ? (
          <span>{value.toFixed(decimals)}</span>
        ) : (
          <span className="text-[var(--text-disabled)]">N/A</span>
        )}
        <span className="font-sans text-xs font-semibold text-[var(--text-muted)] sm:text-sm">
          {unit}
        </span>
      </div>
    </div>
  );
}

interface SimTrendPoint {
  t: number;
  value: number;
}

/* Simulation-local single-node chart: x-axis is simulated seconds (0-240 s),
   only simulation reference lines are drawn. The shared TrendChart is untouched. */
function SimTrendChart({
  data,
  label,
  unit,
  color,
  reference,
}: {
  data: SimTrendPoint[];
  label: string;
  unit: string;
  color: string;
  reference?: { y: number; label: string };
}) {
  const gid = `sim-gradient-${label.replace(/[^A-Za-z0-9]/g, "")}`;
  return (
    <div className="min-w-0 rounded-lg border border-[var(--card-border)] bg-[var(--card-bg)] p-4 shadow-xs sm:p-5">
      <div className="mb-4 flex items-center justify-between gap-3">
        <h3 className="truncate text-sm font-bold tracking-tight text-[var(--text-primary)]">
          {label}
        </h3>
        <span
          className="h-2.5 w-2.5 shrink-0 rounded-full ring-2 ring-[var(--card-border)]"
          style={{ backgroundColor: color }}
        />
      </div>
      <div className="h-64 min-w-0">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 10, right: 28, left: 4, bottom: 0 }}>
            <defs>
              <linearGradient id={gid} x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor={color} stopOpacity={0.25} />
                <stop offset="95%" stopColor={color} stopOpacity={0.02} />
              </linearGradient>
            </defs>
            <CartesianGrid stroke="var(--chart-grid)" vertical={false} />
            <XAxis
              dataKey="t"
              type="number"
              domain={[0, 240]}
              ticks={[0, 60, 120, 180, 240]}
              tickFormatter={(v) => `${v} s`}
              tick={{ fontSize: 11, fill: "var(--chart-labels)" }}
            />
            <YAxis
              domain={["auto", "auto"]}
              tick={{ fontSize: 11, fill: "var(--chart-labels)" }}
              label={{
                value: unit,
                angle: -90,
                position: "insideLeft",
                style: { fill: "var(--chart-labels)", fontSize: 11, textAnchor: "middle" },
              }}
            />
            {reference && (
              <ReferenceLine
                y={reference.y}
                stroke="var(--status-warning)"
                strokeDasharray="3 3"
                label={{
                  position: "insideTopLeft",
                  value: reference.label,
                  fill: "var(--status-warning)",
                  fontSize: 10,
                }}
              />
            )}
            <Tooltip
              contentStyle={{
                backgroundColor: "var(--chart-tooltip-bg)",
                borderColor: "var(--chart-tooltip-border)",
                color: "var(--chart-tooltip-text)",
              }}
              labelFormatter={(v) => `t = ${v} s`}
              formatter={(v) => [`${v} ${unit}`, label]}
            />
            <Area
              type="monotone"
              dataKey="value"
              stroke={color}
              strokeWidth={2}
              fill={`url(#${gid})`}
              dot={false}
              isAnimationActive={false}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

const SIM_TABS: {
  id: SimTabId;
  label: string;
  shortDesc: string;
  icon: React.FC<{ className?: string }>;
}[] = [
  {
    id: "overview",
    label: "1. Overview",
    shortDesc: "Study scope, headline claims & caveats",
    icon: FlaskConical,
  },
  {
    id: "results",
    label: "2. Results",
    shortDesc: "Splits, false alarms (1x vs 360x), SNR & ablations",
    icon: BarChart3,
  },
  {
    id: "robustness",
    label: "3. Robustness",
    shortDesc: "Yaw, shape mismatch, depth/tilt & timescale audit",
    icon: Compass,
  },
  {
    id: "replay",
    label: "4. Replay",
    shortDesc: "6 test runs playback, telemetry & alarm timeline",
    icon: Play,
  },
  {
    id: "parameters",
    label: "5. Parameters",
    shortDesc: "Literature basis, sources & PLACEHOLDER audit",
    icon: BookOpen,
  },
];

const REPLAY_CLASSES: {
  key: ReplayClassKey;
  label: string;
  plainDesc: string;
}[] = [
  {
    key: "normal",
    label: "Normal Baseline",
    plainDesc: "No ground movement; includes benign wind/vehicle/temperature confounders",
  },
  {
    key: "gradual",
    label: "Gradual Subsidence",
    plainDesc: "Smooth S-curve settlement bowl (360x slow-process compression label)",
  },
  {
    key: "rapid",
    label: "Rapid Subsidence",
    plainDesc: "Steep onset settlement ramp (1x uncompressed sensor time)",
  },
  {
    key: "abnormal_vibration",
    label: "Abnormal Vibration",
    plainDesc: "Sustained multi-node high-frequency piezo impulse excitation",
  },
  {
    key: "sensor_fault",
    label: "Sensor Fault",
    plainDesc: "Hardware transducer fault (e.g. single-node IMU bias step / drift / hub stall)",
  },
  {
    key: "comm_fault",
    label: "Communication Fault",
    plainDesc: "LoRa packet drops, burst loss, node unavailability, or high latency",
  },
];

const MODEL_PLAIN_NAMES: Record<string, string> = {
  threshold_rule: "Threshold Rule (Hand-Built)",
  logistic_regression: "Logistic Regression",
  random_forest: "Random Forest (v3)",
};

const SPLIT_PLAIN_NAMES: Record<string, string> = {
  test: "Held-Out Test (120 runs)",
  shifted_test: "Shifted Noise & Bias Test (90 runs)",
  stress: "Combined Multi-Condition Stress (100 runs)",
  broad_basin: "Broad Basin Spread (40 runs)",
  val: "Validation Split (120 runs)",
  train: "Training Split (360 runs)",
};

const CLASS_COLORS: Record<string, string> = {
  normal: "var(--status-success)",
  transition: "var(--text-muted)",
  gradual: "var(--chart-series-3)",
  rapid: "var(--status-danger)",
  abnormal_vibration: "var(--chart-series-4)",
  sensor_fault: "var(--chart-series-5)",
  comm_fault: "var(--chart-series-1)",
};

function fmtNum(v: number | null | undefined, digits = 3): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "N/A";
  return v.toFixed(digits);
}

function fmtCi(
  pt: number | null | undefined,
  lo: number | null | undefined,
  hi: number | null | undefined,
  digits = 3
): string {
  if (pt === null || pt === undefined || Number.isNaN(pt)) return "N/A";
  if (
    lo === null ||
    lo === undefined ||
    hi === null ||
    hi === undefined ||
    Number.isNaN(lo) ||
    Number.isNaN(hi)
  ) {
    return pt.toFixed(digits);
  }
  return `${pt.toFixed(digits)} [${lo.toFixed(digits)}, ${hi.toFixed(digits)}]`;
}

function fmtEveryHours(hours: number | null | undefined): string {
  if (hours === null || hours === undefined || !Number.isFinite(hours)) {
    return "none observed";
  }
  if (hours < 1.0) {
    return `${(hours * 60).toFixed(1)} min`;
  }
  return `${hours.toFixed(1)} h`;
}

function mapSimClassToBadgeStatus(cls: string): SensorReading["status"] {
  if (cls === "normal" || cls === "transition") return "NORMAL";
  if (cls === "rapid") return "CRITICAL";
  return "WARNING";
}

async function fetchAndValidateSimJson<T>(
  fileName: string,
  requiredKeys: string[]
): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`/sim/${fileName}`);
  } catch (err) {
    throw new Error(
      `Failed to load simulation file "/sim/${fileName}": ${
        err instanceof Error ? err.message : "Network error"
      }`
    );
  }
  if (!response.ok) {
    throw new Error(
      `Missing simulation data file "/sim/${fileName}" (HTTP ${response.status}). Run "python simulation/scripts/export_dashboard_data.py" to regenerate.`
    );
  }
  const contentType = response.headers.get("content-type") || "";
  const text = await response.text();
  if (contentType.includes("text/html") || text.trim().startsWith("<!DOCTYPE")) {
    throw new Error(
      `Simulation data file "/sim/${fileName}" was not found (server returned HTML fallback instead of JSON).`
    );
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch (err) {
    throw new Error(
      `Malformed JSON in "/sim/${fileName}": ${
        err instanceof Error ? err.message : "Parse error"
      }`
    );
  }
  validateSimEnvelope(parsed, fileName, requiredKeys);
  return parsed as T;
}

/* ─── Persistent Top Safety Banner shown on every Simulation view ─── */
function SyntheticBanner({
  gitCommit,
  compressionNote,
}: {
  gitCommit?: string;
  compressionNote?: string;
}) {
  return (
    <div
      role="region"
      aria-label="Synthetic simulation disclaimer banner"
      className="mb-6 rounded-xl border-2 border-[var(--status-danger)] bg-[var(--alert-critical-bg)] p-4 shadow-xs"
    >
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2.5">
          <ShieldAlert
            className="h-5 w-5 shrink-0 text-[var(--status-danger)]"
            aria-hidden="true"
          />
          <span className="font-mono text-sm font-extrabold tracking-wide text-[var(--alert-critical-text)]">
            {EXPECTED_SIM_BANNER}
          </span>
        </div>
        <div className="flex flex-wrap items-center gap-2 text-xs font-mono">
          <span className="rounded border border-[var(--alert-critical-border)] bg-[var(--surface)] px-2 py-0.5 font-semibold text-[var(--text-primary)]">
            synthetic: true
          </span>
          <span className="rounded border border-[var(--alert-critical-border)] bg-[var(--surface)] px-2 py-0.5 font-semibold text-[var(--text-primary)]">
            Compression: 360x (Gradual/Normal) · 1x (Rapid & Sensor Noise)
          </span>
          {gitCommit && (
            <span className="rounded border border-[var(--alert-critical-border)] bg-[var(--surface)] px-2 py-0.5 text-[var(--text-secondary)]">
              Sim Commit: {gitCommit.slice(0, 8)}
            </span>
          )}
        </div>
      </div>
      <p className="mt-2 text-xs leading-relaxed text-[var(--text-primary)]">
        <strong>Read-Only Offline Benchmark Archive:</strong> Every metric, chart, and
        replay run in this section comes from mathematical surface equations and
        synthetic sensor/network models in <code className="font-mono">ProjectCanary-Sim</code>{" "}
        (frozen tag <code className="font-mono">v1.0-clean</code>). None of these
        results represent live field telemetry or a real mine warning.{" "}
        {compressionNote ?? ""}
      </p>
    </div>
  );
}

export function SimulationStudy() {
  const { tab: routeTab } = useParams<{ tab?: string }>();
  const [searchParams, setSearchParams] = useSearchParams();
  const navigate = useNavigate();

  const activeTab: SimTabId = useMemo(() => {
    const candidate = (routeTab || searchParams.get("tab") || "overview").toLowerCase();
    if (
      candidate === "overview" ||
      candidate === "results" ||
      candidate === "robustness" ||
      candidate === "replay" ||
      candidate === "parameters"
    ) {
      return candidate;
    }
    return "overview";
  }, [routeTab, searchParams]);

  const handleSelectTab = (tabId: SimTabId) => {
    navigate(`/simulation/${tabId}`);
  };

  const [bundle, setBundle] = useState<SimBundle | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Interactive filters for Results view
  const [selectedSplit, setSelectedSplit] = useState<string>("test");
  const [selectedRegime, setSelectedRegime] = useState<string>(
    "episode_fa<=1 (k=3,m=3)"
  );

  // Interactive filters for Robustness view
  const [yawSnrMode, setYawSnrMode] = useState<"snr_gt_2" | "all_snr">("snr_gt_2");
  const [yawSplit, setYawSplit] = useState<string>("test");

  // Interactive state for Replay view
  const [selectedReplayClass, setSelectedReplayClass] =
    useState<ReplayClassKey>("gradual");
  const [playbackIdx, setPlaybackIdx] = useState<number>(45);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(1);
  const [selectedTrendNode, setSelectedTrendNode] = useState<
    "NODE_01" | "NODE_02" | "NODE_03" | "NODE_04"
  >("NODE_01");

  const loadAllSimData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      if (searchParams.get("simulateError") === "malformed") {
        throw new Error(
          'Malformed JSON in "/sim/summary.json": Unexpected token in JSON at position 0 (simulated diagnostic check).'
        );
      }
      const [
        summary,
        resultsBySplit,
        resultsFalseAlarms,
        resultsRecallVsSnr,
        resultsModelComparison,
        robustness,
        parameters,
        repNormal,
        repGradual,
        repRapid,
        repVib,
        repSensor,
        repComm,
        expNormal,
        expGradual,
        expRapid,
        expVib,
        expSensor,
        expComm,
      ] = await Promise.all([
        fetchAndValidateSimJson<SimSummaryData>("summary.json", [
          "headline_claims",
          "limitations",
          "dataset_overview",
        ]),
        fetchAndValidateSimJson<SimResultsBySplitData>("results_by_split.json", [
          "window_metrics_by_split",
          "v2_vs_v3_all_snr",
          "v2_vs_v3_snr_gt_2",
          "class_share_of_windows_per_split",
        ]),
        fetchAndValidateSimJson<SimResultsFalseAlarmsData>(
          "results_false_alarms.json",
          ["false_alarms_table", "definition_notes"]
        ),
        fetchAndValidateSimJson<SimResultsRecallVsSnrData>(
          "results_recall_vs_snr.json",
          [
            "gradual_rapid_recall_by_snr_v3",
            "gradual_rapid_recall_and_detection_v3",
            "recall_vs_snr_5bins_test",
          ]
        ),
        fetchAndValidateSimJson<SimResultsModelComparisonData>(
          "results_model_comparison.json",
          [
            "rf_vs_threshold_comparison",
            "permutation_importance_groups",
            "rf_ablations_v3",
          ]
        ),
        fetchAndValidateSimJson<SimRobustnessData>("robustness.json", [
          "yaw_sensitivity",
          "shape_mismatch",
          "literature_checks",
          "timescale_audit",
          "unknown_detector_complement",
        ]),
        fetchAndValidateSimJson<SimParametersData>("parameters.json", [
          "literature_table",
          "unsourced_items",
          "timescale_note",
        ]),
        fetchAndValidateSimJson<SimReplayData>("replay_normal.json", [
          "windows",
          "run_metadata",
          "selection_rule",
        ]),
        fetchAndValidateSimJson<SimReplayData>("replay_gradual.json", [
          "windows",
          "run_metadata",
          "selection_rule",
        ]),
        fetchAndValidateSimJson<SimReplayData>("replay_rapid.json", [
          "windows",
          "run_metadata",
          "selection_rule",
        ]),
        fetchAndValidateSimJson<SimReplayData>(
          "replay_abnormal_vibration.json",
          ["windows", "run_metadata", "selection_rule"]
        ),
        fetchAndValidateSimJson<SimReplayData>("replay_sensor_fault.json", [
          "windows",
          "run_metadata",
          "selection_rule",
        ]),
        fetchAndValidateSimJson<SimReplayData>("replay_comm_fault.json", [
          "windows",
          "run_metadata",
          "selection_rule",
        ]),
        fetchAndValidateSimJson<SimReplayExplanationsData>("explanations_normal.json", [
          "explanations",
          "run_class",
        ]),
        fetchAndValidateSimJson<SimReplayExplanationsData>("explanations_gradual.json", [
          "explanations",
          "run_class",
        ]),
        fetchAndValidateSimJson<SimReplayExplanationsData>("explanations_rapid.json", [
          "explanations",
          "run_class",
        ]),
        fetchAndValidateSimJson<SimReplayExplanationsData>(
          "explanations_abnormal_vibration.json",
          ["explanations", "run_class"]
        ),
        fetchAndValidateSimJson<SimReplayExplanationsData>("explanations_sensor_fault.json", [
          "explanations",
          "run_class",
        ]),
        fetchAndValidateSimJson<SimReplayExplanationsData>("explanations_comm_fault.json", [
          "explanations",
          "run_class",
        ]),
      ]);

      setBundle({
        summary,
        resultsBySplit,
        resultsFalseAlarms,
        resultsRecallVsSnr,
        resultsModelComparison,
        robustness,
        parameters: deepSanitizeSim(parameters),
        replays: {
          normal: repNormal,
          gradual: repGradual,
          rapid: repRapid,
          abnormal_vibration: repVib,
          sensor_fault: repSensor,
          comm_fault: repComm,
        },
        explanations: {
          normal: expNormal,
          gradual: expGradual,
          rapid: expRapid,
          abnormal_vibration: expVib,
          sensor_fault: expSensor,
          comm_fault: expComm,
        },
      });
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Failed to load offline simulation study JSON files."
      );
    } finally {
      setLoading(false);
    }
  }, [searchParams]);

  useEffect(() => {
    void loadAllSimData();
  }, [loadAllSimData]);

  // Playback timer for Replay tab
  const currentReplay: SimReplayData | null = useMemo(() => {
    if (!bundle) return null;
    return bundle.replays[selectedReplayClass];
  }, [bundle, selectedReplayClass]);

  useEffect(() => {
    if (!isPlaying || !currentReplay) return;
    const maxIdx = currentReplay.windows.length - 1;
    const intervalMs = Math.max(120, Math.round(450 / playbackSpeed));
    const timer = window.setInterval(() => {
      setPlaybackIdx((prev) => {
        if (prev >= maxIdx) {
          setIsPlaying(false);
          return maxIdx;
        }
        return prev + 1;
      });
    }, intervalMs);
    return () => window.clearInterval(timer);
  }, [isPlaying, currentReplay, playbackSpeed]);

  // Per-node series in simulated seconds for the simulation-local trend charts
  const trendChartHistory = useMemo(() => {
    if (!currentReplay) return [];
    return currentReplay.windows.slice(0, playbackIdx + 1).map((w) => ({
      t: w.t_mid,
      tilt: w.node_mean_tilt_deg[selectedTrendNode] ?? 0,
      piezo: w.piezo_rms[selectedTrendNode] ?? 0,
      lidarCm: Number(
        (Math.abs(w.lidar_range_change_m[selectedTrendNode] ?? 0) * 100).toFixed(3)
      ),
    }));
  }, [currentReplay, playbackIdx, selectedTrendNode]);

  return (
    <main id="main-content" className="dashboard-shell py-8 lg:py-10">
      {/* Persistent safety banner required on every simulation view */}
      <SyntheticBanner
        gitCommit={bundle?.summary.git_commit}
        compressionNote={bundle?.summary.compression_factor.note}
      />

      {/* Page Header & Sub-View Navigation */}
      <header className="mb-8 flex flex-col gap-5">
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <p className="eyebrow mb-1.5">
              Offline Synthetic Benchmark Archive (v1.0-clean · v3 Models)
            </p>
            <h1 className="page-title">
              Simulation Study & Sensitivity Analysis
            </h1>
            <p className="mt-2 max-w-3xl text-sm leading-6 text-[var(--text-secondary)]">
              Read-only evaluation of the 4-corner-node (MPU6050 IMU + ADS1115
              piezo) + central LiDAR hub chunk architecture across 6 synthetic
              classes, 3 models, installation-yaw errors, non-Gaussian trough/slope
              geometries, and literature parameter ranges.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <span className="rounded-full border border-[var(--border)] bg-[var(--surface-alt)] px-3 py-1 font-mono text-xs font-bold tracking-wide text-[var(--text-secondary)]">
              STATIC ARCHIVE
            </span>
            <span className="font-mono text-xs text-[var(--text-muted)]">
              Static JSON · Zero Live Network Calls
            </span>
          </div>
        </div>

        {/* Sub-view tab bar */}
        <nav
          aria-label="Simulation study views"
          className="panel-surface grid grid-cols-1 gap-1.5 rounded-xl p-2 sm:grid-cols-2 lg:grid-cols-5"
        >
          {SIM_TABS.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                type="button"
                onClick={() => handleSelectTab(tab.id)}
                aria-current={isActive ? "page" : undefined}
                className={`flex flex-col items-start gap-1 rounded-lg px-3.5 py-2.5 text-left transition ${
                  isActive
                    ? "bg-[var(--brand-green)] text-white shadow-xs"
                    : "bg-[var(--surface-alt)] text-[var(--text-primary)] hover:bg-[var(--surface-muted)]"
                }`}
              >
                <span className="flex items-center gap-2 text-xs font-bold tracking-wide">
                  <Icon className="h-4 w-4 shrink-0" />
                  {tab.label}
                </span>
                <span
                  className={`text-[0.68rem] leading-tight ${
                    isActive ? "text-white/85" : "text-[var(--text-muted)]"
                  }`}
                >
                  {tab.shortDesc}
                </span>
              </button>
            );
          })}
        </nav>
      </header>

      <DataStatus
        loading={loading}
        error={error}
        loadingLabel="Loading frozen simulation benchmark JSON files…"
      />

      {error && (
        <div className="mt-4 flex justify-center">
          <button
            type="button"
            onClick={() => {
              setSearchParams({});
              void loadAllSimData();
            }}
            className="btn-secondary text-xs"
          >
            Retry Loading Static JSON Files
          </button>
        </div>
      )}

      {!loading && !error && bundle && (
        <div className="space-y-8">
          {activeTab === "overview" && (
            <OverviewSection summary={bundle.summary} />
          )}
          {activeTab === "results" && (
            <ResultsSection
              bySplit={bundle.resultsBySplit}
              falseAlarms={bundle.resultsFalseAlarms}
              recallSnr={bundle.resultsRecallVsSnr}
              modelComp={bundle.resultsModelComparison}
              selectedSplit={selectedSplit}
              onSelectSplit={setSelectedSplit}
              selectedRegime={selectedRegime}
              onSelectRegime={setSelectedRegime}
            />
          )}
          {activeTab === "robustness" && (
            <RobustnessSection
              robustness={bundle.robustness}
              yawSnrMode={yawSnrMode}
              onSelectYawSnrMode={setYawSnrMode}
              yawSplit={yawSplit}
              onSelectYawSplit={setYawSplit}
            />
          )}
          {activeTab === "replay" && currentReplay && (
            <ReplaySection
              replay={currentReplay}
              explanationsData={bundle.explanations[selectedReplayClass]}
              selectedClass={selectedReplayClass}
              onSelectClass={(cls) => {
                setIsPlaying(false);
                setSelectedReplayClass(cls);
                const firstExpWin =
                  bundle.explanations[cls]?.explanations[0]?.window_idx ?? 45;
                setPlaybackIdx(firstExpWin);
              }}
              playbackIdx={playbackIdx}
              onChangePlaybackIdx={(idx) => {
                setIsPlaying(false);
                setPlaybackIdx(idx);
              }}
              isPlaying={isPlaying}
              onTogglePlay={() => {
                if (
                  !isPlaying &&
                  playbackIdx >= currentReplay.windows.length - 1
                ) {
                  setPlaybackIdx(0);
                }
                setIsPlaying((p) => !p);
              }}
              playbackSpeed={playbackSpeed}
              onChangeSpeed={setPlaybackSpeed}
              selectedTrendNode={selectedTrendNode}
              onSelectTrendNode={setSelectedTrendNode}
              trendChartHistory={trendChartHistory}
            />
          )}
          {activeTab === "parameters" && (
            <ParametersSection parameters={bundle.parameters} />
          )}
        </div>
      )}
    </main>
  );
}

/* ══════════════════════════════════════════════════════════════════════════ */
/* VIEW 1: OVERVIEW                                                           */
/* ══════════════════════════════════════════════════════════════════════════ */
function OverviewSection({ summary }: { summary: SimSummaryData }) {
  const ds = summary.dataset_overview;
  const totalRuns = Object.values(ds.splits).reduce((acc, s) => acc + s.runs, 0);

  const findClaim = (id: string) =>
    summary.headline_claims.find((c) => c.id === id);

  const claimDet = findClaim("detection_limit_5x_noise");
  const claimRf = findClaim("v3_vs_threshold_rule");
  const claimFa = findClaim("false_alarms_both_time_definitions");
  const claimGrad = findClaim("gradual_window_recall_low");
  const claimRun = findClaim("run_level_detection_counts_any_alarm");
  const claimYaw = findClaim("yaw_tolerance_10_deg");

  const snrMult = Number(claimDet?.number.snr_threshold_multiple ?? 0);
  const minTiltDeg = Number(claimDet?.number.approx_min_peak_tilt_deg ?? 0);
  const imuNoiseDeg = Number(claimDet?.number.imu_noise_floor_deg ?? 0);

  const rfTestAlarmF1 = Number(claimRf?.number.test_alarm_macro_f1_rf_v3 ?? 0);
  const thTestAlarmF1 = Number(
    claimRf?.number.test_alarm_macro_f1_threshold_v3 ?? 0
  );

  const rfSensorFaPerHour = Number(
    claimFa?.number.rf_B_sensor_fa_per_hour ?? 0
  );
  const rfSensorEveryMin = Number(
    claimFa?.number.rf_B_sensor_every_minutes ?? 0
  );
  const thSensorFaPerHour = Number(
    claimFa?.number.threshold_B_sensor_fa_per_hour ?? 0
  );

  const gradWinRecallRf = Number(
    claimGrad?.number.test_episode_gradual_window_recall_rf ?? 0
  );
  const gradWinRecallTh = Number(
    claimGrad?.number.test_episode_gradual_window_recall_threshold ?? 0
  );
  const gradRunDetectRf = Number(
    claimRun?.number.test_episode_gradual_run_detect_rf ?? 0
  );
  const yawTolDeg = Number(
    claimYaw?.number.recommended_yaw_sd_tolerance_deg ?? 0
  );

  return (
    <div className="space-y-8">
      {/* Start here guidance line */}
      <p
        data-testid="start-here-overview"
        className="rounded-lg border border-[var(--border)] bg-[var(--surface-alt)] px-4 py-2.5 text-xs font-medium text-[var(--text-primary)]"
      >
        <strong className="font-bold text-[var(--brand-green)]">
          Start here:
        </strong>{" "}
        Read the One-Screen Summary (3 headline numbers & 3 key limits) below,
        then scan the 7 Headline Claim Cards before opening any collapsed
        architecture or limitation details.
      </p>

      {/* One-Screen Summary: 3 Headline Numbers + 3 Key Limits (all taken from summary.json) */}
      <section
        aria-label="One-screen summary of simulation study"
        data-testid="overview-one-screen-summary"
        className="panel-surface rounded-2xl border-2 border-[var(--brand-teal)] p-5 sm:p-6 space-y-5"
      >
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div>
            <p className="eyebrow">One-Screen Summary (From summary.json)</p>
            <h2 className="section-title mt-1">
              3 Headline Numbers &amp; 3 Core Limits
            </h2>
          </div>
          <span className="rounded-md border border-[var(--border)] bg-[var(--surface-alt)] px-3 py-1 font-mono text-xs font-semibold text-[var(--text-primary)]">
            {summary.banner}
          </span>
        </div>

        {/* Three Headline Numbers */}
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
          <div className="rounded-xl border border-[var(--border)] bg-[var(--surface-alt)] p-4 space-y-2">
            <span className="font-mono text-xs font-bold uppercase text-[var(--brand-teal)]">
              1. Detection Limit
            </span>
            <div className="font-mono text-2xl font-extrabold text-[var(--text-primary)]">
              ~{snrMult.toFixed(0)}× noise (~{minTiltDeg.toFixed(2)}° tilt)
            </div>
            <p className="text-xs leading-relaxed text-[var(--text-secondary)]">
              Reliable subsidence detection requires a peak surface tilt of
              about {snrMult.toFixed(0)}× the {imuNoiseDeg.toFixed(2)}° IMU
              noise floor (about {minTiltDeg.toFixed(2)}° tilt).
            </p>
          </div>

          <div className="rounded-xl border border-[var(--border)] bg-[var(--surface-alt)] p-4 space-y-2">
            <span className="font-mono text-xs font-bold uppercase text-[var(--brand-teal)]">
              2. Test Alarm Macro-F1 (k=3, m=3)
            </span>
            <div className="font-mono text-2xl font-extrabold text-[var(--text-primary)]">
              RF {rfTestAlarmF1.toFixed(4)} vs. Rule {thTestAlarmF1.toFixed(4)}
            </div>
            <p className="text-xs leading-relaxed text-[var(--text-secondary)]">
              Random Forest ({rfTestAlarmF1.toFixed(4)} alarm macro-F1) is
              comparable to the hand-built threshold rule (
              {thTestAlarmF1.toFixed(4)} alarm macro-F1) on in-domain test runs,
              with gains on broad basins and multi-fault stress runs.
            </p>
          </div>

          <div className="rounded-xl border border-[var(--border)] bg-[var(--surface-alt)] p-4 space-y-2">
            <span className="font-mono text-xs font-bold uppercase text-[var(--brand-teal)]">
              3. False Alarms / Hour (Sensor Time, k=3, m=3)
            </span>
            <div className="font-mono text-2xl font-extrabold text-[var(--status-danger)]">
              {rfSensorFaPerHour.toFixed(2)} FA/h (1 every{" "}
              {rfSensorEveryMin.toFixed(1)} min)
            </div>
            <p className="text-xs leading-relaxed text-[var(--text-secondary)]">
              At the primary k=3, m=3 operating point on test normal runs,
              false alarms occur at {rfSensorFaPerHour.toFixed(2)} per hour in
              1× sensor time for Random Forest ({thSensorFaPerHour.toFixed(2)}{" "}
              per hour for the threshold rule).
            </p>
          </div>
        </div>

        {/* Three Limits */}
        <div className="rounded-xl border border-[var(--status-warning-border)] bg-[var(--status-warning-bg)] p-4 space-y-2.5">
          <h3 className="font-mono text-xs font-bold uppercase tracking-wider text-[var(--status-warning)]">
            Three Mandatory Study Limits
          </h3>
          <ul className="grid grid-cols-1 gap-3 text-xs leading-relaxed text-[var(--text-primary)] lg:grid-cols-3">
            <li className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-3">
              <strong>1. All synthetic and not validated:</strong>{" "}
              {summary.banner} — all ground surfaces, sensor readings, and fault
              injections are synthetic mathematical simulations, not field data.
            </li>
            <li className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-3">
              <strong>2. Gradual window recall is low:</strong> Per-window
              gradual recall on test at k=3, m=3 is only{" "}
              {gradWinRecallRf.toFixed(3)} (RF) and {gradWinRecallTh.toFixed(3)}{" "}
              (threshold rule), whereas run-level detection (
              {gradRunDetectRf.toFixed(3)} RF) counts any single correct
              persisted alarm during the run.
            </li>
            <li className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-3">
              <strong>3. Direction features need 4 nodes &amp; ≤{yawTolDeg.toFixed(0)}° yaw:</strong>{" "}
              World-frame radial convergence requires all four corner nodes and
              installation compass yaw alignment within{" "}
              {yawTolDeg.toFixed(0)}° SD (with no single-node ≥45° blunder).
            </li>
          </ul>
        </div>
      </section>

      {/* Headline Claims with Source File Paths and One-Line Caveats (Always Visible) */}
      <section className="space-y-4">
        <div>
          <p className="eyebrow">Headline Findings (Frozen v3 Evaluation)</p>
          <h2 className="section-title mt-1">
            7 Key Claims, Exact Numbers, Sources &amp; Mandatory Caveats
          </h2>
        </div>

        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          {summary.headline_claims.map((claim, idx) => (
            <article
              key={claim.id}
              className="panel-surface flex flex-col justify-between rounded-xl p-5"
            >
              <div>
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <span className="inline-flex items-center gap-1.5 rounded-full bg-[var(--brand-teal-bg)] px-2.5 py-0.5 font-mono text-xs font-bold text-[var(--brand-teal)]">
                    Claim #{idx + 1} · {claim.id}
                  </span>
                  <span className="inline-flex items-center gap-1 rounded border border-[var(--border)] bg-[var(--surface-alt)] px-2 py-0.5 font-mono text-[0.68rem] text-[var(--text-muted)]">
                    <FileCode2 className="h-3 w-3" />
                    {claim.source_file}
                  </span>
                </div>

                <h3 className="mt-3 text-base font-bold text-[var(--text-primary)]">
                  {claim.title}
                </h3>

                <div className="mt-2.5 rounded-lg border border-[var(--border)] bg-[var(--surface-alt)] px-3.5 py-2.5 font-mono text-xs font-bold text-[var(--brand-green)]">
                  {claim.display_value}
                </div>

                <div className="mt-3 grid grid-cols-2 gap-2 text-xs">
                  {Object.entries(claim.number).map(([k, v]) => (
                    <div
                      key={k}
                      className="rounded border border-[var(--border)] bg-[var(--surface)] px-2.5 py-1.5"
                    >
                      <span className="block font-mono text-[0.65rem] text-[var(--text-muted)]">
                        {k}
                      </span>
                      <span className="font-mono font-semibold text-[var(--text-primary)]">
                        {Array.isArray(v) ? `[${v.join(", ")}]` : String(v)}
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="mt-4 rounded-lg border-l-4 border-l-[var(--status-warning)] bg-[var(--status-warning-bg)] px-3 py-2 text-xs text-[var(--text-primary)]">
                <strong className="font-semibold text-[var(--status-warning)]">
                  Caveat:{" "}
                </strong>
                {claim.caveat}
              </div>
            </article>
          ))}
        </div>
      </section>

      {/* Secondary details blocks collapsed by default */}
      <details className="panel-surface rounded-2xl p-5 sm:p-6">
        <summary className="cursor-pointer font-mono text-xs font-bold text-[var(--text-primary)]">
          What is the benchmark architecture, dataset split breakdown, and live-demo threshold scale difference? (Click to expand)
        </summary>
        <div className="mt-5 space-y-6">
          <div>
            <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
              <div>
                <p className="eyebrow">Benchmark Architecture</p>
                <h2 className="section-title mt-1">{summary.study_title}</h2>
              </div>
              <span className="rounded-md border border-[var(--border)] bg-[var(--surface-alt)] px-3 py-1 font-mono text-xs font-semibold text-[var(--text-primary)]">
                Seed: {summary.seed} · Commit: {summary.git_commit.slice(0, 8)}
              </span>
            </div>
            <p className="text-sm leading-relaxed text-[var(--text-secondary)]">
              {summary.study_description}
            </p>

            <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
              <MetricTile
                label="Total Benchmark Runs"
                value={totalRuns}
                unit="runs"
                decimals={0}
              />
              <MetricTile
                label="Observable v3 Features"
                value={ds.n_observable_features_v3}
                unit="cols"
                decimals={0}
              />
              <MetricTile
                label="Window Length / Step"
                value={ds.window_length_s}
                unit={`s (${ds.window_step_s}s step)`}
                decimals={1}
              />
              <MetricTile
                label="Windows per Run"
                value={ds.windows_per_run}
                unit="windows"
                decimals={0}
              />
              <MetricTile
                label="Detection SNR Limit"
                value={snrMult}
                unit="× IMU noise"
                decimals={1}
              />
              <MetricTile
                label="Compass Yaw Tolerance"
                value={yawTolDeg}
                unit="° SD max"
                decimals={0}
              />
            </div>
          </div>

          <div className="rounded-xl border border-[var(--status-warning-border)] bg-[var(--status-warning-bg)] p-4">
            <div className="flex items-start gap-3">
              <AlertTriangle
                className="mt-0.5 h-5 w-5 shrink-0 text-[var(--status-warning)]"
                aria-hidden="true"
              />
              <div className="space-y-1.5 text-xs leading-relaxed text-[var(--text-primary)]">
                <h3 className="text-sm font-bold text-[var(--text-primary)]">
                  Important Scale Difference: Live Demo Engine vs. Simulation
                  Study Thresholds
                </h3>
                <p>
                  The live dashboard demonstration engine (
                  <code className="font-mono">engine/thresholds.py</code> and{" "}
                  <code className="font-mono">
                    src/services/analysisAdapter.ts
                  </code>
                  ) uses{" "}
                  <strong>
                    bench-demo tilt thresholds of 15.0° (Warning) and 25.0°
                    (Critical)
                  </strong>{" "}
                  so tilting a sensor by hand on a desk visibly triggers alerts.
                  In contrast,{" "}
                  <code className="font-mono">ProjectCanary-Sim</code> models
                  early-warning ground subsidence tilts in{" "}
                  <strong>0.05° to 3.0°</strong> (about 1 to 52 mm/m; high-tilt
                  stress tests up to 12°), and the simulation&apos;s calibrated
                  Threshold Rule triggers at{" "}
                  <strong>~0.607° max node tilt</strong> (
                  <code className="font-mono">
                    basin_tilt_max_deg_th = 0.60727°
                  </code>
                  ). The existing live engine thresholds are intentionally left
                  unchanged.
                </p>
              </div>
            </div>
          </div>

          <div>
            <p className="eyebrow">Frozen Dataset Splits</p>
            <h3 className="subsection-title mt-1">
              Evaluation Splits &amp; Leakage Hygiene
            </h3>
            <p className="mt-1 text-xs text-[var(--text-secondary)]">
              All models, imputers, scalers, and operating points were fit on{" "}
              <code className="font-mono">train</code> and tuned on{" "}
              <code className="font-mono">val</code> only (
              <code className="font-mono">guard_no_test_data</code> enforced).
              No run or window crosses split boundaries.
            </p>
            <div className="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
              {Object.entries(ds.splits).map(([splitName, info]) => (
                <div
                  key={splitName}
                  className="rounded-xl border border-[var(--border)] bg-[var(--surface-alt)] p-3.5"
                >
                  <div className="flex items-center justify-between">
                    <span className="font-mono text-sm font-bold text-[var(--text-primary)]">
                      {splitName}
                    </span>
                    <span className="rounded bg-[var(--surface)] px-2 py-0.5 font-mono text-xs font-semibold text-[var(--brand-teal)]">
                      {info.runs} runs
                    </span>
                  </div>
                  <p className="mt-1.5 text-xs text-[var(--text-secondary)]">
                    {info.description}
                  </p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </details>

      {/* Study Limitations List inside details block */}
      <details className="panel-surface rounded-2xl p-5 sm:p-6">
        <summary className="cursor-pointer font-mono text-xs font-bold text-[var(--text-primary)]">
          Why are these synthetic results not field validation? (6 model &amp; simulation limitations)
        </summary>
        <div className="mt-4 grid grid-cols-1 gap-4 md:grid-cols-2">
          {summary.limitations.map((lim, i) => (
            <div
              key={lim.id}
              className="rounded-xl border border-[var(--border)] bg-[var(--surface-alt)] p-4"
            >
              <div className="flex items-center gap-2">
                <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-[var(--brand-green)] font-mono text-xs font-bold text-white">
                  {i + 1}
                </span>
                <h3 className="text-sm font-bold text-[var(--text-primary)]">
                  {lim.title}
                </h3>
              </div>
              <p className="mt-2 text-xs leading-relaxed text-[var(--text-secondary)]">
                {lim.detail}
              </p>
            </div>
          ))}
        </div>
      </details>
    </div>
  );
}

/* ══════════════════════════════════════════════════════════════════════════ */
/* VIEW 2: RESULTS                                                            */
/* ══════════════════════════════════════════════════════════════════════════ */
function ResultsSection({
  bySplit,
  falseAlarms,
  recallSnr,
  modelComp,
  selectedSplit,
  onSelectSplit,
  selectedRegime,
  onSelectRegime,
}: {
  bySplit: SimResultsBySplitData;
  falseAlarms: SimResultsFalseAlarmsData;
  recallSnr: SimResultsRecallVsSnrData;
  modelComp: SimResultsModelComparisonData;
  selectedSplit: string;
  onSelectSplit: (s: string) => void;
  selectedRegime: string;
  onSelectRegime: (r: string) => void;
}) {
  const splitRows = useMemo(
    () =>
      bySplit.window_metrics_by_split.filter(
        (r) => r.split === selectedSplit && r.regime === selectedRegime
      ),
    [bySplit.window_metrics_by_split, selectedSplit, selectedRegime]
  );

  const v3AllRows = useMemo(
    () =>
      bySplit.v2_vs_v3_all_snr.filter(
        (r) =>
          r.split === selectedSplit &&
          r.regime === selectedRegime &&
          r.version === "v3 (new features)"
      ),
    [bySplit.v2_vs_v3_all_snr, selectedSplit, selectedRegime]
  );

  const splitChartData = useMemo(() => {
    return ["test", "shifted_test", "stress", "broad_basin"].map((sp) => {
      const rows = bySplit.v2_vs_v3_all_snr.filter(
        (r) =>
          r.split === sp &&
          r.regime === selectedRegime &&
          r.version === "v3 (new features)"
      );
      const getVal = (m: string, col: "alarm_macro_f1" | "window_macro_f1") =>
        rows.find((r) => r.model === m)?.[col] ?? 0;
      return {
        split: sp,
        "Threshold Rule": getVal("threshold_rule", "alarm_macro_f1"),
        "Logistic Regression": getVal("logistic_regression", "alarm_macro_f1"),
        "Random Forest (v3)": getVal("random_forest", "alarm_macro_f1"),
      };
    });
  }, [bySplit.v2_vs_v3_all_snr, selectedRegime]);

  const faRows = useMemo(
    () =>
      falseAlarms.false_alarms_table.filter((r) =>
        ["test", "shifted_test"].includes(r.split)
      ),
    [falseAlarms.false_alarms_table]
  );

  const pooledSnrChartData = useMemo(() => {
    const bins = ["<1", "1-2", "2-5", ">5"];
    return bins.map((bin) => {
      const sub = recallSnr.gradual_rapid_recall_by_snr_v3.filter(
        (r) =>
          r.split.startsWith("pooled") &&
          r.regime === "episode_fa<=1 (k=3,m=3)" &&
          r.snr_bin === bin
      );
      const findMetric = (
        m: string,
        cls: string,
        field: "window_recall" | "run_detect_rate"
      ) => sub.find((r) => r.model === m && r.class === cls)?.[field] ?? 0;

      return {
        snr_bin: `SNR ${bin}`,
        "RF Gradual Window Recall": findMetric(
          "random_forest",
          "gradual",
          "window_recall"
        ),
        "RF Gradual Run Detect": findMetric(
          "random_forest",
          "gradual",
          "run_detect_rate"
        ),
        "RF Rapid Window Recall": findMetric(
          "random_forest",
          "rapid",
          "window_recall"
        ),
        "RF Rapid Run Detect": findMetric(
          "random_forest",
          "rapid",
          "run_detect_rate"
        ),
        "Threshold Gradual Run Detect": findMetric(
          "threshold_rule",
          "gradual",
          "run_detect_rate"
        ),
      };
    });
  }, [recallSnr.gradual_rapid_recall_by_snr_v3]);

  return (
    <div className="space-y-8">
      {/* Start here guidance line */}
      <p
        data-testid="start-here-results"
        className="rounded-lg border border-[var(--border)] bg-[var(--surface-alt)] px-4 py-2.5 text-xs font-medium text-[var(--text-primary)]"
      >
        <strong className="font-bold text-[var(--brand-green)]">
          Start here:
        </strong>{" "}
        Look at the Alarm Macro-F1 bar chart across splits first, then open the
        collapsed disclosures below to inspect per-class tables, 1× vs. 360×
        false alarms, SNR recall curves, or feature ablations.
      </p>

      {/* Split & Operating Regime Filter Bar */}
      <section className="panel-surface flex flex-wrap items-end justify-between gap-4 rounded-2xl p-4 sm:p-5">
        <div className="flex flex-wrap gap-4">
          <label className="block">
            <span className="eyebrow mb-1.5 block">Evaluation Split</span>
            <select
              value={selectedSplit}
              onChange={(e) => onSelectSplit(e.target.value)}
              className="rounded-lg border border-[var(--input-border)] bg-[var(--input-bg)] px-3 py-2 font-mono text-xs font-semibold text-[var(--input-text)]"
            >
              {(["test", "shifted_test", "stress", "broad_basin"] as const).map(
                (sp) => (
                  <option key={sp} value={sp}>
                    {SPLIT_PLAIN_NAMES[sp] || sp}
                  </option>
                )
              )}
            </select>
          </label>

          <label className="block">
            <span className="eyebrow mb-1.5 block">
              Alarm Persistence Operating Point
            </span>
            <select
              value={selectedRegime}
              onChange={(e) => onSelectRegime(e.target.value)}
              className="rounded-lg border border-[var(--input-border)] bg-[var(--input-bg)] px-3 py-2 font-mono text-xs font-semibold text-[var(--input-text)]"
            >
              <option value="episode_fa<=1 (k=3,m=3)">
                Episode-Matched FA &lt;= 1/run (k=3, m=3) — Primary v3
              </option>
              <option value="strict_window_fa<=1">
                Strict Window-Matched FA &lt;= 1/run (k=3/5, m=2)
              </option>
              <option value="raw (k=1)">Raw Unpersisted Window (k=1, m=1)</option>
            </select>
          </label>
        </div>

        <div className="text-xs text-[var(--text-muted)] font-mono">
          95% CIs via whole-run bootstrap (B = 300 replicates)
        </div>
      </section>

      {/* Section 2.1: Three Models by Split (Primary Chart visible + Large Table inside details) */}
      <section className="panel-surface rounded-2xl p-5 sm:p-6 space-y-5">
        <div>
          <p className="eyebrow">1. Window &amp; Alarm Metrics by Split</p>
          <h2 className="section-title mt-1">
            Threshold Rule vs. Logistic Regression vs. Random Forest (v3)
          </h2>
          <p className="mt-1 text-xs leading-relaxed text-[var(--text-secondary)]">
            {bySplit.caption_note}
          </p>
        </div>

        <div className="h-72 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={splitChartData}
              margin={{ top: 12, right: 20, left: 0, bottom: 8 }}
            >
              <CartesianGrid stroke="var(--chart-grid)" vertical={false} />
              <XAxis
                dataKey="split"
                tick={{
                  fontSize: 12,
                  fill: "var(--chart-labels)",
                  fontFamily: '"JetBrains Mono", monospace',
                }}
              />
              <YAxis
                domain={[0, 0.8]}
                tick={{
                  fontSize: 11,
                  fill: "var(--chart-labels)",
                  fontFamily: '"JetBrains Mono", monospace',
                }}
                label={{
                  value: "Alarm Macro-F1",
                  angle: -90,
                  position: "insideLeft",
                  style: { fill: "var(--chart-labels)", fontSize: 11 },
                }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "var(--chart-tooltip-bg)",
                  borderColor: "var(--chart-tooltip-border)",
                  color: "var(--chart-tooltip-text)",
                }}
              />
              <Legend />
              <Bar
                dataKey="Threshold Rule"
                fill="var(--chart-series-4)"
                radius={[4, 4, 0, 0]}
              />
              <Bar
                dataKey="Logistic Regression"
                fill="var(--chart-series-3)"
                radius={[4, 4, 0, 0]}
              />
              <Bar
                dataKey="Random Forest (v3)"
                fill="var(--chart-series-1)"
                radius={[4, 4, 0, 0]}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Table of Window Metrics & Per-Class Recall + Run Detection Side by Side */}
        <details className="rounded-xl border border-[var(--border)] bg-[var(--surface-alt)] p-3.5">
          <summary className="cursor-pointer font-mono text-xs font-bold text-[var(--text-primary)]">
            How do per-class precision, window recall, and run-level detection compare on {selectedSplit}?
          </summary>
          <div className="mt-3 overflow-x-auto rounded-xl border border-[var(--table-border)]">
            <table className="w-full text-left text-xs">
              <thead className="bg-[var(--table-header-bg)] font-mono text-[0.7rem] uppercase text-[var(--table-header-text)]">
                <tr>
                  <th className="px-3 py-2.5">Model ({selectedSplit})</th>
                  <th className="px-3 py-2.5">Window Macro-F1 [95% CI]</th>
                  <th className="px-3 py-2.5">Alarm Macro-F1 [95% CI]</th>
                  <th className="px-3 py-2.5">Normal P / R</th>
                  <th className="px-3 py-2.5">
                    Gradual: Win Recall vs Run Detect
                  </th>
                  <th className="px-3 py-2.5">Rapid: Win Recall vs Run Detect</th>
                  <th className="px-3 py-2.5">Sensor Fault P / R</th>
                  <th className="px-3 py-2.5">Comm Fault P / R</th>
                  <th className="px-3 py-2.5">Vibration P / R</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[var(--table-border)] font-mono">
                {splitRows.map((row) => {
                  const v3Row = v3AllRows.find((r) => r.model === row.model);
                  return (
                    <tr
                      key={`${row.split}-${row.model}-${row.regime}`}
                      className="bg-[var(--table-row-bg)] hover:bg-[var(--surface-alt)]"
                    >
                      <td className="px-3 py-2.5 font-sans font-bold text-[var(--text-primary)]">
                        {MODEL_PLAIN_NAMES[row.model] || row.model}
                      </td>
                      <td className="px-3 py-2.5">
                        {fmtCi(
                          row.window_macro_f1,
                          row.window_macro_f1_ci_lo,
                          row.window_macro_f1_ci_hi,
                          3
                        )}
                      </td>
                      <td className="px-3 py-2.5 font-semibold text-[var(--brand-green)]">
                        {v3Row
                          ? fmtCi(
                              v3Row.alarm_macro_f1,
                              v3Row.alarm_macro_f1_ci_lo,
                              v3Row.alarm_macro_f1_ci_hi,
                              3
                            )
                          : "N/A"}
                      </td>
                      <td className="px-3 py-2.5">
                        {fmtNum(row.prec_normal, 2)} / {fmtNum(row.rec_normal, 2)}
                      </td>
                      <td className="px-3 py-2.5">
                        <span className="font-bold text-[var(--status-warning)]">
                          {fmtNum(row.rec_gradual, 2)}
                        </span>{" "}
                        win /{" "}
                        <span className="font-bold text-[var(--brand-teal)]">
                          {fmtNum(row.gradual_run_detect_rate, 2)}
                        </span>{" "}
                        run
                      </td>
                      <td className="px-3 py-2.5">
                        <span className="font-bold text-[var(--status-warning)]">
                          {fmtNum(row.rec_rapid, 2)}
                        </span>{" "}
                        win /{" "}
                        <span className="font-bold text-[var(--brand-teal)]">
                          {fmtNum(row.rapid_run_detect_rate, 2)}
                        </span>{" "}
                        run
                      </td>
                      <td className="px-3 py-2.5">
                        {fmtNum(row.prec_sensor_fault, 2)} /{" "}
                        {fmtNum(row.rec_sensor_fault, 2)}
                      </td>
                      <td className="px-3 py-2.5">
                        {fmtNum(row.prec_comm_fault, 2)} /{" "}
                        {fmtNum(row.rec_comm_fault, 2)}
                      </td>
                      <td className="px-3 py-2.5">
                        {fmtNum(row.prec_abnormal_vibration, 2)} /{" "}
                        {fmtNum(row.rec_abnormal_vibration, 2)}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </details>
      </section>

      {/* Section 2.2: False Alarms per Hour — Collapsed inside details */}
      <details className="panel-surface rounded-2xl p-5 sm:p-6">
        <summary className="cursor-pointer font-mono text-xs font-bold text-[var(--text-primary)]">
          How many false alarms occur per hour in 1× sensor time (Definition B) versus 360× slow-process time (Definition A)?
        </summary>
        <div className="mt-4 space-y-4">
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <div className="rounded-xl border-l-4 border-l-[var(--status-danger)] bg-[var(--surface-alt)] p-4 text-xs">
              <h3 className="font-bold text-[var(--text-primary)]">
                Primary — Definition B: Sensor Time (1x = 240 s = 0.0667 h / run)
              </h3>
              <p className="mt-1 text-[var(--text-secondary)]">
                {falseAlarms.definition_notes.definition_B_sensor_time}
              </p>
            </div>
            <div className="rounded-xl border-l-4 border-l-[var(--brand-teal)] bg-[var(--surface-alt)] p-4 text-xs">
              <h3 className="font-bold text-[var(--text-primary)]">
                Secondary — Definition A: Slow-Process Time (360x = 24 h / run,
                PLACEHOLDER)
              </h3>
              <p className="mt-1 text-[var(--text-secondary)]">
                {falseAlarms.definition_notes.definition_A_slow_process_time}
              </p>
            </div>
          </div>

          <div className="overflow-x-auto rounded-xl border border-[var(--table-border)]">
            <table className="w-full text-left text-xs">
              <thead className="bg-[var(--table-header-bg)] font-mono text-[0.7rem] uppercase text-[var(--table-header-text)]">
                <tr>
                  <th className="px-3 py-2.5">Split</th>
                  <th className="px-3 py-2.5">Model</th>
                  <th className="px-3 py-2.5">Operating Regime</th>
                  <th className="px-3 py-2.5">FA Episodes / Normal Run</th>
                  <th className="px-3 py-2.5 text-[var(--status-danger)]">
                    Def B: FA / Hour (Sensor Time 1x)
                  </th>
                  <th className="px-3 py-2.5 text-[var(--status-danger)]">
                    Def B: 1 FA Every
                  </th>
                  <th className="px-3 py-2.5">
                    Def A: FA / Hour (Slow Time 360x)
                  </th>
                  <th className="px-3 py-2.5">Def A: 1 FA Every</th>
                  <th className="px-3 py-2.5">Missed-Event Rate</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[var(--table-border)] font-mono">
                {faRows
                  .filter((r) => r.regime !== "raw (k=1)")
                  .map((r) => (
                    <tr
                      key={`${r.split}-${r.model}-${r.regime}`}
                      className="bg-[var(--table-row-bg)] hover:bg-[var(--surface-alt)]"
                    >
                      <td className="px-3 py-2 font-bold">{r.split}</td>
                      <td className="px-3 py-2 font-sans font-semibold text-[var(--text-primary)]">
                        {MODEL_PLAIN_NAMES[r.model] || r.model}
                      </td>
                      <td className="px-3 py-2">{r.regime}</td>
                      <td className="px-3 py-2">
                        {fmtCi(
                          r.fa_episodes_per_run,
                          r.fa_episodes_per_run_ci_lo,
                          r.fa_episodes_per_run_ci_hi,
                          3
                        )}
                      </td>
                      <td className="px-3 py-2 font-bold text-[var(--status-danger)]">
                        {fmtCi(
                          r.B_sensor_fa_episodes_per_hour,
                          r.B_sensor_ci_lo,
                          r.B_sensor_ci_hi,
                          2
                        )}
                      </td>
                      <td className="px-3 py-2 font-bold text-[var(--status-danger)]">
                        {fmtEveryHours(r.B_sensor_one_fa_every_hours)}
                      </td>
                      <td className="px-3 py-2">
                        {fmtCi(
                          r.A_slow_fa_episodes_per_hour,
                          r.A_slow_ci_lo,
                          r.A_slow_ci_hi,
                          4
                        )}
                      </td>
                      <td className="px-3 py-2">
                        {fmtEveryHours(r.A_slow_one_fa_every_hours)}
                      </td>
                      <td className="px-3 py-2">
                        {fmtCi(
                          r.missed_event_rate,
                          r.missed_event_ci_lo,
                          r.missed_event_ci_hi,
                          3
                        )}
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
          <p className="text-xs text-[var(--text-muted)]">
            <strong>Note on zero counts:</strong>{" "}
            {falseAlarms.definition_notes.zero_count_caveat}
          </p>
        </div>
      </details>

      {/* Section 2.3: Recall vs SNR Bin & Detection Limit (~5x IMU Noise) — Secondary chart & table inside details */}
      <details className="panel-surface rounded-2xl p-5 sm:p-6">
        <summary className="cursor-pointer font-mono text-xs font-bold text-[var(--text-primary)]">
          Why does reliable detection require peak tilt above ~5× IMU noise (SNR &gt; 5, ~0.95° tilt)?
        </summary>
        <div className="mt-4 space-y-5">
          <p className="text-xs leading-relaxed text-[var(--text-secondary)]">
            {recallSnr.detection_limit_note}
          </p>

          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={pooledSnrChartData}
                margin={{ top: 12, right: 24, left: 4, bottom: 8 }}
              >
                <CartesianGrid stroke="var(--chart-grid)" vertical={false} />
                <XAxis
                  dataKey="snr_bin"
                  tick={{
                    fontSize: 12,
                    fill: "var(--chart-labels)",
                    fontFamily: '"JetBrains Mono", monospace',
                  }}
                />
                <YAxis
                  domain={[0, 1]}
                  tick={{
                    fontSize: 11,
                    fill: "var(--chart-labels)",
                    fontFamily: '"JetBrains Mono", monospace',
                  }}
                  label={{
                    value: "Recall / Run Detection Rate",
                    angle: -90,
                    position: "insideLeft",
                    style: { fill: "var(--chart-labels)", fontSize: 11 },
                  }}
                />
                <ReferenceLine
                  x="SNR >5"
                  stroke="var(--status-warning)"
                  strokeDasharray="4 4"
                  label={{
                    value: "~5x Noise Detection Limit",
                    position: "insideTopRight",
                    fill: "var(--status-warning)",
                    fontSize: 11,
                  }}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "var(--chart-tooltip-bg)",
                    borderColor: "var(--chart-tooltip-border)",
                    color: "var(--chart-tooltip-text)",
                  }}
                />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="RF Gradual Run Detect"
                  stroke="var(--chart-series-1)"
                  strokeWidth={2.5}
                />
                <Line
                  type="monotone"
                  dataKey="RF Gradual Window Recall"
                  stroke="var(--chart-series-1)"
                  strokeDasharray="4 4"
                  strokeWidth={2}
                />
                <Line
                  type="monotone"
                  dataKey="RF Rapid Run Detect"
                  stroke="var(--status-danger)"
                  strokeWidth={2.5}
                />
                <Line
                  type="monotone"
                  dataKey="RF Rapid Window Recall"
                  stroke="var(--status-danger)"
                  strokeDasharray="4 4"
                  strokeWidth={2}
                />
                <Line
                  type="monotone"
                  dataKey="Threshold Gradual Run Detect"
                  stroke="var(--chart-series-4)"
                  strokeWidth={2}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="overflow-x-auto rounded-xl border border-[var(--table-border)]">
            <table className="w-full text-left text-xs">
              <thead className="bg-[var(--table-header-bg)] font-mono text-[0.7rem] uppercase text-[var(--table-header-text)]">
                <tr>
                  <th className="px-3 py-2">Model</th>
                  <th className="px-3 py-2">Regime</th>
                  <th className="px-3 py-2">Class</th>
                  <th className="px-3 py-2">
                    SNR &lt; 1 (Win Recall / Run Detect)
                  </th>
                  <th className="px-3 py-2">
                    SNR 1–2 (Win Recall / Run Detect)
                  </th>
                  <th className="px-3 py-2">
                    SNR 2–5 (Win Recall / Run Detect)
                  </th>
                  <th className="px-3 py-2 text-[var(--brand-green)]">
                    SNR &gt; 5 (Win Recall / Run Detect)
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[var(--table-border)] font-mono">
                {["threshold_rule", "random_forest", "logistic_regression"].flatMap(
                  (m) =>
                    ["gradual", "rapid"].map((cls) => {
                      const bins = ["<1", "1-2", "2-5", ">5"].map((b) =>
                        recallSnr.gradual_rapid_recall_by_snr_v3.find(
                          (r) =>
                            r.split.startsWith("pooled") &&
                            r.model === m &&
                            r.regime === "episode_fa<=1 (k=3,m=3)" &&
                            r.class === cls &&
                            r.snr_bin === b
                        )
                      );
                      return (
                        <tr
                          key={`pooled-${m}-${cls}`}
                          className="bg-[var(--table-row-bg)] hover:bg-[var(--surface-alt)]"
                        >
                          <td className="px-3 py-2 font-sans font-semibold text-[var(--text-primary)]">
                            {MODEL_PLAIN_NAMES[m] || m}
                          </td>
                          <td className="px-3 py-2">episode (k=3,m=3)</td>
                          <td className="px-3 py-2 font-bold">{cls}</td>
                          {bins.map((cell, idx) => (
                            <td
                              key={idx}
                              className={`px-3 py-2 ${
                                idx === 3
                                  ? "font-bold text-[var(--brand-green)]"
                                  : ""
                              }`}
                            >
                              {cell
                                ? `${fmtNum(cell.window_recall, 2)} / ${fmtNum(
                                    cell.run_detect_rate,
                                    2
                                  )}`
                                : "N/A"}
                            </td>
                          ))}
                        </tr>
                      );
                    })
                )}
              </tbody>
            </table>
          </div>
        </div>
      </details>

      {/* Section 2.4: RF vs Threshold Comparison, Permutation Importance & Ablations — Inside details */}
      <details className="panel-surface rounded-2xl p-5 sm:p-6">
        <summary className="cursor-pointer font-mono text-xs font-bold text-[var(--text-primary)]">
          Which feature groups reduce gradual vs. sensor-fault confusions, and where does Random Forest fail to beat the Threshold Rule?
        </summary>
        <div className="mt-4 grid grid-cols-1 gap-6 lg:grid-cols-2">
          {/* Left: Permutation Importance Groups & v3 Ablations */}
          <div className="space-y-4">
            <div>
              <p className="eyebrow">4. Feature Groups &amp; Ablations</p>
              <h3 className="subsection-title mt-1">
                v3 Random Forest Feature-Group Ablations &amp; Importance
              </h3>
              <p className="mt-1 text-xs leading-relaxed text-[var(--text-secondary)]">
                {modelComp.importance_caveat}
              </p>
            </div>

            <div className="overflow-x-auto rounded-xl border border-[var(--table-border)]">
              <table className="w-full text-left text-xs">
                <thead className="bg-[var(--table-header-bg)] font-mono text-[0.7rem] uppercase text-[var(--table-header-text)]">
                  <tr>
                    <th className="px-3 py-2">Removed Group</th>
                    <th className="px-3 py-2">Kept Cols</th>
                    <th className="px-3 py-2">Test Macro-F1 (SNR&gt;2)</th>
                    <th className="px-3 py-2">Delta F1</th>
                    <th className="px-3 py-2">Grad &lt;-&gt; SF Confusions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[var(--table-border)] font-mono">
                  {modelComp.rf_ablations_v3.map((ab) => (
                    <tr
                      key={ab.group_removed}
                      className={`bg-[var(--table-row-bg)] ${
                        ab.group_removed === "direction"
                          ? "bg-[var(--status-danger-bg)] font-bold"
                          : ""
                      }`}
                    >
                      <td className="px-3 py-2">{ab.group_removed}</td>
                      <td className="px-3 py-2">{ab.n_features_kept}</td>
                      <td className="px-3 py-2">
                        {ab.macro_f1_snr_gt_2.toFixed(4)}
                      </td>
                      <td className="px-3 py-2">
                        {ab.delta_macro_f1_snr_gt_2 >= 0 ? "+" : ""}
                        {ab.delta_macro_f1_snr_gt_2.toFixed(4)}
                      </td>
                      <td className="px-3 py-2">
                        {ab.gradual_sf_confusions_snr_gt_2}
                        {ab.group_removed === "direction"
                          ? " (+72, ~3x worse!)"
                          : ""}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="overflow-x-auto rounded-xl border border-[var(--table-border)]">
              <table className="w-full text-left text-xs">
                <thead className="bg-[var(--table-header-bg)] font-mono text-[0.7rem] uppercase text-[var(--table-header-text)]">
                  <tr>
                    <th className="px-3 py-2">Rank</th>
                    <th className="px-3 py-2">Feature Group (val)</th>
                    <th className="px-3 py-2">Cols</th>
                    <th className="px-3 py-2">Mean Macro-F1 Drop</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[var(--table-border)] font-mono">
                  {modelComp.permutation_importance_groups.map((pg) => (
                    <tr key={pg.group} className="bg-[var(--table-row-bg)]">
                      <td className="px-3 py-1.5">#{pg.rank}</td>
                      <td className="px-3 py-1.5 font-bold">{pg.group}</td>
                      <td className="px-3 py-1.5">{pg.feature_count}</td>
                      <td className="px-3 py-1.5">
                        {pg.mean_importance.toFixed(4)} ±{" "}
                        {pg.std_importance.toFixed(4)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Right: Where RF Does NOT Beat the Threshold Rule */}
          <div className="space-y-4">
            <div>
              <p className="eyebrow">5. Paired Bootstrap Comparison</p>
              <h3 className="subsection-title mt-1">
                Where Random Forest Does NOT Beat the Threshold Rule
              </h3>
              <p className="mt-1 text-xs leading-relaxed text-[var(--text-secondary)]">
                Paired run-bootstrap differences (
                <code className="font-mono">RF - Threshold</code>). Rows flagged{" "}
                <code className="font-mono">true</code> indicate metrics where
                Random Forest fails to significantly outperform the simple
                hand-built threshold rule.
              </p>
            </div>

            <div className="max-h-[430px] overflow-y-auto rounded-xl border border-[var(--table-border)]">
              <table className="w-full text-left text-xs">
                <thead className="sticky top-0 bg-[var(--table-header-bg)] font-mono text-[0.7rem] uppercase text-[var(--table-header-text)]">
                  <tr>
                    <th className="px-3 py-2">Split</th>
                    <th className="px-3 py-2">Metric</th>
                    <th className="px-3 py-2">RF</th>
                    <th className="px-3 py-2">Threshold</th>
                    <th className="px-3 py-2">RF - TH [95% CI]</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[var(--table-border)] font-mono">
                  {modelComp.rf_vs_threshold_comparison.map((row, idx) => (
                    <tr
                      key={`${row.split}-${row.metric}-${idx}`}
                      className="bg-[var(--table-row-bg)] hover:bg-[var(--surface-alt)]"
                    >
                      <td className="px-3 py-2 font-bold">{row.split}</td>
                      <td className="px-3 py-2">{row.metric}</td>
                      <td className="px-3 py-2">{row.rf_val.toFixed(3)}</td>
                      <td className="px-3 py-2">{row.th_val.toFixed(3)}</td>
                      <td
                        className={`px-3 py-2 font-semibold ${
                          row.rf_minus_th < 0
                            ? "text-[var(--status-danger)]"
                            : "text-[var(--brand-green)]"
                        }`}
                      >
                        {fmtCi(
                          row.rf_minus_th,
                          row.diff_ci_lo,
                          row.diff_ci_hi,
                          3
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </details>
    </div>
  );
}

/* ══════════════════════════════════════════════════════════════════════════ */
/* VIEW 3: ROBUSTNESS                                                         */
/* ══════════════════════════════════════════════════════════════════════════ */
function RobustnessSection({
  robustness,
  yawSnrMode,
  onSelectYawSnrMode,
  yawSplit,
  onSelectYawSplit,
}: {
  robustness: SimRobustnessData;
  yawSnrMode: "snr_gt_2" | "all_snr";
  onSelectYawSnrMode: (m: "snr_gt_2" | "all_snr") => void;
  yawSplit: string;
  onSelectYawSplit: (s: string) => void;
}) {
  const yawTable = useMemo(() => {
    const src =
      yawSnrMode === "snr_gt_2"
        ? robustness.yaw_sensitivity.snr_gt_2
        : robustness.yaw_sensitivity.all_snr;
    return src.filter((r) => r.split === yawSplit);
  }, [robustness.yaw_sensitivity, yawSnrMode, yawSplit]);

  const yawChartData = useMemo(() => {
    const conditions = [
      "sd_0deg",
      "sd_5deg",
      "sd_10deg",
      "sd_20deg",
      "single_node_45deg",
      "v2_yaw_free_baseline",
    ];
    return conditions.map((cond) => {
      const rfRow = yawTable.find(
        (r) => r.condition === cond && r.model === "random_forest"
      );
      const thRow = yawTable.find(
        (r) => r.condition === cond && r.model === "threshold_rule"
      );
      return {
        condition: cond.replace("_yaw_free_baseline", " (yaw-free)"),
        "RF Alarm Macro-F1": rfRow?.alarm_macro_f1 ?? 0,
        "RF Run Detect Rate": rfRow?.run_exact_detect_rate ?? 0,
        "Threshold Alarm Macro-F1": thRow?.alarm_macro_f1 ?? 0,
        "RF Grad<->SF Confusions": rfRow?.gradual_sf_confusions ?? 0,
      };
    });
  }, [yawTable]);

  const shapeChartData = useMemo(() => {
    const shapes = [
      "gaussian_control",
      "asymmetric",
      "two_basin",
      "trough_panel",
      "planar_slope",
    ];
    return shapes.map((sh) => {
      const rfDet = robustness.shape_mismatch.detection_rates.find(
        (r) =>
          r.shape === sh &&
          r.model === "random_forest" &&
          r.regime === "episode_fa<=1 (k=3,m=3)" &&
          r.subset.startsWith("all_events")
      );
      const thDet = robustness.shape_mismatch.detection_rates.find(
        (r) =>
          r.shape === sh &&
          r.model === "threshold_rule" &&
          r.regime === "episode_fa<=1 (k=3,m=3)" &&
          r.subset.startsWith("all_events")
      );
      const rfConf =
        robustness.shape_mismatch.gradual_sensor_fault_confusion.find(
          (r) =>
            r.shape === sh &&
            r.model === "random_forest" &&
            r.regime === "episode_fa<=1 (k=3,m=3)"
        );
      return {
        shape: sh,
        "RF Exact Detect Rate": rfDet?.exact_detect_rate ?? 0,
        "Threshold Exact Detect Rate": thDet?.exact_detect_rate ?? 0,
        "RF Gradual->SensorFault Windows":
          rfConf?.win_gradual_to_sensor_fault ?? 0,
      };
    });
  }, [robustness.shape_mismatch]);

  return (
    <div className="space-y-8">
      {/* Start here guidance line */}
      <p
        data-testid="start-here-robustness"
        className="rounded-lg border border-[var(--border)] bg-[var(--surface-alt)] px-4 py-2.5 text-xs font-medium text-[var(--text-primary)]"
      >
        <strong className="font-bold text-[var(--brand-green)]">
          Start here:
        </strong>{" "}
        Examine the Installation-Yaw Sensitivity bar chart first (showing why
        compass alignment ≤10° SD is required), then open any collapsed
        disclosure below for shape mismatch, high-tilt/depth checks, timescale
        compression, or the advisory unknown-anomaly detector.
      </p>

      {/* 1. Yaw Sensitivity (Primary Chart visible + Table inside details) */}
      <section className="panel-surface rounded-2xl p-5 sm:p-6 space-y-5">
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <p className="eyebrow">1. Installation-Yaw Sensitivity (v3_yaw)</p>
            <h2 className="section-title mt-1">
              Compass Alignment Error Across All 6 Perturbation Conditions
            </h2>
            <p className="mt-1 max-w-3xl text-xs leading-relaxed text-[var(--text-secondary)]">
              {robustness.yaw_sensitivity.summary_note}
            </p>
          </div>

          <div className="flex flex-wrap gap-3">
            <select
              value={yawSnrMode}
              onChange={(e) =>
                onSelectYawSnrMode(e.target.value as "snr_gt_2" | "all_snr")
              }
              aria-label="Select Yaw SNR subset"
              className="rounded-lg border border-[var(--input-border)] bg-[var(--input-bg)] px-3 py-2 font-mono text-xs font-semibold text-[var(--input-text)]"
            >
              <option value="snr_gt_2">SNR &gt; 2 Subset</option>
              <option value="all_snr">All SNR Runs</option>
            </select>

            <select
              value={yawSplit}
              onChange={(e) => onSelectYawSplit(e.target.value)}
              aria-label="Select Yaw evaluation split"
              className="rounded-lg border border-[var(--input-border)] bg-[var(--input-bg)] px-3 py-2 font-mono text-xs font-semibold text-[var(--input-text)]"
            >
              <option value="test">test</option>
              <option value="broad_basin">broad_basin</option>
              <option value="shifted_test">shifted_test</option>
              {yawSnrMode === "all_snr" && (
                <option value="stress">stress</option>
              )}
            </select>
          </div>
        </div>

        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={yawChartData}
              margin={{ top: 10, right: 20, left: 0, bottom: 8 }}
            >
              <CartesianGrid stroke="var(--chart-grid)" vertical={false} />
              <XAxis
                dataKey="condition"
                tick={{
                  fontSize: 11,
                  fill: "var(--chart-labels)",
                  fontFamily: '"JetBrains Mono", monospace',
                }}
              />
              <YAxis
                domain={[0, 1]}
                tick={{
                  fontSize: 11,
                  fill: "var(--chart-labels)",
                  fontFamily: '"JetBrains Mono", monospace',
                }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "var(--chart-tooltip-bg)",
                  borderColor: "var(--chart-tooltip-border)",
                  color: "var(--chart-tooltip-text)",
                }}
              />
              <Legend />
              <Bar
                dataKey="RF Alarm Macro-F1"
                fill="var(--chart-series-1)"
                radius={[4, 4, 0, 0]}
              />
              <Bar
                dataKey="RF Run Detect Rate"
                fill="var(--chart-series-2)"
                radius={[4, 4, 0, 0]}
              />
              <Bar
                dataKey="Threshold Alarm Macro-F1"
                fill="var(--chart-series-4)"
                radius={[4, 4, 0, 0]}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <details className="rounded-xl border border-[var(--border)] bg-[var(--surface-alt)] p-3.5">
          <summary className="cursor-pointer font-mono text-xs font-bold text-[var(--text-primary)]">
            How do compass yaw errors affect Macro-F1, detection rates, and gradual↔sensor-fault confusions across models?
          </summary>
          <div className="mt-3 overflow-x-auto rounded-xl border border-[var(--table-border)]">
            <table className="w-full text-left text-xs">
              <thead className="bg-[var(--table-header-bg)] font-mono text-[0.7rem] uppercase text-[var(--table-header-text)]">
                <tr>
                  <th className="px-3 py-2">Condition</th>
                  <th className="px-3 py-2">Model</th>
                  <th className="px-3 py-2">Window Macro-F1 [95% CI]</th>
                  <th className="px-3 py-2">Alarm Macro-F1 [95% CI]</th>
                  <th className="px-3 py-2">Run Exact Detect [95% CI]</th>
                  <th className="px-3 py-2">FA Episodes / Run</th>
                  <th className="px-3 py-2">Grad &lt;-&gt; SF Confusions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[var(--table-border)] font-mono">
                {yawTable
                  .filter((r) =>
                    ["random_forest", "threshold_rule", "logistic_regression"].includes(
                      r.model
                    )
                  )
                  .map((r) => (
                    <tr
                      key={`${r.split}-${r.condition}-${r.model}`}
                      className="bg-[var(--table-row-bg)] hover:bg-[var(--surface-alt)]"
                    >
                      <td className="px-3 py-2 font-bold">{r.condition}</td>
                      <td className="px-3 py-2 font-sans font-semibold text-[var(--text-primary)]">
                        {MODEL_PLAIN_NAMES[r.model] || r.model}
                      </td>
                      <td className="px-3 py-2">
                        {fmtCi(
                          r.window_macro_f1,
                          r.window_macro_f1_ci_lo,
                          r.window_macro_f1_ci_hi,
                          3
                        )}
                      </td>
                      <td className="px-3 py-2 font-semibold text-[var(--brand-green)]">
                        {fmtCi(
                          r.alarm_macro_f1,
                          r.alarm_macro_f1_ci_lo,
                          r.alarm_macro_f1_ci_hi,
                          3
                        )}
                      </td>
                      <td className="px-3 py-2">
                        {fmtCi(
                          r.run_exact_detect_rate,
                          r.run_exact_detect_ci_lo,
                          r.run_exact_detect_ci_hi,
                          3
                        )}
                      </td>
                      <td className="px-3 py-2">
                        {fmtNum(r.fa_episodes_per_normal_run, 2)}
                      </td>
                      <td className="px-3 py-2 font-bold">
                        {r.gradual_sf_confusions}
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        </details>
      </section>

      {/* 2. Shape Mismatch Table & Secondary Chart inside details */}
      <details className="panel-surface rounded-2xl p-5 sm:p-6">
        <summary className="cursor-pointer font-mono text-xs font-bold text-[var(--text-primary)]">
          What happens when ground subsidence follows non-Gaussian shapes (asymmetric, two-basin, trough panel, or planar slope)?
        </summary>
        <div className="mt-4 space-y-5">
          <p className="text-xs leading-relaxed text-[var(--text-secondary)]">
            {robustness.shape_mismatch.summary_note}
          </p>

          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={shapeChartData}
                margin={{ top: 10, right: 20, left: 0, bottom: 8 }}
              >
                <CartesianGrid stroke="var(--chart-grid)" vertical={false} />
                <XAxis
                  dataKey="shape"
                  tick={{
                    fontSize: 11,
                    fill: "var(--chart-labels)",
                    fontFamily: '"JetBrains Mono", monospace',
                  }}
                />
                <YAxis
                  tick={{
                    fontSize: 11,
                    fill: "var(--chart-labels)",
                    fontFamily: '"JetBrains Mono", monospace',
                  }}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "var(--chart-tooltip-bg)",
                    borderColor: "var(--chart-tooltip-border)",
                    color: "var(--chart-tooltip-text)",
                  }}
                />
                <Legend />
                <Bar
                  dataKey="RF Exact Detect Rate"
                  fill="var(--chart-series-1)"
                  radius={[4, 4, 0, 0]}
                />
                <Bar
                  dataKey="Threshold Exact Detect Rate"
                  fill="var(--chart-series-4)"
                  radius={[4, 4, 0, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="overflow-x-auto rounded-xl border border-[var(--table-border)]">
            <table className="w-full text-left text-xs">
              <thead className="bg-[var(--table-header-bg)] font-mono text-[0.7rem] uppercase text-[var(--table-header-text)]">
                <tr>
                  <th className="px-3 py-2">Shape Profile</th>
                  <th className="px-3 py-2">RF All-Event Detect [95% CI]</th>
                  <th className="px-3 py-2">RF Gradual / Rapid Detect</th>
                  <th className="px-3 py-2">
                    RF Gradual -&gt; SensorFault Windows
                  </th>
                  <th className="px-3 py-2">
                    SNR&gt;2 Convergence Residual p50 (m)
                  </th>
                  <th className="px-3 py-2">SNR&gt;2 Weighted Cos p50</th>
                  <th className="px-3 py-2">SNR&gt;2 Pairwise Cos p50</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[var(--table-border)] font-mono">
                {[
                  "gaussian_control",
                  "asymmetric",
                  "two_basin",
                  "trough_panel",
                  "planar_slope",
                ].map((sh) => {
                  const rfAll = robustness.shape_mismatch.detection_rates.find(
                    (r) =>
                      r.shape === sh &&
                      r.model === "random_forest" &&
                      r.regime === "episode_fa<=1 (k=3,m=3)" &&
                      r.subset.startsWith("all_events")
                  );
                  const rfGrad = robustness.shape_mismatch.detection_rates.find(
                    (r) =>
                      r.shape === sh &&
                      r.model === "random_forest" &&
                      r.regime === "episode_fa<=1 (k=3,m=3)" &&
                      r.subset.startsWith("gradual")
                  );
                  const rfRap = robustness.shape_mismatch.detection_rates.find(
                    (r) =>
                      r.shape === sh &&
                      r.model === "random_forest" &&
                      r.regime === "episode_fa<=1 (k=3,m=3)" &&
                      r.subset.startsWith("rapid")
                  );
                  const rfConf =
                    robustness.shape_mismatch.gradual_sensor_fault_confusion.find(
                      (r) =>
                        r.shape === sh &&
                        r.model === "random_forest" &&
                        r.regime === "episode_fa<=1 (k=3,m=3)"
                    );
                  const conv =
                    robustness.shape_mismatch.convergence_distributions.find(
                      (r) => r.shape === sh && r.subset === "active_windows_snr_gt_2"
                    );
                  return (
                    <tr
                      key={sh}
                      className="bg-[var(--table-row-bg)] hover:bg-[var(--surface-alt)]"
                    >
                      <td className="px-3 py-2 font-bold">{sh}</td>
                      <td className="px-3 py-2 font-semibold text-[var(--brand-green)]">
                        {rfAll
                          ? fmtCi(
                              rfAll.exact_detect_rate,
                              rfAll.exact_detect_ci_lo,
                              rfAll.exact_detect_ci_hi,
                              3
                            )
                          : "N/A"}
                      </td>
                      <td className="px-3 py-2">
                        {fmtNum(rfGrad?.exact_detect_rate, 3)} /{" "}
                        {fmtNum(rfRap?.exact_detect_rate, 3)}
                      </td>
                      <td className="px-3 py-2 font-bold text-[var(--status-danger)]">
                        {rfConf
                          ? `${rfConf.win_gradual_to_sensor_fault} (${(
                              rfConf.win_gradual_to_sensor_fault_rate * 100
                            ).toFixed(1)}%)`
                          : "N/A"}
                      </td>
                      <td className="px-3 py-2">
                        {conv?.dir_convergence_residual_m_p50 != null
                          ? `${conv.dir_convergence_residual_m_p50.toFixed(2)} m`
                          : "N/A"}
                      </td>
                      <td className="px-3 py-2">
                        {conv?.dir_convergence_weighted_cos_p50 != null
                          ? conv.dir_convergence_weighted_cos_p50.toFixed(3)
                          : "N/A"}
                      </td>
                      <td className="px-3 py-2">
                        {conv?.dir_pairwise_cos_mean_p50 != null
                          ? conv.dir_pairwise_cos_mean_p50.toFixed(3)
                          : "N/A"}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      </details>

      {/* 3. Literature Checks (High Tilt B1 & Realistic Depth B2) inside details */}
      <details className="panel-surface rounded-2xl p-5 sm:p-6">
        <summary className="cursor-pointer font-mono text-xs font-bold text-[var(--text-primary)]">
          How do the models behave under literature high-tilt regimes (3–12°) and realistic overburden depths (H = 50–300 m)?
        </summary>
        <div className="mt-4 space-y-5">
          <p className="text-xs leading-relaxed text-[var(--text-secondary)]">
            {robustness.literature_checks.summary_note}
          </p>

          <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
            {/* High Tilt & Pooled Depth Detection + Confusions */}
            <div className="overflow-x-auto rounded-xl border border-[var(--table-border)]">
              <table className="w-full text-left text-xs">
                <thead className="bg-[var(--table-header-bg)] font-mono text-[0.7rem] uppercase text-[var(--table-header-text)]">
                  <tr>
                    <th className="px-3 py-2">Arm</th>
                    <th className="px-3 py-2">Model</th>
                    <th className="px-3 py-2">All-Event Detect</th>
                    <th className="px-3 py-2">Grad / Rapid Detect</th>
                    <th className="px-3 py-2">Grad -&gt; SF Win</th>
                    <th className="px-3 py-2">Grad -&gt; Rapid Win</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[var(--table-border)] font-mono">
                  {[
                    "gaussian_control",
                    "high_tilt_3_6",
                    "high_tilt_6_12",
                    "depth_all_gt_1p5",
                  ].flatMap((arm) =>
                    ["random_forest", "threshold_rule", "logistic_regression"].map(
                      (m) => {
                        const dAll =
                          robustness.literature_checks.detection_rates.find(
                            (r) =>
                              r.arm === arm &&
                              r.model === m &&
                              r.subset === "all_events"
                          );
                        const dGrad =
                          robustness.literature_checks.detection_rates.find(
                            (r) =>
                              r.arm === arm &&
                              r.model === m &&
                              r.subset === "gradual"
                          );
                        const dRap =
                          robustness.literature_checks.detection_rates.find(
                            (r) =>
                              r.arm === arm &&
                              r.model === m &&
                              r.subset === "rapid"
                          );
                        const cf =
                          robustness.literature_checks.gradual_sensor_fault_confusion.find(
                            (r) => r.arm === arm && r.model === m
                          );
                        return (
                          <tr
                            key={`${arm}-${m}`}
                            className="bg-[var(--table-row-bg)] hover:bg-[var(--surface-alt)]"
                          >
                            <td className="px-3 py-1.5 font-bold">{arm}</td>
                            <td className="px-3 py-1.5 font-sans font-semibold">
                              {MODEL_PLAIN_NAMES[m] || m}
                            </td>
                            <td className="px-3 py-1.5 font-bold text-[var(--brand-green)]">
                              {fmtNum(dAll?.exact_detect_rate, 3)}
                            </td>
                            <td className="px-3 py-1.5">
                              {fmtNum(dGrad?.exact_detect_rate, 2)} /{" "}
                              {fmtNum(dRap?.exact_detect_rate, 2)}
                            </td>
                            <td className="px-3 py-1.5 text-[var(--status-danger)]">
                              {cf
                                ? `${cf.win_gradual_to_sensor_fault} (${(
                                    cf.win_gradual_to_sensor_fault_rate * 100
                                  ).toFixed(0)}%)`
                                : "N/A"}
                            </td>
                            <td className="px-3 py-1.5">
                              {cf ? cf.win_gradual_to_rapid : "N/A"}
                            </td>
                          </tr>
                        );
                      }
                    )
                  )}
                </tbody>
              </table>
            </div>

            {/* 12-Cell Realistic Depth Knothe Spread Grid */}
            <div className="overflow-x-auto rounded-xl border border-[var(--table-border)]">
              <table className="w-full text-left text-xs">
                <thead className="bg-[var(--table-header-bg)] font-mono text-[0.7rem] uppercase text-[var(--table-header-text)]">
                  <tr>
                    <th className="px-3 py-2">Cell ID (PLACEHOLDER)</th>
                    <th className="px-3 py-2">Depth H (m)</th>
                    <th className="px-3 py-2">tan(beta)</th>
                    <th className="px-3 py-2">Sigma (m)</th>
                    <th className="px-3 py-2">Sigma / R_net</th>
                    <th className="px-3 py-2">RF Detect</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[var(--table-border)] font-mono">
                  {robustness.literature_checks.realistic_depth_sigma_grid.map(
                    (cell) => {
                      const rfCell =
                        robustness.literature_checks.detection_rates.find(
                          (r) =>
                            r.arm === cell.cell_id &&
                            r.model === "random_forest" &&
                            r.subset === "all_events"
                        );
                      return (
                        <tr
                          key={cell.cell_id}
                          className="bg-[var(--table-row-bg)] hover:bg-[var(--surface-alt)]"
                        >
                          <td className="px-3 py-1.5 font-bold">
                            {cell.cell_id}
                          </td>
                          <td className="px-3 py-1.5">{cell.H_m}</td>
                          <td className="px-3 py-1.5">{cell.tan_beta}</td>
                          <td className="px-3 py-1.5">
                            {cell.sigma_m.toFixed(2)}
                          </td>
                          <td className="px-3 py-1.5 font-semibold">
                            {cell.sigma_radii.toFixed(2)} R
                          </td>
                          <td className="px-3 py-1.5 text-[var(--brand-green)] font-bold">
                            {rfCell
                              ? fmtNum(rfCell.exact_detect_rate, 2)
                              : "<= 1.5R (in main range)"}
                          </td>
                        </tr>
                      );
                    }
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </details>

      {/* 4. Timescale Audit inside details */}
      <details className="panel-surface rounded-2xl p-5 sm:p-6">
        <summary className="cursor-pointer font-mono text-xs font-bold text-[var(--text-primary)]">
          What does the 360× time-compression placeholder imply for physical tilt rates and slow-process false-alarm rates?
        </summary>
        <div className="mt-4 space-y-4">
          <p className="text-xs leading-relaxed text-[var(--text-secondary)]">
            {robustness.timescale_audit.summary_note}
          </p>

          <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
            <div className="overflow-x-auto rounded-xl border border-[var(--table-border)]">
              <table className="w-full text-left text-xs">
                <thead className="bg-[var(--table-header-bg)] font-mono text-[0.7rem] uppercase text-[var(--table-header-text)]">
                  <tr>
                    <th className="px-3 py-2">Subset</th>
                    <th className="px-3 py-2">Gradual Runs</th>
                    <th className="px-3 py-2">&gt; 0.1 deg/day</th>
                    <th className="px-3 py-2">Implied Rate p50 (deg/day)</th>
                    <th className="px-3 py-2">Speedup vs 0.1 deg/day (p50)</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[var(--table-border)] font-mono">
                  {robustness.timescale_audit.gradual_rate_summary.map((row) => (
                    <tr
                      key={row.subset}
                      className="bg-[var(--table-row-bg)] hover:bg-[var(--surface-alt)]"
                    >
                      <td className="px-3 py-2 font-bold">{row.subset}</td>
                      <td className="px-3 py-2">{row.n_gradual_runs}</td>
                      <td className="px-3 py-2 text-[var(--status-danger)] font-bold">
                        {row.n_faster_than_0p1_deg_per_day} / {row.n_gradual_runs}{" "}
                        ({(row.frac_faster_than_0p1_deg_per_day * 100).toFixed(1)}
                        %)
                      </td>
                      <td className="px-3 py-2">
                        {row.rate_deg_per_day_p50_median.toFixed(3)}
                      </td>
                      <td className="px-3 py-2 font-semibold">
                        {row.factor_vs_0p1_p50_median.toFixed(2)}×
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="overflow-x-auto rounded-xl border border-[var(--table-border)]">
              <table className="w-full text-left text-xs">
                <thead className="bg-[var(--table-header-bg)] font-mono text-[0.7rem] uppercase text-[var(--table-header-text)]">
                  <tr>
                    <th className="px-3 py-2">Model (Regime)</th>
                    <th className="px-3 py-2 text-[var(--status-danger)]">
                      Def B (1x Sensor)
                    </th>
                    <th className="px-3 py-2">Def A (360x, 24h)</th>
                    <th className="px-3 py-2">Alt A (10,800x = 30d)</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[var(--table-border)] font-mono">
                  {robustness.timescale_audit.false_alarms_under_alt_compression
                    .filter((r) => r.regime !== "raw (k=1)")
                    .map((r) => (
                      <tr
                        key={`${r.model}-${r.regime}`}
                        className="bg-[var(--table-row-bg)] hover:bg-[var(--surface-alt)]"
                      >
                        <td className="px-3 py-2 font-sans font-semibold">
                          {MODEL_PLAIN_NAMES[r.model] || r.model} ({r.regime})
                        </td>
                        <td className="px-3 py-2 font-bold text-[var(--status-danger)]">
                          {r.B_sensor_fa_per_hour_1x.toFixed(2)}/h (
                          {r.B_sensor_one_fa_every})
                        </td>
                        <td className="px-3 py-2">
                          {r.A_slow_fa_per_hour_360x.toFixed(4)}/h (
                          {r.A_slow_one_fa_every_360x})
                        </td>
                        <td className="px-3 py-2">
                          {r.A_slow_fa_per_hour_10800x.toFixed(6)}/h (
                          {r.A_slow_one_fa_every_10800x})
                        </td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </details>

      {/* 5. Unsupervised Unknown-Detector Complement Table (Marked ADVISORY) inside details */}
      <details className="panel-surface rounded-2xl border-2 border-[var(--status-warning-border)] p-5 sm:p-6">
        <summary className="cursor-pointer font-mono text-xs font-bold text-[var(--text-primary)]">
          Can an advisory unsupervised &ldquo;unknown_anomaly&rdquo; detector catch out-of-distribution runs missed by the classifier, and how many extra false alarms does it add?
        </summary>
        <div className="mt-4 space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <span className="inline-flex items-center gap-1.5 rounded-full bg-[var(--status-warning-bg)] px-3 py-0.5 font-mono text-xs font-bold text-[var(--status-warning)]">
              ADVISORY SIDE-STUDY · NOT A VALIDATED HAZARD DETECTOR
            </span>
          </div>

          <p className="rounded-lg border border-[var(--status-warning-border)] bg-[var(--status-warning-bg)] p-3 text-xs text-[var(--text-primary)]">
            {robustness.unknown_detector_complement.advisory_banner}
          </p>

          <div className="overflow-x-auto rounded-xl border border-[var(--table-border)]">
            <table className="w-full text-left text-xs">
              <thead className="bg-[var(--table-header-bg)] font-mono text-[0.7rem] uppercase text-[var(--table-header-text)]">
                <tr>
                  <th className="px-3 py-2">Split / Arm</th>
                  <th className="px-3 py-2">Classifier + Detector</th>
                  <th className="px-3 py-2">Clf Exact Detect</th>
                  <th className="px-3 py-2">Missed Runs Flagged by Det</th>
                  <th className="px-3 py-2">Rule 1 (Clf OR Det) Detect</th>
                  <th className="px-3 py-2">Rule 2 (Clf OR Gated) Detect</th>
                  <th className="px-3 py-2 text-[var(--status-danger)]">
                    Extra FA / Run (OR / Gated)
                  </th>
                  <th className="px-3 py-2 text-[var(--status-danger)]">
                    Def B Extra Sensor FA/h (OR / Gated)
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[var(--table-border)] font-mono">
                {robustness.unknown_detector_complement.complement_table
                  .filter((r) => r.classifier === "random_forest")
                  .map((r) => (
                    <tr
                      key={`${r.split_or_arm}-${r.classifier}-${r.detector}`}
                      className="bg-[var(--table-row-bg)] hover:bg-[var(--surface-alt)]"
                    >
                      <td className="px-3 py-2 font-bold">{r.split_or_arm}</td>
                      <td className="px-3 py-2">
                        {r.classifier} + {r.detector}
                      </td>
                      <td className="px-3 py-2">
                        {r.clf_exact_detect_rate.toFixed(3)}
                      </td>
                      <td className="px-3 py-2 font-bold text-[var(--brand-teal)]">
                        {r.n_missed_flagged_by_det} / {r.n_clf_exact_missed} (
                        {r.frac_of_clf_exact_missed_flagged_by_det !== null
                          ? `${(
                              r.frac_of_clf_exact_missed_flagged_by_det * 100
                            ).toFixed(1)}%`
                          : "N/A"}
                        )
                      </td>
                      <td className="px-3 py-2 font-bold text-[var(--brand-green)]">
                        {fmtCi(
                          r.combined_or_detect_rate,
                          r.combined_or_detect_ci_lo,
                          r.combined_or_detect_ci_hi,
                          3
                        )}
                      </td>
                      <td className="px-3 py-2">
                        {fmtCi(
                          r.gated_unknown_detect_rate,
                          r.gated_unknown_detect_ci_lo,
                          r.gated_unknown_detect_ci_hi,
                          3
                        )}
                      </td>
                      <td className="px-3 py-2 text-[var(--status-danger)]">
                        {r.extra_fa_per_normal_run_or !== null
                          ? `+${r.extra_fa_per_normal_run_or.toFixed(
                              2
                            )} / +${r.extra_fa_per_normal_run_gated?.toFixed(2)}`
                          : "N/A"}
                      </td>
                      <td className="px-3 py-2 font-bold text-[var(--status-danger)]">
                        {r.extra_B_sensor_fa_per_hour_or !== null
                          ? `+${r.extra_B_sensor_fa_per_hour_or.toFixed(
                              1
                            )} / +${r.extra_B_sensor_fa_per_hour_gated?.toFixed(
                              1
                            )} /h`
                          : "N/A"}
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        </div>
      </details>
    </div>
  );
}

/* ══════════════════════════════════════════════════════════════════════════ */
/* VIEW 4: REPLAY                                                             */
/* ══════════════════════════════════════════════════════════════════════════ */
function ReplaySection({
  replay,
  explanationsData,
  selectedClass,
  onSelectClass,
  playbackIdx,
  onChangePlaybackIdx,
  isPlaying,
  onTogglePlay,
  playbackSpeed,
  onChangeSpeed,
  selectedTrendNode,
  onSelectTrendNode,
  trendChartHistory,
}: {
  replay: SimReplayData;
  explanationsData?: SimReplayExplanationsData;
  selectedClass: ReplayClassKey;
  onSelectClass: (cls: ReplayClassKey) => void;
  playbackIdx: number;
  onChangePlaybackIdx: (idx: number) => void;
  isPlaying: boolean;
  onTogglePlay: () => void;
  playbackSpeed: number;
  onChangeSpeed: (s: number) => void;
  selectedTrendNode: "NODE_01" | "NODE_02" | "NODE_03" | "NODE_04";
  onSelectTrendNode: (n: "NODE_01" | "NODE_02" | "NODE_03" | "NODE_04") => void;
  trendChartHistory: { t: number; tilt: number; piezo: number; lidarCm: number }[];
}) {
  const safeIdx = Math.min(
    Math.max(0, playbackIdx),
    replay.windows.length - 1
  );
  const currentWin = replay.windows[safeIdx];

  const explanationByIdx = useMemo(() => {
    const map = new Map<number, SimReplayExplanationsData["explanations"][number]>();
    if (explanationsData?.explanations) {
      for (const exp of explanationsData.explanations) {
        map.set(exp.window_idx, exp);
      }
    }
    return map;
  }, [explanationsData]);

  const currentExplanation = explanationByIdx.get(currentWin.window_idx);

  const chartSeries = useMemo(
    () =>
      replay.windows.map((w) => ({
        window_idx: w.window_idx,
        t_mid: w.t_mid,
        NODE_01_tilt: w.node_mean_tilt_deg.NODE_01,
        NODE_02_tilt: w.node_mean_tilt_deg.NODE_02,
        NODE_03_tilt: w.node_mean_tilt_deg.NODE_03,
        NODE_04_tilt: w.node_mean_tilt_deg.NODE_04,
        piezo_net: w.piezo_rms.network_mean,
        NODE_01_piezo: w.piezo_rms.NODE_01,
        NODE_02_piezo: w.piezo_rms.NODE_02,
        NODE_03_piezo: w.piezo_rms.NODE_03,
        NODE_04_piezo: w.piezo_rms.NODE_04,
        NODE_01_lidar: w.lidar_range_change_m.NODE_01,
        NODE_02_lidar: w.lidar_range_change_m.NODE_02,
        NODE_03_lidar: w.lidar_range_change_m.NODE_03,
        NODE_04_lidar: w.lidar_range_change_m.NODE_04,
        hub_common_mode: w.lidar_range_change_m.hub_common_mode_m,
        comm_missing_net: w.comm_missing_fraction.network,
        model_score: w.model_score,
        predicted_class_model_score: w.predicted_class_model_score,
        anomaly_score: w.anomaly_score,
        unknown_anomaly_num: w.unknown_anomaly ? 1 : 0,
      })),
    [replay.windows]
  );

  return (
    <div className="space-y-8">
      {/* Start here guidance line */}
      <p
        data-testid="start-here-replay"
        className="rounded-lg border border-[var(--border)] bg-[var(--surface-alt)] px-4 py-2.5 text-xs font-medium text-[var(--text-primary)]"
      >
        <strong className="font-bold text-[var(--brand-green)]">
          Start here:
        </strong>{" "}
        Pick one of the 6 scenario classes, scrub or play the 95-window timeline
        ribbon, and read the advisory explanation card for the selected window.
      </p>

      {/* Run Selector & Fixed Rule Documentation */}
      <section className="panel-surface rounded-2xl p-5 sm:p-6 space-y-4">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="eyebrow">
              Deterministic Test-Split Replay (6 Classes)
            </p>
            <h2 className="section-title mt-1">
              Example Run Playback:{" "}
              <span className="font-mono text-[var(--brand-teal)]">
                {replay.simulation_id}
              </span>
            </h2>
            <p className="mt-1 text-xs text-[var(--text-secondary)]">
              <strong>Selection Rule:</strong> {replay.selection_rule}
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2 font-mono text-xs">
            <span className="rounded border border-[var(--border)] bg-[var(--surface-alt)] px-2.5 py-1">
              Seed: {replay.run_metadata.seed}
            </span>
            <span className="rounded border border-[var(--border)] bg-[var(--surface-alt)] px-2.5 py-1">
              SNR: {replay.run_metadata.snr_vs_imu_noise.toFixed(2)}
            </span>
            <span className="rounded border border-[var(--border)] bg-[var(--surface-alt)] px-2.5 py-1">
              Peak Tilt: {replay.run_metadata.peak_tilt_deg.toFixed(3)}°
            </span>
            <span className="rounded border border-[var(--border)] bg-[var(--surface-alt)] px-2.5 py-1">
              Compression: {replay.run_metadata.run_compression_factor}×
            </span>
          </div>
        </div>

        {/* 6 Class Buttons */}
        <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-6">
          {REPLAY_CLASSES.map((item) => {
            const active = selectedClass === item.key;
            return (
              <button
                key={item.key}
                type="button"
                onClick={() => onSelectClass(item.key)}
                className={`flex flex-col items-start rounded-xl border p-3 text-left transition ${
                  active
                    ? "border-[var(--brand-green)] bg-[var(--brand-green)] text-white shadow-xs"
                    : "border-[var(--border)] bg-[var(--surface-alt)] text-[var(--text-primary)] hover:border-[var(--brand-teal)]"
                }`}
              >
                <span className="font-mono text-xs font-bold">{item.key}</span>
                <span
                  className={`mt-0.5 text-xs font-semibold ${
                    active ? "text-white" : "text-[var(--text-primary)]"
                  }`}
                >
                  {item.label}
                </span>
                <span
                  className={`mt-1 text-[0.66rem] leading-tight ${
                    active ? "text-white/85" : "text-[var(--text-muted)]"
                  }`}
                >
                  {item.plainDesc}
                </span>
              </button>
            );
          })}
        </div>
      </section>

      {/* Playback Controls & Scrubber */}
      <section className="panel-surface rounded-2xl p-4 sm:p-5 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={onTogglePlay}
              className="btn-primary px-4 py-2 text-xs"
              aria-label={isPlaying ? "Pause replay" : "Play replay"}
            >
              {isPlaying ? (
                <>
                  <Pause className="h-4 w-4" /> Pause
                </>
              ) : (
                <>
                  <Play className="h-4 w-4" /> Play
                </>
              )}
            </button>
            <button
              type="button"
              onClick={() => onChangePlaybackIdx(Math.max(0, safeIdx - 1))}
              className="btn-secondary px-3 py-2 text-xs"
              aria-label="Step back one window"
            >
              <SkipBack className="h-3.5 w-3.5" /> Prev
            </button>
            <button
              type="button"
              onClick={() =>
                onChangePlaybackIdx(
                  Math.min(replay.windows.length - 1, safeIdx + 1)
                )
              }
              className="btn-secondary px-3 py-2 text-xs"
              aria-label="Step forward one window"
            >
              Next <SkipForward className="h-3.5 w-3.5" />
            </button>
            <button
              type="button"
              onClick={() => onChangePlaybackIdx(0)}
              className="btn-secondary px-3 py-2 text-xs"
              aria-label="Reset replay to window 0"
            >
              <RotateCcw className="h-3.5 w-3.5" /> Reset
            </button>
          </div>

          <div className="flex items-center gap-2">
            <span className="font-mono text-xs text-[var(--text-muted)]">
              Speed:
            </span>
            {[1, 2, 4].map((spd) => (
              <button
                key={spd}
                type="button"
                onClick={() => onChangeSpeed(spd)}
                className={`rounded px-2.5 py-1 font-mono text-xs font-bold ${
                  playbackSpeed === spd
                    ? "bg-[var(--brand-teal)] text-white"
                    : "bg-[var(--surface-alt)] text-[var(--text-secondary)]"
                }`}
              >
                {spd}×
              </button>
            ))}
          </div>

          <div className="font-mono text-xs font-bold text-[var(--text-primary)]">
            Window #{currentWin.window_idx} / {replay.windows.length - 1} · t ={" "}
            {currentWin.t_start.toFixed(1)}s – {currentWin.t_end.toFixed(1)}s
            (mid {currentWin.t_mid.toFixed(2)}s)
          </div>
        </div>

        {/* Scrubber Slider */}
        <div>
          <label htmlFor="replay-scrubber" className="sr-only">
            Window timeline scrubber
          </label>
          <input
            id="replay-scrubber"
            type="range"
            min={0}
            max={replay.windows.length - 1}
            value={safeIdx}
            onChange={(e) => onChangePlaybackIdx(Number(e.target.value))}
            className="w-full accent-[var(--brand-green)] cursor-pointer"
          />
        </div>

        {/* Window Classification vs Persisted Alarm vs Advisory Unknown Flag Strip */}
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-5">
          <div className="rounded-xl border border-[var(--border)] bg-[var(--surface-alt)] p-3">
            <span className="block text-[0.65rem] font-bold uppercase tracking-wider text-[var(--text-muted)]">
              Ground-Truth Window Label
            </span>
            <div className="mt-1.5 flex items-center justify-between">
              <span className="font-mono text-sm font-extrabold text-[var(--text-primary)]">
                {currentWin.true_label}
              </span>
              <StatusBadge
                status={mapSimClassToBadgeStatus(currentWin.true_label)}
              />
            </div>
          </div>

          <div className="rounded-xl border border-[var(--border)] bg-[var(--surface-alt)] p-3">
            <span className="block text-[0.65rem] font-bold uppercase tracking-wider text-[var(--text-muted)]">
              Frozen RF Window Predicted Class (ep)
            </span>
            <div className="mt-1.5 flex items-center justify-between">
              <span className="font-mono text-sm font-extrabold text-[var(--brand-teal)]">
                {currentWin.predicted_class}
              </span>
              <span className="font-mono text-xs text-[var(--text-muted)]">
                raw: {currentWin.predicted_class_raw}
              </span>
            </div>
          </div>

          <div className="rounded-xl border border-[var(--border)] bg-[var(--surface-alt)] p-3">
            <span className="block text-[0.65rem] font-bold uppercase tracking-wider text-[var(--text-muted)]">
              Persisted Alarm State (k=3, m=3)
            </span>
            <div className="mt-1.5 flex items-center justify-between">
              <span className="font-mono text-sm font-extrabold text-[var(--text-primary)]">
                {currentWin.alarm_state}
              </span>
              <StatusBadge
                status={mapSimClassToBadgeStatus(currentWin.alarm_state)}
              />
            </div>
          </div>

          <div className="rounded-xl border border-[var(--border)] bg-[var(--surface-alt)] p-3">
            <span className="block text-[0.65rem] font-bold uppercase tracking-wider text-[var(--text-muted)]">
              Frozen RF model_score (Top-1)
            </span>
            <div className="mt-1.5 flex items-baseline justify-between font-mono">
              <span className="text-lg font-extrabold text-[var(--brand-green)]">
                {currentWin.model_score.toFixed(3)}
              </span>
              <span className="text-xs text-[var(--text-muted)]">
                pred score: {currentWin.predicted_class_model_score.toFixed(3)}
              </span>
            </div>
          </div>

          <div className="rounded-xl border border-[var(--border)] bg-[var(--surface-alt)] p-3">
            <span className="block text-[0.65rem] font-bold uppercase tracking-wider text-[var(--text-muted)]">
              Advisory unknown_anomaly Flag
            </span>
            <div className="mt-1.5 flex items-center justify-between font-mono">
              <span
                className={`text-sm font-extrabold ${
                  currentWin.unknown_anomaly
                    ? "text-[var(--status-warning)]"
                    : "text-[var(--text-muted)]"
                }`}
              >
                {currentWin.unknown_anomaly ? "FLAGGED (true)" : "false"}
              </span>
              <span className="text-xs text-[var(--text-muted)]">
                score: {currentWin.anomaly_score.toFixed(2)}
              </span>
            </div>
          </div>
        </div>

        {/* Visual Timeline Ribbon across all 95 windows */}
        <div className="space-y-1.5 pt-2">
          <div className="flex flex-wrap items-center justify-between gap-2 text-[0.7rem] font-mono text-[var(--text-muted)]">
            <span>
              Timeline Tracks (click any window cell to jump): True Label ·
              Predicted Class · Alarm (k=3,m=3) · unknown_anomaly
            </span>
            <div className="flex flex-wrap gap-2">
              {Object.entries(CLASS_COLORS).map(([clsName, col]) => (
                <span key={clsName} className="inline-flex items-center gap-1">
                  <span
                    className="h-2.5 w-2.5 rounded-xs"
                    style={{ backgroundColor: col }}
                  />
                  {clsName}
                </span>
              ))}
            </div>
          </div>

          {(
            [
              { label: "True Label", getter: (w: typeof currentWin) => w.true_label },
              {
                label: "RF Pred (ep)",
                getter: (w: typeof currentWin) => w.predicted_class,
              },
              {
                label: "Alarm (k3m3)",
                getter: (w: typeof currentWin) => w.alarm_state,
              },
              {
                label: "Unknown Flag",
                getter: (w: typeof currentWin) =>
                  w.unknown_anomaly ? "sensor_fault" : "normal",
              },
            ] as const
          ).map((track) => (
            <div key={track.label} className="flex items-center gap-2">
              <span className="w-24 shrink-0 font-mono text-[0.68rem] font-semibold text-[var(--text-secondary)]">
                {track.label}
              </span>
              <div className="grid flex-1 grid-cols-95 gap-px overflow-hidden rounded bg-[var(--border)] p-0.5">
                {replay.windows.map((w, idx) => {
                  const val = track.getter(w);
                  const bg = CLASS_COLORS[val] || "var(--surface-muted)";
                  return (
                    <button
                      key={w.window_idx}
                      type="button"
                      onClick={() => onChangePlaybackIdx(idx)}
                      title={`Win #${w.window_idx} (${w.t_mid}s): ${val}`}
                      style={{ backgroundColor: bg }}
                      className={`h-3.5 w-full transition-opacity ${
                        idx === safeIdx
                          ? "ring-2 ring-[var(--text-primary)] z-10"
                          : "opacity-85 hover:opacity-100"
                      }`}
                    />
                  );
                })}
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Advisory AI / Template Explanation Panel for Current Window (when one exists) */}
      <section className="panel-surface rounded-2xl p-4 sm:p-5 space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <Info className="h-4 w-4 text-[var(--brand-teal)]" />
            <h3 className="subsection-title">
              Explanation (advisory)
            </h3>
          </div>
          {explanationsData && explanationsData.explanations.length > 0 && (
            <div className="flex flex-wrap items-center gap-1.5 text-xs font-mono">
              <span className="text-[var(--text-muted)]">
                Explained windows ({explanationsData.explanations.length}):
              </span>
              {explanationsData.explanations.map((exp) => {
                const isCur = exp.window_idx === currentWin.window_idx;
                return (
                  <button
                    key={exp.window_idx}
                    type="button"
                    onClick={() => onChangePlaybackIdx(exp.window_idx)}
                    className={`rounded px-2 py-0.5 font-mono text-xs font-bold transition ${
                      isCur
                        ? "bg-[var(--brand-green)] text-white"
                        : "border border-[var(--border)] bg-[var(--surface-alt)] text-[var(--text-secondary)] hover:border-[var(--brand-teal)]"
                    }`}
                  >
                    #{exp.window_idx}
                  </button>
                );
              })}
            </div>
          )}
        </div>

        {currentExplanation ? (
          <div
            role="region"
            aria-label="Explanation (advisory)"
            className="rounded-xl border border-[var(--border)] bg-[var(--surface-alt)] p-4 space-y-3"
          >
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="flex flex-wrap items-center gap-2">
                <span
                  className={`inline-flex items-center rounded-full px-2.5 py-0.5 font-mono text-xs font-bold ${
                    currentExplanation.source === "ai"
                      ? "bg-[var(--status-success-bg)] text-[var(--status-success)] border border-[var(--status-success-border)]"
                      : "bg-[var(--status-warning-bg)] text-[var(--status-warning)] border border-[var(--status-warning-border)]"
                  }`}
                >
                  {currentExplanation.source === "ai"
                    ? "AI-generated"
                    : "Template (offline fallback)"}
                </span>
                <span className="font-mono text-xs text-[var(--text-muted)]">
                  Model: {currentExplanation.model_id} · Guard:{" "}
                  {currentExplanation.guard_result.passed ? "PASSED" : "FALLBACK"}
                </span>
              </div>
              <span className="rounded border border-[var(--border)] bg-[var(--surface)] px-2.5 py-0.5 font-mono text-xs font-semibold text-[var(--text-secondary)]">
                Advisory text only. It does not change the alarm.
              </span>
            </div>

            <p className="text-sm leading-relaxed text-[var(--text-primary)]">
              {currentExplanation.summary}
            </p>

            <div className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-3 text-xs text-[var(--text-secondary)]">
              <strong className="text-[var(--text-primary)]">
                Suggested Check:
              </strong>{" "}
              {currentExplanation.suggested_check}
            </div>

            {currentWin.true_label !== currentWin.predicted_class && (
              <p
                data-testid="replay-explainer-mismatch-note"
                className="font-mono text-xs font-semibold text-[var(--status-warning)]"
              >
                Explains the classifier&apos;s output only; it does not know the true state.
              </p>
            )}

            <details className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-3 text-xs">
              <summary className="cursor-pointer font-mono text-xs font-bold text-[var(--text-primary)]">
                Packet values used to build this explanation (Window #
                {currentExplanation.window_idx} · {currentExplanation.window_id})
              </summary>
              <div className="mt-3 grid grid-cols-1 gap-2 font-mono text-[0.72rem] text-[var(--text-secondary)] sm:grid-cols-2 lg:grid-cols-3">
                <div>
                  <strong>Window / Run:</strong> {currentExplanation.packet.window_id} (
                  {currentExplanation.packet.run_id})
                </div>
                <div>
                  <strong>Simulated Time:</strong>{" "}
                  {currentExplanation.packet.time_s.t_start}s –{" "}
                  {currentExplanation.packet.time_s.t_end}s (mid{" "}
                  {currentExplanation.packet.time_s.t_mid}s)
                </div>
                <div>
                  <strong>RF Predicted Class:</strong>{" "}
                  {currentExplanation.packet.predicted_class} (model_score{" "}
                  {currentExplanation.packet.model_score})
                </div>
                <div>
                  <strong>Persisted Alarm (k=3,m=3):</strong>{" "}
                  {currentExplanation.packet.alarm_state}
                </div>
                <div>
                  <strong>Threshold-Rule Class:</strong>{" "}
                  {currentExplanation.packet.threshold_rule_class}
                </div>
                <div>
                  <strong>Unknown Anomaly:</strong>{" "}
                  {String(currentExplanation.packet.unknown_anomaly)} (anomaly_score{" "}
                  {currentExplanation.packet.anomaly_score})
                </div>
                <div>
                  <strong>Node Mean Tilt (deg):</strong> N1{" "}
                  {currentExplanation.packet.node_mean_tilt_deg.NODE_01}, N2{" "}
                  {currentExplanation.packet.node_mean_tilt_deg.NODE_02}, N3{" "}
                  {currentExplanation.packet.node_mean_tilt_deg.NODE_03}, N4{" "}
                  {currentExplanation.packet.node_mean_tilt_deg.NODE_04}
                </div>
                <div>
                  <strong>Piezo RMS (V):</strong> net{" "}
                  {currentExplanation.packet.piezo_rms.network_mean} (N1{" "}
                  {currentExplanation.packet.piezo_rms.NODE_01}, N2{" "}
                  {currentExplanation.packet.piezo_rms.NODE_02}, N3{" "}
                  {currentExplanation.packet.piezo_rms.NODE_03}, N4{" "}
                  {currentExplanation.packet.piezo_rms.NODE_04})
                </div>
                <div>
                  <strong>LiDAR Range Change (m):</strong> hub{" "}
                  {currentExplanation.packet.lidar_range_change_m.hub_common_mode_m}{" "}
                  (N1 {currentExplanation.packet.lidar_range_change_m.NODE_01}, N2{" "}
                  {currentExplanation.packet.lidar_range_change_m.NODE_02}, N3{" "}
                  {currentExplanation.packet.lidar_range_change_m.NODE_03}, N4{" "}
                  {currentExplanation.packet.lidar_range_change_m.NODE_04})
                </div>
                <div>
                  <strong>Comm Missing Fraction:</strong> net{" "}
                  {currentExplanation.packet.comm.missing_fraction_network} (N1{" "}
                  {currentExplanation.packet.comm.NODE_01_missing_fraction}, N2{" "}
                  {currentExplanation.packet.comm.NODE_02_missing_fraction}, N3{" "}
                  {currentExplanation.packet.comm.NODE_03_missing_fraction}, N4{" "}
                  {currentExplanation.packet.comm.NODE_04_missing_fraction})
                </div>
                <div>
                  <strong>Comm Latency (s):</strong> mean{" "}
                  {currentExplanation.packet.comm.latency_mean_s}s, max{" "}
                  {currentExplanation.packet.comm.latency_max_s}s
                </div>
                <div>
                  <strong>Radial Convergence & SNR Bin:</strong> residual{" "}
                  {currentExplanation.packet.radial_convergence_residual_m}m, cos{" "}
                  {currentExplanation.packet.radial_convergence_weighted_cos}, SNR bin{" "}
                  {currentExplanation.packet.snr_vs_imu_noise_bin}
                </div>
              </div>
            </details>
          </div>
        ) : (
          <p className="text-xs text-[var(--text-muted)]">
            No advisory explanation generated for Window #{currentWin.window_idx} (explanations are
            pre-computed only for windows where the alarm state changes, the unknown_anomaly flag turns
            on, or the classifier&apos;s first correct or missed detection occurs; click one of the
            explained window badges above to view).
          </p>
        )}
      </section>

      {/* Current Window Sensor MetricTiles (Reusing existing MetricTile component) */}
      <section>
        <h3 className="subsection-title mb-3">
          Current Window Per-Node Tilt & Multi-Sensor Readings (Window #
          {currentWin.window_idx})
        </h3>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 lg:grid-cols-8">
          <MetricTile
            label="NODE_01 Tilt"
            value={currentWin.node_mean_tilt_deg.NODE_01}
            unit="°"
            decimals={3}
          />
          <MetricTile
            label="NODE_02 Tilt"
            value={currentWin.node_mean_tilt_deg.NODE_02}
            unit="°"
            decimals={3}
          />
          <MetricTile
            label="NODE_03 Tilt"
            value={currentWin.node_mean_tilt_deg.NODE_03}
            unit="°"
            decimals={3}
          />
          <MetricTile
            label="NODE_04 Tilt"
            value={currentWin.node_mean_tilt_deg.NODE_04}
            unit="°"
            decimals={3}
          />
          <MetricTile
            label="Piezo RMS Net"
            value={currentWin.piezo_rms.network_mean}
            unit="V"
            decimals={4}
          />
          <MetricTile
            label="LiDAR Max Res"
            value={currentWin.lidar_range_change_m.hub_residual_abs_max_m}
            unit="m"
            decimals={3}
          />
          <MetricTile
            label="Comm Missing"
            value={currentWin.comm_missing_fraction.network * 100}
            unit="%"
            decimals={1}
          />
          <MetricTile
            label="Conv Residual"
            value={currentWin.dir_convergence_residual_m}
            unit="m"
            decimals={2}
          />
        </div>
      </section>

      {/* Multi-Node Recharts Timelines (Primary Tilt & Model Score charts visible; secondary Piezo, LiDAR & Single-Node charts inside details) */}
      <section className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Chart 1: Per-Node Mean Tilt vs Time */}
        <div className="panel-surface rounded-2xl p-4 sm:p-5">
          <div className="mb-3 flex items-center justify-between">
            <h3 className="text-sm font-bold text-[var(--text-primary)]">
              Per-Node Mean Tilt vs. Time (deg) — Sim Threshold ~0.607°
            </h3>
            <span className="font-mono text-xs text-[var(--text-muted)]">
              4 Corner Nodes
            </span>
          </div>
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartSeries}>
                <CartesianGrid stroke="var(--chart-grid)" vertical={false} />
                <XAxis
                  dataKey="t_mid"
                  type="number"
                  domain={[0, 240]}
                  ticks={[0, 60, 120, 180, 240]}
                  unit=" s"
                  tick={{ fontSize: 11, fill: "var(--chart-labels)" }}
                />
                <YAxis
                  tick={{ fontSize: 11, fill: "var(--chart-labels)" }}
                  unit="°"
                />
                <ReferenceLine
                  y={0.607}
                  stroke="var(--status-warning)"
                  strokeDasharray="3 3"
                  label={{
                    value: "Sim Basin Th (0.61°)",
                    fill: "var(--status-warning)",
                    fontSize: 10,
                  }}
                />
                <ReferenceLine
                  x={currentWin.t_mid}
                  stroke="var(--status-danger)"
                  strokeWidth={1.5}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "var(--chart-tooltip-bg)",
                    borderColor: "var(--chart-tooltip-border)",
                    color: "var(--chart-tooltip-text)",
                  }}
                />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="NODE_01_tilt"
                  stroke="var(--chart-series-1)"
                  dot={false}
                  strokeWidth={2}
                />
                <Line
                  type="monotone"
                  dataKey="NODE_02_tilt"
                  stroke="var(--chart-series-2)"
                  dot={false}
                  strokeWidth={2}
                />
                <Line
                  type="monotone"
                  dataKey="NODE_03_tilt"
                  stroke="var(--chart-series-3)"
                  dot={false}
                  strokeWidth={2}
                />
                <Line
                  type="monotone"
                  dataKey="NODE_04_tilt"
                  stroke="var(--chart-series-4)"
                  dot={false}
                  strokeWidth={2}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Chart 4: Frozen RF model_score & Comm Missing Fraction vs Time */}
        <div className="panel-surface rounded-2xl p-4 sm:p-5">
          <div className="mb-3 flex items-center justify-between">
            <h3 className="text-sm font-bold text-[var(--text-primary)]">
              Frozen RF model_score &amp; Comm Missing Fraction vs. Time
            </h3>
            <span className="font-mono text-xs text-[var(--text-muted)]">
              model_score (0–1) &amp; Packet Loss
            </span>
          </div>
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartSeries}>
                <CartesianGrid stroke="var(--chart-grid)" vertical={false} />
                <XAxis
                  dataKey="t_mid"
                  type="number"
                  domain={[0, 240]}
                  ticks={[0, 60, 120, 180, 240]}
                  unit=" s"
                  tick={{ fontSize: 11, fill: "var(--chart-labels)" }}
                />
                <YAxis
                  domain={[0, 1]}
                  tick={{ fontSize: 11, fill: "var(--chart-labels)" }}
                />
                <ReferenceLine
                  x={currentWin.t_mid}
                  stroke="var(--status-danger)"
                  strokeWidth={1.5}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "var(--chart-tooltip-bg)",
                    borderColor: "var(--chart-tooltip-border)",
                    color: "var(--chart-tooltip-text)",
                  }}
                />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="model_score"
                  name="Frozen RF Top-1 model_score"
                  stroke="var(--brand-green)"
                  dot={false}
                  strokeWidth={2.5}
                />
                <Line
                  type="monotone"
                  dataKey="comm_missing_net"
                  name="Network Comm Missing Fraction"
                  stroke="var(--chart-series-4)"
                  dot={false}
                  strokeWidth={2}
                />
                <Line
                  type="stepAfter"
                  dataKey="unknown_anomaly_num"
                  name="unknown_anomaly Flag (0/1)"
                  stroke="var(--chart-series-5)"
                  dot={false}
                  strokeWidth={1.8}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </section>

      {/* Secondary Charts: Piezo RMS and LiDAR Hub Range Change collapsed inside details */}
      <details className="panel-surface rounded-2xl p-5 sm:p-6">
        <summary className="cursor-pointer font-mono text-xs font-bold text-[var(--text-primary)]">
          How do piezo acoustic vibration energy (V) and LiDAR hub range change (m) evolve over the 240 s run?
        </summary>
        <div className="mt-4 grid grid-cols-1 gap-6 lg:grid-cols-2">
          {/* Chart 2: Piezo RMS vs Time */}
          <div className="rounded-xl border border-[var(--border)] bg-[var(--surface-alt)] p-4">
            <div className="mb-3 flex items-center justify-between">
              <h3 className="text-sm font-bold text-[var(--text-primary)]">
                Piezo RMS Acoustic Energy vs. Time (V)
              </h3>
              <span className="font-mono text-xs text-[var(--text-muted)]">
                Network Mean &amp; Per-Node
              </span>
            </div>
            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartSeries}>
                  <CartesianGrid stroke="var(--chart-grid)" vertical={false} />
                  <XAxis
                    dataKey="t_mid"
                    type="number"
                    domain={[0, 240]}
                    ticks={[0, 60, 120, 180, 240]}
                    unit=" s"
                    tick={{ fontSize: 11, fill: "var(--chart-labels)" }}
                  />
                  <YAxis tick={{ fontSize: 11, fill: "var(--chart-labels)" }} />
                  <ReferenceLine
                    y={0.028}
                    stroke="var(--status-warning)"
                    strokeDasharray="3 3"
                    label={{
                      value: "Piezo Th (0.028)",
                      fill: "var(--status-warning)",
                      fontSize: 10,
                    }}
                  />
                  <ReferenceLine
                    x={currentWin.t_mid}
                    stroke="var(--status-danger)"
                    strokeWidth={1.5}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "var(--chart-tooltip-bg)",
                      borderColor: "var(--chart-tooltip-border)",
                      color: "var(--chart-tooltip-text)",
                    }}
                  />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="piezo_net"
                    name="Network Mean Piezo RMS"
                    stroke="var(--status-danger)"
                    dot={false}
                    strokeWidth={2.5}
                  />
                  <Line
                    type="monotone"
                    dataKey="NODE_01_piezo"
                    stroke="var(--chart-series-1)"
                    dot={false}
                    strokeWidth={1.5}
                  />
                  <Line
                    type="monotone"
                    dataKey="NODE_02_piezo"
                    stroke="var(--chart-series-3)"
                    dot={false}
                    strokeWidth={1.5}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Chart 3: LiDAR Hub Range Change vs Time */}
          <div className="rounded-xl border border-[var(--border)] bg-[var(--surface-alt)] p-4">
            <div className="mb-3 flex items-center justify-between">
              <h3 className="text-sm font-bold text-[var(--text-primary)]">
                LiDAR Hub Range Change vs. Time (m)
              </h3>
              <span className="font-mono text-xs text-[var(--text-muted)]">
                4 Corner Targets + Hub Common Mode
              </span>
            </div>
            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartSeries}>
                  <CartesianGrid stroke="var(--chart-grid)" vertical={false} />
                  <XAxis
                    dataKey="t_mid"
                    type="number"
                    domain={[0, 240]}
                    ticks={[0, 60, 120, 180, 240]}
                    unit=" s"
                    tick={{ fontSize: 11, fill: "var(--chart-labels)" }}
                  />
                  <YAxis
                    tick={{ fontSize: 11, fill: "var(--chart-labels)" }}
                    unit="m"
                  />
                  <ReferenceLine
                    x={currentWin.t_mid}
                    stroke="var(--status-danger)"
                    strokeWidth={1.5}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "var(--chart-tooltip-bg)",
                      borderColor: "var(--chart-tooltip-border)",
                      color: "var(--chart-tooltip-text)",
                    }}
                  />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="NODE_01_lidar"
                    stroke="var(--chart-series-1)"
                    dot={false}
                    strokeWidth={2}
                  />
                  <Line
                    type="monotone"
                    dataKey="NODE_02_lidar"
                    stroke="var(--chart-series-2)"
                    dot={false}
                    strokeWidth={2}
                  />
                  <Line
                    type="monotone"
                    dataKey="NODE_03_lidar"
                    stroke="var(--chart-series-3)"
                    dot={false}
                    strokeWidth={2}
                  />
                  <Line
                    type="monotone"
                    dataKey="NODE_04_lidar"
                    stroke="var(--chart-series-4)"
                    dot={false}
                    strokeWidth={2}
                  />
                  <Line
                    type="monotone"
                    dataKey="hub_common_mode"
                    stroke="var(--chart-series-5)"
                    strokeDasharray="4 4"
                    dot={false}
                    strokeWidth={1.8}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </details>

      {/* Single-Node Inspection Secondary Charts inside details */}
      <details className="panel-surface rounded-2xl p-5 sm:p-6">
        <summary className="cursor-pointer font-mono text-xs font-bold text-[var(--text-primary)]">
          What do the cumulative single-node tilt, piezo, and LiDAR traces look like up to the current playback window?
        </summary>
        <div className="mt-4 space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-2">
              <Sliders className="h-4 w-4 text-[var(--brand-teal)]" />
              <h3 className="subsection-title">
                Single-Node Inspection (Up to Current Playback Window, simulated
                seconds)
              </h3>
            </div>
            <div className="flex items-center gap-1.5">
              {(["NODE_01", "NODE_02", "NODE_03", "NODE_04"] as const).map(
                (nd) => (
                  <button
                    key={nd}
                    type="button"
                    onClick={() => onSelectTrendNode(nd)}
                    className={`rounded px-3 py-1 font-mono text-xs font-bold ${
                      selectedTrendNode === nd
                        ? "bg-[var(--brand-green)] text-white"
                        : "bg-[var(--surface-alt)] text-[var(--text-secondary)]"
                    }`}
                  >
                    {nd}
                  </button>
                )
              )}
            </div>
          </div>

          <div className="grid grid-cols-1 gap-6 md:grid-cols-3">
            <SimTrendChart
              data={trendChartHistory.map((p) => ({ t: p.t, value: p.tilt }))}
              label={`${selectedTrendNode} Mean Tilt (°)`}
              unit="°"
              color="var(--chart-series-3)"
              reference={{ y: 0.607, label: "Sim basin threshold 0.607°" }}
            />
            <SimTrendChart
              data={trendChartHistory.map((p) => ({ t: p.t, value: p.piezo }))}
              label={`${selectedTrendNode} Piezo RMS (V)`}
              unit="V"
              color="var(--chart-series-5)"
              reference={{ y: 0.028, label: "Sim piezo threshold 0.028 V" }}
            />
            <SimTrendChart
              data={trendChartHistory.map((p) => ({
                t: p.t,
                value: p.lidarCm,
              }))}
              label={`${selectedTrendNode} LiDAR Range Change (cm)`}
              unit="cm"
              color="var(--chart-series-1)"
            />
          </div>
        </div>
      </details>
    </div>
  );
}

/* ══════════════════════════════════════════════════════════════════════════ */
/* VIEW 5: PARAMETERS                                                         */
/* ══════════════════════════════════════════════════════════════════════════ */
function ParametersSection({ parameters }: { parameters: SimParametersData }) {
  return (
    <div className="space-y-8">
      {/* Start here guidance line */}
      <p
        data-testid="start-here-parameters"
        className="rounded-lg border border-[var(--border)] bg-[var(--surface-alt)] px-4 py-2.5 text-xs font-medium text-[var(--text-primary)]"
      >
        <strong className="font-bold text-[var(--brand-green)]">
          Start here:
        </strong>{" "}
        Review the 6 unsourced parameter cards that must be calibrated before
        field use, then expand the full literature provenance table below.
      </p>

      {/* What Is Unsourced List (Visible cards) */}
      <section className="panel-surface rounded-2xl p-5 sm:p-6 space-y-4">
        <div className="flex items-center gap-2">
          <Layers className="h-5 w-5 text-[var(--status-warning)]" />
          <div>
            <p className="eyebrow">Unsourced Items &amp; Literature Leads</p>
            <h2 className="subsection-title">
              Parameters That Must Be Sourced Before Claiming Site Calibration
            </h2>
          </div>
        </div>

        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          {parameters.unsourced_items.map((u) => (
            <div
              key={u.item}
              className="rounded-xl border border-[var(--border)] bg-[var(--surface-alt)] p-4 space-y-2"
            >
              <div className="flex items-start justify-between gap-2">
                <h3 className="text-sm font-bold text-[var(--text-primary)]">
                  {u.item}
                </h3>
                <span className="shrink-0 rounded border border-[var(--status-warning-border)] bg-[var(--status-warning-bg)] px-2 py-0.5 font-mono text-[0.65rem] font-bold text-[var(--status-warning)]">
                  {u.status}
                </span>
              </div>
              <p className="text-xs text-[var(--text-secondary)]">
                <strong>Literature Leads:</strong> {u.literature_leads}
              </p>
              <p className="text-xs text-[var(--text-muted)]">
                <strong>Impact:</strong> {u.impact}
              </p>
            </div>
          ))}
        </div>
      </section>

      {/* Literature Basis Table with PLACEHOLDER tags collapsed inside details */}
      <details className="panel-surface rounded-2xl p-5 sm:p-6">
        <summary className="cursor-pointer font-mono text-xs font-bold text-[var(--text-primary)]">
          Which simulator parameter ranges have cited longwall literature sources versus PLACEHOLDER status ({parameters.source_file})?
        </summary>
        <div className="mt-4 space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <span className="rounded-full bg-[var(--status-warning-bg)] px-3 py-1 font-mono text-xs font-bold text-[var(--status-warning)]">
              All Unsourced Values Explicitly Tagged PLACEHOLDER
            </span>
          </div>

          <div className="overflow-x-auto rounded-xl border border-[var(--table-border)]">
            <table className="w-full text-left text-xs">
              <thead className="bg-[var(--table-header-bg)] font-mono text-[0.7rem] uppercase text-[var(--table-header-text)]">
                <tr>
                  <th className="px-3.5 py-3">Parameter / Topic</th>
                  <th className="px-3.5 py-3">Status Tag</th>
                  <th className="px-3.5 py-3">Simulator Setting &amp; Status</th>
                  <th className="px-3.5 py-3">
                    Literature Basis &amp; Cited Sources
                  </th>
                  <th className="px-3.5 py-3">Assessment / Scope Note</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[var(--table-border)]">
                {parameters.literature_table.map((row) => (
                  <tr
                    key={row.parameter_or_topic}
                    className="bg-[var(--table-row-bg)] hover:bg-[var(--surface-alt)]"
                  >
                    <td className="px-3.5 py-3 font-bold text-[var(--text-primary)]">
                      {row.parameter_or_topic}
                    </td>
                    <td className="px-3.5 py-3">
                      <span
                        className={`inline-flex items-center rounded-full px-2.5 py-0.5 font-mono text-[0.68rem] font-bold ${
                          row.is_placeholder
                            ? "bg-[var(--status-warning-bg)] text-[var(--status-warning)] border border-[var(--status-warning-border)]"
                            : "bg-[var(--status-success-bg)] text-[var(--status-success)] border border-[var(--status-success-border)]"
                        }`}
                      >
                        {row.status_tag}
                      </span>
                    </td>
                    <td className="px-3.5 py-3 font-mono text-[var(--text-primary)]">
                      {row.simulator_setting_and_status}
                    </td>
                    <td className="px-3.5 py-3 leading-relaxed text-[var(--text-secondary)]">
                      {row.literature_basis_and_sources}
                    </td>
                    <td className="px-3.5 py-3 leading-relaxed text-[var(--text-secondary)]">
                      {row.assessment_or_scope_note}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </details>

      {/* Timescale Note */}
      <section className="panel-surface rounded-2xl p-5 sm:p-6 space-y-2">
        <p className="eyebrow">Timescale Specification Note</p>
        <h3 className="subsection-title">
          240 s Sensor Time vs. 24 h (360×) Slow-Process Label vs. 30-Day
          Longwall Duration
        </h3>
        <p className="text-xs leading-relaxed text-[var(--text-secondary)]">
          {parameters.timescale_note}
        </p>
      </section>
    </div>
  );
}
