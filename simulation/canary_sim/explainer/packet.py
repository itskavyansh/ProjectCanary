"""Packet schema (Pydantic) and loaders for exported evaluation samples and dashboard replay files."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from canary_sim import evaluate as EV, features as F, models as M

ValidClass = Literal[
    "normal",
    "gradual",
    "rapid",
    "abnormal_vibration",
    "sensor_fault",
    "comm_fault",
]

VALID_CLASSES: tuple[str, ...] = (
    "normal",
    "gradual",
    "rapid",
    "abnormal_vibration",
    "sensor_fault",
    "comm_fault",
)

SNR_BIN_VALUES: tuple[str, ...] = ("<0.5", "0.5-1", "1-2", "2-5", ">5")


class PacketTime(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    t_start: float
    t_end: float
    t_mid: float


class NodeTiltMap(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    NODE_01: float
    NODE_02: float
    NODE_03: float
    NODE_04: float


class PiezoRmsMap(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    network_mean: float
    NODE_01: float
    NODE_02: float
    NODE_03: float
    NODE_04: float


class LidarRangeChangeMap(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    hub_common_mode_m: float
    NODE_01: float
    NODE_02: float
    NODE_03: float
    NODE_04: float


class CommMetrics(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    missing_fraction_network: float
    NODE_01_missing_fraction: float
    NODE_02_missing_fraction: float
    NODE_03_missing_fraction: float
    NODE_04_missing_fraction: float
    latency_mean_s: float
    latency_max_s: float


class ExplainerPacket(BaseModel):
    """Immutable alarm packet passed to the advisory explainer.

    Contains ONLY observable window summaries and frozen classifier/detector outputs.
    Never holds ground-truth labels (`truth` / `true_label`).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    window_id: str
    run_id: str
    time_s: PacketTime
    predicted_class: ValidClass
    model_score: float = Field(ge=0.0, le=1.0)
    alarm_state: ValidClass
    threshold_rule_class: ValidClass
    unknown_anomaly: bool
    anomaly_score: float
    node_mean_tilt_deg: NodeTiltMap
    piezo_rms: PiezoRmsMap
    lidar_range_change_m: LidarRangeChangeMap
    comm: CommMetrics
    radial_convergence_residual_m: float
    radial_convergence_weighted_cos: float
    snr_vs_imu_noise_bin: str

    def packet_hash(self) -> str:
        raw = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def all_numeric_values(self) -> list[float]:
        """Extract every numeric value in the packet (plus integer node indices 1..4 and k=3, m=3)
        for deterministic numerical verification by the guard."""
        nums: list[float] = [
            float(self.time_s.t_start),
            float(self.time_s.t_end),
            float(self.time_s.t_mid),
            float(self.model_score),
            float(self.anomaly_score),
            float(self.node_mean_tilt_deg.NODE_01),
            float(self.node_mean_tilt_deg.NODE_02),
            float(self.node_mean_tilt_deg.NODE_03),
            float(self.node_mean_tilt_deg.NODE_04),
            float(self.piezo_rms.network_mean),
            float(self.piezo_rms.NODE_01),
            float(self.piezo_rms.NODE_02),
            float(self.piezo_rms.NODE_03),
            float(self.piezo_rms.NODE_04),
            float(self.lidar_range_change_m.hub_common_mode_m),
            float(self.lidar_range_change_m.NODE_01),
            float(self.lidar_range_change_m.NODE_02),
            float(self.lidar_range_change_m.NODE_03),
            float(self.lidar_range_change_m.NODE_04),
            float(self.comm.missing_fraction_network),
            float(self.comm.NODE_01_missing_fraction),
            float(self.comm.NODE_02_missing_fraction),
            float(self.comm.NODE_03_missing_fraction),
            float(self.comm.NODE_04_missing_fraction),
            float(self.comm.latency_mean_s),
            float(self.comm.latency_max_s),
            float(self.radial_convergence_residual_m),
            float(self.radial_convergence_weighted_cos),
        ]
        # Add SNR bin bounds if referenced verbatim from snr_vs_imu_noise_bin
        for tok in self.snr_vs_imu_noise_bin.replace("<", "").replace(">", "").split("-"):
            try:
                nums.append(float(tok))
            except ValueError:
                pass
        return [v for v in nums if math.isfinite(v)]


def _snr_to_bin(snr_val: float) -> str:
    f = float(snr_val)
    for b_lbl, lo, hi in EV.SNR_BINS:
        if lo <= f < hi:
            return b_lbl
    return ">5" if f >= 5.0 else "<0.5"


def load_packets_from_sample_json(path: Path | str) -> list[ExplainerPacket]:
    """Load strictly sanitized `ExplainerPacket` objects from `outputs/explainer_sample/packets.json`.
    Ignores the top-level `"truth"` dictionary completely."""
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    raw_packets = data["packets"] if isinstance(data, dict) and "packets" in data else data
    return [ExplainerPacket.model_validate(item) for item in raw_packets]


def load_packets_from_replay_json(
    replay_path: Path | str,
    features_test_df: Any | None = None,
    threshold_rule_model: Any | None = None,
) -> list[ExplainerPacket]:
    """Build `ExplainerPacket` objects for all 95 windows of a `public/sim/replay_<class>.json` file.

    Uses the fields directly in the replay JSON window record, and supplements
    `threshold_rule_class` and `comm_latency_*` from the frozen `features_test.csv` +
    `ThresholdRuleClassifier` if available (or derives them deterministically from the window fields).
    """
    p = Path(replay_path)
    rep = json.loads(p.read_text(encoding="utf-8"))
    sim_id = str(rep["simulation_id"])
    run_cls = str(rep.get("run_class", "unknown"))
    # Anonymise run_id so no class name appears in run_id
    run_seed = int(rep.get("run_metadata", {}).get("seed", 0))
    anon_run_id = f"run_replay_{run_seed}"
    snr_val = float(rep.get("run_metadata", {}).get("snr_vs_imu_noise", 2.0))
    snr_bin = _snr_to_bin(snr_val)

    # Optional lookup from frozen test DataFrame for exact comm latency and threshold_rule_class
    th_by_win: dict[int, str] = {}
    lat_mean_by_win: dict[int, float] = {}
    lat_max_by_win: dict[int, float] = {}

    if features_test_df is not None:
        sub = features_test_df[features_test_df["simulation_id"] == sim_id].copy()
        if len(sub) > 0:
            if threshold_rule_model is not None:
                th_preds = threshold_rule_model.predict(sub)
                for w_i, pr in zip(sub["window_idx"], th_preds):
                    th_by_win[int(w_i)] = str(pr)
            for _, r in sub.iterrows():
                w_i = int(r["window_idx"])
                lat_mean_by_win[w_i] = round(float(r["comm_latency_mean_s"]), 4)
                lat_max_by_win[w_i] = round(float(r["comm_latency_max_s"]), 4)

    packets: list[ExplainerPacket] = []
    for w in rep["windows"]:
        w_idx = int(w["window_idx"])
        window_id = f"{anon_run_id}_w{w_idx:02d}"
        pred_cls = str(w["predicted_class"])
        pred_score = round(float(w.get("predicted_class_model_score", w["model_score"])), 4)

        # Deterministic threshold-rule fallback if frozen table not passed
        if w_idx in th_by_win:
            th_cls = th_by_win[w_idx]
        else:
            th_cls = _fallback_threshold_rule_from_replay_window(w)

        comm_block = w["comm_missing_fraction"]
        lat_mean = lat_mean_by_win.get(w_idx, round(float(comm_block.get("latency_mean_s", 0.05)), 4))
        lat_max = lat_max_by_win.get(w_idx, round(float(comm_block.get("latency_max_s", 0.12)), 4))

        pkt = ExplainerPacket(
            window_id=window_id,
            run_id=anon_run_id,
            time_s=PacketTime(
                t_start=round(float(w["t_start"]), 2),
                t_end=round(float(w["t_end"]), 2),
                t_mid=round(float(w["t_mid"]), 2),
            ),
            predicted_class=pred_cls,  # type: ignore[arg-type]
            model_score=pred_score,
            alarm_state=str(w["alarm_state"]),  # type: ignore[arg-type]
            threshold_rule_class=th_cls,  # type: ignore[arg-type]
            unknown_anomaly=bool(w["unknown_anomaly"]),
            anomaly_score=round(float(w["anomaly_score"]), 4),
            node_mean_tilt_deg=NodeTiltMap(
                NODE_01=round(float(w["node_mean_tilt_deg"]["NODE_01"]), 4),
                NODE_02=round(float(w["node_mean_tilt_deg"]["NODE_02"]), 4),
                NODE_03=round(float(w["node_mean_tilt_deg"]["NODE_03"]), 4),
                NODE_04=round(float(w["node_mean_tilt_deg"]["NODE_04"]), 4),
            ),
            piezo_rms=PiezoRmsMap(
                network_mean=round(float(w["piezo_rms"]["network_mean"]), 5),
                NODE_01=round(float(w["piezo_rms"]["NODE_01"]), 5),
                NODE_02=round(float(w["piezo_rms"]["NODE_02"]), 5),
                NODE_03=round(float(w["piezo_rms"]["NODE_03"]), 5),
                NODE_04=round(float(w["piezo_rms"]["NODE_04"]), 5),
            ),
            lidar_range_change_m=LidarRangeChangeMap(
                hub_common_mode_m=round(float(w["lidar_range_change_m"]["hub_common_mode_m"]), 4),
                NODE_01=round(float(w["lidar_range_change_m"]["NODE_01"]), 4),
                NODE_02=round(float(w["lidar_range_change_m"]["NODE_02"]), 4),
                NODE_03=round(float(w["lidar_range_change_m"]["NODE_03"]), 4),
                NODE_04=round(float(w["lidar_range_change_m"]["NODE_04"]), 4),
            ),
            comm=CommMetrics(
                missing_fraction_network=round(float(comm_block["network"]), 4),
                NODE_01_missing_fraction=round(float(comm_block["NODE_01"]), 4),
                NODE_02_missing_fraction=round(float(comm_block["NODE_02"]), 4),
                NODE_03_missing_fraction=round(float(comm_block["NODE_03"]), 4),
                NODE_04_missing_fraction=round(float(comm_block["NODE_04"]), 4),
                latency_mean_s=lat_mean,
                latency_max_s=lat_max,
            ),
            radial_convergence_residual_m=round(float(w["dir_convergence_residual_m"]), 3),
            radial_convergence_weighted_cos=round(float(w["dir_convergence_weighted_cos"]), 4),
            snr_vs_imu_noise_bin=snr_bin,
        )
        packets.append(pkt)
    return packets


def _fallback_threshold_rule_from_replay_window(w: dict[str, Any]) -> str:
    """Deterministic approximation of ThresholdRuleClassifier when reading a standalone replay JSON
    without `features_test.csv`."""
    comm = w["comm_missing_fraction"]
    if float(comm["network"]) >= 0.12 or float(comm.get("max_burst_len", 0)) >= 2.0:
        return "comm_fault"
    if float(w["piezo_rms"]["network_mean"]) >= 0.028:
        return "abnormal_vibration"
    tilts = sorted(float(w["node_mean_tilt_deg"][f"NODE_0{i}"]) for i in range(1, 5))
    t_max = tilts[-1]
    t_med = 0.5 * (tilts[1] + tilts[2])
    t_sec = tilts[2]
    hub_cm = abs(float(w["lidar_range_change_m"]["hub_common_mode_m"]))
    hub_res = float(w["lidar_range_change_m"].get("hub_residual_abs_max_m", 0.0))
    if hub_cm >= 0.09 or hub_res >= 0.11 or t_max >= 3.5:
        return "sensor_fault"
    if t_max >= 0.67 and (t_max / max(t_sec, 0.15)) >= 2.4:
        return "sensor_fault"
    if t_max >= 0.607 or t_med >= 0.51:
        return str(w["predicted_class"]) if str(w["predicted_class"]) in ("gradual", "rapid") else "gradual"
    return "normal"
