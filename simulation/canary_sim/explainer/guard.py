"""Deterministic guard that verifies every advisory explanation before it can be shown."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import re
from typing import Any

from .packet import VALID_CLASSES, ExplainerPacket

CONFIG_PATH = Path(__file__).resolve().parent / "explainer_config.json"

# Match standalone integers and decimals (with optional leading minus sign),
# while ignoring numbers glued inside identifiers like NODE_01 or run_test_014_w20.
NUMBER_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9_])-?\d+(?:\.\d+)?(?![A-Za-z0-9_])")
NODE_ID_RE = re.compile(r"\bNODE_0[1-4]\b")
WINDOW_OR_RUN_ID_RE = re.compile(r"\brun_[A-Za-z0-9_]+\b")


@dataclass(frozen=True)
class GuardResult:
    passed: bool
    reasons: list[str] = field(default_factory=list)
    extracted_numbers: list[float] = field(default_factory=list)
    unmatched_numbers: list[float] = field(default_factory=list)
    sentence_count_summary: int = 0
    sentence_count_check: int = 0
    char_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "reasons": list(self.reasons),
            "extracted_numbers": list(self.extracted_numbers),
            "unmatched_numbers": list(self.unmatched_numbers),
            "sentence_count_summary": self.sentence_count_summary,
            "sentence_count_check": self.sentence_count_check,
            "char_count": self.char_count,
        }


def load_explainer_config(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    return json.loads(config_path.read_text(encoding="utf-8"))


def count_sentences(text: str) -> int:
    """Count plain sentences while ignoring decimal points inside numbers (e.g. 0.343)."""
    cleaned = text.strip()
    if not cleaned:
        return 0
    # Replace decimal points between digits with a placeholder
    no_dec = re.sub(r"(\d)\.(\d)", r"\1_DEC_\2", cleaned)
    # Replace common abbreviations like deg. or s. at mid-sentence if followed by lowercase
    parts = [p.strip() for p in re.split(r"[.!?]+(?:\s+|$)", no_dec) if p.strip()]
    return len(parts)


def extract_numbers_from_text(text: str, packet: ExplainerPacket) -> list[float]:
    """Extract all standalone numeric tokens from `text`, stripping out `window_id`, `run_id`,
    and `NODE_01..NODE_04` identifiers first."""
    scrubbed = text.replace(packet.window_id, " ").replace(packet.run_id, " ")
    scrubbed = NODE_ID_RE.sub(" ", scrubbed)
    scrubbed = WINDOW_OR_RUN_ID_RE.sub(" ", scrubbed)

    out: list[float] = []
    for m in NUMBER_TOKEN_RE.finditer(scrubbed):
        tok = m.group(0)
        try:
            out.append(float(tok))
        except ValueError:
            continue
    return out


def number_matches_packet(val: float, packet_numbers: list[float], raw_token: str | None = None) -> bool:
    """Check whether `val` matches any number in `packet_numbers` within rounding tolerance
    of the token's decimal places (or 1e-3 absolute tolerance for rounded display)."""
    for p_val in packet_numbers:
        if math.isclose(val, p_val, rel_tol=1e-4, abs_tol=1e-4):
            return True
        # Check standard rounding to 1, 2, 3, 4, or 5 decimal places
        for nd in (1, 2, 3, 4, 5):
            if round(p_val, nd) == val:
                return True
        # Also allow abs(p_val) match if text writes "range change of 0.012 m" for -0.012 m
        if math.isclose(abs(val), abs(p_val), rel_tol=1e-4, abs_tol=1e-4):
            return True
        for nd in (2, 3, 4, 5):
            if round(abs(p_val), nd) == abs(val):
                return True
    return False


def check_label_value_bindings(text: str, packet: ExplainerPacket) -> list[str]:
    """Verify that whenever a node identifier (`NODE_01`..`NODE_04`) or sensor/metric name
    appears bound to a numeric value, that number matches the specific field of `packet`."""
    errors: list[str] = []
    scrubbed = text.replace(packet.window_id, " ").replace(packet.run_id, " ")
    scrubbed = WINDOW_OR_RUN_ID_RE.sub(" ", scrubbed)

    # 1. Scalar metric bindings (e.g. model_score, anomaly_score, hub_common_mode_m, latencies, etc.)
    scalar_fields: list[tuple[str, list[float], str]] = [
        ("model_score", [float(packet.model_score)], r"model_score"),
        ("anomaly_score", [float(packet.anomaly_score)], r"anomaly_score"),
        (
            "hub_common_mode_m",
            [float(packet.lidar_range_change_m.hub_common_mode_m)],
            r"hub_common_mode_m|hub\s+common[- ]mode(?:\s+change|\s+shift|\s+offset)?",
        ),
        (
            "radial_convergence_residual_m",
            [float(packet.radial_convergence_residual_m)],
            r"radial_convergence_residual_m|(?:radial\s+)?convergence\s+residual|radial\s+residual",
        ),
        (
            "radial_convergence_weighted_cos",
            [float(packet.radial_convergence_weighted_cos)],
            r"radial_convergence_weighted_cos|weighted\s+cosine",
        ),
        (
            "missing_fraction_network",
            [float(packet.comm.missing_fraction_network)],
            r"missing_fraction_network|network\s+missing(?:\s+packet)?\s+fraction",
        ),
        (
            "latency_mean_s",
            [float(packet.comm.latency_mean_s)],
            r"latency_mean_s|mean\s+latency",
        ),
        (
            "latency_max_s",
            [float(packet.comm.latency_max_s)],
            r"latency_max_s|max(?:imum)?\s+latency",
        ),
        (
            "piezo_rms.network_mean",
            [float(packet.piezo_rms.network_mean)],
            r"piezo_rms\s+network_mean|network(?:\s+mean)?\s+piezo(?:\s+rms)?|network_mean\s+piezo_rms",
        ),
    ]
    for fname, allowed_vals, label_pat in scalar_fields:
        fwd = (
            rf"\b(?:{label_pat})\b(?:\s+(?:is|of|at|was|reported\s+as|equals|=|:))?\s*\(?"
            rf"(?P<num>-?\d+(?:\.\d+)?)(?![A-Za-z0-9_])"
        )
        for m in re.finditer(fwd, scrubbed, re.IGNORECASE):
            val = float(m.group("num"))
            if not number_matches_packet(val, allowed_vals):
                errors.append(
                    f"unfaithful_label_binding: {fname} bound to {val}, expected {allowed_vals}"
                )
        rev = (
            rf"(?<![A-Za-z0-9_])(?P<num>-?\d+(?:\.\d+)?)\s+(?:m\s+|s\s+|V\s+|deg\s+)?"
            rf"(?:for\s+|on\s+)?\b(?:{label_pat})\b"
        )
        for m in re.finditer(rev, scrubbed, re.IGNORECASE):
            val = float(m.group("num"))
            if not number_matches_packet(val, allowed_vals):
                errors.append(
                    f"unfaithful_label_binding: {fname} bound to {val}, expected {allowed_vals}"
                )

    # 2. Node-specific bindings (`NODE_01`..`NODE_04`)
    for node_idx in (1, 2, 3, 4):
        node_id = f"NODE_0{node_idx}"
        node_tilt = float(getattr(packet.node_mean_tilt_deg, node_id))
        node_piezo = float(getattr(packet.piezo_rms, node_id))
        node_lidar = float(getattr(packet.lidar_range_change_m, node_id))
        node_miss = float(getattr(packet.comm, f"{node_id}_missing_fraction"))
        node_all = [node_tilt, node_piezo, node_lidar, node_miss]

        # Explicit `<NODE_0X>_missing_fraction`
        miss_fwd = (
            rf"\b{node_id}_missing_fraction\b(?:\s+(?:is|of|at|was|reported\s+as|equals|=|:))?\s*\(?"
            rf"(?P<num>-?\d+(?:\.\d+)?)(?![A-Za-z0-9_])"
        )
        for m in re.finditer(miss_fwd, scrubbed, re.IGNORECASE):
            val = float(m.group("num"))
            if not number_matches_packet(val, [node_miss]):
                errors.append(
                    f"unfaithful_label_binding: {node_id}_missing_fraction bound to {val}, expected {node_miss}"
                )
        miss_rev = (
            rf"(?<![A-Za-z0-9_])(?P<num>-?\d+(?:\.\d+)?)\s+(?:for\s+|on\s+)?\b{node_id}_missing_fraction\b"
        )
        for m in re.finditer(miss_rev, scrubbed, re.IGNORECASE):
            val = float(m.group("num"))
            if not number_matches_packet(val, [node_miss]):
                errors.append(
                    f"unfaithful_label_binding: {node_id}_missing_fraction bound to {val}, expected {node_miss}"
                )

        # Forward: `NODE_0X [optional sensor label] <number>` without crossing conjunctions or other metrics
        fwd_pat = (
            rf"\b{node_id}\b(?!_missing_fraction)"
            rf"(?P<mid>(?:(?!\bNODE_0[1-4]\b|\b(?:and|with|while|against|versus|vs|or|around|from|to|between|through|window|"
            rf"anomaly_score|model_score|hub_common_mode_m|radial_convergence|missing_fraction_network|latency_|network_mean)\b|"
            rf"[,.;!?])[^0-9\n]){{0,45}}?)"
            rf"(?P<num>-?\d+(?:\.\d+)?)(?![A-Za-z0-9_])"
        )
        for m in re.finditer(fwd_pat, scrubbed, re.IGNORECASE):
            mid = m.group("mid").lower()
            val = float(m.group("num"))
            before = scrubbed[max(0, m.start() - 35) : m.start()].lower()
            has_other_mid = any(k in mid for k in ("tilt", "deg", "piezo", "lidar", "range", "missing"))
            if "tilt" in mid or "deg" in mid or (not has_other_mid and "tilt" in before):
                allowed = [node_tilt]
            elif "piezo" in mid or (not has_other_mid and "piezo" in before):
                allowed = [node_piezo]
            elif "lidar" in mid or "range" in mid or (not has_other_mid and ("lidar" in before or "range" in before)):
                allowed = [node_lidar]
            elif "missing" in mid or (not has_other_mid and "missing" in before):
                allowed = [node_miss]
            else:
                allowed = node_all
            if not number_matches_packet(val, allowed):
                errors.append(
                    f"unfaithful_label_binding: {node_id} bound to {val}, expected {allowed}"
                )

        # Reverse: `<number> [optional unit] (on|at|for|in|of) NODE_0X`
        rev_node_pat = (
            rf"(?<![A-Za-z0-9_])(?P<num>-?\d+(?:\.\d+)?)"
            rf"(?P<mid>\s*(?:deg|V|m)?\s+(?:on|at|for|in|of)\s+)\b{node_id}\b(?!_missing_fraction)"
        )
        for m in re.finditer(rev_node_pat, scrubbed, re.IGNORECASE):
            val = float(m.group("num"))
            mid = m.group("mid").lower()
            if "deg" in mid:
                allowed = [node_tilt]
            elif "v" in mid.split():
                allowed = [node_piezo]
            else:
                allowed = node_all
            if not number_matches_packet(val, allowed):
                errors.append(
                    f"unfaithful_label_binding: {val} bound to {node_id}, expected {allowed}"
                )

    # 3. Sensor-family bindings when a sensor name is directly bound to a number
    all_tilts = [
        float(packet.node_mean_tilt_deg.NODE_01),
        float(packet.node_mean_tilt_deg.NODE_02),
        float(packet.node_mean_tilt_deg.NODE_03),
        float(packet.node_mean_tilt_deg.NODE_04),
    ]
    all_piezos = [
        float(packet.piezo_rms.network_mean),
        float(packet.piezo_rms.NODE_01),
        float(packet.piezo_rms.NODE_02),
        float(packet.piezo_rms.NODE_03),
        float(packet.piezo_rms.NODE_04),
    ]
    all_lidars = [
        float(packet.lidar_range_change_m.hub_common_mode_m),
        float(packet.lidar_range_change_m.NODE_01),
        float(packet.lidar_range_change_m.NODE_02),
        float(packet.lidar_range_change_m.NODE_03),
        float(packet.lidar_range_change_m.NODE_04),
    ]
    all_missing = [
        float(packet.comm.missing_fraction_network),
        float(packet.comm.NODE_01_missing_fraction),
        float(packet.comm.NODE_02_missing_fraction),
        float(packet.comm.NODE_03_missing_fraction),
        float(packet.comm.NODE_04_missing_fraction),
    ]
    sensor_families: list[tuple[str, list[float], str]] = [
        ("node_mean_tilt_deg", all_tilts, r"(?:node_mean_tilt_deg|mean\s+tilt|tilt(?:\s+reading|\s+value)?)"),
        ("piezo_rms", all_piezos, r"(?:piezo_rms|piezo\s+rms)"),
        (
            "lidar_range_change_m",
            all_lidars,
            r"(?:lidar_range_change_m|lidar\s+range\s+change|lidar\s+change|lidar\s+value)",
        ),
        ("missing_fraction", all_missing, r"(?:missing\s+packet\s+fraction|missing\s+fraction)"),
    ]
    for sname, svals, spat in sensor_families:
        sf_fwd = (
            rf"\b{spat}\b(?:\s+(?:is|of|at|was|reported\s+as|equals|=|:))?\s*\(?"
            rf"(?P<num>-?\d+(?:\.\d+)?)(?![A-Za-z0-9_])"
        )
        for m in re.finditer(sf_fwd, scrubbed, re.IGNORECASE):
            val = float(m.group("num"))
            if not number_matches_packet(val, svals):
                errors.append(
                    f"unfaithful_label_binding: {sname} bound to {val}, expected one of {svals}"
                )
        sf_rev = (
            rf"(?<![A-Za-z0-9_])(?P<num>-?\d+(?:\.\d+)?)\s+(?:deg\s+|V\s+|m\s+)?(?:for\s+|on\s+)?\b{spat}\b"
        )
        for m in re.finditer(sf_rev, scrubbed, re.IGNORECASE):
            val = float(m.group("num"))
            if not number_matches_packet(val, svals):
                errors.append(
                    f"unfaithful_label_binding: {sname} bound to {val}, expected one of {svals}"
                )

    return errors


def packet_supports_trend(packet: ExplainerPacket | dict[str, Any], supporting_fields: list[str]) -> bool:
    """Return True only if the packet explicitly contains a trend or rate-of-change field."""
    pkt_dict = packet.model_dump(mode="json") if isinstance(packet, ExplainerPacket) else dict(packet)
    for key in supporting_fields:
        if key in pkt_dict and pkt_dict[key] is not None:
            return True
    return False


class ExplanationGuard:
    """Deterministic safety and faithfulness guard for advisory explanations."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        enable_label_binding: bool = True,
        enable_trend_check: bool = True,
    ) -> None:
        cfg = config if config is not None else load_explainer_config()
        g_cfg = cfg["guard"]
        self.min_sentences_summary = int(g_cfg["min_sentences_summary"])
        self.max_sentences_summary = int(g_cfg["max_sentences_summary"])
        self.max_sentences_check = int(g_cfg["max_sentences_suggested_check"])
        self.min_total_chars = int(g_cfg["min_total_chars"])
        self.max_total_chars = int(g_cfg["max_total_chars"])
        self.required_sim_words = [w.lower() for w in g_cfg["required_simulation_words"]]
        self.forbidden_claim_patterns = [
            re.compile(pat, re.IGNORECASE) for pat in g_cfg["forbidden_claim_regexes"]
        ]
        self.forbidden_score_patterns = [
            re.compile(pat, re.IGNORECASE) for pat in g_cfg["forbidden_score_terms"]
        ]
        self.forbidden_alarm_patterns = [
            re.compile(pat, re.IGNORECASE) for pat in g_cfg["forbidden_alarm_change_regexes"]
        ]
        self.forbidden_trend_patterns = [
            re.compile(pat, re.IGNORECASE) for pat in g_cfg.get("forbidden_trend_regexes", [])
        ]
        self.trend_supporting_fields = list(
            g_cfg.get(
                "trend_supporting_fields",
                ["trend", "rate_of_change", "slope", "derivative", "tilt_rate", "velocity", "acceleration"],
            )
        )
        self.enable_label_binding = enable_label_binding
        self.enable_trend_check = enable_trend_check

    def check(self, explanation: dict[str, Any] | None, packet: ExplainerPacket) -> GuardResult:
        reasons: list[str] = []

        if not isinstance(explanation, dict):
            return GuardResult(passed=False, reasons=["invalid_json_object"])

        summary = explanation.get("summary")
        suggested_check = explanation.get("suggested_check")
        if not isinstance(summary, str) or not summary.strip():
            reasons.append("missing_or_empty_summary")
            summary = ""
        if not isinstance(suggested_check, str) or not suggested_check.strip():
            reasons.append("missing_or_empty_suggested_check")
            suggested_check = ""

        combined = f"{summary.strip()} {suggested_check.strip()}".strip()
        lower_combined = combined.lower()
        char_count = len(combined)

        # (a) Number faithfulness check: every number in the text must match a packet value within rounding
        packet_nums = packet.all_numeric_values()
        extracted_nums = extract_numbers_from_text(combined, packet)
        unmatched_nums: list[float] = []
        for num in extracted_nums:
            if not number_matches_packet(num, packet_nums):
                unmatched_nums.append(num)
        if unmatched_nums:
            reasons.append(f"unfaithful_numbers: {unmatched_nums}")

        # (b) Class faithfulness check: must name packet.predicted_class and MUST NOT name a different class
        pred_cls = packet.predicted_class
        # Check exact class token or space-separated form for predicted_class
        pred_pattern = re.compile(r"\b" + re.escape(pred_cls) + r"\b", re.IGNORECASE)
        if not pred_pattern.search(combined):
            reasons.append(f"missing_predicted_class: expected '{pred_cls}'")

        other_classes_mentioned: list[str] = []
        for cls_name in VALID_CLASSES:
            if cls_name == pred_cls:
                continue
            # Check both snake_case (`sensor_fault`) and space-separated (`sensor fault`, `comm fault`, `abnormal vibration`)
            variants = {cls_name}
            if "_" in cls_name:
                variants.add(cls_name.replace("_", " "))
            for var in variants:
                pat = re.compile(r"\b" + re.escape(var) + r"\b", re.IGNORECASE)
                if pat.search(combined):
                    other_classes_mentioned.append(cls_name)
                    break
        if other_classes_mentioned:
            reasons.append(
                f"wrong_class_mentioned: named {other_classes_mentioned} instead of only '{pred_cls}'"
            )

        # (c) Forbidden claims: real-world certainty, safety advice, or "probability"/"confidence"
        for pat in self.forbidden_claim_patterns:
            m = pat.search(combined)
            if m:
                reasons.append(f"forbidden_claim: matched '{m.group(0)}'")

        for pat in self.forbidden_score_patterns:
            m = pat.search(combined)
            if m:
                reasons.append(f"forbidden_score_term: matched '{m.group(0)}'")

        # (d) Length limits (sentences and characters)
        n_sent_summary = count_sentences(summary)
        n_sent_check = count_sentences(suggested_check)
        if n_sent_summary < self.min_sentences_summary or n_sent_summary > self.max_sentences_summary:
            reasons.append(
                f"summary_sentence_count_out_of_range: got {n_sent_summary}, expected {self.min_sentences_summary}..{self.max_sentences_summary}"
            )
        if n_sent_check < 1 or n_sent_check > self.max_sentences_check:
            reasons.append(
                f"suggested_check_sentence_count_out_of_range: got {n_sent_check}, expected 1..{self.max_sentences_check}"
            )
        if char_count < self.min_total_chars or char_count > self.max_total_chars:
            reasons.append(
                f"char_length_out_of_range: got {char_count}, expected {self.min_total_chars}..{self.max_total_chars}"
            )

        # (e) Must contain the word "simulated" or "synthetic"
        if not any(w in lower_combined for w in self.required_sim_words):
            reasons.append("missing_simulated_or_synthetic_keyword")

        # (f) Must not state or imply a changed alarm state
        for pat in self.forbidden_alarm_patterns:
            m = pat.search(combined)
            if m:
                reasons.append(f"alarm_mutation_claim: matched '{m.group(0)}'")

        # (g) Label-value binding: when a node name or sensor name appears next to a number,
        # the number must match that specific field of the packet
        if self.enable_label_binding:
            binding_errors = check_label_value_bindings(combined, packet)
            reasons.extend(binding_errors)

        # (h) Unsupported trend / derived words (rising, falling, increasing, decreasing,
        # doubled, spiking, worsening, accelerating, steady/steadily) unless packet has a supporting field
        if self.enable_trend_check and not packet_supports_trend(packet, self.trend_supporting_fields):
            for pat in self.forbidden_trend_patterns:
                m = pat.search(combined)
                if m:
                    reasons.append(f"unsupported_trend_term: matched '{m.group(0)}'")

        return GuardResult(
            passed=(len(reasons) == 0),
            reasons=reasons,
            extracted_numbers=extracted_nums,
            unmatched_numbers=unmatched_nums,
            sentence_count_summary=n_sent_summary,
            sentence_count_check=n_sent_check,
            char_count=char_count,
        )
