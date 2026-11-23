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


class ExplanationGuard:
    """Deterministic safety and faithfulness guard for advisory explanations."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
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

        return GuardResult(
            passed=(len(reasons) == 0),
            reasons=reasons,
            extracted_numbers=extracted_nums,
            unmatched_numbers=unmatched_nums,
            sentence_count_summary=n_sent_summary,
            sentence_count_check=n_sent_check,
            char_count=char_count,
        )
