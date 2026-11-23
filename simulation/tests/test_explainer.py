"""Comprehensive unit tests for simulation/canary_sim/explainer/ with MockChatClient.

Verifies all Step B.7 requirements:
1. Guard rejects a wrong number (hallucinated or derived number not in packet).
2. Guard rejects a wrong class (missing predicted_class or naming a different class).
3. Guard rejects a forbidden phrase (real-world claim, safety/evacuation advice,
   "probability", "confidence", or implied alarm state mutation).
4. Guard rejects an over-long text (or wrong sentence count) and missing "simulated"/"synthetic".
5. Guard accepts a faithful text.
6. Retry works once with the guard's error list; on second failure uses deterministic template fallback.
7. Fallback is used on API error, missing key, and budget cap.
8. Explainer never mutates the packet (immutable Pydantic schema + deep-equality check).
9. Packet-hash cache works and avoids duplicate API spend.
10. No API key string ever appears in logs, cache files, or outputs.
11. The template path and replay/sample packet loaders work fully offline with no network.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from canary_sim.explainer import (  # noqa: E402
    AdvisoryExplainer,
    BudgetTracker,
    ChatClient,
    ExplainerPacket,
    ExplanationGuard,
    MockChatClient,
    generate_template_explanation,
    load_packets_from_replay_json,
    load_packets_from_sample_json,
)


def _sample_packet(
    predicted_class: str = "gradual",
    unknown_anomaly: bool = False,
) -> ExplainerPacket:
    return ExplainerPacket.model_validate(
        {
            "window_id": "run_test_010_w45",
            "run_id": "run_test_010",
            "time_s": {"t_start": 112.5, "t_end": 117.5, "t_mid": 115.0},
            "predicted_class": predicted_class,
            "model_score": 0.7421,
            "alarm_state": predicted_class,
            "threshold_rule_class": predicted_class,
            "unknown_anomaly": unknown_anomaly,
            "anomaly_score": 6.412,
            "node_mean_tilt_deg": {
                "NODE_01": 0.4123,
                "NODE_02": 0.3891,
                "NODE_03": 0.3512,
                "NODE_04": 0.4015,
            },
            "piezo_rms": {
                "network_mean": 0.00412,
                "NODE_01": 0.00405,
                "NODE_02": 0.00418,
                "NODE_03": 0.00411,
                "NODE_04": 0.00414,
            },
            "lidar_range_change_m": {
                "hub_common_mode_m": 0.0012,
                "NODE_01": -0.0145,
                "NODE_02": -0.0132,
                "NODE_03": -0.0118,
                "NODE_04": -0.0139,
            },
            "comm": {
                "missing_fraction_network": 0.0125,
                "NODE_01_missing_fraction": 0.0,
                "NODE_02_missing_fraction": 0.05,
                "NODE_03_missing_fraction": 0.0,
                "NODE_04_missing_fraction": 0.0,
                "latency_mean_s": 0.0512,
                "latency_max_s": 0.142,
            },
            "radial_convergence_residual_m": 2.341,
            "radial_convergence_weighted_cos": 0.9124,
            "snr_vs_imu_noise_bin": "2-5",
        }
    )


def test_guard_accepts_faithful_text_and_all_templates() -> None:
    guard = ExplanationGuard()
    for cls in ("normal", "gradual", "rapid", "abnormal_vibration", "sensor_fault", "comm_fault"):
        for unk in (False, True):
            pkt = _sample_packet(predicted_class=cls, unknown_anomaly=unk)
            tmpl = generate_template_explanation(pkt)
            res = guard.check(tmpl, pkt)
            assert res.passed, f"Template for ({cls}, unk={unk}) failed guard: {res.reasons}"


def test_guard_rejects_wrong_number() -> None:
    guard = ExplanationGuard()
    pkt = _sample_packet(predicted_class="gradual")
    # 0.9999 and 18.5 do not exist in packet
    bad_num_explanation = {
        "summary": (
            "In this simulated window from 112.5 s to 117.5 s, the classifier output is consistent with "
            "gradual movement at model_score 0.7421. Simulated NODE_01 tilt is 0.9999 deg with a derived "
            "difference of 18.5 deg across nodes."
        ),
        "suggested_check": "Inspect simulated NODE_01 tilt (0.4123 deg) and radial convergence residual (2.341 m).",
    }
    res = guard.check(bad_num_explanation, pkt)
    assert not res.passed
    assert any("unfaithful_numbers" in r for r in res.reasons)
    assert 0.9999 in res.unmatched_numbers
    assert 18.5 in res.unmatched_numbers


def test_guard_rejects_wrong_class() -> None:
    guard = ExplanationGuard()
    pkt = _sample_packet(predicted_class="gradual")

    # Names sensor_fault instead of gradual
    wrong_cls_1 = {
        "summary": (
            "In this simulated window from 112.5 s to 117.5 s, the classifier output is consistent with "
            "sensor_fault at model_score 0.7421. Simulated NODE_01 mean tilt is 0.4123 deg."
        ),
        "suggested_check": "Check simulated NODE_01 tilt (0.4123 deg).",
    }
    res1 = guard.check(wrong_cls_1, pkt)
    assert not res1.passed
    assert any("missing_predicted_class" in r for r in res1.reasons)
    assert any("wrong_class_mentioned" in r for r in res1.reasons)

    # Names both gradual AND rapid (contradictory/extra class mention)
    wrong_cls_2 = {
        "summary": (
            "In this simulated window from 112.5 s to 117.5 s, the classifier predicts gradual at "
            "model_score 0.7421, although it looks like rapid movement. Simulated NODE_01 mean tilt is 0.4123 deg."
        ),
        "suggested_check": "Check simulated NODE_01 tilt (0.4123 deg).",
    }
    res2 = guard.check(wrong_cls_2, pkt)
    assert not res2.passed
    assert any("wrong_class_mentioned" in r for r in res2.reasons)


@pytest.mark.parametrize(
    "forbidden_snippet,expected_tag",
    [
        ("Evacuate the area immediately.", "forbidden_claim"),
        ("Collapse imminent on the longwall panel.", "forbidden_claim"),
        ("The ground is subsiding rapidly.", "forbidden_claim"),
        ("The mine is currently unsafe.", "forbidden_claim"),
        ("The classifier probability is 0.7421.", "forbidden_score_term"),
        ("We have high confidence in this reading.", "forbidden_score_term"),
        ("This explanation will downgrade the alarm state.", "alarm_mutation_claim"),
    ],
)
def test_guard_rejects_forbidden_phrases_and_score_terms(
    forbidden_snippet: str, expected_tag: str
) -> None:
    guard = ExplanationGuard()
    pkt = _sample_packet(predicted_class="gradual")
    bad = {
        "summary": (
            f"In this simulated window from 112.5 s to 117.5 s, the output is consistent with gradual "
            f"at model_score 0.7421 and NODE_01 tilt 0.4123 deg. {forbidden_snippet}"
        ),
        "suggested_check": "Inspect simulated NODE_01 tilt (0.4123 deg).",
    }
    res = guard.check(bad, pkt)
    assert not res.passed
    assert any(expected_tag in r for r in res.reasons), f"Expected {expected_tag} in {res.reasons}"


def test_guard_rejects_over_long_text_and_missing_simulated() -> None:
    guard = ExplanationGuard()
    pkt = _sample_packet(predicted_class="gradual")

    # Missing "simulated" / "synthetic"
    no_sim = {
        "summary": (
            "In this window from 112.5 s to 117.5 s, the classifier output is consistent with gradual "
            "at model_score 0.7421. NODE_01 mean tilt is 0.4123 deg and NODE_02 mean tilt is 0.3891 deg."
        ),
        "suggested_check": "Check NODE_01 mean tilt (0.4123 deg) and convergence residual (2.341 m).",
    }
    res_no_sim = guard.check(no_sim, pkt)
    assert not res_no_sim.passed
    assert "missing_simulated_or_synthetic_keyword" in res_no_sim.reasons

    # Over-long text (>3 sentences in summary and >750 chars)
    s = "In this simulated window from 112.5 s to 117.5 s, the output is consistent with gradual at model_score 0.7421 and NODE_01 tilt 0.4123 deg."
    over_long = {
        "summary": f"{s} {s} {s} {s} {s} {s}",
        "suggested_check": "Inspect simulated NODE_01 tilt (0.4123 deg). Also inspect NODE_02 tilt (0.3891 deg).",
    }
    res_long = guard.check(over_long, pkt)
    assert not res_long.passed
    assert any("summary_sentence_count_out_of_range" in r for r in res_long.reasons)
    assert any("char_length_out_of_range" in r for r in res_long.reasons)


def test_retry_on_first_failure_and_fallback_on_second_failure(tmp_path: Path) -> None:
    pkt = _sample_packet(predicted_class="gradual")
    bad_attempt = {
        "summary": (
            "In this simulated window from 112.5 s to 117.5 s, the output is consistent with gradual "
            "at model_score 0.7421. Hallucinated tilt is 9.876 deg."
        ),
        "suggested_check": "Check simulated NODE_01 tilt (0.4123 deg).",
    }
    good_attempt = generate_template_explanation(pkt)

    # Case 1: 1st attempt fails guard, 2nd attempt succeeds -> source == "ai", retried == True
    client_recover = MockChatClient(scripted_responses=[bad_attempt, good_attempt])
    budget1 = BudgetTracker(cap_usd=5.0, cache_dir=tmp_path / "c1", log_path=tmp_path / "log1.jsonl")
    explainer1 = AdvisoryExplainer(client=client_recover, budget=budget1, use_cache=False)
    out1 = explainer1.explain(pkt)
    assert out1.source == "ai"
    assert out1.retried is True
    assert out1.attempts_made == 2
    assert out1.guard_result.passed is True
    assert out1.first_guard_result is not None and not out1.first_guard_result.passed
    # Verify retry prompt included the guard's error list
    assert "unfaithful_numbers" in client_recover.calls[1]["user_prompt"]

    # Case 2: Both attempts fail guard -> deterministic fallback to template
    client_fail2 = MockChatClient(scripted_responses=[bad_attempt, bad_attempt])
    budget2 = BudgetTracker(cap_usd=5.0, cache_dir=tmp_path / "c2", log_path=tmp_path / "log2.jsonl")
    explainer2 = AdvisoryExplainer(client=client_fail2, budget=budget2, use_cache=False)
    out2 = explainer2.explain(pkt)
    assert out2.source == "template"
    assert out2.model_id == "template"
    assert out2.retried is True
    assert out2.attempts_made == 2
    assert out2.guard_result.passed is True
    assert out2.fallback_reason is not None and "guard_rejected_twice" in out2.fallback_reason


def test_fallback_on_api_error_missing_key_and_budget_cap(tmp_path: Path) -> None:
    pkt = _sample_packet(predicted_class="sensor_fault")

    # 1. API error
    c_err = MockChatClient(simulate_api_error=True)
    e_err = AdvisoryExplainer(client=c_err, budget=BudgetTracker(cap_usd=5.0), use_cache=False)
    out_err = e_err.explain(pkt)
    assert out_err.source == "template"
    assert out_err.guard_result.passed is True
    assert out_err.fallback_reason is not None and "api_error" in out_err.fallback_reason

    # 2. Missing key
    c_nokey = MockChatClient(simulate_missing_key=True)
    e_nokey = AdvisoryExplainer(client=c_nokey, budget=BudgetTracker(cap_usd=5.0), use_cache=False)
    out_nokey = e_nokey.explain(pkt)
    assert out_nokey.source == "template"
    assert out_nokey.attempts_made == 0
    assert out_nokey.fallback_reason == "missing_api_key"

    # 3. Budget cap reached (cap = $0.0001, first call spends more, second packet falls back immediately)
    c_cap = MockChatClient(prompt_tokens_per_call=100_000, completion_tokens_per_call=50_000)
    b_cap = BudgetTracker(cap_usd=0.001, cache_dir=tmp_path / "ccap")
    e_cap = AdvisoryExplainer(client=c_cap, budget=b_cap, use_cache=False)
    out_first = e_cap.explain(pkt)
    assert out_first.source == "ai"
    assert b_cap.total_cost_usd >= 0.001

    out_second = e_cap.explain(_sample_packet(predicted_class="comm_fault"))
    assert out_second.source == "template"
    assert out_second.attempts_made == 0
    assert out_second.fallback_reason is not None and "budget_cap_reached" in out_second.fallback_reason


def test_explainer_never_mutates_packet_and_cache_works(tmp_path: Path) -> None:
    pkt = _sample_packet(predicted_class="rapid", unknown_anomaly=True)
    snapshot_before = copy.deepcopy(pkt.model_dump(mode="json"))
    hash_before = pkt.packet_hash()

    # Verify packet is frozen/immutable
    with pytest.raises(Exception):
        pkt.predicted_class = "normal"  # type: ignore[misc]

    client = MockChatClient()
    budget = BudgetTracker(cap_usd=5.0, cache_dir=tmp_path / "cache", log_path=tmp_path / "calls.jsonl")
    explainer = AdvisoryExplainer(client=client, budget=budget, use_cache=True)

    res1 = explainer.explain(pkt)
    assert res1.source == "ai"
    assert res1.from_cache is False
    assert client.call_count == 1
    cost_after_first = budget.total_cost_usd

    # Second call with identical packet hits cache without calling client or spending money
    res2 = explainer.explain(pkt)
    assert res2.source == "ai"
    assert res2.from_cache is True
    assert res2.summary == res1.summary
    assert client.call_count == 1
    assert budget.total_cost_usd == cost_after_first

    assert pkt.model_dump(mode="json") == snapshot_before
    assert pkt.packet_hash() == hash_before


def test_no_key_string_ever_appears_in_logs_or_outputs(tmp_path: Path) -> None:
    fake_secret = "SECRET_TEST_META_KEY_998877665544332211"

    class FailingHttpStub:
        def post(self, url: str, headers: dict, json: dict, timeout: float):
            raise RuntimeError(f"Connection refused for Authorization: Bearer {fake_secret}")

    client = ChatClient(
        model="muse-spark-1.3",
        http_client=FailingHttpStub(),
        api_key=fake_secret,
        max_retries=0,
    )
    log_file = tmp_path / "calls.jsonl"
    cache_dir = tmp_path / "cache"
    budget = BudgetTracker(cap_usd=5.0, cache_dir=cache_dir, log_path=log_file)
    explainer = AdvisoryExplainer(client=client, budget=budget, use_cache=True)

    pkt = _sample_packet(predicted_class="comm_fault")
    out = explainer.explain(pkt)

    assert out.source == "template"
    out_json = json.dumps(out.to_dict())
    assert fake_secret not in out_json

    log_text = log_file.read_text(encoding="utf-8")
    assert fake_secret not in log_text
    assert "[REDACTED]" in log_text


def test_offline_template_and_replay_loaders_work_without_network() -> None:
    replay_path = ROOT.parent / "public" / "sim" / "replay_gradual.json"
    if replay_path.is_file():
        packets = load_packets_from_replay_json(replay_path)
        assert len(packets) == 95
        guard = ExplanationGuard()
        for pkt in (packets[0], packets[45], packets[94]):
            tmpl = generate_template_explanation(pkt)
            g_res = guard.check(tmpl, pkt)
            assert g_res.passed, f"Replay packet {pkt.window_id} template failed guard: {g_res.reasons}"

    sample_path = ROOT / "outputs" / "explainer_sample" / "packets.json"
    if sample_path.is_file():
        sample_pkts = load_packets_from_sample_json(sample_path)
        assert len(sample_pkts) == 150
        guard = ExplanationGuard()
        for pkt in sample_pkts[:10]:
            tmpl = generate_template_explanation(pkt)
            assert guard.check(tmpl, pkt).passed
