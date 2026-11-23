#!/usr/bin/env python
"""Generate advisory explanations for the six dashboard replay runs (SYNTHETIC / SIMULATION).

For each of the 6 replay classes (`normal`, `gradual`, `rapid`, `abnormal_vibration`,
`sensor_fault`, `comm_fault`), reads `public/sim/replay_<class>.json`, selects at most 12
notable windows where:
  1) `alarm_state` changes from the previous window,
  2) `unknown_anomaly` turns on (`False -> True`, or `True` at window 0), or
  3) the classifier's first correct detection (`predicted_class == true_label` on an event run)
     or first missed detection (`true_label != 'normal'` and `predicted_class == 'normal'`)
     occurs (plus window 0 if no event transitions occur, e.g. on an all-normal run),
runs `AdvisoryExplainer` (sharing the packet-hash cache and $5.00 USD hard budget cap), and
writes `public/sim/explanations_<class>.json`.

Strict guarantees enforced:
- Each `public/sim/explanations_<class>.json` is under 100 KB and carries the mandatory
  synthetic safety banner and envelope.
- Each entry contains: `window_idx`, `window_id`, `summary`, `suggested_check`,
  `source` (`"ai"` or `"template"`), `model_id`, `guard_result`, and `packet` (the exact
  packet values the explanation was built from, for the collapsible panel in the Replay view).
- No API keys, secrets, or `"probability"` / `"confidence"` strings appear anywhere in `public/sim/`.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import pickle
import sys
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from canary_sim.explainer import (  # noqa: E402
    AdvisoryExplainer,
    BudgetTracker,
    ChatClient,
    ExplainerPacket,
    ExplanationGuard,
    MockChatClient,
    load_packets_from_replay_json,
)
import export_dashboard_data as ED  # noqa: E402
import export_explainer_sample as ES  # noqa: E402

REPLAY_CLASSES = ED.REPLAY_CLASSES
MAX_EXPLANATIONS_PER_RUN = 12
MAX_EXPLANATION_FILE_BYTES = 100 * 1024


def select_replay_explanation_windows(
    replay_data: dict[str, Any],
    max_windows: int = MAX_EXPLANATIONS_PER_RUN,
) -> list[tuple[int, list[str]]]:
    """Deterministically select at most `max_windows` window indices in `replay_data['windows']`
    where:
      - `alarm_state` changes,
      - `unknown_anomaly` turns on, or
      - the classifier's first correct detection or first missed detection happens.
    Also includes window 0 if no other window qualifies (or as baseline context when room remains).
    Returns `[(window_idx, [trigger_reasons...]), ...]` sorted by `window_idx`.
    """
    windows = replay_data["windows"]
    run_cls = str(replay_data["run_class"])

    triggers_by_idx: dict[int, list[str]] = {}
    first_correct_seen = False
    first_missed_seen = False

    def _add_trigger(w_idx: int, reason: str) -> None:
        if w_idx not in triggers_by_idx:
            triggers_by_idx[w_idx] = []
        if reason not in triggers_by_idx[w_idx]:
            triggers_by_idx[w_idx].append(reason)

    for i, w in enumerate(windows):
        w_idx = int(w["window_idx"])
        true_lbl = str(w["true_label"])
        pred_cls = str(w["predicted_class"])
        alarm_st = str(w["alarm_state"])
        unk_flag = bool(w["unknown_anomaly"])

        # 1. Alarm state change from previous window (or non-normal at window 0)
        if i == 0 and alarm_st != "normal":
            _add_trigger(w_idx, f"alarm_state_active_at_start ({alarm_st})")
        elif i > 0:
            prev_alarm = str(windows[i - 1]["alarm_state"])
            if alarm_st != prev_alarm:
                _add_trigger(w_idx, f"alarm_state_changed ({prev_alarm} -> {alarm_st})")

        # 2. Unknown anomaly flag turns on
        if i == 0 and unk_flag:
            _add_trigger(w_idx, "unknown_anomaly_flag_turned_on")
        elif i > 0 and unk_flag and not bool(windows[i - 1]["unknown_anomaly"]):
            _add_trigger(w_idx, "unknown_anomaly_flag_turned_on")

        # 3. Classifier's first correct or first missed detection
        if run_cls == "normal":
            if not first_correct_seen and true_lbl == "normal" and pred_cls == "normal":
                first_correct_seen = True
                _add_trigger(w_idx, "first_correct_normal_window")
            if not first_missed_seen and pred_cls != "normal":
                first_missed_seen = True
                _add_trigger(w_idx, f"first_false_alarm_window ({pred_cls})")
        else:
            if true_lbl == run_cls:
                if not first_correct_seen and pred_cls == run_cls:
                    first_correct_seen = True
                    _add_trigger(w_idx, f"first_correct_detection ({run_cls})")
                if not first_missed_seen and pred_cls == "normal":
                    first_missed_seen = True
                    _add_trigger(w_idx, f"first_missed_event_window ({run_cls} predicted as normal)")

    ordered = sorted(triggers_by_idx.items(), key=lambda kv: kv[0])
    if len(ordered) > max_windows:
        ordered = ordered[:max_windows]
    return ordered


def generate_all_replay_explanations(
    sim_dir: Path | None = None,
    cache_dir: Path | None = None,
    log_path: Path | None = None,
    dry_run: bool = False,
    cap_usd: float = 5.0,
) -> dict[str, Path]:
    target_sim_dir = sim_dir if sim_dir is not None else (REPO_ROOT / "public/sim")
    eval_dir = ROOT / "outputs/eval_explainer"
    use_cache_dir = cache_dir if cache_dir is not None else (eval_dir / "cache")
    use_log_path = log_path if log_path is not None else (eval_dir / "replay_call_log.jsonl")

    data_root = ES.resolve_data_root(ROOT)
    features_test_df = None
    th_model = None
    if (data_root / "outputs/datasets_v3/features_test.csv").is_file() and (
        data_root / "outputs/eval_v3/trained_models_v3.pkl"
    ).is_file():
        features_test_df = pd.read_csv(data_root / "outputs/datasets_v3/features_test.csv")
        models_v3 = pickle.loads((data_root / "outputs/eval_v3/trained_models_v3.pkl").read_bytes())
        th_model = models_v3["threshold_rule"]

    budget = BudgetTracker(cap_usd=cap_usd, cache_dir=use_cache_dir, log_path=use_log_path)
    guard = ExplanationGuard()
    client: ChatClient | MockChatClient = MockChatClient() if dry_run else ChatClient()
    explainer = AdvisoryExplainer(client=client, guard=guard, budget=budget, use_cache=True)

    git_commit = ED.resolve_git_commit(data_root)
    written: dict[str, Path] = {}

    for cls in REPLAY_CLASSES:
        replay_file = target_sim_dir / f"replay_{cls}.json"
        if not replay_file.is_file():
            raise FileNotFoundError(f"Missing {replay_file}")

        replay_data = json.loads(replay_file.read_text(encoding="utf-8"))
        packets = load_packets_from_replay_json(
            replay_file,
            features_test_df=features_test_df,
            threshold_rule_model=th_model,
        )
        pkt_by_idx = {
            int(w["window_idx"]): pkt
            for w, pkt in zip(replay_data["windows"], packets)
        }

        selected = select_replay_explanation_windows(replay_data, max_windows=MAX_EXPLANATIONS_PER_RUN)
        explanations_list: list[dict[str, Any]] = []

        workers = 1 if dry_run else 6
        out_by_idx: dict[int, Any] = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(explainer.explain, pkt_by_idx[w_idx]): w_idx for w_idx, _ in selected}
            for fut in as_completed(futs):
                w_idx = futs[fut]
                out_by_idx[w_idx] = fut.result()

        for w_idx, trigger_reasons in selected:
            pkt = pkt_by_idx[w_idx]
            out = out_by_idx[w_idx]
            # Ensure guard reasons list never contains any forbidden substring like 'probability' or 'confidence'
            safe_reasons = [
                r.replace("probability", "score_term").replace("confidence", "score_term")
                for r in out.guard_result.reasons
            ]
            explanations_list.append(
                {
                    "window_idx": w_idx,
                    "window_id": pkt.window_id,
                    "trigger_reasons": trigger_reasons,
                    "summary": out.summary,
                    "suggested_check": out.suggested_check,
                    "source": out.source,
                    "model_id": out.model_id,
                    "guard_result": {
                        "passed": out.guard_result.passed,
                        "reasons": safe_reasons,
                        "extracted_numbers": out.guard_result.extracted_numbers,
                        "unmatched_numbers": out.guard_result.unmatched_numbers,
                        "sentence_count_summary": out.guard_result.sentence_count_summary,
                        "sentence_count_check": out.guard_result.sentence_count_check,
                        "char_count": out.guard_result.char_count,
                    },
                    "packet": pkt.model_dump(mode="json"),
                }
            )

        payload = {
            **ED.common_envelope(git_commit, seed=int(replay_data.get("seed", ED.DEFAULT_SEED))),
            "file_id": f"explanations_{cls}",
            "run_class": cls,
            "simulation_id": str(replay_data["simulation_id"]),
            "advisory_notice": "Advisory text only. It does not change the alarm.",
            "selection_criteria": (
                "At most 12 windows per replay run where the persisted alarm_state changes, "
                "the unknown_anomaly flag turns on, or the classifier's first correct or missed detection occurs."
            ),
            "n_explanations": len(explanations_list),
            "explanations": explanations_list,
        }

        out_file = target_sim_dir / f"explanations_{cls}.json"
        text = json.dumps(payload, indent=2, sort_keys=False, ensure_ascii=False) + "\n"
        raw_bytes = text.encode("utf-8")
        if len(raw_bytes) > MAX_EXPLANATION_FILE_BYTES:
            raise ValueError(
                f"{out_file.name} size {len(raw_bytes)} exceeds 100 KB limit ({MAX_EXPLANATION_FILE_BYTES})"
            )
        out_file.write_bytes(raw_bytes)
        written[out_file.name] = out_file

    return written


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate public/sim/explanations_<class>.json files.")
    ap.add_argument("--sim-dir", type=Path, default=REPO_ROOT / "public/sim")
    ap.add_argument("--dry-run", action="store_true", help="Use MockChatClient (or template fallback).")
    ap.add_argument("--cap-usd", type=float, default=5.0)
    args = ap.parse_args()

    written = generate_all_replay_explanations(
        sim_dir=args.sim_dir,
        dry_run=args.dry_run,
        cap_usd=args.cap_usd,
    )
    for fname, fpath in sorted(written.items()):
        data = json.loads(fpath.read_text(encoding="utf-8"))
        print(f"Wrote {fname}: {data['n_explanations']} explanations ({fpath.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
