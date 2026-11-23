"""System prompt, user prompt builder, and structured JSON schema for the advisory alarm explainer."""
from __future__ import annotations

import json
from typing import Any

from .packet import ExplainerPacket

SYSTEM_PROMPT = """You are a cautious advisory assistant explaining a SIMULATED monitoring alarm packet to an operator.

STRICT RULES YOU MUST FOLLOW:
1. SIMULATION ONLY: Every explanation is about a synthetic / simulated run. You MUST include the word "simulated" (or "synthetic") in `summary` and use hedged wording such as "consistent with".
2. EXACT NUMBERS ONLY: Use ONLY exact numbers that appear verbatim in the JSON packet provided by the user. NEVER compute, round, add, subtract, average, or convert numbers (do not compute percentages from fractions, do not compute differences between nodes, do not invent thresholds). Copy any number you cite character-for-character from the packet.
3. EXACT CLASS NAMING: Name the classifier's `predicted_class` exactly as given in the packet (in snake_case, e.g., `gradual`, `rapid`, `abnormal_vibration`, `sensor_fault`, `comm_fault`, or `normal`). You may also report `alarm_state` or `threshold_rule_class` if they equal `predicted_class`, or you may omit mentioning other class names altogether so you never name a conflicting class. NEVER contradict the classifier.
4. MODEL SCORE TERMINOLOGY: Refer to `model_score` strictly as `model_score`. NEVER use the words "probability", "probabilities", "confidence", "confident", "certain", "certainty", or "guaranteed".
5. NO REAL-WORLD CLAIMS OR SAFETY ADVICE: Never state what real ground or a mine is actually doing. Never use words like "evacuate", "evacuation", "collapse imminent", "safe", "unsafe", "danger now", "the ground is", "the mine", "roof fall", or "pillar failure".
6. NO ALARM MUTATION: Never state or imply that the alarm state is changed, upgraded, downgraded, cleared, overridden, or suppressed.
7. LENGTH & FORMAT: Output strict JSON matching `{ "summary": "...", "suggested_check": "..." }`:
   - `summary`: 2 to 3 plain sentences (total explanation length between 60 and 650 characters).
   - `suggested_check`: Exactly 1 plain sentence suggesting one field or telemetry verification check consistent with the simulated packet.
"""

EXPLAINER_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "explainer_advisory_output",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "2 to 3 plain sentences explaining the simulated packet using only verbatim numbers from the packet and naming predicted_class.",
                },
                "suggested_check": {
                    "type": "string",
                    "description": "Exactly 1 plain sentence suggesting one telemetry or sensor check.",
                },
            },
            "required": ["summary", "suggested_check"],
            "additionalProperties": False,
        },
    },
}


def build_user_prompt(packet: ExplainerPacket, retry_errors: list[str] | None = None) -> str:
    """Format the packet JSON and optional guard error feedback for a retry call."""
    pkt_dict = packet.model_dump(mode="json")
    # If alarm_state or threshold_rule_class differs from predicted_class, remind the model
    # NOT to mention any class name other than predicted_class so it cannot fail the class check.
    pkt_json = json.dumps(pkt_dict, indent=2)

    base = (
        f"Explain the following simulated monitoring packet in JSON `{{ \"summary\": \"...\", \"suggested_check\": \"...\" }}`.\n"
        f"- Be concise and direct (do not over-deliberate).\n"
        f"- You MUST include the word \"simulated\" and use hedged phrasing (\"consistent with\").\n"
        f"- You MUST name the predicted class `{packet.predicted_class}` and MUST NOT mention any other class name "
        f"(do not mention any of {[c for c in ('normal', 'gradual', 'rapid', 'abnormal_vibration', 'sensor_fault', 'comm_fault') if c != packet.predicted_class]}).\n"
        f"- Copy every number you cite character-for-character from the packet below. Do not compute differences, sums, or percentages, and do not mention numbers not in the packet.\n"
        f"- Refer to `model_score` ({packet.model_score}) only as `model_score` (never probability or confidence).\n\n"
        f"PACKET JSON:\n{pkt_json}\n"
    )
    if retry_errors:
        err_bullet = "\n".join(f"  * {e}" for e in retry_errors)
        base += (
            f"\nYOUR PREVIOUS ATTEMPT WAS REJECTED BY THE DETERMINISTIC GUARD FOR THESE ERRORS:\n"
            f"{err_bullet}\n"
            f"Fix every listed error immediately while strictly following all rules."
        )
    return base
