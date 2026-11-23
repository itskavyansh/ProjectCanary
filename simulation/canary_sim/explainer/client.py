"""Small HTTP chat client for the advisory alarm explainer.

Copied in spirit from Canary-Analyst's HTTP request code (`https://api.meta.ai/v1/chat/completions`,
`Bearer <META_API_KEY>`, model from `AGENT_MODEL` defaulting to `muse-spark-1.3`), with:
- Configurable timeout and at most 1 HTTP retry on transient errors.
- Token usage recording and secret scrubbing (never prints, logs, or writes API keys).
- A deterministic `MockChatClient` for `--dry-run` and unit testing.
- Zero imports from Canary-Analyst.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from typing import Any, Callable

import httpx
from dotenv import load_dotenv

from .budget import scrub_secrets
from .guard import load_explainer_config
from .packet import ExplainerPacket
from .templates import generate_template_explanation


def _load_env_quietly() -> None:
    """Load `.env` from workspace or `simulation/.env` without ever printing values."""
    sim_dir = Path(__file__).resolve().parents[2]
    root_dir = sim_dir.parent
    for env_candidate in (sim_dir / ".env", root_dir / ".env"):
        if env_candidate.is_file():
            load_dotenv(env_candidate, override=False)


_load_env_quietly()


@dataclass
class ChatClientResponse:
    ok: bool
    parsed: dict[str, Any] | None
    raw_content: str
    usage: dict[str, Any]
    latency_s: float
    model_id: str
    error: str | None = None


class ChatClient:
    """Minimal HTTP chat completions client using `META_API_KEY` and `AGENT_MODEL`."""

    def __init__(
        self,
        model: str | None = None,
        api_url: str | None = None,
        timeout_s: float | None = None,
        max_retries: int | None = None,
        http_client: Any | None = None,
        api_key: str | None = None,
    ) -> None:
        cfg = load_explainer_config()
        self.model = model or os.getenv("AGENT_MODEL") or str(cfg.get("model_default", "muse-spark-1.3"))
        self.api_url = api_url or str(cfg.get("api_endpoint", "https://api.meta.ai/v1/chat/completions"))
        self.timeout_s = float(timeout_s if timeout_s is not None else cfg.get("timeout_s", 45.0))
        self.max_retries = int(max_retries if max_retries is not None else cfg.get("max_http_retries", 1))
        self.http_client = http_client
        self._explicit_api_key = api_key
        self._omit_temperature = False
        self._omit_reasoning_effort = False
        self.reasoning_effort = str(cfg.get("reasoning_effort", "low"))

    def _get_api_key(self) -> str:
        if self._explicit_api_key is not None:
            return self._explicit_api_key
        return os.getenv("META_API_KEY", "").strip()

    def has_api_key(self) -> bool:
        return bool(self._get_api_key())

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: dict[str, Any] | None = None,
        temperature: float = 0.1,
        max_completion_tokens: int = 2048,
    ) -> ChatClientResponse:
        t0 = time.time()
        key = self._get_api_key()
        if not key:
            return ChatClientResponse(
                ok=False,
                parsed=None,
                raw_content="",
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                latency_s=0.0,
                model_id=self.model,
                error="missing_api_key",
            )

        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        attempts_allowed = 1 + max(0, self.max_retries)
        last_err = "unknown_error"

        for attempt in range(attempts_allowed):
            payload: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "max_completion_tokens": max_completion_tokens,
            }
            if response_format is not None:
                payload["response_format"] = response_format
            if not self._omit_temperature and temperature is not None:
                payload["temperature"] = temperature
            if not self._omit_reasoning_effort and self.reasoning_effort:
                payload["reasoning_effort"] = self.reasoning_effort

            try:
                if self.http_client is not None:
                    resp = self.http_client.post(
                        self.api_url, headers=headers, json=payload, timeout=self.timeout_s
                    )
                else:
                    with httpx.Client(timeout=self.timeout_s) as client:
                        resp = client.post(self.api_url, headers=headers, json=payload)

                if resp.status_code == 200:
                    data = resp.json()
                    choices = data.get("choices") or []
                    if not choices:
                        return ChatClientResponse(
                            ok=False,
                            parsed=None,
                            raw_content="",
                            usage=data.get("usage") or {},
                            latency_s=time.time() - t0,
                            model_id=self.model,
                            error="empty_choices",
                        )
                    msg = choices[0].get("message") or {}
                    raw_content = str(msg.get("content") or "")
                    usage = data.get("usage") or {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    }
                    parsed = _parse_json_content(raw_content)
                    return ChatClientResponse(
                        ok=(parsed is not None),
                        parsed=parsed,
                        raw_content=raw_content,
                        usage=usage,
                        latency_s=time.time() - t0,
                        model_id=self.model,
                        error=None if parsed is not None else "json_parse_error",
                    )

                err_body = scrub_secrets(resp.text[:300], key)
                if resp.status_code == 400 and "reasoning_effort" in err_body.lower() and not self._omit_reasoning_effort:
                    self._omit_reasoning_effort = True
                    continue
                if resp.status_code == 400 and "temperature" in err_body.lower() and not self._omit_temperature:
                    self._omit_temperature = True
                    continue
                last_err = f"http_{resp.status_code}: {err_body}"
                if resp.status_code in (429, 500, 502, 503, 504) and attempt < attempts_allowed - 1:
                    time.sleep(0.5)
                    continue
                break
            except Exception as exc:
                last_err = scrub_secrets(str(exc), key)
                if attempt < attempts_allowed - 1:
                    time.sleep(0.5)
                    continue

        return ChatClientResponse(
            ok=False,
            parsed=None,
            raw_content="",
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            latency_s=time.time() - t0,
            model_id=self.model,
            error=last_err,
        )


def _parse_json_content(raw: str) -> dict[str, Any] | None:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    try:
        obj = json.loads(cleaned)
        return obj if isinstance(obj, dict) else None
    except Exception:
        # Try extracting first {...} block
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end > start:
            try:
                obj = json.loads(cleaned[start : end + 1])
                return obj if isinstance(obj, dict) else None
            except Exception:
                return None
        return None


class MockChatClient:
    """Deterministic mock client for `--dry-run` and unit tests.

    Supports scripted response queues or custom generator functions so tests can verify
    guard rejection on wrong numbers, wrong classes, forbidden phrases, over-long text,
    missing 'simulated', retry recovery, and fallback on API failure.
    """

    def __init__(
        self,
        scripted_responses: list[dict[str, Any] | ChatClientResponse] | None = None,
        handler: Callable[[str, str, int], dict[str, Any] | ChatClientResponse] | None = None,
        model: str = "mock-muse-spark-1.3",
        simulate_api_error: bool = False,
        simulate_missing_key: bool = False,
        prompt_tokens_per_call: int = 450,
        completion_tokens_per_call: int = 95,
    ) -> None:
        self.scripted_responses = list(scripted_responses) if scripted_responses else []
        self.handler = handler
        self.model = model
        self.simulate_api_error = simulate_api_error
        self.simulate_missing_key = simulate_missing_key
        self.prompt_tokens_per_call = prompt_tokens_per_call
        self.completion_tokens_per_call = completion_tokens_per_call
        self.call_count = 0
        self.calls: list[dict[str, Any]] = []

    def has_api_key(self) -> bool:
        return not self.simulate_missing_key

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: dict[str, Any] | None = None,
        temperature: float = 0.1,
        max_completion_tokens: int = 2048,
    ) -> ChatClientResponse:
        self.call_count += 1
        self.calls.append(
            {
                "attempt": self.call_count,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }
        )
        if self.simulate_missing_key:
            return ChatClientResponse(
                ok=False,
                parsed=None,
                raw_content="",
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                latency_s=0.001,
                model_id=self.model,
                error="missing_api_key",
            )
        if self.simulate_api_error:
            return ChatClientResponse(
                ok=False,
                parsed=None,
                raw_content="",
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                latency_s=0.001,
                model_id=self.model,
                error="simulated_http_503",
            )

        usage = {
            "prompt_tokens": self.prompt_tokens_per_call,
            "completion_tokens": self.completion_tokens_per_call,
            "total_tokens": self.prompt_tokens_per_call + self.completion_tokens_per_call,
            "prompt_tokens_details": {"cached_tokens": 0},
        }

        if self.scripted_responses:
            item = self.scripted_responses.pop(0)
            if isinstance(item, ChatClientResponse):
                return item
            raw_str = json.dumps(item)
            return ChatClientResponse(
                ok=True,
                parsed=item,
                raw_content=raw_str,
                usage=usage,
                latency_s=0.005,
                model_id=self.model,
                error=None,
            )

        if self.handler is not None:
            res = self.handler(system_prompt, user_prompt, self.call_count)
            if isinstance(res, ChatClientResponse):
                return res
            return ChatClientResponse(
                ok=True,
                parsed=res,
                raw_content=json.dumps(res),
                usage=usage,
                latency_s=0.005,
                model_id=self.model,
                error=None,
            )

        # Default dry-run behavior: extract packet JSON from user_prompt and return a faithful explanation
        pkt_start = user_prompt.find("PACKET JSON:\n")
        if pkt_start != -1:
            after = user_prompt[pkt_start + len("PACKET JSON:\n") :]
            end_brace = after.find("\n\nYOUR PREVIOUS")
            pkt_raw = after[:end_brace].strip() if end_brace != -1 else after.strip()
            pkt = ExplainerPacket.model_validate_json(pkt_raw)
            faithful = generate_template_explanation(pkt)
            return ChatClientResponse(
                ok=True,
                parsed=faithful,
                raw_content=json.dumps(faithful),
                usage=usage,
                latency_s=0.005,
                model_id=self.model,
                error=None,
            )

        fallback_obj = {
            "summary": "This simulated packet is consistent with normal conditions. All numbers match the simulated window.",
            "suggested_check": "Verify simulated node telemetry logs.",
        }
        return ChatClientResponse(
            ok=True,
            parsed=fallback_obj,
            raw_content=json.dumps(fallback_obj),
            usage=usage,
            latency_s=0.005,
            model_id=self.model,
            error=None,
        )
