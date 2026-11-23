"""Advisory AI Explainer package for ProjectCanary-Sim alarm packets.

Strictly read-only over frozen classifier outputs:
- Never modifies `predicted_class`, `alarm_state`, `model_score`, or any numeric value in a packet.
- Validates every AI response with `ExplanationGuard`; retries at most once with guard error feedback;
  falls back deterministically to `generate_template_explanation` on API error, missing key,
  budget cap, or second guard rejection.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Literal

from .budget import BudgetTracker
from .client import ChatClient, MockChatClient
from .guard import ExplanationGuard, GuardResult
from .packet import (
    ExplainerPacket,
    load_packets_from_replay_json,
    load_packets_from_sample_json,
)
from .prompt import EXPLAINER_OUTPUT_SCHEMA, SYSTEM_PROMPT, build_user_prompt
from .templates import generate_template_explanation

__all__ = [
    "AdvisoryExplainer",
    "BudgetTracker",
    "ChatClient",
    "ExplainerPacket",
    "ExplanationGuard",
    "ExplanationOutput",
    "GuardResult",
    "MockChatClient",
    "generate_template_explanation",
    "load_packets_from_replay_json",
    "load_packets_from_sample_json",
]


@dataclass(frozen=True)
class ExplanationOutput:
    window_id: str
    summary: str
    suggested_check: str
    source: Literal["ai", "template"]
    model_id: str
    guard_result: GuardResult
    attempts_made: int
    retried: bool
    fallback_reason: str | None
    from_cache: bool
    first_raw_output: dict[str, Any] | None
    first_guard_result: GuardResult | None
    total_tokens: int
    cost_usd: float
    latency_s: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id,
            "summary": self.summary,
            "suggested_check": self.suggested_check,
            "source": self.source,
            "model_id": self.model_id,
            "guard_result": self.guard_result.to_dict(),
            "attempts_made": self.attempts_made,
            "retried": self.retried,
            "fallback_reason": self.fallback_reason,
            "from_cache": self.from_cache,
            "first_raw_output": self.first_raw_output,
            "first_guard_result": self.first_guard_result.to_dict() if self.first_guard_result else None,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.cost_usd, 8),
            "latency_s": round(self.latency_s, 4),
        }


class AdvisoryExplainer:
    """Orchestrates prompt construction, chat client call, deterministic guard check,
    single retry with error list, budget tracking, caching, and offline template fallback."""

    def __init__(
        self,
        client: ChatClient | MockChatClient | None = None,
        guard: ExplanationGuard | None = None,
        budget: BudgetTracker | None = None,
        use_cache: bool = True,
    ) -> None:
        self.client = client if client is not None else ChatClient()
        self.guard = guard if guard is not None else ExplanationGuard()
        self.budget = budget if budget is not None else BudgetTracker()
        self.use_cache = use_cache

    def explain(self, packet: ExplainerPacket) -> ExplanationOutput:
        """Explain `packet` without mutating it."""
        pkt_hash = packet.packet_hash()
        model_id = self.client.model

        # 1. Check cache first
        if self.use_cache:
            cached = self.budget.get_cached(model_id, pkt_hash)
            if cached is not None:
                g_res = self.guard.check(
                    {"summary": cached["summary"], "suggested_check": cached["suggested_check"]},
                    packet,
                )
                if g_res.passed:
                    first_g = (
                        GuardResult(**cached["first_guard_result"])
                        if isinstance(cached.get("first_guard_result"), dict)
                        else g_res
                    )
                    self.budget.record_call(
                        packet_id=packet.window_id,
                        packet_hash=pkt_hash,
                        model_id=cached.get("model_id", model_id),
                        attempt=0,
                        usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                        guard_passed=True,
                        guard_reasons=[],
                        source=cached.get("source", "ai"),
                        latency_s=0.0,
                        from_cache=True,
                    )
                    return ExplanationOutput(
                        window_id=packet.window_id,
                        summary=cached["summary"],
                        suggested_check=cached["suggested_check"],
                        source=cached.get("source", "ai"),
                        model_id=cached.get("model_id", model_id),
                        guard_result=g_res,
                        attempts_made=int(cached.get("attempts_made", 1)),
                        retried=bool(cached.get("retried", False)),
                        fallback_reason=cached.get("fallback_reason"),
                        from_cache=True,
                        first_raw_output=cached.get("first_raw_output"),
                        first_guard_result=first_g,
                        total_tokens=int(cached.get("total_tokens", 0)),
                        cost_usd=float(cached.get("cost_usd", 0.0)),
                        latency_s=float(cached.get("latency_s", 0.0)),
                    )

        # 2. Check API key availability and budget cap before calling API
        if not self.client.has_api_key():
            return self._use_template_fallback(
                packet=packet,
                pkt_hash=pkt_hash,
                fallback_reason="missing_api_key",
                attempts_made=0,
                retried=False,
                first_raw_output=None,
                first_guard_result=None,
                acc_tokens=0,
                acc_cost=0.0,
                acc_latency=0.0,
            )

        can_afford, budget_reason = self.budget.can_afford_next_call()
        if not can_afford:
            return self._use_template_fallback(
                packet=packet,
                pkt_hash=pkt_hash,
                fallback_reason=budget_reason,
                attempts_made=0,
                retried=False,
                first_raw_output=None,
                first_guard_result=None,
                acc_tokens=0,
                acc_cost=0.0,
                acc_latency=0.0,
            )

        # 3. Attempt 1
        user_prompt_1 = build_user_prompt(packet, retry_errors=None)
        resp_1 = self.client.complete(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt_1,
            response_format=EXPLAINER_OUTPUT_SCHEMA,
        )
        acc_latency = resp_1.latency_s

        if not resp_1.ok or resp_1.parsed is None:
            entry_1 = self.budget.record_call(
                packet_id=packet.window_id,
                packet_hash=pkt_hash,
                model_id=model_id,
                attempt=1,
                usage=resp_1.usage,
                guard_passed=False,
                guard_reasons=[resp_1.error or "api_error"],
                source="api_error",
                latency_s=resp_1.latency_s,
            )
            return self._use_template_fallback(
                packet=packet,
                pkt_hash=pkt_hash,
                fallback_reason=f"api_error: {resp_1.error}",
                attempts_made=1,
                retried=False,
                first_raw_output=None,
                first_guard_result=None,
                acc_tokens=entry_1.total_tokens,
                acc_cost=entry_1.cost_usd,
                acc_latency=acc_latency,
            )

        first_raw = dict(resp_1.parsed)
        g_res_1 = self.guard.check(first_raw, packet)
        entry_1 = self.budget.record_call(
            packet_id=packet.window_id,
            packet_hash=pkt_hash,
            model_id=model_id,
            attempt=1,
            usage=resp_1.usage,
            guard_passed=g_res_1.passed,
            guard_reasons=g_res_1.reasons,
            source="ai" if g_res_1.passed else "ai_rejected_attempt_1",
            latency_s=resp_1.latency_s,
        )
        acc_tokens = entry_1.total_tokens
        acc_cost = entry_1.cost_usd

        if g_res_1.passed:
            out = ExplanationOutput(
                window_id=packet.window_id,
                summary=str(first_raw["summary"]).strip(),
                suggested_check=str(first_raw["suggested_check"]).strip(),
                source="ai",
                model_id=model_id,
                guard_result=g_res_1,
                attempts_made=1,
                retried=False,
                fallback_reason=None,
                from_cache=False,
                first_raw_output=first_raw,
                first_guard_result=g_res_1,
                total_tokens=acc_tokens,
                cost_usd=acc_cost,
                latency_s=acc_latency,
            )
            if self.use_cache:
                self.budget.set_cached(model_id, pkt_hash, out.to_dict())
            return out

        # 4. Attempt 2 (Retry once with the guard's error list)
        can_afford_2, budget_reason_2 = self.budget.can_afford_next_call()
        if not can_afford_2:
            return self._use_template_fallback(
                packet=packet,
                pkt_hash=pkt_hash,
                fallback_reason=budget_reason_2,
                attempts_made=1,
                retried=False,
                first_raw_output=first_raw,
                first_guard_result=g_res_1,
                acc_tokens=acc_tokens,
                acc_cost=acc_cost,
                acc_latency=acc_latency,
            )

        user_prompt_2 = build_user_prompt(packet, retry_errors=g_res_1.reasons)
        resp_2 = self.client.complete(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt_2,
            response_format=EXPLAINER_OUTPUT_SCHEMA,
        )
        acc_latency += resp_2.latency_s

        if not resp_2.ok or resp_2.parsed is None:
            entry_2 = self.budget.record_call(
                packet_id=packet.window_id,
                packet_hash=pkt_hash,
                model_id=model_id,
                attempt=2,
                usage=resp_2.usage,
                guard_passed=False,
                guard_reasons=[resp_2.error or "api_error_on_retry"],
                source="api_error_on_retry",
                latency_s=resp_2.latency_s,
            )
            return self._use_template_fallback(
                packet=packet,
                pkt_hash=pkt_hash,
                fallback_reason=f"api_error_on_retry: {resp_2.error}",
                attempts_made=2,
                retried=True,
                first_raw_output=first_raw,
                first_guard_result=g_res_1,
                acc_tokens=acc_tokens + entry_2.total_tokens,
                acc_cost=acc_cost + entry_2.cost_usd,
                acc_latency=acc_latency,
            )

        second_raw = dict(resp_2.parsed)
        g_res_2 = self.guard.check(second_raw, packet)
        entry_2 = self.budget.record_call(
            packet_id=packet.window_id,
            packet_hash=pkt_hash,
            model_id=model_id,
            attempt=2,
            usage=resp_2.usage,
            guard_passed=g_res_2.passed,
            guard_reasons=g_res_2.reasons,
            source="ai" if g_res_2.passed else "ai_rejected_attempt_2",
            latency_s=resp_2.latency_s,
        )
        acc_tokens += entry_2.total_tokens
        acc_cost += entry_2.cost_usd

        if g_res_2.passed:
            out = ExplanationOutput(
                window_id=packet.window_id,
                summary=str(second_raw["summary"]).strip(),
                suggested_check=str(second_raw["suggested_check"]).strip(),
                source="ai",
                model_id=model_id,
                guard_result=g_res_2,
                attempts_made=2,
                retried=True,
                fallback_reason=None,
                from_cache=False,
                first_raw_output=first_raw,
                first_guard_result=g_res_1,
                total_tokens=acc_tokens,
                cost_usd=acc_cost,
                latency_s=acc_latency,
            )
            if self.use_cache:
                self.budget.set_cached(model_id, pkt_hash, out.to_dict())
            return out

        # 5. Second failure -> deterministic template fallback
        return self._use_template_fallback(
            packet=packet,
            pkt_hash=pkt_hash,
            fallback_reason=f"guard_rejected_twice: {g_res_2.reasons}",
            attempts_made=2,
            retried=True,
            first_raw_output=first_raw,
            first_guard_result=g_res_1,
            acc_tokens=acc_tokens,
            acc_cost=acc_cost,
            acc_latency=acc_latency,
        )

    def _use_template_fallback(
        self,
        *,
        packet: ExplainerPacket,
        pkt_hash: str,
        fallback_reason: str,
        attempts_made: int,
        retried: bool,
        first_raw_output: dict[str, Any] | None,
        first_guard_result: GuardResult | None,
        acc_tokens: int,
        acc_cost: float,
        acc_latency: float,
    ) -> ExplanationOutput:
        t0 = time.time()
        tmpl = generate_template_explanation(packet)
        g_res_tmpl = self.guard.check(tmpl, packet)
        assert g_res_tmpl.passed, f"Template fallback failed guard: {g_res_tmpl.reasons}"
        dt = time.time() - t0

        self.budget.record_call(
            packet_id=packet.window_id,
            packet_hash=pkt_hash,
            model_id="template",
            attempt=attempts_made + 1,
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            guard_passed=True,
            guard_reasons=[],
            source="template",
            latency_s=dt,
            from_cache=True,  # Do not increment paid API call count
        )

        return ExplanationOutput(
            window_id=packet.window_id,
            summary=tmpl["summary"],
            suggested_check=tmpl["suggested_check"],
            source="template",
            model_id="template",
            guard_result=g_res_tmpl,
            attempts_made=attempts_made,
            retried=retried,
            fallback_reason=fallback_reason,
            from_cache=False,
            first_raw_output=first_raw_output,
            first_guard_result=first_guard_result,
            total_tokens=acc_tokens,
            cost_usd=acc_cost,
            latency_s=acc_latency + dt,
        )
