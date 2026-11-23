"""Spend cap, secret-free JSONL call logging, and packet-hash disk cache for the advisory explainer."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import threading
import time
from typing import Any

from .guard import load_explainer_config

SECRET_SCRUB_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"Bearer\s+[A-Za-z0-9._\-]+", re.IGNORECASE),
    re.compile(r"META_API_KEY\s*=\s*\S+", re.IGNORECASE),
    re.compile(r"eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}"),
    re.compile(r"sk-[A-Za-z0-9_-]{16,}"),
)


def scrub_secrets(text: str, explicit_secret: str | None = None) -> str:
    """Remove any secret-looking substring or explicit API key from a string before logging."""
    out = text
    if explicit_secret and len(explicit_secret) >= 6:
        out = out.replace(explicit_secret, "[REDACTED]")
    for pat in SECRET_SCRUB_PATTERNS:
        out = pat.sub("[REDACTED]", out)
    return out


@dataclass
class CallLogEntry:
    timestamp: float
    packet_id: str
    packet_hash: str
    model_id: str
    attempt: int
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    total_tokens: int
    cost_usd: float
    cumulative_cost_usd: float
    guard_passed: bool
    guard_reasons: list[str]
    source: str
    latency_s: float
    from_cache: bool


class BudgetTracker:
    """Enforces a hard USD spend cap (default 5.00 USD) and call/token caps, logs every call to JSONL
    without secrets, and caches validated explanations keyed by `(model_id, packet_hash)`."""

    def __init__(
        self,
        cap_usd: float | None = None,
        cache_dir: Path | str | None = None,
        log_path: Path | str | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        cfg = config if config is not None else load_explainer_config()
        b_cfg = cfg["budget"]
        self.cap_usd = float(cap_usd if cap_usd is not None else b_cfg.get("default_cap_usd", 5.0))
        self.cost_per_m_input = float(b_cfg.get("cost_per_m_input_usd", 1.25))
        self.cost_per_m_cached_input = float(b_cfg.get("cost_per_m_cached_input_usd", 0.15))
        self.cost_per_m_output = float(b_cfg.get("cost_per_m_output_usd", 4.25))
        self.max_calls = int(b_cfg.get("max_calls", 600))
        self.max_total_tokens = int(b_cfg.get("max_total_tokens", 2_000_000))

        self.total_cost_usd: float = 0.0
        self.total_calls: int = 0
        self.total_tokens: int = 0

        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.log_path = Path(log_path) if log_path is not None else None
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

        self._mem_cache: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def compute_cost_usd(self, usage: dict[str, Any]) -> float:
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        prompt_details = usage.get("prompt_tokens_details") or {}
        cached_tokens = int(prompt_details.get("cached_tokens", 0) or 0)
        uncached_prompt = max(0, prompt_tokens - cached_tokens)

        return (
            (uncached_prompt / 1_000_000.0) * self.cost_per_m_input
            + (cached_tokens / 1_000_000.0) * self.cost_per_m_cached_input
            + (completion_tokens / 1_000_000.0) * self.cost_per_m_output
        )

    def can_afford_next_call(self) -> tuple[bool, str]:
        with self._lock:
            if self.total_cost_usd >= self.cap_usd:
                return False, f"budget_cap_reached: spent ${self.total_cost_usd:.4f} >= cap ${self.cap_usd:.2f}"
            if self.total_calls >= self.max_calls:
                return False, f"call_cap_reached: {self.total_calls} >= {self.max_calls}"
            if self.total_tokens >= self.max_total_tokens:
                return False, f"token_cap_reached: {self.total_tokens} >= {self.max_total_tokens}"
            return True, "ok"

    def cache_key(self, model_id: str, packet_hash: str) -> str:
        safe_model = re.sub(r"[^A-Za-z0-9_.-]", "_", model_id)
        return f"{safe_model}_{packet_hash}"

    def get_cached(self, model_id: str, packet_hash: str) -> dict[str, Any] | None:
        key = self.cache_key(model_id, packet_hash)
        with self._lock:
            if key in self._mem_cache:
                return dict(self._mem_cache[key])
        if self.cache_dir is not None:
            cfile = self.cache_dir / f"{key}.json"
            if cfile.is_file():
                try:
                    data = json.loads(cfile.read_text(encoding="utf-8"))
                    with self._lock:
                        self._mem_cache[key] = data
                    return dict(data)
                except Exception:
                    return None
        return None

    def set_cached(self, model_id: str, packet_hash: str, payload: dict[str, Any]) -> None:
        key = self.cache_key(model_id, packet_hash)
        # Never store secrets in cache
        clean_json = scrub_secrets(json.dumps(payload, sort_keys=False))
        clean_payload = json.loads(clean_json)
        with self._lock:
            self._mem_cache[key] = clean_payload
            if self.cache_dir is not None:
                cfile = self.cache_dir / f"{key}.json"
                tmp = cfile.with_suffix(".tmp")
                tmp.write_text(json.dumps(clean_payload, indent=2) + "\n", encoding="utf-8")
                tmp.replace(cfile)

    def record_call(
        self,
        *,
        packet_id: str,
        packet_hash: str,
        model_id: str,
        attempt: int,
        usage: dict[str, Any],
        guard_passed: bool,
        guard_reasons: list[str],
        source: str,
        latency_s: float,
        from_cache: bool = False,
        explicit_secret: str | None = None,
    ) -> CallLogEntry:
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        cached_tokens = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0)
        total_tok = int(usage.get("total_tokens", prompt_tokens + completion_tokens) or 0)

        with self._lock:
            if from_cache:
                cost = 0.0
            else:
                cost = self.compute_cost_usd(usage)
                self.total_cost_usd += cost
                self.total_calls += 1
                self.total_tokens += total_tok
            cum_cost = self.total_cost_usd

            entry = CallLogEntry(
                timestamp=time.time(),
                packet_id=packet_id,
                packet_hash=packet_hash,
                model_id=model_id,
                attempt=attempt,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cached_tokens=cached_tokens,
                total_tokens=total_tok,
                cost_usd=round(cost, 8),
                cumulative_cost_usd=round(cum_cost, 8),
                guard_passed=guard_passed,
                guard_reasons=[scrub_secrets(r, explicit_secret) for r in guard_reasons],
                source=source,
                latency_s=round(latency_s, 4),
                from_cache=from_cache,
            )

            if self.log_path is not None:
                line = scrub_secrets(json.dumps(asdict(entry)), explicit_secret)
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")

        return entry
