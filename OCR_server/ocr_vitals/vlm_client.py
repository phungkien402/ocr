"""VLM HTTP client với retry + circuit breaker.

Tách khỏi ocr_engine.py để dễ test + tái sử dụng. Stateful — giữ 1 instance
module-level cho cả app lifetime.

Behavior:
- Retry chỉ với transient errors (connection, timeout, 5xx). 4xx KHÔNG retry
  (vd 401 auth, 400 bad payload — retry vô nghĩa, chỉ thêm latency).
- Circuit breaker bao quanh retry: nếu nhiều request liên tiếp fail, OPEN
  circuit để fail-fast (skip VLM, parser trả null) thay vì để mỗi request
  block đợi timeout đầy đủ.
- Async-safe: asyncio.Lock cho state mutation.

States:
    CLOSED      → bình thường, count failures
    OPEN        → fail fast (raise CircuitOpenError), không gọi VLM
    HALF_OPEN   → cho 1 request thử, success → CLOSED, fail → OPEN lại

Env vars (default sensible cho vLLM 1× V100):
    VLM_MAX_RETRIES         = 2      (3 attempts total)
    VLM_RETRY_BASE_DELAY    = 0.5    (s) backoff = base * 2^attempt + jitter
    VLM_RETRY_MAX_DELAY     = 8      (s) cap cho mỗi backoff
    VLM_CB_THRESHOLD        = 5      (consecutive failures → open)
    VLM_CB_COOLDOWN         = 60     (s) open → half_open transition
"""
from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Awaitable, Callable, Optional

import httpx

from .obs import get_timings

logger = logging.getLogger(__name__)


# ─── Config ───────────────────────────────────────────────────────────────

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


MAX_RETRIES      = _env_int("VLM_MAX_RETRIES", 2)
RETRY_BASE_DELAY = _env_float("VLM_RETRY_BASE_DELAY", 0.5)
RETRY_MAX_DELAY  = _env_float("VLM_RETRY_MAX_DELAY", 8.0)
CB_THRESHOLD     = _env_int("VLM_CB_THRESHOLD", 5)
CB_COOLDOWN_S    = _env_float("VLM_CB_COOLDOWN", 60.0)


# ─── Exceptions ───────────────────────────────────────────────────────────

class CircuitOpenError(Exception):
    """Raised khi CB đang OPEN, không thử gọi VLM."""


class NonRetryableError(Exception):
    """4xx hoặc error logic không nên retry."""


# ─── State machine ───────────────────────────────────────────────────────

class CircuitState(str, Enum):
    CLOSED    = "closed"
    OPEN      = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    """Async-safe circuit breaker.

    Thread-safety: KHÔNG cần — chỉ asyncio. asyncio.Lock đủ cho event loop concurrency.
    """
    threshold:    int   = CB_THRESHOLD
    cooldown_s:   float = CB_COOLDOWN_S
    state:        CircuitState = CircuitState.CLOSED
    failure_count: int  = 0
    opened_at:    Optional[float] = None
    _lock:        asyncio.Lock = field(default_factory=asyncio.Lock)

    async def before_call(self) -> None:
        """Gọi trước khi attempt — raise CircuitOpenError nếu đang OPEN.
        Tự chuyển OPEN → HALF_OPEN khi đã qua cooldown.
        """
        async with self._lock:
            if self.state == CircuitState.OPEN:
                if self.opened_at is None or (time.monotonic() - self.opened_at) >= self.cooldown_s:
                    self.state = CircuitState.HALF_OPEN
                    logger.info("[CB] transitioning OPEN → HALF_OPEN (cooldown %.0fs đã qua)", self.cooldown_s)
                else:
                    elapsed = time.monotonic() - self.opened_at
                    raise CircuitOpenError(
                        f"VLM circuit OPEN, fail-fast (còn {self.cooldown_s - elapsed:.0f}s cooldown)"
                    )

    async def on_success(self) -> None:
        async with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                logger.info("[CB] HALF_OPEN → CLOSED (recovery confirmed)")
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.opened_at = None

    async def on_failure(self) -> None:
        async with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                # Recovery probe fail → OPEN ngay, reset cooldown timer
                self.state = CircuitState.OPEN
                self.opened_at = time.monotonic()
                logger.warning("[CB] HALF_OPEN → OPEN (probe fail, reset cooldown %.0fs)", self.cooldown_s)
                return
            self.failure_count += 1
            if self.state == CircuitState.CLOSED and self.failure_count >= self.threshold:
                self.state = CircuitState.OPEN
                self.opened_at = time.monotonic()
                logger.warning(
                    "[CB] CLOSED → OPEN (failures=%d ≥ threshold=%d, cooldown %.0fs)",
                    self.failure_count, self.threshold, self.cooldown_s,
                )

    def snapshot(self) -> dict:
        """Plain dict cho /health endpoint."""
        return {
            "state":         self.state.value,
            "failure_count": self.failure_count,
            "threshold":     self.threshold,
            "cooldown_s":    self.cooldown_s,
            "opened_at_ago": (time.monotonic() - self.opened_at) if self.opened_at else None,
        }


# Module-level singleton — app instantiate sớm, reset trong tests
_breaker: Optional[CircuitBreaker] = None


def get_breaker() -> CircuitBreaker:
    global _breaker
    if _breaker is None:
        _breaker = CircuitBreaker()
    return _breaker


def reset_breaker_for_test(threshold: int = CB_THRESHOLD,
                            cooldown_s: float = CB_COOLDOWN_S) -> CircuitBreaker:
    """Tests gọi để có fresh state. KHÔNG dùng trong production code."""
    global _breaker
    _breaker = CircuitBreaker(threshold=threshold, cooldown_s=cooldown_s)
    return _breaker


# ─── Retry helper ─────────────────────────────────────────────────────────

# Errors coi là transient (retry):
#   - Mạng: ConnectError, ConnectTimeout, ReadTimeout, WriteTimeout, PoolTimeout
#   - Protocol: RemoteProtocolError (server close half-way)
#   - HTTP 5xx (xử lý riêng trong _is_retryable_response)
_RETRYABLE_EXC = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.WriteTimeout,
    httpx.PoolTimeout,
    httpx.RemoteProtocolError,
)


def _is_retryable_response(resp: httpx.Response) -> bool:
    """500/502/503/504 nên retry. 4xx KHÔNG retry."""
    return 500 <= resp.status_code < 600


def _backoff_delay(attempt: int) -> float:
    """Exponential + jitter: base * 2^attempt + uniform(0, 0.25 * base)."""
    delay = min(RETRY_BASE_DELAY * (2 ** attempt), RETRY_MAX_DELAY)
    jitter = random.uniform(0, RETRY_BASE_DELAY * 0.25)
    return delay + jitter


async def post_with_retry(
    url: str,
    json_payload: dict,
    timeout_s: float,
    *,
    client_factory: Callable[[], httpx.AsyncClient] = None,
) -> httpx.Response:
    """POST JSON với retry + circuit breaker.

    Raises:
        CircuitOpenError: khi CB OPEN, không thử gọi.
        NonRetryableError: 4xx response (sau khi đã đọc body).
        httpx.HTTPError: subclass — sau khi exhausted retries vẫn fail.

    Caller (ocr_engine.extract_vitals_async) chịu trách nhiệm catch all 3
    rồi return "" để pipeline không crash.
    """
    cb = get_breaker()
    await cb.before_call()  # raise CircuitOpenError nếu cần

    if client_factory is None:
        client_factory = lambda: httpx.AsyncClient(timeout=timeout_s)

    timings = get_timings()                    # record retry count + cb state into log
    if timings is not None:
        timings.add("vlm_cb_state_open", 0)    # mark có gọi VLM thật (CB closed/half-open)

    last_exc: Optional[BaseException] = None
    attempts_done = 0
    for attempt in range(MAX_RETRIES + 1):
        attempts_done = attempt + 1
        try:
            async with client_factory() as client:
                r = await client.post(url, json=json_payload)
            if r.status_code < 400:
                await cb.on_success()
                if timings is not None:
                    timings.add("vlm_attempts", attempts_done)
                return r
            if _is_retryable_response(r):
                # 5xx — retry
                logger.warning("[VLM] HTTP %d attempt=%d/%d", r.status_code, attempt + 1, MAX_RETRIES + 1)
                last_exc = httpx.HTTPStatusError(f"server {r.status_code}", request=r.request, response=r)
            else:
                # 4xx — không retry, bump CB cho transient miss config
                await cb.on_failure()
                raise NonRetryableError(f"HTTP {r.status_code} (non-retryable)")
        except _RETRYABLE_EXC as e:
            logger.warning("[VLM] %s attempt=%d/%d", type(e).__name__, attempt + 1, MAX_RETRIES + 1)
            last_exc = e
        # backoff giữa retries (không sleep sau attempt cuối)
        if attempt < MAX_RETRIES:
            await asyncio.sleep(_backoff_delay(attempt))

    # Exhausted retries
    await cb.on_failure()
    if timings is not None:
        timings.add("vlm_attempts", attempts_done)
    assert last_exc is not None
    raise last_exc
