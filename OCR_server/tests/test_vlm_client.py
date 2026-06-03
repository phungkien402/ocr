"""Tests cho vlm_client — circuit breaker transitions + retry behavior.

Không cần vLLM thật chạy. Dùng httpx.MockTransport + custom client_factory để
control mọi response.
"""
from __future__ import annotations

import asyncio
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import httpx
import pytest

from ocr_vitals import vlm_client
from ocr_vitals.vlm_client import (
    CircuitOpenError,
    CircuitState,
    NonRetryableError,
    post_with_retry,
    reset_breaker_for_test,
)


# ─── Helpers ───────────────────────────────────────────────────────────────

def _mock_client(handler):
    """Trả factory tạo AsyncClient với MockTransport(handler)."""
    transport = httpx.MockTransport(handler)
    return lambda: httpx.AsyncClient(transport=transport, timeout=5.0)


def _ok_handler(request):
    return httpx.Response(
        200,
        json={"choices": [{"message": {"content": "mạch: 80"}}]},
    )


def _500_handler(request):
    return httpx.Response(500, text="upstream broken")


def _401_handler(request):
    return httpx.Response(401, text="auth")


def _network_error_handler(request):
    raise httpx.ConnectError("network down", request=request)


@pytest.fixture(autouse=True)
def fast_backoff(monkeypatch):
    """Tăng tốc test — backoff thật mất 0.5-8s, tests cần sub-second."""
    monkeypatch.setattr(vlm_client, "RETRY_BASE_DELAY", 0.001)
    monkeypatch.setattr(vlm_client, "RETRY_MAX_DELAY", 0.01)


@pytest.fixture
def fresh_cb():
    """Fresh CB với threshold thấp để test nhanh."""
    return reset_breaker_for_test(threshold=3, cooldown_s=0.1)


# ─── Happy path ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_success_closed_state(fresh_cb):
    r = await post_with_retry(
        "http://fake/v1/chat/completions",
        {"prompt": "x"},
        timeout_s=5,
        client_factory=_mock_client(_ok_handler),
    )
    assert r.status_code == 200
    assert fresh_cb.state == CircuitState.CLOSED
    assert fresh_cb.failure_count == 0


# ─── Retry on transient errors ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_retry_on_500_then_succeed(fresh_cb, monkeypatch):
    monkeypatch.setattr(vlm_client, "MAX_RETRIES", 2)
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503, text="loading")
        return _ok_handler(request)

    r = await post_with_retry("http://fake/x", {}, 5, client_factory=_mock_client(handler))
    assert r.status_code == 200
    assert calls["n"] == 3                     # 1 try + 2 retries
    assert fresh_cb.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_retry_on_network_error(fresh_cb, monkeypatch):
    monkeypatch.setattr(vlm_client, "MAX_RETRIES", 2)
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] < 3:
            raise httpx.ConnectTimeout("slow", request=request)
        return _ok_handler(request)

    r = await post_with_retry("http://fake/x", {}, 5, client_factory=_mock_client(handler))
    assert r.status_code == 200
    assert calls["n"] == 3


# ─── 4xx KHÔNG retry ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_4xx_not_retried(fresh_cb, monkeypatch):
    monkeypatch.setattr(vlm_client, "MAX_RETRIES", 5)
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return _401_handler(request)

    with pytest.raises(NonRetryableError):
        await post_with_retry("http://fake/x", {}, 5, client_factory=_mock_client(handler))
    assert calls["n"] == 1                     # KHÔNG retry — đúng 1 call


# ─── Circuit breaker transitions ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_cb_closed_to_open_after_threshold(fresh_cb, monkeypatch):
    """N failure liên tiếp → OPEN, request tiếp theo fail-fast."""
    monkeypatch.setattr(vlm_client, "MAX_RETRIES", 0)  # fail ngay, không retry

    # 3 failures = threshold
    for _ in range(3):
        with pytest.raises(httpx.HTTPError):
            await post_with_retry("http://fake/x", {}, 5,
                                  client_factory=_mock_client(_500_handler))

    assert fresh_cb.state == CircuitState.OPEN
    assert fresh_cb.failure_count == 3

    # Request thứ 4 — fail-fast, KHÔNG gọi handler
    handler_calls = {"n": 0}

    def spy_handler(request):
        handler_calls["n"] += 1
        return _ok_handler(request)

    with pytest.raises(CircuitOpenError):
        await post_with_retry("http://fake/x", {}, 5, client_factory=_mock_client(spy_handler))
    assert handler_calls["n"] == 0             # confirm fail-fast


@pytest.mark.asyncio
async def test_cb_open_to_half_open_after_cooldown(fresh_cb, monkeypatch):
    monkeypatch.setattr(vlm_client, "MAX_RETRIES", 0)
    # Force OPEN
    for _ in range(3):
        with pytest.raises(httpx.HTTPError):
            await post_with_retry("http://fake/x", {}, 5, client_factory=_mock_client(_500_handler))
    assert fresh_cb.state == CircuitState.OPEN

    # Đợi qua cooldown (0.1s từ fixture)
    await asyncio.sleep(0.15)

    # Request sau cooldown → HALF_OPEN probe. Success → CLOSED.
    r = await post_with_retry("http://fake/x", {}, 5, client_factory=_mock_client(_ok_handler))
    assert r.status_code == 200
    assert fresh_cb.state == CircuitState.CLOSED
    assert fresh_cb.failure_count == 0


@pytest.mark.asyncio
async def test_cb_half_open_probe_fail_reopens(fresh_cb, monkeypatch):
    monkeypatch.setattr(vlm_client, "MAX_RETRIES", 0)
    for _ in range(3):
        with pytest.raises(httpx.HTTPError):
            await post_with_retry("http://fake/x", {}, 5, client_factory=_mock_client(_500_handler))
    assert fresh_cb.state == CircuitState.OPEN
    opened_at_before = fresh_cb.opened_at

    await asyncio.sleep(0.15)

    # Probe request fail → OPEN lại, opened_at reset
    with pytest.raises(httpx.HTTPError):
        await post_with_retry("http://fake/x", {}, 5, client_factory=_mock_client(_500_handler))
    assert fresh_cb.state == CircuitState.OPEN
    assert fresh_cb.opened_at > opened_at_before    # cooldown timer reset


# ─── Snapshot cho /health ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_snapshot_shape(fresh_cb):
    snap = fresh_cb.snapshot()
    assert snap["state"] == "closed"
    assert snap["failure_count"] == 0
    assert snap["threshold"] == 3
    assert snap["cooldown_s"] == 0.1
    assert snap["opened_at_ago"] is None
