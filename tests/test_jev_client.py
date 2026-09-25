"""Tests for the Decisions API client (retries, errors, usage metering)."""

from __future__ import annotations

import json

import httpx
import pytest

from hollingsbot.jev import client as client_mod
from hollingsbot.jev.client import DEFAULT_MODEL, DecisionsClient, JevError, Usage

ANSWERS = {"q": {"type": "noul", "noul": 0.9}}


def _ok(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"answers": ANSWERS, "usage": {"input_tokens": 300, "cost": 0.00001}})


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    async def instant(_seconds):
        return None

    monkeypatch.setattr(client_mod.asyncio, "sleep", instant)


def make(handler, **kw) -> DecisionsClient:
    return DecisionsClient(api_key="k", transport=httpx.MockTransport(handler), **kw)


async def test_sends_model_state_questions_and_bearer_key():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers["Authorization"]
        seen["body"] = json.loads(request.content)
        seen["url"] = str(request.url)
        return _ok(request)

    usage = Usage()
    answers = await make(handler).ask({"chat": []}, {"q": {"type": "noul", "instructions": "?"}}, usage)
    assert answers == ANSWERS
    assert seen["auth"] == "Bearer k"
    assert seen["url"] == client_mod.DECISIONS_URL
    assert seen["body"]["model"] == DEFAULT_MODEL
    assert seen["body"]["state"] == {"chat": []}
    assert usage.calls == 1 and usage.input_tokens == 300 and usage.cost == pytest.approx(0.00001)


async def test_retries_rate_limit_then_succeeds():
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(429, headers={"Retry-After": "0"}, text="slow down")
        return _ok(request)

    assert await make(handler).ask("s", {}) == ANSWERS
    assert len(calls) == 3


async def test_gives_up_after_retries():
    def handler(request):
        return httpx.Response(502, text="bad gateway")

    with pytest.raises(JevError, match="502"):
        await make(handler, retries=2).ask("s", {})


async def test_client_errors_are_not_retried():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(400, text="model does not exist")

    with pytest.raises(JevError, match="400"):
        await make(handler).ask("s", {})
    assert len(calls) == 1


async def test_transport_errors_retry_then_raise():
    def handler(request):
        raise httpx.ConnectError("down", request=request)

    with pytest.raises(JevError, match="unreachable"):
        await make(handler, retries=1).ask("s", {})


def test_needs_a_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(JevError):
        DecisionsClient()
