"""Tests for Jev's reply log / daily spend."""

from __future__ import annotations

import contextlib
import json
import sqlite3
from datetime import datetime, timezone

from hollingsbot.jev.client import Usage
from hollingsbot.jev.ledger import JevLedger
from hollingsbot.jev.writer import ChatLine, Reply, Step

CHAT = [ChatLine("Hollings", "Jev capital of France?")]


def reply(cost: float, text: str = "paris") -> Reply:
    usage = Usage(calls=4, input_tokens=70000, cost=cost, seconds=2.5)
    return Reply(text, [text], [Step(text, 0.8, 0.7, 0.0, 1.0, 55)], "complete", usage, model="jev-1.13")


def test_spent_today_sums_only_today(temp_db):
    ledger = JevLedger(temp_db)
    today = datetime(2026, 9, 25, 12, tzinfo=timezone.utc)
    yesterday = datetime(2026, 9, 24, 23, tzinfo=timezone.utc)
    ledger.record(reply(0.01), CHAT, channel_id=1, message_id=2, now=today)
    ledger.record(reply(0.02), CHAT, channel_id=1, message_id=3, now=today)
    ledger.record(reply(5.0), CHAT, channel_id=1, message_id=4, now=yesterday)
    assert abs(ledger.spent_today(now=today) - 0.03) < 1e-9
    assert JevLedger(temp_db).spent_today(now=datetime(2026, 9, 26, tzinfo=timezone.utc)) == 0.0


def test_record_keeps_the_trace_and_override_reason(temp_db):
    JevLedger(temp_db).record(reply(0.01), CHAT, channel_id=7, message_id=None, stop_reason="interrupted")
    with contextlib.closing(sqlite3.connect(temp_db)) as conn:
        row = conn.execute(
            "SELECT reply, stop_reason, words, calls, chat_json, steps_json, model FROM jev_replies"
        ).fetchone()
    text, reason, words, calls, chat_json, steps_json, model = row
    assert (text, reason, words, calls, model) == ("paris", "interrupted", 1, 4, "jev-1.13")
    assert json.loads(chat_json) == [{"speaker": "Hollings", "text": "Jev capital of France?"}]
    assert json.loads(steps_json)[0]["word"] == "paris"
