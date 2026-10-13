import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from hollingsbot.summarization.summary_cache import CachedMessage
from hollingsbot.summarization.summary_worker import SummaryWorker


async def test_messages_arriving_during_generation_trigger_another_pass():
    worker = SummaryWorker(Mock(), Mock())
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def run(channel):
        calls.append(channel)
        if len(calls) == 1:
            entered.set()
            await release.wait()

    worker._run_summarization = run
    first = asyncio.create_task(worker.trigger_summarization(1))
    await entered.wait()
    await worker.trigger_summarization(1)
    await worker.trigger_summarization(1)
    release.set()
    await first
    assert calls == [1, 1]


async def test_empty_summary_is_not_saved_as_permanently_complete():
    cache = Mock()
    summarizer = Mock(summarize_messages=AsyncMock(return_value="  "))
    worker = SummaryWorker(cache, summarizer)
    message = CachedMessage(1, 1, 1, "A", "hello", 1)
    with pytest.raises(ValueError, match="empty"):
        await worker._generate_summary_for_group(1, 1, [message])
    cache.save_message_group.assert_not_called()
