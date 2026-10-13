import asyncio
import threading
from unittest.mock import Mock

import pytest

from hollingsbot.utils.celery_utils import run_task


async def test_broker_calls_run_off_the_event_loop():
    owner = threading.get_ident()
    seen = []

    def called(value):
        seen.append(threading.get_ident())
        return value

    result = Mock()
    result.ready.side_effect = lambda: called(True)
    result.get.side_effect = lambda **kw: called("hello")
    task = Mock()
    task.apply_async.side_effect = lambda *a, **kw: called(result)
    assert await run_task(task, ("prompt",), timeout=1) == "hello"
    assert seen and all(thread != owner for thread in seen)
    result.revoke.assert_not_called()


async def test_timeout_revokes_the_remote_job():
    result = Mock()
    result.ready.return_value = False
    task = Mock()
    task.apply_async.return_value = result
    with pytest.raises(asyncio.TimeoutError):
        await run_task(task, (), timeout=0.02, poll_interval=0.001)
    result.revoke.assert_called_once_with(terminate=True)


async def test_cancellation_during_publish_still_revokes_the_job():
    started, release = threading.Event(), threading.Event()
    result = Mock()

    def publish(*args, **kwargs):
        started.set()
        assert release.wait(2)
        return result

    task = Mock()
    task.apply_async.side_effect = publish
    waiting = asyncio.create_task(run_task(task, (), timeout=1))
    assert await asyncio.to_thread(started.wait, 2)
    waiting.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    result.revoke.assert_called_once_with(terminate=True)


async def test_provider_failure_is_preserved():
    result = Mock()
    result.ready.return_value = True
    result.get.side_effect = ValueError("provider rejected prompt")
    task = Mock()
    task.apply_async.return_value = result
    with pytest.raises(ValueError, match="provider rejected"):
        await run_task(task, (), timeout=1)
    result.revoke.assert_not_called()
