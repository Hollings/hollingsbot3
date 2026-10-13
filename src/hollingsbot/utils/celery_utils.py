"""Async bridge to Celery's synchronous publish and result APIs."""

import asyncio
import logging

_LOG = logging.getLogger(__name__)


async def run_task(task, args, *, kwargs=None, timeout: float, poll_interval: float = 0.5, queue=None):
    """Publish off-thread, await a result, and revoke work abandoned by its caller."""
    options = {"kwargs": kwargs or {}}
    if queue is not None:
        options["queue"] = queue
    submission = asyncio.create_task(asyncio.to_thread(task.apply_async, args, **options))
    result = None
    try:
        # A cancelled to_thread keeps publishing; retain its result so that the
        # task can still be revoked instead of silently running without a caller.
        result = await asyncio.shield(submission)

        async def wait():
            while not await asyncio.to_thread(result.ready):
                await asyncio.sleep(poll_interval)
            return await asyncio.to_thread(result.get, timeout=0.1)

        return await asyncio.wait_for(wait(), timeout)
    except (asyncio.CancelledError, asyncio.TimeoutError):
        try:
            if result is None:
                result = await submission
            await asyncio.to_thread(result.revoke, terminate=True)
        except Exception:
            _LOG.exception("Could not revoke abandoned Celery task")
        raise
