"""Async bridge to Celery's synchronous publish and result APIs."""

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from celery.exceptions import Terminated, TimeoutError as CeleryTimeoutError, WorkerLostError
from kombu.exceptions import OperationalError
from redis.exceptions import RedisError

_LOG = logging.getLogger(__name__)
_REVOKE_GRACE = 0.5


class RemoteOutcomeUnknown(TimeoutError):
    """A remote operation may have run; starting another would risk duplication."""


async def run_task(task, args, *, kwargs=None, timeout: float, poll_interval: float = 0.5, queue=None):
    """Bound publish and result polling together; expire and revoke abandoned work."""
    options = {"kwargs": kwargs or {}, "expires": datetime.now(timezone.utc) + timedelta(seconds=timeout)}
    if queue is not None:
        options["queue"] = queue
    submission = asyncio.create_task(asyncio.to_thread(task.apply_async, args, **options))

    async def wait():
        # Cancelling to_thread does not cancel the publish itself.
        try:
            result = await asyncio.shield(submission)
        except Exception as exc:
            raise RemoteOutcomeUnknown("Could not confirm Celery publication.") from exc
        try:
            while not await asyncio.to_thread(result.ready):
                await asyncio.sleep(poll_interval)
            return await asyncio.to_thread(result.get, timeout=0.1)
        except (CeleryTimeoutError, OperationalError, RedisError, Terminated, WorkerLostError) as exc:
            raise RemoteOutcomeUnknown("Could not confirm the Celery task outcome.") from exc

    async def revoke():
        try:
            result = await asyncio.wait_for(asyncio.shield(submission), _REVOKE_GRACE)
            # Revoke queued work, but do not kill a worker process which may
            # already have moved on to another task. Running work is bounded
            # by its worker-side limits, not guaranteed to have stopped.
            await asyncio.wait_for(asyncio.to_thread(result.revoke, terminate=False), _REVOKE_GRACE)
        except asyncio.TimeoutError:
            # A late publish still carries an absolute expiration time. Consume
            # any late exception even when this caller can no longer wait.
            submission.add_done_callback(lambda task: None if task.cancelled() else task.exception())
            _LOG.warning("Timed out revoking abandoned Celery task")
        except Exception:
            _LOG.exception("Could not revoke abandoned Celery task")

    try:
        return await asyncio.wait_for(wait(), timeout)
    except asyncio.CancelledError:
        await revoke()
        raise
    except (asyncio.TimeoutError, RemoteOutcomeUnknown) as exc:
        await revoke()
        if isinstance(exc, RemoteOutcomeUnknown):
            raise
        raise RemoteOutcomeUnknown("Timed out waiting for the Celery task outcome.") from exc
