"""Per-event-loop cache for async SDK clients.

Celery tasks run coroutines with ``asyncio.run()``, which creates and destroys
a fresh event loop per task. An async client cached across tasks keeps its
httpx connection pool bound to the already-closed loop, so every later request
in that worker process fails with ``RuntimeError: Event loop is closed``.

Keying the cache by the running loop gives each loop its own client while
still reusing the client (and its connection pool) for the lifetime of a
long-lived loop such as the main bot process.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

# One slot per client name: (loop it was created on, client). A stale entry
# from a finished Celery-task loop is simply replaced, so the cache stays
# bounded at one client per name.
_cache: dict[str, tuple[asyncio.AbstractEventLoop, Any]] = {}


def get_client(name: str, factory: Callable[[], Any]) -> Any:
    """Return the cached client for *name*, rebuilding it on a new event loop.

    Must be called from a running event loop (i.e. inside a coroutine).
    """
    loop = asyncio.get_running_loop()
    entry = _cache.get(name)
    if entry is not None and entry[0] is loop:
        return entry[1]
    client = factory()
    _cache[name] = (loop, client)
    return client
