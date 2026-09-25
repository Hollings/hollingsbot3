"""A Discord message that grows as its text is written.

The first :meth:`LiveMessage.update` sends the message; later updates edit it,
at most once per ``interval`` seconds, from a background task so a slow or
rate-limited edit never holds up whoever is producing the text.
:meth:`LiveMessage.finish` writes the final text and stops the background task.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Awaitable, Callable

import discord

_LOG = logging.getLogger(__name__)

Send = Callable[[str], Awaitable[discord.Message]]
Edit = Callable[[discord.Message, str], Awaitable[object]]


class LiveMessage:
    def __init__(self, send: Send, edit: Edit, *, interval: float = 1.0) -> None:
        self._send = send
        self._edit = edit
        self._interval = interval
        self.message: discord.Message | None = None
        self._shown = ""
        self._latest = ""
        self._last_write = 0.0
        self._flusher: asyncio.Task | None = None
        self._lock = asyncio.Lock()

    async def update(self, text: str) -> None:
        """Show ``text`` soon. The first call sends the message and waits for it."""
        self._latest = text
        if self.message is None:
            async with self._lock:
                if self.message is None:
                    self.message = await self._send(text)
                    self._shown = text
                    self._last_write = time.monotonic()
            return
        if self._flusher is None or self._flusher.done():
            self._flusher = asyncio.create_task(self._flush_soon())

    async def finish(self, text: str) -> discord.Message | None:
        """Write ``text`` now (sending it if nothing was sent yet) and stop editing."""
        self._latest = text
        if self._flusher is not None and not self._flusher.done():
            self._flusher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._flusher
        async with self._lock:
            if self.message is None:
                if text:
                    self.message = await self._send(text)
                    self._shown = text
            elif text and text != self._shown:
                await self._edit(self.message, text)
                self._shown = text
        return self.message

    async def _flush_soon(self) -> None:
        while self._latest != self._shown:
            wait = self._last_write + self._interval - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            async with self._lock:
                text = self._latest
                if self.message is None or text == self._shown:
                    return
                try:
                    await self._edit(self.message, text)
                except discord.HTTPException:
                    # A dropped intermediate edit is harmless: finish() writes the final text.
                    _LOG.warning("live message edit failed", exc_info=True)
                self._shown = text
                self._last_write = time.monotonic()
