"""Temp bot package.

Re-exports :class:`TempBotManager` so existing import sites
(``from hollingsbot.cogs.chat_bots.temp_bot import TempBotManager``) keep
working after the file-to-package refactor.
"""

from .manager import TempBotManager

__all__ = ["TempBotManager"]
