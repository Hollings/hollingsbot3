"""Post a test prompt and a real Jev reply in a channel, without joining the gateway.

Checks the Discord half of JevBot end to end (webhook found or created with its
avatar, the typing indicator, the reply posted once) using the bot's REST API
only, so it can run while the real bot is online. What it cannot check is the
on_message trigger: that needs a human to type in the channel.

The prompt is posted as the bot, marked as a smoke test. Replies are logged to
a throwaway ledger, not the bot DB.

    python scripts/jev_discord_smoke.py 1473033550805598253 "Jev, what's the capital of France?"
    python scripts/jev_discord_smoke.py 1473033550805598253 "hi Jev2" --name Jev2   # a copy's webhook
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import discord
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hollingsbot.cogs.chat_bots.jev_bot import JevBot, JevBotSettings
from hollingsbot.cogs.typing_tracker import TypingTracker
from hollingsbot.jev import ChatLine
from hollingsbot.jev.ledger import JevLedger
from hollingsbot.jev.lexicon import Lexicon


class _Coordinator:
    async def _add_response_to_history(self, *args) -> None:
        pass


async def run(channel_id: int, prompt: str, speaker: str, name: str | None = None) -> None:
    import os

    client = discord.Client(intents=discord.Intents.none())
    await client.login(os.environ["DISCORD_TOKEN"])  # REST only: no gateway, no events
    try:
        channel = await client.fetch_channel(channel_id)
        print(f"channel #{channel.name}; bot user {client.user}")
        await channel.send(f"(Jev smoke test, asking as {speaker}) {prompt}")

        settings = JevBotSettings.from_env()
        if name:  # a spawn-only copy, e.g. Jev2: its own webhook and the name it's told it has
            settings = settings.copy_named(name)
        settings = dataclasses.replace(settings, channels=frozenset({channel_id}))
        jev = JevBot(client, _Coordinator(), TypingTracker(), settings)  # no gateway: never sees typing
        with tempfile.TemporaryDirectory() as tmp:
            jev.ledger = JevLedger(Path(tmp) / "smoke.db")
            jev.lexicon = Lexicon(Path(tmp) / "smoke.db")
            jev.lexicon.learn(prompt, speaker=speaker)

            async def add_reaction(emoji):
                print(f"  reaction {emoji!r}")

            message = SimpleNamespace(channel=channel, add_reaction=add_reaction)
            result = await jev._write_and_send(message, [ChatLine(speaker, prompt)])
            print("result:", result)
            if result:
                sent = await channel.fetch_message(result["message_id"])
                print(f"in channel: {sent.author} (webhook {sent.webhook_id}): {sent.content!r}")
                print(f"  edited: {sent.edited_at}")  # None: posted once, whole
            print(f"spent ${jev.ledger.spent_today():.4f}")
    finally:
        await client.close()


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("channel_id", type=int)
    ap.add_argument("prompt")
    ap.add_argument("--as", dest="speaker", default="Hollings")
    ap.add_argument("--name", help="reply as a copy of Jev under this name (e.g. Jev2)")
    args = ap.parse_args()
    asyncio.run(run(args.channel_id, args.prompt, args.speaker, args.name))


if __name__ == "__main__":
    main()
