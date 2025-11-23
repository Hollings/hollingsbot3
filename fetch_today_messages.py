"""Fetch all messages from today for a specific channel."""

import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path

import discord
from discord.ext import commands


async def fetch_messages():
    """Fetch messages from today for the specified channel."""
    token = os.getenv("DISCORD_TOKEN")
    channel_id = 1050900592031178752

    # Create a minimal bot
    intents = discord.Intents.default()
    intents.message_content = True
    bot = commands.Bot(command_prefix="!", intents=intents)

    @bot.event
    async def on_ready():
        print(f"Bot logged in as {bot.user}")

        # Get the channel
        channel = bot.get_channel(channel_id)
        if not channel:
            print(f"Channel {channel_id} not found")
            await bot.close()
            return

        print(f"Fetching messages from #{channel.name}")

        # Get today's date (start of day in UTC)
        now = datetime.now(timezone.utc)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

        print(f"Fetching messages since {today_start}")

        # Fetch messages
        messages = []
        async for message in channel.history(limit=None, after=today_start, oldest_first=True):
            messages.append(message)

        print(f"Found {len(messages)} messages from today")

        # Format messages
        output_lines = []
        output_lines.append(f"Messages from #{channel.name} on {now.strftime('%Y-%m-%d')}")
        output_lines.append("=" * 80)
        output_lines.append("")

        for msg in messages:
            timestamp = msg.created_at.strftime("%H:%M:%S")
            author = msg.author.display_name

            # Handle bot/webhook markers
            markers = []
            if msg.author.bot:
                markers.append("BOT")
            if msg.webhook_id:
                markers.append("WEBHOOK")
            marker_str = f" [{', '.join(markers)}]" if markers else ""

            # Format content
            content = msg.content.strip()

            # Add attachments info
            if msg.attachments:
                attachment_info = ", ".join([f"<{att.filename}>" for att in msg.attachments])
                content += f" [Attachments: {attachment_info}]"

            output_lines.append(f"[{timestamp}] {author}{marker_str}:")
            if content:
                # Indent multi-line messages
                for line in content.split("\n"):
                    output_lines.append(f"  {line}")
            else:
                output_lines.append("  [no content]")
            output_lines.append("")

        # Write to file
        output_path = Path("generated") / f"messages_{now.strftime('%Y%m%d')}.txt"
        output_path.parent.mkdir(exist_ok=True)
        output_path.write_text("\n".join(output_lines), encoding="utf-8")

        print(f"Saved {len(messages)} messages to {output_path}")

        await bot.close()

    await bot.start(token)


if __name__ == "__main__":
    asyncio.run(fetch_messages())
