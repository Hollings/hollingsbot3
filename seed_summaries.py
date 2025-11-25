"""Seed the summary database with historical messages and generate initial summaries.

This script uses the message-count-based summarization system:
- Level-1: Groups of 5 raw messages
- Level-2: Groups of 5 level-1 summaries (25 messages total)
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

import discord
from discord.ext import commands

from hollingsbot.summarization import (
    GROUP_SIZE,
    CachedMessage,
    MessageGroup,
    Summarizer,
    SummaryCache,
)
from hollingsbot.text_generators.anthropic import AnthropicTextGenerator


class SummarizerLLM:
    """Adapter for AnthropicTextGenerator to work with Summarizer."""

    def __init__(self, model: str = "claude-haiku-4-5"):
        self._gen = AnthropicTextGenerator(model=model)

    async def generate(self, prompt: str) -> str:
        return await self._gen.generate(prompt)


async def seed_summaries(
    channel_id: int,
    hours_back: int = 24,
    dry_run: bool = False,
):
    """Fetch historical messages and generate summaries.

    Args:
        channel_id: Discord channel ID to fetch from
        hours_back: How many hours of history to fetch
        dry_run: If True, only cache messages without generating summaries
    """
    token = os.getenv("DISCORD_TOKEN")
    if not token:
        print("ERROR: DISCORD_TOKEN not set")
        return

    db_path = os.getenv("SUMMARY_DB_PATH", "data/summaries.db")
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"Database path: {db_path}")
    print(f"Channel ID: {channel_id}")
    print(f"Hours back: {hours_back}")
    print(f"Dry run: {dry_run}")
    print(f"Group size: {GROUP_SIZE}")
    print()

    # Initialize components
    cache = SummaryCache(db_path)

    # Create bot to fetch messages
    intents = discord.Intents.default()
    intents.message_content = True
    bot = commands.Bot(command_prefix="!", intents=intents)

    @bot.event
    async def on_ready():
        print(f"Bot logged in as {bot.user}")

        channel = bot.get_channel(channel_id)
        if not channel:
            print(f"ERROR: Channel {channel_id} not found")
            await bot.close()
            return

        print(f"Fetching messages from #{channel.name}")

        # Calculate time range
        now = datetime.now(timezone.utc)
        after_time = now - timedelta(hours=hours_back)

        print(f"Fetching messages from {after_time} to {now}")

        # Fetch messages
        messages = []
        try:
            async for msg in channel.history(limit=None, after=after_time, oldest_first=True):
                messages.append(msg)
        except Exception as e:
            print(f"ERROR fetching messages: {e}")
            await bot.close()
            return

        print(f"Found {len(messages)} messages")

        # Cache messages
        cached_count = 0
        for msg in messages:
            # Skip empty messages
            if not msg.content.strip() and not msg.attachments:
                continue

            # Skip bot commands
            if msg.content.startswith(("!", "$", "edit:")):
                continue

            try:
                cached = CachedMessage(
                    channel_id=channel_id,
                    message_id=msg.id,
                    author_id=msg.author.id,
                    author_name=msg.author.display_name,
                    content=msg.content,
                    timestamp=int(msg.created_at.timestamp()),
                    has_images=any(
                        att.content_type and att.content_type.startswith("image/") for att in msg.attachments
                    ),
                    has_attachments=bool(msg.attachments),
                )
                cache.cache_message(cached)
                cached_count += 1
            except Exception as e:
                print(f"Error caching message {msg.id}: {e}")

        print(f"Cached {cached_count} messages")

        if dry_run:
            print("Dry run - skipping summary generation")
            await bot.close()
            return

        # Generate summaries
        print()
        print("Generating summaries...")

        summary_model = os.getenv("LLM_SUMMARY_MODEL", "claude-haiku-4-5")
        print(f"Using model: {summary_model}")

        llm = SummarizerLLM(summary_model)
        summarizer = Summarizer(llm)

        # Get all cached messages for this channel
        all_messages = cache.get_all_messages_ordered(channel_id)
        print(f"Total cached messages: {len(all_messages)}")

        if len(all_messages) < GROUP_SIZE * 2:
            print(f"Not enough messages for summarization (need at least {GROUP_SIZE * 2})")
            await bot.close()
            return

        # Keep the most recent GROUP_SIZE messages unsummarized (creates 2-msg overlap with 7 raw)
        messages_to_summarize = all_messages[:-GROUP_SIZE]
        print(f"Messages available for summarization: {len(messages_to_summarize)}")

        # Generate Level-1 summaries (groups of GROUP_SIZE messages)
        level_1_created = 0
        level_1_groups = []

        for i in range(0, len(messages_to_summarize) - GROUP_SIZE + 1, GROUP_SIZE):
            chunk = messages_to_summarize[i : i + GROUP_SIZE]
            if len(chunk) < GROUP_SIZE:
                break

            start_id = chunk[0].message_id
            end_id = chunk[-1].message_id

            # Check if group already exists
            if cache.group_exists(channel_id, 1, start_id):
                # Get existing group for level-2 processing
                existing = cache.get_groups_by_level(channel_id, 1)
                for g in existing:
                    if g.start_message_id == start_id:
                        level_1_groups.append(g)
                        break
                continue

            print(f"  Summarizing messages {start_id}-{end_id} ({len(chunk)} messages)...")

            try:
                summary_text = await summarizer.summarize_messages(chunk)

                group = MessageGroup(
                    id=None,
                    channel_id=channel_id,
                    level=1,
                    start_message_id=start_id,
                    end_message_id=end_id,
                    summary_text=summary_text,
                    message_count=len(chunk),
                    start_timestamp=chunk[0].timestamp,
                    end_timestamp=chunk[-1].timestamp,
                )
                group_id = cache.save_message_group(group)
                group.id = group_id
                level_1_groups.append(group)
                level_1_created += 1
                print(f"    Created level-1 summary ({len(summary_text)} chars)")
            except Exception as e:
                print(f"    ERROR: {e}")

        print()
        print(f"Created {level_1_created} Level-1 summaries")

        # Generate Level-2 summaries (groups of GROUP_SIZE level-1 summaries)
        # Re-fetch all level-1 groups to include any pre-existing ones
        all_level_1 = cache.get_summarized_groups(channel_id, 1)
        print(f"Total level-1 groups with summaries: {len(all_level_1)}")

        level_2_created = 0
        level_2_groups = []

        for i in range(0, len(all_level_1) - GROUP_SIZE + 1, GROUP_SIZE):
            chunk = all_level_1[i : i + GROUP_SIZE]
            if len(chunk) < GROUP_SIZE:
                break

            start_id = chunk[0].start_message_id
            end_id = chunk[-1].end_message_id

            # Check if group already exists
            if cache.group_exists(channel_id, 2, start_id):
                existing = cache.get_groups_by_level(channel_id, 2)
                for g in existing:
                    if g.start_message_id == start_id:
                        level_2_groups.append(g)
                        break
                continue

            total_messages = sum(g.message_count for g in chunk)
            print(f"  Combining {GROUP_SIZE} level-1 summaries ({total_messages} messages)...")

            try:
                summary_text = await summarizer.summarize_groups(chunk)

                group = MessageGroup(
                    id=None,
                    channel_id=channel_id,
                    level=2,
                    start_message_id=start_id,
                    end_message_id=end_id,
                    summary_text=summary_text,
                    message_count=total_messages,
                    start_timestamp=chunk[0].start_timestamp,
                    end_timestamp=chunk[-1].end_timestamp,
                )
                group_id = cache.save_message_group(group)
                group.id = group_id
                level_2_groups.append(group)
                level_2_created += 1
                print(f"    Created level-2 summary ({len(summary_text)} chars)")
            except Exception as e:
                print(f"    ERROR: {e}")

        print()
        print(f"Created {level_2_created} Level-2 summaries")

        # Print summary stats
        print()
        print("Summary statistics:")
        for level in [1, 2]:
            groups = cache.get_groups_by_level(channel_id, level)
            summarized = [g for g in groups if g.summary_text]
            total_coverage = sum(g.message_count for g in summarized)
            print(f"  Level {level}: {len(summarized)} summaries covering {total_coverage} messages")

        await bot.close()

    await bot.start(token)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Seed summary database with historical messages")
    parser.add_argument(
        "--channel", type=int, default=1050900592031178752, help="Discord channel ID (default: LLM chat channel)"
    )
    parser.add_argument("--hours", type=int, default=24, help="Hours of history to fetch (default: 24)")
    parser.add_argument("--dry-run", action="store_true", help="Only cache messages, don't generate summaries")

    args = parser.parse_args()

    asyncio.run(
        seed_summaries(
            channel_id=args.channel,
            hours_back=args.hours,
            dry_run=args.dry_run,
        )
    )
