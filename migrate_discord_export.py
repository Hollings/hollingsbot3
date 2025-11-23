#!/usr/bin/env python3
"""Migration script to import DiscordChatExporter JSON into message history."""

import json
import sys
from pathlib import Path

# Add src to path so we can import hollingsbot modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

from hollingsbot.prompt_db import bulk_add_messages


def parse_discord_export(json_path: str) -> list[dict]:
    """Parse DiscordChatExporter JSON and convert to message history format."""

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    channel_id = int(data["channel"]["id"])
    guild_id = int(data["guild"]["id"]) if data.get("guild") else None

    messages = []
    for msg in data["messages"]:
        # Extract attachment URLs
        attachment_urls = [att["url"] for att in msg.get("attachments", [])]

        # Extract reply reference
        reply_to_id = None
        if msg.get("reference") and msg["reference"].get("messageId"):
            reply_to_id = int(msg["reference"]["messageId"])

        # Parse reactions
        reactions = []
        for reaction in msg.get("reactions", []):
            emoji_info = reaction["emoji"]
            user_ids = [int(u["id"]) for u in reaction.get("users", [])]
            reactions.append(
                {
                    "emoji": emoji_info.get("name", ""),
                    "emoji_id": emoji_info.get("id", ""),
                    "count": reaction.get("count", 0),
                    "user_ids": user_ids,
                }
            )

        # Determine if webhook (Discord webhooks appear as bots)
        author = msg["author"]
        is_bot = author.get("isBot", False)
        # Webhooks typically don't have discriminator or have specific patterns
        is_webhook = is_bot and (author.get("discriminator") == "0000" or msg.get("type") == "WebhookMessage")

        messages.append(
            {
                "message_id": int(msg["id"]),
                "channel_id": channel_id,
                "guild_id": guild_id,
                "timestamp": msg["timestamp"],
                "author_id": int(author["id"]),
                "author_nickname": author.get("nickname") or author.get("name"),
                "is_bot": is_bot,
                "is_webhook": is_webhook,
                "content": msg.get("content"),
                "attachment_urls": attachment_urls,
                "reply_to_id": reply_to_id,
                "reactions": reactions,
            }
        )

    return messages


def main():
    if len(sys.argv) < 2:
        print("Usage: python migrate_discord_export.py <path_to_export.json>")
        print("\nExample: python migrate_discord_export.py dump.json")
        sys.exit(1)

    json_path = sys.argv[1]

    if not Path(json_path).exists():
        print(f"Error: File not found: {json_path}")
        sys.exit(1)

    print(f"Reading {json_path}...")
    messages = parse_discord_export(json_path)

    print(f"Parsed {len(messages)} messages")
    print("Inserting into database...")

    count = bulk_add_messages(messages)

    print(f"Successfully inserted {count} messages into message_history table")


if __name__ == "__main__":
    main()
