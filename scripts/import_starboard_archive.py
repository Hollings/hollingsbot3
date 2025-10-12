#!/usr/bin/env python3
"""Backfill or update starboard posts from an exported Discord channel dump."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from pathlib import Path
from typing import Iterable, Mapping

DEFAULT_DB = Path(os.getenv("PROMPT_DB_PATH", Path(__file__).resolve().parents[1] / "src/hollingsbot/prompts.db"))


def _load_json(path: Path) -> Mapping[str, object]:
    data = json.loads(path.read_text("utf8"))
    if "messages" not in data:
        raise ValueError("export file missing 'messages' array")
    return data


def _parse_message_link(link: str) -> tuple[int, int] | None:
    parts = link.strip().split("/")
    if len(parts) < 7:
        return None
    try:
        channel_id = int(parts[5])
        message_id = int(parts[6])
    except ValueError:
        return None
    return channel_id, message_id


def _attachments_from_message(msg: Mapping[str, object]) -> list[dict[str, object | None]]:
    out: list[dict[str, object | None]] = []

    for att in msg.get("attachments", []) or []:
        if not isinstance(att, Mapping):
            continue
        url = att.get("url") or att.get("proxyUrl") or att.get("proxiedUrl")
        proxy = att.get("proxyUrl") or att.get("proxiedUrl") or att.get("url")
        if not url and not proxy:
            continue
        out.append(
            {
                "filename": att.get("name") or att.get("filename"),
                "url": url,
                "proxy_url": proxy,
                "content_type": att.get("contentType") or att.get("type"),
                "width": att.get("width"),
                "height": att.get("height"),
            }
        )

    embeds = msg.get("embeds", []) or []
    for embed in embeds:
        if not isinstance(embed, Mapping):
            continue
        image = None
        if isinstance(embed.get("image"), Mapping):
            image = embed["image"]
        elif isinstance(embed.get("thumbnail"), Mapping):
            image = embed["thumbnail"]
        if not isinstance(image, Mapping):
            continue
        url = image.get("url")
        if not url:
            continue
        out.append(
            {
                "filename": embed.get("title") or "embed",
                "url": embed.get("url") or url,
                "proxy_url": url,
                "content_type": "image/embed",
                "width": image.get("width"),
                "height": image.get("height"),
            }
        )
    return out


def import_archive(json_path: Path, db_path: Path, update_existing: bool) -> tuple[int, int]:
    data = _load_json(json_path)
    guild_id = int(data.get("guild", {}).get("id"))
    starboard_channel_id = int(data.get("channel", {}).get("id"))
    messages: Iterable[Mapping[str, object]] = data.get("messages", [])  # type: ignore[assignment]

    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
    except sqlite3.OperationalError:
        pass

    inserted = 0
    updated = 0

    try:
        with conn:
            existing: dict[int, int] = {
                int(row[0]): int(row[1])
                for row in conn.execute(
                    "SELECT original_message_id, id FROM starboard_posts WHERE starboard_channel_id = ?",
                    (starboard_channel_id,),
                )
            }

            for raw_msg in messages:
                if not isinstance(raw_msg, Mapping):
                    continue
                content = raw_msg.get("content") or ""
                if not isinstance(content, str):
                    continue
                lines = [line.strip() for line in content.splitlines() if line.strip()]
                if not lines:
                    continue
                parsed = _parse_message_link(lines[0])
                if parsed is None:
                    continue
                source_channel_id, original_message_id = parsed
                try:
                    starboard_message_id = int(str(raw_msg.get("id")))
                except (TypeError, ValueError):
                    continue
                attachments = _attachments_from_message(raw_msg)
                attachments_json = json.dumps(attachments)

                author = raw_msg.get("author") or {}
                original_author_id = int(author.get("id")) if isinstance(author, Mapping) and author.get("id") else 0
                original_author_name = ""
                if isinstance(author, Mapping):
                    original_author_name = (
                        author.get("nickname")
                        or author.get("globalName")
                        or author.get("name")
                        or ""
                    )
                body = "\n".join(lines[1:]) if len(lines) > 1 else ""
                created_at = raw_msg.get("timestamp") if isinstance(raw_msg.get("timestamp"), str) else None

                existing_row = existing.get(original_message_id)
                if existing_row is not None:
                    if update_existing:
                        conn.execute(
                            """
                            UPDATE starboard_posts
                               SET attachment_urls = ?,
                                   content = CASE WHEN ? != '' THEN ? ELSE content END,
                                   original_author_id = CASE WHEN ? != 0 THEN ? ELSE original_author_id END,
                                   original_author_name = CASE WHEN ? != '' THEN ? ELSE original_author_name END,
                                   created_at = COALESCE(?, created_at)
                             WHERE id = ?
                            """,
                            (
                                attachments_json,
                                body,
                                body,
                                original_author_id,
                                original_author_id,
                                original_author_name,
                                original_author_name,
                                created_at,
                                existing_row,
                            ),
                        )
                        updated += 1
                    continue

                conn.execute(
                    """
                    INSERT INTO starboard_posts (
                        guild_id,
                        source_channel_id,
                        starboard_channel_id,
                        original_message_id,
                        starboard_message_id,
                        reactor_user_id,
                        reaction_emoji,
                        original_author_id,
                        original_author_name,
                        jump_url,
                        content,
                        attachment_urls,
                        created_at
                    ) VALUES (?, ?, ?, ?, ?, 0, NULL, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        guild_id,
                        source_channel_id,
                        starboard_channel_id,
                        original_message_id,
                        starboard_message_id,
                        original_author_id,
                        original_author_name,
                        lines[0],
                        body,
                        attachments_json,
                        created_at,
                    ),
                )
                existing[original_message_id] = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
                inserted += 1
    finally:
        conn.close()

    return inserted, updated


def main() -> None:
    parser = argparse.ArgumentParser(description="Import or update starboard history from a JSON export")
    parser.add_argument("export", type=Path, help="Path to starboard.json export")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="SQLite database path (default: %(default)s)")
    parser.add_argument(
        "--update-existing",
        action="store_true",
        help="Update attachment/content metadata for rows already present",
    )
    args = parser.parse_args()

    inserted, updated = import_archive(args.export, args.db, args.update_existing)
    print(f"Inserted {inserted} rows; updated {updated} rows")


if __name__ == "__main__":
    main()
