"""Agentic search tool using Claude Code CLI to query message history."""

from __future__ import annotations

import json
import logging
import sqlite3
import subprocess
from pathlib import Path

_LOG = logging.getLogger(__name__)

# Database path (same as prompt_db)
from hollingsbot.prompt_db import DB_PATH


def _get_schema_and_sample() -> str:
    """Get schema info and sample data for context."""
    schema = """
CREATE TABLE cached_messages (
    channel_id INTEGER NOT NULL,
    message_id INTEGER NOT NULL,
    author_id INTEGER,
    author_name TEXT,               -- Display name
    content TEXT,
    timestamp INTEGER NOT NULL,     -- Unix timestamp (seconds since epoch)
    has_images BOOLEAN DEFAULT 0,
    has_attachments BOOLEAN DEFAULT 0,
    PRIMARY KEY (channel_id, message_id)
);
"""
    return schema


def _run_sql_query(query: str) -> str:
    """Execute a read-only SQL query. Returns formatted results."""
    # Security: Only allow SELECT
    query_stripped = query.strip().upper()
    if not query_stripped.startswith("SELECT"):
        return "Error: Only SELECT queries are allowed"

    dangerous = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE", "EXEC"]
    for kw in dangerous:
        if kw in query_stripped:
            return f"Error: {kw} not allowed"

    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query).fetchall()

            if not rows:
                return "No results"

            cols = rows[0].keys()
            lines = [f"Found {len(rows)} row(s):"]
            for i, row in enumerate(rows[:30]):
                data = {c: row[c] for c in cols}
                if "content" in data and data["content"] and len(str(data["content"])) > 150:
                    data["content"] = str(data["content"])[:150] + "..."
                lines.append(f"{i+1}. {json.dumps(data, default=str)}")
            if len(rows) > 30:
                lines.append(f"... +{len(rows)-30} more")
            return "\n".join(lines)
    except Exception as e:
        return f"SQL Error: {e}"


def search_history_agent(thought: str = "", question: str = "") -> str:
    """Try to remember something from message history using Claude Code CLI.

    Args:
        thought: What to try to remember (preferred parameter name)
        question: Legacy parameter name, same as thought

    Returns:
        What was remembered from the database search
    """
    # Support both parameter names
    query = thought or question
    from hollingsbot.tools.parser import get_current_context
    context = get_current_context()
    channel_id = context.get("channel_id") if context else None

    if not channel_id:
        return "Error: No channel context"

    # Build the prompt for Claude Code
    schema = _get_schema_and_sample()

    prompt = f"""You are querying a message database to answer a question for "Wendy's Mobile Oracle".

IMPORTANT: Any mention of "I", "me", or "my" refers to Wendy's Mobile Oracle (author_id = 771821437199581204).

Question: {query}

Database: SQLite at {DB_PATH}
Channel ID: {channel_id}

Schema:
{schema}

Instructions:
1. Query the database using: sqlite3 {DB_PATH}
3. Answer the question using the SPECIFIC DATA you found (quote messages, names, timestamps)
4. If nothing found, say "I can't find that"

CRITICAL: Your answer MUST include actual data from the database. Don't just say "I found it" - include WHAT you found. Quote the relevant message content, say WHO said it and WHEN.

Rules:
- Filter by channel_id = {channel_id}
- For "I/me/my" queries, filter by author_id = 771821437199581204
- Only SELECT queries
- Max 500 characters response
- Respond in first person as Wendy
- You may use multiple rounds of queries to investigate the answer, but don't spend more than 5 turns trying to dig into it

Example good response: "Yeah, on Nov 15th deltaryz asked 'can you count the h's in this?' and I replied with..."
Example bad response: "I found it! That happened!" (NO - this doesn't answer anything)"""

    try:
        # Write prompt to temp file (avoids shell escaping issues)
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(prompt)
            prompt_file = f.name

        # Make readable by claude user
        import os
        os.chmod(prompt_file, 0o644)

        try:
            # Run Claude Code as non-root user (root can't use --dangerously-skip-permissions)
            # Use JSON output to capture full agent trace
            result = subprocess.run(
                [
                    "su", "-s", "/bin/bash", "-c",
                    f'claude -p "$(cat {prompt_file})" --output-format json --dangerously-skip-permissions --model sonnet',
                    "claude"
                ],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(Path(DB_PATH).parent),
            )
        finally:
            # Clean up temp file
            os.unlink(prompt_file)

        if result.returncode != 0:
            _LOG.error("Claude CLI failed (code %d): %s", result.returncode, result.stderr)
            return _fallback_search(query, channel_id)

        raw_output = result.stdout.strip()
        if not raw_output:
            _LOG.warning("Claude CLI returned empty response")
            return _fallback_search(query, channel_id)

        # Parse JSON output to extract final answer and log full trace
        try:
            data = json.loads(raw_output)

            # Log the full agent trace for debugging
            _LOG.info("=== AGENT TRACE ===")
            if isinstance(data, dict):
                # Log messages/conversation if present
                messages = data.get("messages", data.get("conversation", []))
                for i, msg in enumerate(messages if isinstance(messages, list) else []):
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        # Handle content blocks
                        for block in content:
                            if isinstance(block, dict):
                                block_type = block.get("type", "")
                                if block_type == "tool_use":
                                    _LOG.info(f"  [{i}] {role} -> TOOL: {block.get('name')} | input: {json.dumps(block.get('input', {}))[:200]}")
                                elif block_type == "tool_result":
                                    _LOG.info(f"  [{i}] {role} -> TOOL_RESULT: {str(block.get('content', ''))[:200]}")
                                elif block_type == "text":
                                    _LOG.info(f"  [{i}] {role}: {block.get('text', '')[:200]}")
                    else:
                        _LOG.info(f"  [{i}] {role}: {str(content)[:200]}")

                # Extract final response text
                response = data.get("result", data.get("response", data.get("text", "")))
                if not response and messages:
                    # Get last assistant message
                    for msg in reversed(messages if isinstance(messages, list) else []):
                        if msg.get("role") == "assistant":
                            content = msg.get("content", "")
                            if isinstance(content, str):
                                response = content
                            elif isinstance(content, list):
                                for block in content:
                                    if isinstance(block, dict) and block.get("type") == "text":
                                        response = block.get("text", "")
                                        break
                            break
            else:
                response = str(data)

            _LOG.info("=== END AGENT TRACE ===")

        except json.JSONDecodeError:
            # Fallback: treat as plain text
            _LOG.warning("Failed to parse JSON output, using raw text")
            response = raw_output

        return response.strip() if response else _fallback_search(query, channel_id)

    except FileNotFoundError:
        _LOG.warning("Claude CLI not found, using fallback search")
        return _fallback_search(query, channel_id)
    except subprocess.TimeoutExpired:
        _LOG.error("Claude CLI timed out")
        return "Search timed out"
    except Exception as e:
        _LOG.exception("Claude CLI error: %s", e)
        return _fallback_search(query, channel_id)


def _fallback_search(thought: str, channel_id: int | None) -> str:
    """Simple keyword search fallback when Claude CLI fails."""
    if not channel_id:
        return "I can't remember - no channel context"

    keywords = [w for w in thought.lower().split() if len(w) > 3]

    results = []
    for kw in keywords[:3]:
        # Use parameterized query to prevent SQL injection
        sql = """
            SELECT author_name, content, datetime(timestamp, 'unixepoch') as time
            FROM cached_messages
            WHERE channel_id = ? AND LOWER(content) LIKE ?
            ORDER BY timestamp DESC LIMIT 5
        """
        try:
            with sqlite3.connect(DB_PATH) as conn:
                conn.row_factory = sqlite3.Row
                # Escape any existing % or _ in keyword and wrap with %
                escaped_kw = kw.replace('%', r'\%').replace('_', r'\_')
                rows = conn.execute(sql, (channel_id, f'%{escaped_kw}%')).fetchall()

                if not rows:
                    continue

                cols = rows[0].keys()
                lines = [f"Found {len(rows)} row(s):"]
                for i, row in enumerate(rows[:5]):
                    data = {c: row[c] for c in cols}
                    if "content" in data and data["content"] and len(str(data["content"])) > 150:
                        data["content"] = str(data["content"])[:150] + "..."
                    lines.append(f"{i+1}. {json.dumps(data, default=str)}")
                results.append(f"'{kw}':\n" + "\n".join(lines))
        except Exception as e:
            _LOG.warning("Fallback search error for keyword '%s': %s", kw, e)
            continue

    return "\n\n".join(results) if results else "I can't remember that"
