#!/usr/bin/env python3
"""Execute read-only SQL queries against the message database.

Security measures:
1. Database opened in read-only mode (?mode=ro)
2. PRAGMA query_only = ON (rejects all write operations)
3. Authorizer callback rejects INSERT, UPDATE, DELETE, DROP, ALTER, CREATE
4. Statement must start with SELECT or WITH (CTE)

Usage:
    python query_db.py "SELECT * FROM cached_messages LIMIT 10"
    python query_db.py "SELECT author_name, COUNT(*) FROM cached_messages GROUP BY author_name"
    echo "SELECT ..." | python query_db.py --stdin

Examples:
    # Find messages by a user
    python query_db.py "SELECT * FROM cached_messages WHERE author_name LIKE '%John%' LIMIT 20"

    # Count messages per channel
    python query_db.py "SELECT channel_id, COUNT(*) as cnt FROM cached_messages GROUP BY channel_id ORDER BY cnt DESC"

    # Search message content
    python query_db.py "SELECT author_name, content FROM cached_messages WHERE content LIKE '%hello%' LIMIT 10"

    # Join with message_history for more details
    python query_db.py "SELECT h.author_nickname, h.content, h.reactions FROM message_history h WHERE h.content LIKE '%thanks%' LIMIT 10"
"""
import argparse
import json
import sqlite3
import sys

DB_PATH = "/data/hollingsbot.db"

# SQL operations to allow (read-only)
ALLOWED_OPERATIONS = {
    sqlite3.SQLITE_SELECT,      # SELECT statements
    sqlite3.SQLITE_READ,        # Reading a table
    sqlite3.SQLITE_FUNCTION,    # Using functions
    sqlite3.SQLITE_PRAGMA,      # PRAGMA (we'll filter dangerous ones)
}

# Dangerous PRAGMAs that could modify state
DANGEROUS_PRAGMAS = {
    'journal_mode', 'synchronous', 'cache_size', 'page_size',
    'auto_vacuum', 'incremental_vacuum', 'secure_delete',
    'wal_checkpoint', 'optimize', 'shrink_memory',
}


def authorizer(action, arg1, arg2, db_name, trigger_name):
    """SQLite authorizer callback - reject any write operations."""
    # Allow read operations
    if action in ALLOWED_OPERATIONS:
        # Extra check for PRAGMA
        if action == sqlite3.SQLITE_PRAGMA and arg1:
            if arg1.lower() in DANGEROUS_PRAGMAS:
                return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    # Deny everything else (INSERT, UPDATE, DELETE, CREATE, DROP, etc.)
    return sqlite3.SQLITE_DENY


def execute_query(query: str, limit: int = 1000) -> dict:
    """Execute a read-only query and return results."""
    # Validate query starts with SELECT or WITH (for CTEs)
    query_upper = query.strip().upper()
    if not (query_upper.startswith('SELECT') or query_upper.startswith('WITH')):
        return {
            "error": "Only SELECT queries allowed. Query must start with SELECT or WITH.",
            "query": query[:100],
        }

    # Check for obviously dangerous keywords (defense in depth)
    dangerous_keywords = ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'ALTER', 'CREATE', 'ATTACH', 'DETACH']
    for keyword in dangerous_keywords:
        # Check for keyword as a standalone word (not part of column name)
        if f' {keyword} ' in f' {query_upper} ' or query_upper.startswith(f'{keyword} '):
            return {
                "error": f"Query contains disallowed keyword: {keyword}",
                "query": query[:100],
            }

    try:
        # Open in read-only mode
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row

        # Set query_only pragma (belt and suspenders)
        conn.execute("PRAGMA query_only = ON")

        # Install authorizer (another layer of protection)
        conn.set_authorizer(authorizer)

        # Add LIMIT if not present to prevent huge results
        if 'LIMIT' not in query_upper:
            query = f"{query.rstrip(';')} LIMIT {limit}"

        cursor = conn.execute(query)
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = cursor.fetchall()

        results = []
        for row in rows:
            results.append({col: row[col] for col in columns})

        conn.close()

        return {
            "success": True,
            "columns": columns,
            "row_count": len(results),
            "results": results,
        }

    except sqlite3.OperationalError as e:
        return {"error": f"SQL error: {e}", "query": query[:200]}
    except Exception as e:
        return {"error": f"Error: {e}", "query": query[:200]}


def get_schema() -> dict:
    """Get database schema information."""
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    # Get all tables
    tables = conn.execute(
        "SELECT name, sql FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()

    schema = {}
    for table in tables:
        # Get column info
        columns = conn.execute(f"PRAGMA table_info({table['name']})").fetchall()
        schema[table['name']] = {
            "columns": [{"name": c['name'], "type": c['type']} for c in columns],
            "sql": table['sql'],
        }

    conn.close()
    return schema


def main():
    parser = argparse.ArgumentParser(description="Execute read-only SQL queries")
    parser.add_argument("query", nargs="?", help="SQL query to execute")
    parser.add_argument("--stdin", action="store_true", help="Read query from stdin")
    parser.add_argument("--schema", action="store_true", help="Show database schema")
    parser.add_argument("--limit", type=int, default=100, help="Max rows (default 100)")

    args = parser.parse_args()

    if args.schema:
        schema = get_schema()
        print(json.dumps(schema, indent=2))
        return

    if args.stdin:
        query = sys.stdin.read().strip()
    elif args.query:
        query = args.query
    else:
        parser.print_help()
        sys.exit(1)

    result = execute_query(query, limit=args.limit)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
