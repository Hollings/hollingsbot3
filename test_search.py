#!/usr/bin/env python3
"""Test the search_messages tool."""

import sys

sys.path.insert(0, "src")

from hollingsbot.tools import AVAILABLE_TOOLS
from hollingsbot.tools.parser import set_current_context

# Set the context (simulating being in the text-ai-vestibule channel)
set_current_context({"channel_id": 1050900592031178752})

# Get the search tool
search_tool = AVAILABLE_TOOLS["search_messages"]

# Test 1: Search for "wendy"
print("=" * 60)
print("TEST 1: Search for 'wendy' messages (5 most recent)")
print("=" * 60)
result = search_tool.function(query="wendy")
print(result)

print("\n" + "=" * 60)
print("TEST 2: Get recent messages (no query)")
print("=" * 60)
result = search_tool.function(query="")
print(result)

print("\n" + "=" * 60)
print("TEST 3: Search for messages about 'toilet'")
print("=" * 60)
result = search_tool.function(query="toilet")
print(result)

print("\n✅ All tests completed successfully!")
