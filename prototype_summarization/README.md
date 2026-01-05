# Time-Based Progressive Summarization Prototype

This is a **test-driven prototype** implementation of the time-based progressive summarization system for Discord bot conversation history. It demonstrates the core logic without integration to Discord, AI APIs, or the actual bot.

## Test Results

```
81 tests passing, 2 skipped
Test coverage across all core components
```

## Architecture Overview

The system uses **fixed 30-minute time windows** aligned to wall clock (`:00-:29`, `:30-:59`) and progressively combines summaries at exponentially increasing levels.

### Key Features Implemented

✅ **Time Window Management** - Wall-clock aligned 30-min boundaries
✅ **Empty Window Handling** - Skips windows with no messages
✅ **Gap Detection** - Identifies and notes time gaps when combining summaries
✅ **Character Limits** - Drops oldest summaries to stay within limits
✅ **Cascade Logic** - Automatically combines summaries into higher levels
✅ **Atomic Operations** - Batch saves with rollback on error
✅ **Duplicate Prevention** - Checks for existing summaries before creating

## Components

### 1. `time_windows.py` (15 tests)

Handles time window boundary calculations.

**Key Functions:**
- `get_window_at_time(timestamp)` - Returns (start, end) for a given timestamp
- `get_current_window_boundary()` - Returns current 30-min window
- `calculate_summary_level(start, end)` - Determines level from time span

**Example:**
```python
from time_windows import get_window_at_time

# 1:15 PM falls in 1:00-1:29 window
start, end = get_window_at_time(timestamp_1_15pm)
# Returns: (timestamp_1_00pm, timestamp_1_29_59pm)
```

### 2. `summary_cache.py` (18 tests)

SQLite database operations for summaries and cached messages.

**Database Schema:**
```sql
-- Summaries with flexible time spans
conversation_summaries (
    channel_id, start_time, end_time, summary_level,
    summary_text, message_count, created_at
)

-- Cached Discord messages
cached_messages (
    channel_id, message_id, author_id, author_name,
    content, timestamp, has_images, has_attachments
)
```

**Key Methods:**
- `save_summary(summary)` - Save single summary
- `save_summaries_batch(summaries)` - Atomic batch save
- `get_summaries_for_context(channel_id, char_limit=10000)` - Get summaries for LLM
- `cache_message(message)` - Cache raw Discord message
- `get_cached_messages(channel_id, start, end)` - Retrieve messages in range

**Example:**
```python
from summary_cache import SummaryCache, Summary

cache = SummaryCache("test.db")

# Save a summary
summary = Summary(
    channel_id=123,
    start_time=1000,
    end_time=2799,
    summary_level=1,
    summary_text="User discussed authentication...",
    message_count=25
)
cache.save_summary(summary)

# Retrieve for context (respects char limit)
summaries = cache.get_summaries_for_context(123, char_limit=5000)
```

### 3. `summarizer.py` (19 tests)

Generates summaries using an LLM interface.

**Key Components:**
- `build_level_1_prompt(messages)` - Prompt for raw messages
- `build_meta_summary_prompt(summary1, summary2)` - Prompt for combining
- `Summarizer.summarize_messages()` - Generate Level-1 summary
- `Summarizer.summarize_summaries()` - Generate meta-summary

**Prompt Features:**
- Overlap buffer instructions for smooth transitions
- Gap detection (notes time gaps > 1 hour)
- Timestamp formatting for context

**Example:**
```python
from summarizer import Summarizer, build_level_1_prompt
from summary_cache import CachedMessage

messages = [
    CachedMessage(123, 1, 456, "Alice", "Hello!", 1000),
    CachedMessage(123, 2, 789, "Bob", "Hi Alice", 1100),
]

# Generate prompt
prompt = build_level_1_prompt(messages, overlap_buffer=7)

# Use with LLM
summarizer = Summarizer(llm=your_llm)
summary_text = await summarizer.summarize_messages(messages, 1000, 1800)
```

### 4. `summary_worker.py` (29 tests)

Background cascade worker that combines summaries.

**Key Features:**
- Finds combinable pairs at each level
- Generates Level-1 summaries from messages
- Combines summaries into higher levels
- Prevents duplicate summarization
- Non-blocking background operation

**Cascade Algorithm:**
```
1. Identify closed windows with messages but no summaries
2. Generate Level-1 summaries for those windows
3. Find pairs of Level-1 summaries → create Level-2
4. Find pairs of Level-2 summaries → create Level-3
5. Continue until no more pairs
6. Commit all new summaries atomically
```

**Example:**
```python
from summary_worker import SummaryWorker

worker = SummaryWorker(cache, summarizer)

# Trigger background summarization
await worker.trigger_summarization(channel_id=123)

# Result: All combinable summaries are merged into higher levels
```

## How It Works: Example Flow

### Scenario: 2 hours of conversation

```
Time:        Messages:                    Summaries:
1:00-1:29    30 messages                  → Level-1 (30 msgs)
1:30-1:59    25 messages                  → Level-1 (25 msgs)
2:00-2:29    [empty]                      (skipped)
2:30-2:59    40 messages                  → Level-1 (40 msgs)

Cascade runs:
1. Two Level-1s (1:00-1:29, 1:30-1:59) → Level-2 (1:00-1:59, 55 msgs)
2. Level-2 + Level-1 (2:30-2:59) wait for pair

Result in LLM context:
[Summary Level-2: 1:00-1:59] "User discussed X and Y..."
[Note: 30-min gap with no messages]
[Summary Level-1: 2:30-2:59] "User implemented Z..."
<raw messages from current window>
```

## Empty Window Handling

**Problem:** Many 30-min windows have 0 messages.

**Solution:** Skip empty windows entirely. When combining summaries:
- Pair summaries sequentially by time (not requiring adjacency)
- Note gaps > 1 hour in meta-summary prompts
- Combined summary spans the full time range

**Example:**
```python
# Window 1:00-1:29 has messages → Level-1 summary
# Windows 1:30-3:29 are empty → skipped
# Window 3:30-3:59 has messages → Level-1 summary

# When combining:
# Summary 1 (1:00-1:29) + Summary 2 (3:30-3:59)
# Result: Level-2 summary spanning 1:00-3:59
# Prompt includes: "Note: 2 hour gap between summaries"
```

## Testing

All components use TDD (Test-Driven Development):

```bash
# Run all tests
pytest prototype_summarization/tests/ -v

# Run specific component
pytest prototype_summarization/tests/test_time_windows.py -v
pytest prototype_summarization/tests/test_summary_cache.py -v
pytest prototype_summarization/tests/test_summarizer.py -v
pytest prototype_summarization/tests/test_summary_worker.py -v
```

### Test Coverage

- **time_windows.py**: Window calculations, level detection, edge cases
- **summary_cache.py**: CRUD operations, filtering, char limits, atomicity
- **summarizer.py**: Prompt generation, LLM integration (mocked), gap detection
- **summary_worker.py**: Pairing logic, cascade execution, deduplication

### Dummy LLM

Tests use a `DummyLLM` class that:
- Tracks call count
- Records prompts for assertion
- Returns predictable responses
- No external API calls

## Next Steps for Integration

### 1. Connect to Real LLM
Replace `DummyLLM` with actual Claude Haiku integration:
```python
from anthropic import Anthropic

class ClaudeLLM:
    def __init__(self):
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    async def generate(self, prompt: str) -> str:
        response = self.client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
```

### 2. Integrate with Discord Bot

Add to `llm_chat.py`:
```python
from hollingsbot.conversation import SummaryWorker, SummaryCache, Summarizer

class LLMChatNewCog:
    def __init__(self, bot):
        self.cache = SummaryCache(db_path)
        self.summarizer = Summarizer(ClaudeLLM())
        self.worker = SummaryWorker(self.cache, self.summarizer)

    async def on_message(self, message):
        # Cache the message
        self.cache.cache_message(message)

        # Trigger background summarization
        asyncio.create_task(self.worker.trigger_summarization(message.channel.id))

        # Build context with summaries
        context = await self._build_context(message.channel.id)

        # Generate response...
```

### 3. Environment Variables

Add to `.env`:
```bash
LLM_SUMMARY_WINDOW_MINUTES=30
LLM_SUMMARY_PROVIDER=anthropic
LLM_SUMMARY_MODEL=claude-haiku-4-5
LLM_CONTEXT_CHAR_LIMIT=10000
LLM_SUMMARY_OVERLAP_MESSAGES=7
```

### 4. Database Migration

Create migration script to add tables to existing `prompts.db`:
```python
def migrate_add_summarization_tables(db_path):
    cache = SummaryCache(db_path)
    # Tables are created automatically in __init__
```

## Performance Estimates

Based on Claude Haiku pricing:

**Level-1 Summary (30 min window, ~30 messages):**
- Input: ~3,000 tokens
- Output: ~300 tokens
- Cost: ~$0.0009

**Full 8-hour cascade:**
- 16 Level-1 summaries
- 8 Level-2 summaries
- 4 Level-3 summaries
- 2 Level-4 summaries
- **Total: ~$0.02 per 8 hours of conversation**

**Latency:**
- Background cascade: 30-60s for 8 hours
- User impact: None (runs async)
- Context building: <50ms added latency

## Design Decisions

### Why Time-Based Instead of Message-Count?
- **Deterministic caching**: "1pm-1:30pm" never changes
- **No re-summarization waste**: Once closed, window is final
- **Natural conversation flow**: Humans think in time, not message counts

### Why Fixed 30-Min Windows?
- **Predictable**: Easy to reason about and debug
- **Cache-friendly**: Window boundaries are deterministic
- **Flexible**: Handles both high and low traffic channels

### Why Skip Empty Windows?
- **Efficiency**: No storage/tokens for empty periods
- **Natural**: Silence is meaningful context
- **Flexible levels**: Level ≠ strict time span

## Limitations & Future Work

### Current Limitations
1. Level-1 generation not automatically triggered (manual pre-population needed)
2. No cross-channel context
3. Images/files ignored in summaries
4. Fixed 30-min windows (no dynamic sizing)

### Future Enhancements
- **Smart window sizing**: Adjust based on message rate
- **Topic extraction**: Include topics in summaries
- **Image descriptions**: Describe images in summary text
- **User commands**: `!summarize`, `!context`, `!export-conversation`
- **Cross-channel**: Optional related channel summaries

## License

Part of hollingsbot3 project.
