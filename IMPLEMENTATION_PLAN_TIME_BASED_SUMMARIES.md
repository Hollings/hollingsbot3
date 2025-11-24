# Implementation Plan: Time-Based Progressive Summarization

## Overview
Replace message-count-based history with time-windowed progressive summarization. Recent messages stay raw, older messages are progressively compressed into summaries at exponentially increasing time scales.

---

## Architecture

### Time Window Strategy

**Fixed 30-minute boundaries aligned to wall clock:**
- Windows: `1:00-1:29`, `1:30-1:59`, `2:00-2:29`, `2:30-2:59`, etc.
- Most recent window (current 30-min boundary) stays **raw** (unsummarized)
- Closed windows (older than current boundary) get summarized

**Summary levels (exponential grouping by 1/2):**
- **Level 0**: Raw messages (no summary)
- **Level 1**: Summary of 1 non-empty window (variable time span)
- **Level 2**: Summary of 2 Level-1 summaries combined (variable time span)
- **Level 3**: Summary of 2 Level-2 summaries combined (variable time span)
- **Level 4**: Summary of 2 Level-3 summaries combined (variable time span)
- ...continues exponentially

**Important:** Empty windows (no messages) are **skipped entirely**. We only create summaries for windows with actual messages. When combining summaries, we pair them sequentially even if there are time gaps between them.

**Overlap buffer (Option B):**
When generating summaries, include instruction:
```
"Summarize these messages. Note: the last 5-10 messages continue into
the next conversation window, so ensure smooth handoff and context continuity."
```

---

## Database Schema

### New Table: `conversation_summaries`

```sql
CREATE TABLE conversation_summaries (
    channel_id INTEGER NOT NULL,
    start_time INTEGER NOT NULL,  -- Unix timestamp (actual start of first message)
    end_time INTEGER NOT NULL,    -- Unix timestamp (actual end of last message)
    summary_level INTEGER NOT NULL,  -- 1=single window, 2=two combined, 3=four combined, etc.
    summary_text TEXT NOT NULL,
    message_count INTEGER,        -- Number of original messages summarized
    created_at INTEGER NOT NULL,
    PRIMARY KEY (channel_id, start_time, end_time, summary_level)
);

CREATE INDEX idx_summaries_channel_time
    ON conversation_summaries(channel_id, start_time, end_time);
CREATE INDEX idx_summaries_level
    ON conversation_summaries(channel_id, summary_level);
```

### Optional Table: `cached_messages`

Store raw Discord messages so we can:
1. Fetch only new messages using `channel.history(after=...)`
2. Rebuild summaries if needed
3. Avoid re-fetching from Discord API

```sql
CREATE TABLE cached_messages (
    channel_id INTEGER NOT NULL,
    message_id INTEGER NOT NULL,
    author_id INTEGER,
    author_name TEXT,
    content TEXT,
    timestamp INTEGER NOT NULL,
    has_images BOOLEAN DEFAULT 0,
    has_attachments BOOLEAN DEFAULT 0,
    PRIMARY KEY (channel_id, message_id)
);

CREATE INDEX idx_messages_channel_time
    ON cached_messages(channel_id, timestamp);
```

---

## Configuration (Environment Variables)

```bash
# Time-based summarization settings
LLM_SUMMARY_WINDOW_MINUTES=30          # Base window size (default: 30)
LLM_SUMMARY_PROVIDER=anthropic         # Provider for summarization (default: anthropic)
LLM_SUMMARY_MODEL=claude-haiku-4-5     # Model for summarization (default: claude-haiku-4-5)
LLM_CONTEXT_CHAR_LIMIT=10000           # Max chars in context (default: 10000)
LLM_SUMMARY_OVERLAP_MESSAGES=7         # Messages to mention in handoff (default: 7)

# Deprecated (remove after migration)
# LLM_HISTORY_LIMIT=50
# LLM_MAX_TURNS_SENT=8
```

---

## Core Components

### 1. Time Window Manager (`src/hollingsbot/conversation/time_windows.py`)

**Responsibilities:**
- Calculate current window boundary based on wall clock
- Get time range for a given window and level
- Determine which windows need summarization

**Key functions:**
```python
def get_current_window_boundary() -> tuple[int, int]:
    """Return (start_timestamp, end_timestamp) for current 30-min window."""

def get_window_at_time(timestamp: int) -> tuple[int, int]:
    """Return window boundaries for a given timestamp."""

def get_windows_needing_summary(
    channel_id: int,
    current_time: int
) -> list[tuple[int, int, int]]:
    """Return list of (start, end, level) tuples that need summarization."""

def calculate_summary_level(start: int, end: int) -> int:
    """Calculate what level a summary should be based on time span."""
```

### 2. Summary Cache Manager (`src/hollingsbot/conversation/summary_cache.py`)

**Responsibilities:**
- Read/write summaries to SQLite database
- Cache raw messages to avoid re-fetching from Discord
- Query summaries by time range and level

**Key functions:**
```python
def get_summary(
    channel_id: int,
    start: int,
    end: int,
    level: int
) -> str | None:
    """Retrieve cached summary, or None if not found."""

def save_summary(
    channel_id: int,
    start: int,
    end: int,
    level: int,
    summary_text: str,
    message_count: int
) -> None:
    """Save summary to database."""

def get_summaries_for_context(
    channel_id: int,
    before_timestamp: int,
    char_limit: int
) -> list[dict]:
    """Get all summaries before a timestamp, respecting char limit."""

def cache_message(message: discord.Message) -> None:
    """Cache a Discord message to local database."""

def get_cached_messages(
    channel_id: int,
    start: int,
    end: int
) -> list[dict]:
    """Retrieve cached messages in a time range."""

def get_latest_cached_message(channel_id: int) -> int | None:
    """Get message_id of most recent cached message for this channel."""
```

### 3. Summary Generator (`src/hollingsbot/conversation/summarizer.py`)

**Responsibilities:**
- Generate Level-1 summaries from raw messages
- Generate Level-N summaries from Level-(N-1) summaries
- Use Claude Haiku for all summarization
- Handle overlap buffer instructions

**Key functions:**
```python
async def summarize_messages(
    messages: list[dict],
    window_start: int,
    window_end: int,
    overlap_buffer: int = 7
) -> str:
    """Generate Level-1 summary from raw messages."""

async def summarize_summaries(
    summaries: list[str],
    window_start: int,
    window_end: int,
    target_level: int
) -> str:
    """Generate Level-N summary from two Level-(N-1) summaries."""

def build_summary_prompt(
    content: list[dict] | list[str],
    is_meta_summary: bool,
    overlap_buffer: int
) -> str:
    """Build the prompt for Claude Haiku to generate summary."""
```

**Prompt template for Level-1 (raw messages):**
```
You are summarizing a 30-minute window of a Discord conversation for context retention.

Summarize the key points, decisions, and important context from these messages.
Focus on information that would be useful for continuing the conversation later.

Note: The last {overlap_buffer} messages continue into the next window, so ensure
your summary provides smooth handoff and context continuity.

Messages:
{formatted_messages}

Provide a concise summary (2-4 paragraphs).
```

**Prompt template for Level-N (meta-summary):**
```
You are creating a higher-level summary by combining two time-window summaries.

Previous summaries cover:
- Window 1: {start1} to {end1}
- Window 2: {start2} to {end2}

{gap_note}

Combine these into a single coherent summary that captures the essential context
from both time periods.

Summary 1:
{summary1}

Summary 2:
{summary2}

Provide a concise combined summary (2-4 paragraphs).
```

Where `{gap_note}` is:
- If gap > 1 hour: "Note: There is a {X} hour gap between these summaries with no messages."
- Otherwise: empty string

### 4. Background Summarization Worker (`src/hollingsbot/conversation/summary_worker.py`)

**Responsibilities:**
- Run summarization cascade in background (non-blocking)
- Handle atomic commits (all-or-nothing for cascade)
- Prevent duplicate summarization work

**Key functions:**
```python
class SummaryWorker:
    def __init__(self):
        self._active_jobs: dict[int, asyncio.Task] = {}
        self._job_locks: dict[int, asyncio.Lock] = {}

    async def trigger_summarization(self, channel_id: int) -> None:
        """Trigger background summarization cascade for a channel."""

    async def _run_summarization_cascade(self, channel_id: int) -> None:
        """Execute full cascade: Level-1 → Level-2 → Level-3 → ... → commit all."""

    def _find_combinable_pairs(
        self,
        summaries: list[dict]
    ) -> list[tuple[dict, dict]]:
        """
        Find same-level summaries that can be combined.
        Pairs them sequentially by time, even if there are gaps between them.
        """
```

**Cascade algorithm:**
```python
1. Identify closed windows (older than current 30-min boundary) that have messages
2. Generate Level-1 summaries for windows without existing summaries (skip empty windows)
3. Find pairs of Level-1 summaries (sorted by time, paired sequentially) → generate Level-2
4. Find pairs of Level-2 summaries (sorted by time, paired sequentially) → generate Level-3
5. Continue until no more pairs can be formed
6. Commit all newly generated summaries atomically to database
```

**Important:** The pairing logic doesn't require adjacent time windows - it pairs summaries sequentially after sorting by start time. If there's a gap (empty windows), the combined summary will naturally span a longer time period.

### 5. Modified LLM Chat Cog (`src/hollingsbot/cogs/llm_chat.py`)

**Changes needed:**

#### Remove deprecated features:
- ❌ `channel_histories: dict[int, Deque[ConversationTurn]]` (replaced by time windows)
- ❌ `history_limit` (replaced by time windows)
- ❌ `max_turns_sent` (replaced by char limit)
- ❌ `_ensure_channel_warm()` (replaced by new loading logic)

#### Add new features:
- ✅ `context_char_limit` from env var
- ✅ `summary_worker: SummaryWorker` instance
- ✅ Integration with `SummaryCache` and `TimeWindowManager`

#### Modified methods:

**`on_message()`:**
```python
async def on_message(self, message: discord.Message) -> None:
    # 1. Cache the incoming message
    cache_message(message)

    # 2. Trigger background summarization (non-blocking)
    asyncio.create_task(self.summary_worker.trigger_summarization(channel_id))

    # 3. Build context from summaries + raw messages
    context = await self._build_context(channel_id, message.created_at)

    # 4. Generate response (existing logic)
    ...
```

**New method: `_build_context()`:**
```python
async def _build_context(
    self,
    channel_id: int,
    current_time: datetime
) -> list[dict[str, object]]:
    """
    Build conversation context with:
    1. System prompt
    2. Progressive summaries (oldest to newest)
    3. Raw messages from current window

    Respects context_char_limit.
    """

    current_window_start, current_window_end = get_current_window_boundary()

    # Get raw messages from current window
    raw_messages = await self._fetch_raw_window_messages(
        channel_id,
        current_window_start,
        current_time.timestamp()
    )

    # Get summaries before current window (respecting char limit)
    summaries = get_summaries_for_context(
        channel_id,
        before_timestamp=current_window_start,
        char_limit=self.context_char_limit
    )

    # Build conversation payload
    conversation = [
        {"role": "system", "text": self.system_prompt, "images": []}
    ]

    # Add summaries as user messages (with gap detection)
    prev_summary = None
    for summary in summaries:
        # Detect time gap between summaries
        if prev_summary:
            gap_seconds = summary['start_time'] - prev_summary['end_time']
            if gap_seconds > 3600:  # More than 1 hour gap
                gap_hours = gap_seconds / 3600
                conversation.append({
                    "role": "user",
                    "text": f"[{gap_hours:.1f} hours of silence]",
                    "images": []
                })

        summary_text = (
            f"[summarized {summary['message_count']} messages "
            f"from {format_time(summary['start_time'])} to {format_time(summary['end_time'])}]\n"
            f"{summary['summary_text']}"
        )
        conversation.append({
            "role": "user",
            "text": summary_text,
            "images": []
        })
        prev_summary = summary

    # Add raw messages
    for msg in raw_messages:
        conversation.append({
            "role": msg["role"],
            "text": msg["content"],
            "images": msg.get("images", [])
        })

    return conversation
```

**New method: `_fetch_raw_window_messages()`:**
```python
async def _fetch_raw_window_messages(
    self,
    channel_id: int,
    window_start: int,
    window_end: int
) -> list[dict]:
    """
    Fetch raw messages for current window.
    First check cache, then fetch from Discord if needed.
    """

    # Try cache first
    cached = get_cached_messages(channel_id, window_start, window_end)
    if cached:
        return cached

    # Fetch from Discord
    channel = self.bot.get_channel(channel_id)
    messages = []

    # Get latest cached message to avoid re-fetching old messages
    latest_cached_id = get_latest_cached_message(channel_id)

    if latest_cached_id:
        # Only fetch messages after our cache
        async for msg in channel.history(
            after=discord.Object(id=latest_cached_id),
            limit=None,
            oldest_first=True
        ):
            if window_start <= msg.created_at.timestamp() <= window_end:
                cache_message(msg)
                messages.append(await self._message_to_dict(msg))
    else:
        # First time: fetch all messages in window
        async for msg in channel.history(
            after=datetime.fromtimestamp(window_start),
            before=datetime.fromtimestamp(window_end),
            limit=None,
            oldest_first=True
        ):
            cache_message(msg)
            messages.append(await self._message_to_dict(msg))

    return messages
```

---

## Migration Strategy

### Phase 1: Database Setup
1. Create migration script: `migrations/add_time_based_summaries.py`
2. Add new tables: `conversation_summaries`, `cached_messages`
3. Run migration on bot startup (check for table existence)

### Phase 2: Implement Core Components (in order)
1. `time_windows.py` - No dependencies
2. `summary_cache.py` - Depends on database schema
3. `summarizer.py` - Depends on Celery tasks (or inline async)
4. `summary_worker.py` - Depends on summarizer + cache

### Phase 3: Integration
1. Modify `llm_chat.py` to use new system
2. Add env var support for new configuration
3. Keep old `channel_histories` for 1-2 releases (feature flag?)

### Phase 4: Testing
1. Test window boundary calculations
2. Test summarization cascade with mock messages
3. Test character limit enforcement
4. Test background worker doesn't block responses
5. Test incremental Discord message fetching

### Phase 5: Deployment
1. Update `.env` with new config vars
2. Update `CLAUDE.md` documentation
3. Deploy with migration
4. Monitor summarization performance (latency, token usage)

---

## Edge Cases & Considerations

### 1. Empty Windows ✅ RESOLVED
**Problem:** What if a 30-min window has no messages?
**Solution:** Skip empty windows entirely - don't create summaries for them. When combining summaries, pair them sequentially by time even if there are gaps. The summary prompt will note any time gaps > 1 hour between windows being combined.

### 2. Very Active Channels
**Problem:** Current window might have 100+ messages
**Solution:**
- Enforce character limit on raw messages too (truncate oldest in window)
- OR dynamically adjust to 15-min windows if message rate is high (future enhancement)

### 3. Bot Restart During Summarization
**Problem:** Cascade in progress when bot restarts
**Solution:**
- Use transaction for atomic commit
- On startup, check for incomplete cascades and re-run

### 4. Summary Generation Failure
**Problem:** Claude Haiku API fails during cascade
**Solution:**
- Retry with exponential backoff
- Log failure and continue (gap in summaries is acceptable)
- User can trigger manual re-summarization via command

### 5. Character Limit Exceeded
**Problem:** Even with summaries, we exceed 10k char limit
**Solution:**
- Drop oldest summaries first (FIFO eviction)
- Always keep current window raw messages
- Log warning if we're dropping Level-4+ summaries (very old context)

### 6. Clock Boundaries and User Timezone
**Problem:** Windows are UTC-based but users think in local time
**Solution:**
- Always use UTC for window calculations (consistent across servers)
- Display times in human-readable format in summary headers
- Document this behavior

### 7. Image and File Attachments
**Problem:** How to handle images in summarized messages?
**Solution:**
- **Phase 1 (this implementation):** Ignore images/files in summaries (text only)
- **Future enhancement:** Track "N images shared" in summary metadata

---

## Testing Plan

### Unit Tests

**`test_time_windows.py`:**
```python
def test_get_current_window_boundary()
def test_get_window_at_time()
def test_calculate_summary_level()
def test_windows_needing_summary()
```

**`test_summary_cache.py`:**
```python
def test_save_and_retrieve_summary()
def test_get_summaries_for_context_respects_char_limit()
def test_cache_message()
def test_get_latest_cached_message()
```

**`test_summarizer.py`:**
```python
async def test_summarize_messages()
async def test_summarize_summaries()
def test_build_summary_prompt()
```

**`test_summary_worker.py`:**
```python
async def test_cascade_generates_all_levels()
async def test_cascade_commits_atomically()
async def test_no_duplicate_work()
```

### Integration Tests

**`test_llm_chat_integration.py`:**
```python
async def test_context_building_with_summaries()
async def test_incremental_message_fetching()
async def test_background_summarization_doesnt_block()
```

### Manual Testing Checklist
- [ ] Send 100 messages over 2 hours, verify summaries generated
- [ ] Verify overlap buffer creates smooth transitions
- [ ] Restart bot mid-conversation, verify context preserved
- [ ] Test with very active channel (200+ msgs/hour)
- [ ] Test with sparse channel (1 msg/day)
- [ ] Verify character limit enforcement
- [ ] Check token usage for Claude Haiku (should be <10k tokens/cascade)

---

## Performance Considerations

### Token Usage Estimates

**Level-1 summary (30 min window, ~30 messages):**
- Input: ~3,000 tokens (messages + prompt)
- Output: ~300 tokens (summary)
- Cost: ~$0.0009 per window (Claude Haiku pricing)

**Level-2 summary (1 hour, 2 Level-1 summaries):**
- Input: ~800 tokens (2 summaries + prompt)
- Output: ~300 tokens
- Cost: ~$0.0003

**Full cascade (8 hours of conversation, 16 windows):**
- 16 Level-1 summaries: ~$0.0144
- 8 Level-2 summaries: ~$0.0024
- 4 Level-3 summaries: ~$0.0012
- 2 Level-4 summaries: ~$0.0006
- **Total: ~$0.0186** (less than 2 cents for 8 hours)

### Latency

**Background summarization:**
- Level-1: ~2-3s per window (Claude Haiku)
- Full cascade: ~30-60s for 8 hours of conversation
- **User impact:** None (runs in background)

**Context building:**
- Database query: <10ms
- Total: ~10-50ms added to response time

---

## Future Enhancements

### 1. Smart Window Sizing
Dynamically adjust window size based on message rate:
- High traffic: 15-min windows
- Low traffic: 60-min windows

### 2. Topic-Based Summarization
Include topic extraction in summaries:
```
[summarized 30 messages, topics: authentication, database migration]
User discussed implementing JWT authentication and migrating to PostgreSQL...
```

### 3. User-Triggered Commands
```
!summarize <time_range>  - Manually trigger summarization
!context                 - Show current context breakdown (X summaries, Y raw msgs)
!rebuild-summaries       - Regenerate all summaries for channel
```

### 4. Cross-Channel Context
For servers with multiple related channels, optionally include summaries from related channels.

### 5. Export Conversation
```
!export-conversation <start_date> <end_date>
```
Generate a markdown file with:
- All summaries in range
- All raw messages in range
- Formatted for readability

---

## Open Questions

1. **Should we run summarization in Celery or inline async?**
   - Celery: Better isolation, can run on separate worker
   - Inline async: Simpler, no queue overhead
   - **Recommendation:** Start with inline async, move to Celery if needed

2. **Should we summarize on every new message or on a timer?**
   - Every message: More responsive, but might cascade too often
   - Timer (e.g., every 5 min): Less overhead, but summaries lag
   - **Recommendation:** Trigger on message, but debounce (only run if >5min since last cascade)

3. **Should we cache Discord messages in database or just fetch on-demand?**
   - Cache: Faster, less Discord API load, enables incremental fetch
   - On-demand: Simpler, no storage overhead
   - **Recommendation:** Cache (we need it for incremental fetching)

4. **How to handle very old channels (years of history)?**
   - Summarize everything on first load: Expensive, slow
   - Only summarize recent history: Faster, but loses old context
   - **Recommendation:** Start with last 7 days, add "!load-history" command for older

---

## Success Metrics

- ✅ Bot can maintain context across 8+ hours of conversation
- ✅ Response latency remains <500ms (excluding LLM generation)
- ✅ Token usage for summarization <$0.05 per channel per day
- ✅ No user-visible interruptions during summarization
- ✅ Context character limit stays under 10k consistently
- ✅ Summaries provide smooth handoff (tested via user feedback)

---

## File Structure

```
src/hollingsbot/
├── conversation/
│   ├── __init__.py
│   ├── time_windows.py          # Window boundary calculations
│   ├── summary_cache.py         # Database interactions
│   ├── summarizer.py            # Summary generation logic
│   └── summary_worker.py        # Background cascade worker
├── cogs/
│   └── llm_chat.py              # Modified to use time-based system
├── migrations/
│   └── add_time_based_summaries.py
└── tasks.py                     # (Optional) Celery tasks for summarization

tests/
├── conversation/
│   ├── test_time_windows.py
│   ├── test_summary_cache.py
│   ├── test_summarizer.py
│   └── test_summary_worker.py
└── integration/
    └── test_llm_chat_integration.py
```

---

**END OF IMPLEMENTATION PLAN**
