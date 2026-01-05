# Summarization Integration Plan

## Overview

Integrate the prototype_summarization system into the main bot to:
1. Reduce token usage via summarization (~75% input cost savings)
2. Enable prompt caching on stable summary context (~additional 90% cache hits)

## Current Architecture

```
Message → ChatCoordinator (in-memory deque) → WendyBot._build_conversation_payload()
                                                    ↓
                                            [system prompt]
                                            [last N turns from deque]  ← ROLLING, can't cache
                                            [current turn]
                                                    ↓
                                              Anthropic API
```

## Target Architecture

```
Message → ChatCoordinator
              ├─→ In-memory deque (unchanged, for temp bots/quick access)
              └─→ SQLite cache_message() (NEW)
                      ↓
              Background SummaryWorker (NEW)
                      ↓
              conversation_summaries table

WendyBot._build_conversation_payload():
    [system prompt]                    ← CACHED (already done)
    [summaries - older windows]        ← CACHED (stable, immutable)
    [raw messages - current window]    ← NOT CACHED (rolling)
    [current turn]
```

## Implementation Steps

### Phase 1: Move Prototype to Main Codebase

1. Copy `prototype_summarization/` modules to `src/hollingsbot/summarization/`:
   - `time_windows.py`
   - `summary_cache.py`
   - `summarizer.py`
   - `summary_worker.py`

2. Update imports to use new package location

### Phase 2: Create Haiku Summarizer

Create `src/hollingsbot/summarization/haiku_llm.py`:
- Implement `LLMProtocol` using Anthropic Haiku
- Use shared `AsyncAnthropic` client from text_generators
- Add retry logic and error handling

### Phase 3: Integrate with ChatCoordinator

Modify `chat_coordinator.py`:

1. **Initialize summarization components**:
   ```python
   def __init__(self):
       # ... existing code ...
       self.summary_cache = SummaryCache(db_path)
       self.summarizer = Summarizer(HaikuLLM())
       self.summary_worker = SummaryWorker(self.summary_cache, self.summarizer)
   ```

2. **Cache messages on arrival**:
   ```python
   def _add_to_history(self, channel_id, turn):
       # ... existing deque append ...

       # Also cache for summarization
       self.summary_cache.cache_message(CachedMessage(
           channel_id=channel_id,
           message_id=turn.message_id,
           author_id=turn.author_id,
           author_name=turn.author_name,
           content=turn.content,
           timestamp=int(time.time()),
           has_images=bool(turn.images),
       ))
   ```

3. **Trigger background summarization**:
   ```python
   async def _process_message(self, message):
       # ... existing code ...

       # Trigger summarization in background (non-blocking)
       asyncio.create_task(
           self.summary_worker.trigger_summarization(message.channel.id)
       )
   ```

4. **Provide context method for bots**:
   ```python
   def get_summarized_context(self, channel_id: int) -> dict:
       """Get summaries + recent messages for LLM context."""
       current_time = int(time.time())
       window_start, _ = get_current_window_boundary()

       # Get summaries (older than current window)
       summaries = self.summary_cache.get_summaries_for_context(
           channel_id,
           before_timestamp=window_start,
           char_limit=8000,  # ~2000 tokens
       )

       # Get recent messages (current window + buffer)
       recent = self.summary_cache.get_cached_messages(
           channel_id,
           start_time=window_start - 1800,  # 30 min buffer
           end_time=current_time,
       )

       return {
           "summaries": summaries,
           "recent_messages": recent,
       }
   ```

### Phase 4: Modify WendyBot Context Building

Modify `wendy_bot.py`:

1. **Update `_build_conversation_payload()`**:
   ```python
   def _build_conversation_payload(
       self,
       history: list[ConversationTurn],  # Still used as fallback
       current_turn: ModelTurn,
       system_prompt: str,
   ) -> list[dict[str, Any]]:

       # Try to get summarized context
       summarized = self.coordinator.get_summarized_context(channel_id)

       if summarized["summaries"]:
           # Build with summaries + recent messages
           return self._build_with_summaries(
               summarized, current_turn, system_prompt
           )
       else:
           # Fallback to rolling history (no summaries yet)
           return self._build_with_history(
               history, current_turn, system_prompt
           )
   ```

2. **New method `_build_with_summaries()`**:
   ```python
   def _build_with_summaries(
       self,
       summarized: dict,
       current_turn: ModelTurn,
       system_prompt: str,
   ) -> list[dict[str, Any]]:
       conversation = [
           {"role": "system", "text": system_prompt, "images": []},
       ]

       # Add summaries as context (will be marked cacheable)
       summary_text = self._format_summaries(summarized["summaries"])
       if summary_text:
           conversation.append({
               "role": "user",
               "text": f"[Previous conversation context]\n{summary_text}",
               "images": [],
               "_cacheable": True,  # Flag for anthropic.py
           })
           conversation.append({
               "role": "assistant",
               "text": "I understand the context from our previous conversation.",
               "images": [],
               "_cacheable": True,
           })

       # Add recent raw messages (not cacheable)
       for msg in summarized["recent_messages"]:
           role = "assistant" if msg.author_id == self.bot.user.id else "user"
           conversation.append({
               "role": role,
               "text": f"{msg.author_name}: {msg.content}",
               "images": [],
           })

       # Add current turn
       conversation.append({
           "role": current_turn.role,
           "text": current_turn.text,
           "images": [img.to_payload() for img in current_turn.images],
       })

       return conversation
   ```

### Phase 5: Add Cache Control to Anthropic

Modify `anthropic.py` to mark cacheable messages:

```python
async def _call_sdk(msgs, system, temp):
    # Process messages with cache_control
    processed_msgs = []
    for m in msgs:
        msg_dict = {"role": m["role"], "content": m["content"]}

        # Add cache_control for cacheable messages
        if m.get("_cacheable"):
            # For multi-part content, add cache_control to last block
            if isinstance(msg_dict["content"], list):
                msg_dict["content"][-1]["cache_control"] = {"type": "ephemeral"}
            else:
                msg_dict["content"] = [{
                    "type": "text",
                    "text": msg_dict["content"],
                    "cache_control": {"type": "ephemeral"},
                }]

        processed_msgs.append(msg_dict)

    # ... rest of API call
```

### Phase 6: Environment Variables

Add to `.env`:
```bash
# Summarization
LLM_SUMMARY_ENABLED=1
LLM_SUMMARY_MODEL=claude-haiku-4-5
LLM_SUMMARY_CHAR_LIMIT=8000
LLM_SUMMARY_WINDOW_MINUTES=30
```

### Phase 7: Database Migration

The summary tables will be auto-created in the existing `prompts.db` or a new `summaries.db`.

## Rollout Strategy

1. **Feature flag**: `LLM_SUMMARY_ENABLED=0` by default
2. **Test in single channel first**: Add channel ID to test list
3. **Monitor**:
   - Haiku costs for summarization
   - Cache hit rates (Anthropic dashboard)
   - Response quality (manual review)
4. **Gradual rollout**: Enable for more channels

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Summary quality issues | Keep fallback to raw history; manual review |
| Haiku costs spike | Set rate limits; monitor costs |
| Database growth | Add cleanup job for old summaries |
| Bot restart loses context | Summaries persist in SQLite |
| Cache misses (5-min TTL) | Accept ~80% hit rate is fine |

## Expected Cost Savings

**Current (100 messages/day):**
- Input: 1.1M tokens × $3/MTok = $3.34

**With summarization + caching:**
- Summaries: ~2000 tokens (vs 10,000)
- Current window: ~1500 tokens
- Cache hits on summaries: ~80%
- Effective input: ~500K tokens × ~$1.50/MTok = $0.75

**Summarization overhead:**
- Haiku: ~$0.02/8 hours

**Net savings: ~$2.50/day (75%)**

## Files to Modify

1. `src/hollingsbot/summarization/` (new directory)
   - `__init__.py`
   - `time_windows.py` (from prototype)
   - `summary_cache.py` (from prototype)
   - `summarizer.py` (from prototype)
   - `summary_worker.py` (from prototype)
   - `haiku_llm.py` (new)

2. `src/hollingsbot/cogs/chat_coordinator.py`
   - Add summarization initialization
   - Add message caching
   - Add background summarization trigger
   - Add `get_summarized_context()` method

3. `src/hollingsbot/cogs/chat_bots/wendy_bot.py`
   - Modify `_build_conversation_payload()`
   - Add `_build_with_summaries()`
   - Add `_format_summaries()`

4. `src/hollingsbot/text_generators/anthropic.py`
   - Add cache_control handling for `_cacheable` messages

5. `.env`
   - Add summarization config variables
