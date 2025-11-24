"""Simulate incremental summarization to measure LLM API calls and cache hits."""

import random
import tempfile
import os
from datetime import datetime, timedelta
from summary_cache import SummaryCache, CachedMessage, Summary


class MockLLM:
    """Mock LLM that tracks API calls."""

    def __init__(self):
        self.call_count = 0
        self.calls_by_level = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}
        self.call_log = []

    async def generate_level_1_summary(self, messages, window_start, window_end):
        """Generate Level-1 summary from messages."""
        self.call_count += 1
        self.calls_by_level[1] += 1

        user_count = len(set(msg.author_name for msg in messages))
        msg_count = len(messages)

        summary = f"{user_count} users, {msg_count} msgs"
        self.call_log.append(f"L1: {format_time(window_start)}-{format_time(window_end)} ({msg_count} msgs) -> API call #{self.call_count}")

        return summary

    async def generate_meta_summary(self, summary1, summary2, target_level):
        """Generate meta-summary from two summaries."""
        self.call_count += 1
        self.calls_by_level[target_level] += 1

        total_msgs = summary1.message_count + summary2.message_count
        span_hours = (summary2.end_time - summary1.start_time) / 3600

        summary = f"Combined: {total_msgs} msgs over {span_hours:.1f}h"
        self.call_log.append(f"L{target_level}: {format_time(summary1.start_time)}-{format_time(summary2.end_time)} -> API call #{self.call_count}")

        return summary


class IncrementalWorker:
    """Simulates the real SummaryWorker with incremental cascading."""

    def __init__(self, cache, llm):
        self.cache = cache
        self.llm = llm
        self.chunk_size = 1800  # 30 minutes

    async def process_window_closed(self, channel_id, window_start, window_end):
        """
        Called when a 30-minute window closes.

        This is the entry point that would be triggered every 30 minutes.
        """
        print(f"\nWindow closed: {format_time(window_start)}-{format_time(window_end)}")

        # Check if Level-1 summary already exists (CACHE CHECK)
        existing = self.cache.get_summary(channel_id, window_start, window_end, 1)
        if existing:
            print(f"   * Cache hit: Level-1 summary already exists")
            return

        # Get messages for this window
        messages = self.cache.get_cached_messages(channel_id, window_start, window_end)

        if not messages:
            print(f"   * No messages in window, skipping")
            return

        # Generate Level-1 summary (LLM API CALL)
        print(f"   [API] Generating Level-1 summary for {len(messages)} messages...")
        summary_text = await self.llm.generate_level_1_summary(messages, window_start, window_end)

        level_1_summary = Summary(
            channel_id=channel_id,
            start_time=window_start,
            end_time=window_end,
            summary_level=1,
            summary_text=summary_text,
            message_count=len(messages),
        )

        self.cache.save_summary(level_1_summary)
        print(f"   * Saved Level-1 summary")

        # Now trigger cascade
        await self._cascade_upward(channel_id)

    async def _cascade_upward(self, channel_id):
        """
        Cascade summaries upward through levels.

        This runs after each Level-1 summary is created.
        """
        print(f"   Running cascade...")

        current_level = 1
        cascade_count = 0

        for iteration in range(10):  # Max 10 levels
            # Find pairs at current level
            summaries = self.cache.get_summaries_by_level(channel_id, current_level)
            summaries.sort(key=lambda s: s.start_time)

            if len(summaries) < 2:
                # Not enough summaries to pair
                current_level += 1
                continue

            # Pair them: [0,1], [2,3], [4,5], etc.
            pairs = []
            for i in range(0, len(summaries) - 1, 2):
                pairs.append((summaries[i], summaries[i + 1]))

            if not pairs:
                current_level += 1
                continue

            # Process each pair
            for s1, s2 in pairs:
                target_level = current_level + 1

                # Check if meta-summary already exists (CACHE CHECK)
                existing = self.cache.get_summary(
                    channel_id,
                    s1.start_time,
                    s2.end_time,
                    target_level
                )

                if existing:
                    print(f"      * Cache hit: Level-{target_level} summary exists")
                    continue

                # Generate meta-summary (LLM API CALL)
                print(f"      [API] Combining Level-{current_level} -> Level-{target_level}")
                summary_text = await self.llm.generate_meta_summary(s1, s2, target_level)

                meta_summary = Summary(
                    channel_id=channel_id,
                    start_time=s1.start_time,
                    end_time=s2.end_time,
                    summary_level=target_level,
                    summary_text=summary_text,
                    message_count=s1.message_count + s2.message_count,
                )

                self.cache.save_summary(meta_summary)
                cascade_count += 1

            current_level += 1

        if cascade_count > 0:
            print(f"   * Cascade complete: {cascade_count} new summaries created")
        else:
            print(f"   * No cascade needed")


def format_time(ts):
    """Format timestamp as readable time."""
    base = datetime(2025, 1, 7, 12, 0, 0)
    dt = base + timedelta(seconds=ts)
    hours = ts / 3600
    if hours >= 24:
        days = int(hours // 24)
        return f"D{days+1} {dt.strftime('%H:%M')}"
    return dt.strftime("%H:%M")


def generate_messages(channel_id, start_time, end_time, message_id_start):
    """Generate random messages for a time window."""
    num_messages = random.randint(3, 8)
    messages = []

    usernames = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank"]

    for i in range(num_messages):
        timestamp = random.randint(start_time, end_time)
        msg = CachedMessage(
            channel_id=channel_id,
            message_id=message_id_start + i,
            author_id=random.randint(1000, 1999),
            author_name=random.choice(usernames),
            content=f"Message {message_id_start + i}",
            timestamp=timestamp,
        )
        messages.append(msg)

    return messages


async def simulate_incremental(hours=24):
    """
    Simulate incremental summarization over time.

    Shows exactly when LLM API calls happen vs when cache is hit.
    """
    # Setup
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)

    try:
        cache = SummaryCache(db_path)
        llm = MockLLM()
        worker = IncrementalWorker(cache, llm)

        chunk_size = 1800  # 30 minutes
        num_chunks = hours * 2
        channel_id = 123

        print("="*80)
        print(f"  INCREMENTAL SUMMARIZATION SIMULATION")
        print(f"  Simulating {hours} hours ({num_chunks} 30-minute windows)")
        print("="*80)

        message_id = 1

        # Simulate time passing, one 30-minute window at a time
        for chunk_idx in range(num_chunks):
            window_start = chunk_idx * chunk_size
            window_end = window_start + chunk_size - 1

            print(f"\n{'='*80}")
            print(f"WINDOW {chunk_idx + 1}/{num_chunks}: {format_time(window_start)}-{format_time(window_end)}")
            print(f"{'='*80}")

            # Messages arrive during this window
            messages = generate_messages(channel_id, window_start, window_end, message_id)
            print(f"{len(messages)} messages arrived")

            # Cache them as they arrive
            for msg in messages:
                cache.cache_message(msg)

            message_id += len(messages)

            # Window closes - trigger summarization
            await worker.process_window_closed(channel_id, window_start, window_end)

            # Show stats after each window
            print(f"\nStats after window {chunk_idx + 1}:")
            print(f"   Total LLM API calls: {llm.call_count}")
            print(f"   Breakdown: L1={llm.calls_by_level[1]}, L2={llm.calls_by_level[2]}, "
                  f"L3={llm.calls_by_level[3]}, L4={llm.calls_by_level[4]}, L5={llm.calls_by_level[5]}")

        # Final summary
        print(f"\n\n{'='*80}")
        print(f"  FINAL SUMMARY")
        print(f"{'='*80}")
        print(f"\nSimulated: {hours} hours ({num_chunks} windows)")
        print(f"Total LLM API calls: {llm.call_count}")
        print(f"\nCalls by level:")
        for level in range(1, 9):
            if llm.calls_by_level[level] > 0:
                print(f"   Level-{level}: {llm.calls_by_level[level]} calls")

        print(f"\nCost estimation (assuming $0.003 per call):")
        cost = llm.call_count * 0.003
        print(f"   ${cost:.2f} for {hours} hours")
        print(f"   ${cost / hours:.4f} per hour")

        # Count summaries in database
        total_summaries = 0
        for level in range(1, 6):
            count = len(cache.get_summaries_by_level(channel_id, level))
            if count > 0:
                total_summaries += count
                print(f"\nLevel-{level} summaries in DB: {count}")

        print(f"\nTotal summaries stored: {total_summaries}")

        print(f"\n\nKey Insights:")
        print(f"   * API calls are ONLY made once per summary")
        print(f"   * Summaries are cached forever in the database")
        print(f"   * Re-requesting context = 0 additional API calls")
        print(f"   * Each new 30-min window triggers ~log2(N) cascade calls")

        if hours >= 24:
            print(f"\nFor 24-hour periods:")
            print(f"   * {llm.calls_by_level[1]} Level-1 summaries (1 per window)")
            print(f"   * Cascade creates ~{llm.call_count - llm.calls_by_level[1]} meta-summaries")
            print(f"   * Total: {llm.call_count} API calls for 24 hours of chat")

    finally:
        try:
            os.unlink(db_path)
        except:
            pass


if __name__ == "__main__":
    import asyncio
    import sys

    random.seed(42)

    hours = 24
    if len(sys.argv) > 1:
        hours = int(sys.argv[1])

    asyncio.run(simulate_incremental(hours))
