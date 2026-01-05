"""Terminal simulator for hierarchical summarization system."""

import random
import tempfile
import os
from datetime import datetime, timedelta
from summary_cache import SummaryCache, CachedMessage, Summary


# Mock messages to simulate realistic Discord conversations
MESSAGE_TEMPLATES = [
    "Hey everyone! {topic}",
    "Anyone want to {activity}?",
    "I just {achievement}",
    "Does anyone know {question}?",
    "That's {reaction}!",
    "{topic} is really {adjective} today",
    "I'm working on {project}",
    "Thanks for {favor}!",
    "Can someone help me with {problem}?",
    "Just finished {task}, feeling great",
]

TOPICS = ["the weather", "the game", "this project", "the new update", "today's news"]
ACTIVITIES = ["play games", "grab lunch", "chat about AI", "watch a movie", "start a project"]
ACHIEVEMENTS = ["hit level 50", "finished my essay", "learned Rust", "fixed the bug", "ran 5k"]
QUESTIONS = ["how to center a div", "when the meeting is", "if we have tests today", "where the docs are"]
REACTIONS = ["awesome", "wild", "hilarious", "concerning", "amazing", "unexpected"]
ADJECTIVES = ["interesting", "busy", "quiet", "hectic", "fun", "productive"]
PROJECTS = ["a web scraper", "a Discord bot", "my portfolio", "the backend API", "the UI redesign"]
FAVORS = ["the help", "sharing that link", "the advice", "explaining that", "your time"]
PROBLEMS = ["my code", "this error", "understanding async", "Git conflicts", "Docker setup"]
TASKS = ["my homework", "the refactor", "the documentation", "debugging", "code review"]

USERNAMES = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry"]


def generate_message(timestamp: int, message_id: int) -> CachedMessage:
    """Generate a random Discord-like message."""
    template = random.choice(MESSAGE_TEMPLATES)

    # Fill in template with random values
    content = template.format(
        topic=random.choice(TOPICS),
        activity=random.choice(ACTIVITIES),
        achievement=random.choice(ACHIEVEMENTS),
        question=random.choice(QUESTIONS),
        reaction=random.choice(REACTIONS),
        adjective=random.choice(ADJECTIVES),
        project=random.choice(PROJECTS),
        favor=random.choice(FAVORS),
        problem=random.choice(PROBLEMS),
        task=random.choice(TASKS),
    )

    return CachedMessage(
        channel_id=123,
        message_id=message_id,
        author_id=random.randint(1000, 1999),
        author_name=random.choice(USERNAMES),
        content=content,
        timestamp=timestamp,
    )


def create_realistic_summary(messages: list[CachedMessage]) -> str:
    """Create a realistic-looking summary from messages."""
    if not messages:
        return "No activity during this period."

    # Count unique users
    unique_users = set(msg.author_name for msg in messages)

    # Extract key topics/words
    all_content = " ".join(msg.content for msg in messages)

    # Create a summary
    summary_parts = []
    summary_parts.append(f"{len(unique_users)} user(s) active ({', '.join(sorted(unique_users)[:3])}{'...' if len(unique_users) > 3 else ''})")
    summary_parts.append(f"{len(messages)} messages")

    # Add some content hints
    if "game" in all_content.lower() or "play" in all_content.lower():
        summary_parts.append("discussing games")
    if "project" in all_content.lower() or "working" in all_content.lower():
        summary_parts.append("project work")
    if "help" in all_content.lower() or "?" in all_content:
        summary_parts.append("Q&A")

    return " | ".join(summary_parts)


def create_meta_summary(summary1: Summary, summary2: Summary) -> str:
    """Create a meta-summary from two summaries."""
    # Extract message counts
    total_messages = summary1.message_count + summary2.message_count

    # Combine summaries
    return f"Combined period: {total_messages} total messages spanning {(summary2.end_time - summary1.start_time) / 3600:.1f} hours"


def format_timestamp(ts: int) -> str:
    """Format a Unix timestamp as a readable time."""
    # Use a fixed base date for consistency
    base = datetime(2025, 1, 7, 12, 0, 0)
    dt = base + timedelta(seconds=ts)

    # For times spanning multiple days, include day info
    hours_elapsed = ts / 3600
    if hours_elapsed >= 24:
        days = int(hours_elapsed // 24)
        return f"Day{days+1} {dt.strftime('%H:%M')}"
    return dt.strftime("%H:%M")


def format_time_range(start: int, end: int) -> str:
    """Format a time range."""
    return f"{format_timestamp(start)}-{format_timestamp(end)}"


def print_separator(char="=", width=80):
    """Print a separator line."""
    print(char * width)


def print_section_header(title: str):
    """Print a section header."""
    print_separator()
    print(f"  {title}")
    print_separator()


def simulate_12_hours(hours=12):
    """Simulate Discord chat with hierarchical summaries."""
    # Create temp database
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)

    try:
        cache = SummaryCache(db_path)

        # Simulation parameters
        chunk_size = 1800  # 30 minutes
        num_chunks = hours * 2  # 2 chunks per hour
        start_time = 0
        messages_per_chunk = random.randint(3, 8)

        print_section_header(f"SIMULATION: {hours} Hours of Discord Chat")
        print(f"\nConfig: {chunk_size}s chunks ({chunk_size/60}min), {num_chunks} chunks total")
        print(f"Time range: {format_timestamp(start_time)} - {format_timestamp(start_time + chunk_size * num_chunks)}")
        print()

        # Phase 1: Generate messages
        print_section_header("PHASE 1: Generating Messages")

        message_id = 1
        for chunk_idx in range(num_chunks):
            chunk_start = start_time + (chunk_idx * chunk_size)
            chunk_end = chunk_start + chunk_size - 1

            # Generate random number of messages for this chunk
            num_messages = random.randint(3, 8)

            print(f"\n[{format_time_range(chunk_start, chunk_end)}] Chunk {chunk_idx + 1}/{num_chunks}:")

            for _ in range(num_messages):
                # Random timestamp within the chunk
                msg_time = random.randint(chunk_start, chunk_end)
                msg = generate_message(msg_time, message_id)
                cache.cache_message(msg)

                print(f"  {format_timestamp(msg_time)} [{msg.author_name}] {msg.content}")
                message_id += 1

        # Phase 2: Create Level-1 summaries
        print_section_header("PHASE 2: Creating Level-1 Summaries (30min chunks)")

        for chunk_idx in range(num_chunks):
            chunk_start = start_time + (chunk_idx * chunk_size)
            chunk_end = chunk_start + chunk_size - 1

            # Get messages in this chunk
            messages = cache.get_cached_messages(123, chunk_start, chunk_end)

            if messages:
                summary_text = create_realistic_summary(messages)
                summary = Summary(
                    channel_id=123,
                    start_time=chunk_start,
                    end_time=chunk_end,
                    summary_level=1,
                    summary_text=summary_text,
                    message_count=len(messages),
                )
                cache.save_summary(summary)

                print(f"\n[{format_time_range(chunk_start, chunk_end)}] Level-1 Summary:")
                print(f"  {summary_text}")

        # Phase 3: Create Level-2 summaries (combine pairs of Level-1)
        print_section_header("PHASE 3: Creating Level-2 Summaries (1 hour chunks)")

        level_1_summaries = cache.get_summaries_by_level(123, 1)

        # Pair them up
        for i in range(0, len(level_1_summaries) - 1, 2):
            s1 = level_1_summaries[i]
            s2 = level_1_summaries[i + 1]

            meta_text = create_meta_summary(s1, s2)
            meta_summary = Summary(
                channel_id=123,
                start_time=s1.start_time,
                end_time=s2.end_time,
                summary_level=2,
                summary_text=meta_text,
                message_count=s1.message_count + s2.message_count,
            )
            cache.save_summary(meta_summary)

            print(f"\n[{format_time_range(s1.start_time, s2.end_time)}] Level-2 Summary:")
            print(f"  {meta_text}")

        # Phase 4: Create Level-3 summaries (combine pairs of Level-2)
        print_section_header("PHASE 4: Creating Level-3 Summaries (2 hour chunks)")

        level_2_summaries = cache.get_summaries_by_level(123, 2)

        # Pair them up
        for i in range(0, len(level_2_summaries) - 1, 2):
            s1 = level_2_summaries[i]
            s2 = level_2_summaries[i + 1]

            meta_text = create_meta_summary(s1, s2)
            meta_summary = Summary(
                channel_id=123,
                start_time=s1.start_time,
                end_time=s2.end_time,
                summary_level=3,
                summary_text=meta_text,
                message_count=s1.message_count + s2.message_count,
            )
            cache.save_summary(meta_summary)

            print(f"\n[{format_time_range(s1.start_time, s2.end_time)}] Level-3 Summary:")
            print(f"  {meta_text}")

        # Phase 5: Create Level-4 summaries (combine pairs of Level-3)
        print_section_header("PHASE 5: Creating Level-4 Summaries (4 hour chunks)")

        level_3_summaries = cache.get_summaries_by_level(123, 3)

        # Pair them up
        for i in range(0, len(level_3_summaries) - 1, 2):
            s1 = level_3_summaries[i]
            s2 = level_3_summaries[i + 1]

            meta_text = create_meta_summary(s1, s2)
            meta_summary = Summary(
                channel_id=123,
                start_time=s1.start_time,
                end_time=s2.end_time,
                summary_level=4,
                summary_text=meta_text,
                message_count=s1.message_count + s2.message_count,
            )
            cache.save_summary(meta_summary)

            print(f"\n[{format_time_range(s1.start_time, s2.end_time)}] Level-4 Summary:")
            print(f"  {meta_text}")

        # Phase 6: Create Level-5 summaries (combine pairs of Level-4)
        print_section_header("PHASE 6: Creating Level-5 Summaries (8 hour chunks)")

        level_4_summaries = cache.get_summaries_by_level(123, 4)

        # Pair them up
        for i in range(0, len(level_4_summaries) - 1, 2):
            s1 = level_4_summaries[i]
            s2 = level_4_summaries[i + 1]

            meta_text = create_meta_summary(s1, s2)
            meta_summary = Summary(
                channel_id=123,
                start_time=s1.start_time,
                end_time=s2.end_time,
                summary_level=5,
                summary_text=meta_text,
                message_count=s1.message_count + s2.message_count,
            )
            cache.save_summary(meta_summary)

            print(f"\n[{format_time_range(s1.start_time, s2.end_time)}] Level-5 Summary:")
            print(f"  {meta_text}")

        # Phase 7: Show context windows at different points
        print_section_header("PHASE 7: Context Windows at Different Times")

        # Show context at different checkpoints based on total hours
        if hours <= 12:
            times_to_check = [
                (7200, "2 hours in"),
                (21600, "6 hours in"),
                (hours * 3600, f"{hours} hours in (end)"),
            ]
        elif hours <= 24:
            times_to_check = [
                (7200, "2 hours in"),
                (43200, "12 hours in"),
                (hours * 3600, f"{hours} hours in (end)"),
            ]
        else:
            times_to_check = [
                (43200, "12 hours in"),
                (86400, "24 hours in"),
                (hours * 3600, f"{hours} hours in (end)"),
            ]

        for current_time, label in times_to_check:
            print(f"\n\n{'='*80}")
            print(f"  CONTEXT WINDOW AT: {format_timestamp(current_time)} ({label})")
            print(f"{'='*80}")

            context = cache.get_context_windows(
                channel_id=123,
                current_time=current_time,
                chunk_size=1800,
                chunks_per_level=4,
                chunk_size_multiplier=2,
            )

            print(f"\nTotal context depth: {context['config']['total_depth_seconds']/3600:.1f} hours")

            # Show raw messages
            print(f"\n--- LEVEL 0: Raw Messages (most recent 1 hour) ---")
            if context["raw_messages"]:
                for msg in context["raw_messages"][-10:]:  # Show last 10
                    print(f"  {format_timestamp(msg.timestamp)} [{msg.author_name}] {msg.content}")
                if len(context["raw_messages"]) > 10:
                    print(f"  ... and {len(context['raw_messages']) - 10} more messages")
            else:
                print("  (no messages)")

            # Show summaries at each level
            for level in range(1, 6):
                summaries = context.get(f"level_{level}", [])
                if summaries:
                    print(f"\n--- LEVEL {level}: Summaries ({len(summaries)} chunks) ---")
                    for summary in summaries[:5]:  # Show first 5
                        print(f"  [{format_time_range(summary.start_time, summary.end_time)}] {summary.summary_text}")
                    if len(summaries) > 5:
                        print(f"  ... and {len(summaries) - 5} more summaries")

        print("\n" + "="*80)
        print("  SIMULATION COMPLETE")
        print("="*80)

    finally:
        try:
            os.unlink(db_path)
        except:
            pass


if __name__ == "__main__":
    import sys
    random.seed(42)  # For reproducible output

    # Check for command line argument
    hours = 12
    if len(sys.argv) > 1:
        try:
            hours = int(sys.argv[1])
        except ValueError:
            print(f"Invalid hours: {sys.argv[1]}, using default of 12")

    simulate_12_hours(hours)
