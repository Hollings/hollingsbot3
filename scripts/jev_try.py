"""Ask Jev something from the terminal, with the same writer the bot uses.

Needs OPENROUTER_API_KEY (read from .env). Each reply costs about $0.002 per word.

    python scripts/jev_try.py "Jev what's the capital of France?"
    python scripts/jev_try.py --as Mallory "how do rainbows work, Jev?" --top-p 0.8
    python scripts/jev_try.py --chat "Hollings: got a puppy" --chat "Mallory: cute" "Jev name her?"
    python scripts/jev_try.py --samples      # the prompt set src/hollingsbot/jev/README.md was tuned on
    python scripts/jev_try.py --samples --suggest --llm-pages 3 --own-page 1   # pages; words tagged [p2], [own]
    python scripts/jev_try.py --samples --units pieces --stop choice          # raw LLM pieces, not words
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import random
import sys
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hollingsbot.jev import ChatLine, DecisionsClient, WriterConfig
from hollingsbot.jev.pieces import make_writer
from hollingsbot.jev.suggest import DEFAULT_SUGGEST_MODEL, NextWordSuggester

if TYPE_CHECKING:
    from hollingsbot.jev.writer import Step

SAMPLES = {
    # the first things people actually said to Jev in #wendy-dev
    "hi": [ChatLine("Hollings", "Hi")],
    "whatup": [ChatLine("Hollings", "Hi"), ChatLine("Jev", "hi bro"), ChatLine("Hollings", "What up dog")],
    "food": [ChatLine("Hollings", "hey Jev, what's your favorite food?")],
    "capital": [ChatLine("Hollings", "Jev what is the capital of France?")],
    "sentient": [ChatLine("Hollings", "jev are you sentient? be honest")],
    "dog": [
        ChatLine("Hollings", "just got back from the beach, it was so windy"),
        ChatLine("Mallory", "nice lol. did you bring the new puppy"),
        ChatLine("Hollings", "yeah she loved it. Jev what should i name her?"),
    ],
    "joke": [ChatLine("Hollings", "Jev tell me a joke")],
    "rainbow": [ChatLine("Mallory", "how do rainbows actually work? Jev?")],
    "weekend": [
        ChatLine("Hollings", "anyone doing anything fun this weekend"),
        ChatLine("Mallory", "probably just sleeping lol"),
        ChatLine("Hollings", "what about you Jev?"),
    ],
    "interview": [ChatLine("Mallory", "Jev I'm nervous about my job interview tomorrow, any advice?")],
    "hotdog": [ChatLine("Hollings", "jev settle this. is a hotdog a sandwich")],
}


async def run(
    chats: dict[str, list[ChatLine]], config: WriterConfig, seed: int | None, trace: bool, suggest: str | None
) -> None:
    client = DecisionsClient()
    if config.units == "pieces" and not suggest:
        suggest = DEFAULT_SUGGEST_MODEL  # the LLM's tokens are the menu
    suggester = NextWordSuggester(model=suggest) if suggest else None
    writer = make_writer(client, name="Jev", config=config, rng=random.Random(seed), suggester=suggester)
    paging = suggester is not None and (config.llm_pages > 1 or config.own_page)
    total = 0.0
    words = 0
    by_page: Counter[str] = Counter()
    try:
        for label, chat in chats.items():
            print(f"\n[{label}] {chat[-1].speaker}: {chat[-1].text}")

            async def show(text: str) -> None:
                print(f"\r  Jev: {text}", end="", flush=True)

            reply = await writer.write(chat, on_word=show if sys.stdout.isatty() else None)
            u = reply.usage
            total += u.cost
            words += len(reply.words)
            print(f"\r  Jev: {reply.text}")
            print(
                f"  ({len(reply.words)} words, stop={reply.stop_reason}, {u.calls} calls, {u.seconds:.1f}s, ${u.cost:.4f})"
            )
            if paging:
                # every word Jev turned the page for, tagged with where it found it (a gave_up
                # reply's last step is the word it rejected, not part of the reply)
                kept = reply.steps[: len(reply.words)]
                print("  pages: " + " ".join(s.word if s.page == 0 else f"{s.word}[{_page(s)}]" for s in kept))
                by_page.update(_page(s) for s in kept)
            if trace:
                for s in reply.steps:
                    print(
                        f"     {s.word!r:14} choice={s.choice:.2f} fluency={s.fluency:.2f} llm={s.suggested:.2f} "
                        f"send={s.send:.2f} sense={s.sense:.2f} of {s.options} page={_page(s)} other={s.other:.2f}"
                    )
                if reply.final_check:
                    print(f"     (stop check: send={reply.final_check[0]:.2f} sense={reply.final_check[1]:.2f})")
    finally:
        await client.aclose()
        if suggester is not None:
            await suggester.aclose()
    if len(chats) > 1:
        n = len(chats)
        print(f"\ntotal ${total:.4f} | mean {words / n:.1f} words, ${total / n:.4f} per reply")
    if by_page:
        counted = sum(by_page.values())
        print("words by page: " + "  ".join(f"{k} {v / counted:.0%}" for k, v in sorted(by_page.items())))


def _page(step: Step) -> str:
    """p1, p2... for the LLM's pages, own for Jev's own menu."""
    return "own" if step.source == "own" else f"p{step.page + 1}"


def _truthy(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "on")


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("message", nargs="?", help="the message to answer")
    ap.add_argument("--as", dest="speaker", default="Hollings", help="who sends the message")
    ap.add_argument("--chat", action="append", default=[], help='earlier message, "Name: text" (repeatable)')
    ap.add_argument("--samples", action="store_true", help="run the sample prompt set")
    ap.add_argument("--only", help=f"comma-separated subset of samples: {','.join(SAMPLES)}")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--trace", action="store_true", help="print each step's scores")
    ap.add_argument(
        "--suggest",
        nargs="?",
        const=DEFAULT_SUGGEST_MODEL,
        help=f"let an LLM propose each next word (default model {DEFAULT_SUGGEST_MODEL})",
    )
    for f in dataclasses.fields(WriterConfig):
        kind = _truthy if isinstance(f.default, bool) else float if f.default is None else type(f.default)
        ap.add_argument("--" + f.name.replace("_", "-"), type=kind, default=f.default)
    args = ap.parse_args()

    config = WriterConfig(**{f.name: getattr(args, f.name) for f in dataclasses.fields(WriterConfig)})
    if args.samples:
        chats = {k: v for k, v in SAMPLES.items() if not args.only or k in args.only.split(",")}
    elif args.message:
        earlier = [ChatLine(*(part.strip() for part in line.split(":", 1))) for line in args.chat]
        chats = {"message": [*earlier, ChatLine(args.speaker, args.message)]}
    else:
        ap.error("give a message or --samples")
    asyncio.run(run(chats, config, args.seed, args.trace, args.suggest))


if __name__ == "__main__":
    main()
