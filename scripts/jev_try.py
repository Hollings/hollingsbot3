"""Ask Jev something from the terminal, with the same writer the bot uses.

Needs OPENROUTER_API_KEY (read from .env). Each reply costs about $0.002 per word.

    python scripts/jev_try.py "Jev what's the capital of France?"
    python scripts/jev_try.py --as Mallory "how do rainbows work, Jev?" --top-p 0.8
    python scripts/jev_try.py --chat "Hollings: got a puppy" --chat "Mallory: cute" "Jev name her?"
    python scripts/jev_try.py --samples      # the prompt set src/hollingsbot/jev/README.md was tuned on
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import random
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hollingsbot.jev import ChatLine, DecisionsClient, JevWriter, WriterConfig

SAMPLES = {
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


async def run(chats: dict[str, list[ChatLine]], config: WriterConfig, seed: int | None, trace: bool) -> None:
    client = DecisionsClient()
    writer = JevWriter(client, config=config, rng=random.Random(seed))
    total = 0.0
    try:
        for label, chat in chats.items():
            print(f"\n[{label}] {chat[-1].speaker}: {chat[-1].text}")

            async def show(text: str) -> None:
                print(f"\r  Jev: {text}", end="", flush=True)

            reply = await writer.write(chat, on_word=show if sys.stdout.isatty() else None)
            u = reply.usage
            total += u.cost
            print(f"\r  Jev: {reply.text}")
            print(
                f"  ({len(reply.words)} words, stop={reply.stop_reason}, {u.calls} calls, {u.seconds:.1f}s, ${u.cost:.4f})"
            )
            if trace:
                for s in reply.steps:
                    print(
                        f"     {s.word!r:14} choice={s.choice:.2f} fluency={s.fluency:.2f} "
                        f"send={s.send:.2f} sense={s.sense:.2f} of {s.options}"
                    )
                if reply.final_check:
                    print(f"     (stop check: send={reply.final_check[0]:.2f} sense={reply.final_check[1]:.2f})")
    finally:
        await client.aclose()
    if len(chats) > 1:
        print(f"\ntotal ${total:.4f}")


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("message", nargs="?", help="the message to answer")
    ap.add_argument("--as", dest="speaker", default="Hollings", help="who sends the message")
    ap.add_argument("--chat", action="append", default=[], help='earlier message, "Name: text" (repeatable)')
    ap.add_argument("--samples", action="store_true", help="run the sample prompt set")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--trace", action="store_true", help="print each step's scores")
    for f in dataclasses.fields(WriterConfig):
        ap.add_argument("--" + f.name.replace("_", "-"), type=type(f.default), default=f.default)
    args = ap.parse_args()

    config = WriterConfig(**{f.name: getattr(args, f.name) for f in dataclasses.fields(WriterConfig)})
    if args.samples:
        chats = SAMPLES
    elif args.message:
        earlier = [ChatLine(*(part.strip() for part in line.split(":", 1))) for line in args.chat]
        chats = {"message": [*earlier, ChatLine(args.speaker, args.message)]}
    else:
        ap.error("give a message or --samples")
    asyncio.run(run(chats, config, args.seed, args.trace))


if __name__ == "__main__":
    main()
