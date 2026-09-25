"""Regenerate src/hollingsbot/jev/vocab_en.txt, Jev's word list.

Jev writes by picking the next word from a vocabulary, so this file is every
word it can say (plus whatever words appear in the chat it is replying to).
The list is English by frequency from ``wordfreq``, whose sources include
Twitter and Reddit, so chat words (lol, gonna, idk) rank where people use them.

wordfreq is only needed to run this script, not by the bot:

    pip install wordfreq
    python scripts/build_jev_vocab.py            # default 8000 words
    python scripts/build_jev_vocab.py --size 10000
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from wordfreq import top_n_list

OUT = Path(__file__).resolve().parents[1] / "src" / "hollingsbot" / "jev" / "vocab_en.txt"

# A word the bot can say at random, with no one having said it first. wordfreq's
# top lists already drop slurs; this keeps out the rest of what should never
# come out of a dice roll. (Words from the chat itself still reach Jev.)
EXCLUDE = {
    "rape",
    "raped",
    "rapes",
    "raping",
    "rapist",
    "retard",
    "retarded",
    "negro",
    "cunt",
    "whore",
    "slut",
    "tranny",
    "fag",
    "faggot",
}
SINGLE_LETTER_WORDS = {"a", "i", "u"}
WORD = re.compile(r"[a-z0-9][a-z0-9']*")


def build(size: int) -> list[str]:
    words: list[str] = []
    for w in top_n_list("en", size * 2):
        if not WORD.fullmatch(w) or w in EXCLUDE:
            continue
        if len(w) == 1 and w not in SINGLE_LETTER_WORDS and not w.isdigit():
            continue
        words.append(w)
        if len(words) == size:
            break
    return words


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=int, default=8000)
    args = ap.parse_args()
    words = build(args.size)
    header = f"# Jev vocabulary: top {len(words)} English words by frequency (wordfreq). Rebuild: scripts/build_jev_vocab.py\n"
    OUT.write_text(header + "\n".join(words) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {len(words)} words to {OUT}")


if __name__ == "__main__":
    main()
