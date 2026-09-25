"""Decoding rules for Jev's word-by-word writing.

Jev judges; these rules only decide what it is asked about and how its
judgments are turned into one pick. They are the same knobs every LLM sampler
has (repetition bans, a repetition penalty, temperature, nucleus sampling),
applied to Jev's probabilities instead of token logits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from hollingsbot.jev.text import PUNCTUATION

if TYPE_CHECKING:
    import random
    from collections.abc import Iterable, Mapping, Sequence


def finalists(probs: Mapping[str, float], mass: float, cap: int) -> list[str]:
    """A bucket's leaders: best-first until ``mass`` probability or ``cap`` words."""
    out: list[str] = []
    acc = 0.0
    for word, p in sorted(probs.items(), key=lambda kv: -kv[1]):
        out.append(word)
        acc += p
        if acc >= mass or len(out) >= cap:
            break
    return out


def allowed(candidates: Iterable[str], words: Sequence[str]) -> list[str]:
    """Drop candidates the text so far rules out.

    - never the same word twice in a row
    - never a three-word run that already appeared (breaks phrase loops)
    - punctuation never first, and never right after punctuation
    """
    last = words[-1] if words else None
    trigrams = {tuple(words[i : i + 3]) for i in range(len(words) - 2)}
    out: list[str] = []
    for w in dict.fromkeys(candidates):
        if w == last:
            continue
        if len(words) >= 2 and (words[-2], words[-1], w) in trigrams:
            continue
        if w in PUNCTUATION and (last is None or last in PUNCTUATION):
            continue
        out.append(w)
    return out


def repetition_factor(
    word: str,
    words: Sequence[str],
    common: frozenset[str],
    *,
    penalty: float,
    common_penalty: float,
    window: int,
) -> float:
    """Score multiplier that makes repeats less likely.

    A content word pays ``penalty`` per earlier use anywhere in the reply. A
    common word ("the", "is", "more") pays the milder ``common_penalty`` per use
    in the last ``window`` words only: grammar needs it again, but not in a
    stutter ("more mo er more er more"). Punctuation is free.
    """
    if word in PUNCTUATION:
        return 1.0
    if word in common:
        return common_penalty ** list(words[-window:]).count(word)
    return penalty ** list(words).count(word)


def pick(scores: Mapping[str, float], rng: random.Random, temperature: float, top_p: float) -> str:
    """Sample one option: nucleus (``top_p``) first, then temperature.

    ``temperature <= 0`` is greedy. Scores need not sum to 1.
    """
    items = sorted(((w, s) for w, s in scores.items() if s > 0), key=lambda kv: -kv[1])
    if not items:
        # Every option scored zero: choose uniformly rather than fail.
        return rng.choice(list(scores))
    if temperature <= 0:
        return items[0][0]
    total = sum(s for _, s in items)
    kept: list[tuple[str, float]] = []
    acc = 0.0
    for w, s in items:
        kept.append((w, s))
        acc += s / total
        if acc >= top_p:
            break
    weights = [s ** (1.0 / temperature) for _, s in kept]
    return rng.choices([w for w, _ in kept], weights=weights)[0]
