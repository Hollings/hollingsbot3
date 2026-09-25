"""Tests for Jev's text helpers and decoding rules."""

from __future__ import annotations

import random
from collections import Counter

from hollingsbot.jev.decoding import allowed, finalists, pick, repetition_factor
from hollingsbot.jev.text import context_words, join_words, words_in


class TestText:
    def test_join_attaches_punctuation(self):
        assert join_words(["i", "like", "pizza", ".", "you", "?"]) == "i like pizza. you?"

    def test_join_ellipsis_and_empty(self):
        assert join_words(["well", "...", "no"]) == "well... no"
        assert join_words([]) == ""

    def test_words_in_lowercases_and_keeps_apostrophes(self):
        assert words_in("Jev, what's UP? It\u2019s 2 AM") == ["jev", "what's", "up", "it's", "2", "am"]

    def test_context_words_dedupes_in_order_and_limits(self):
        assert context_words(["b a", "a c d"], limit=3) == ["b", "a", "c"]


class TestFinalists:
    def test_stops_at_mass(self):
        assert finalists({"a": 0.5, "b": 0.3, "c": 0.2}, mass=0.7, cap=4) == ["a", "b"]

    def test_stops_at_cap(self):
        probs = dict.fromkeys("abcdefghij", 0.1)
        assert len(finalists(probs, mass=0.99, cap=4)) == 4


class TestAllowed:
    def test_no_immediate_repeat(self):
        assert allowed(["the", "cat"], ["the"]) == ["cat"]

    def test_no_repeated_trigram(self):
        words = ["to", "the", "other", "side", "to", "the"]
        # "to the other" already happened, so "other" may not follow "to the" again.
        assert allowed(["other", "far"], words) == ["far"]

    def test_punctuation_not_first_nor_after_punctuation(self):
        assert allowed([".", "hi"], []) == ["hi"]
        assert allowed(["?", "so"], ["no", "."]) == ["so"]
        assert allowed(["?"], ["no"]) == ["?"]

    def test_dedupes_candidates(self):
        assert allowed(["a", "b", "a"], []) == ["a", "b"]


class TestRepetition:
    COMMON = frozenset({"the", "more"})

    def factor(self, word, words):
        return repetition_factor(word, words, self.COMMON, penalty=0.3, common_penalty=0.5, window=3)

    def test_content_words_get_cheaper_each_use_anywhere(self):
        words = ["pizza", "a", "b", "c", "d", "pizza"]
        assert self.factor("pizza", words) == 0.3**2
        assert self.factor("new", words) == 1.0
        assert self.factor(".", [".", "."]) == 1.0

    def test_common_words_only_pay_for_recent_uses(self):
        assert self.factor("more", ["more", "er", "more"]) == 0.5**2
        assert self.factor("the", ["the", "x", "y", "z"]) == 1.0  # outside the window


class TestPick:
    def test_greedy(self):
        assert pick({"a": 0.2, "b": 0.7, "c": 0.1}, random.Random(0), temperature=0, top_p=1.0) == "b"

    def test_nucleus_excludes_the_tail(self):
        scores = {"a": 0.5, "b": 0.3, "c": 0.1, "d": 0.1}
        rng = random.Random(0)
        seen = Counter(pick(scores, rng, temperature=1.0, top_p=0.7) for _ in range(500))
        assert set(seen) == {"a", "b"}

    def test_samples_proportionally_at_temperature_one(self):
        rng = random.Random(1)
        seen = Counter(pick({"a": 0.75, "b": 0.25}, rng, temperature=1.0, top_p=1.0) for _ in range(4000))
        assert 0.7 < seen["a"] / 4000 < 0.8

    def test_all_zero_scores_still_pick_something(self):
        assert pick({"a": 0.0, "b": 0.0}, random.Random(0), temperature=0.7, top_p=0.6) in {"a", "b"}
