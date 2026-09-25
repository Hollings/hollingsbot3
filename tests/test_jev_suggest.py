"""Tests for the LLM next-word suggester (token -> word mapping and the API call)."""

from __future__ import annotations

import json
import math

import httpx
import pytest

from hollingsbot.jev.client import JevError, Usage
from hollingsbot.jev.suggest import NextWordSuggester, continuation, words_from_tokens
from hollingsbot.jev.writer import ChatLine

DICT = ["i", "like", "the", "there", "pizza", "ram", "both", "them", "spicy"]


def words(tokens, heard=()):
    # The first 4 dictionary words count as "most common" (never completed); "ram" is not.
    return words_from_tokens(tokens, whole_words=DICT, complete_from=heard, protect_top=4)


def test_whole_words_pass_and_case_merges():
    assert words([(" Pizza", 0.5), ("pizza", 0.2), ("both", 0.3)]) == [("pizza", 0.7), ("both", 0.3)]


def test_pieces_of_rare_heard_words_become_those_words():
    got = dict(words([("ram", 0.6), ("sp", 0.1), ("xq", 0.1)], heard=["ramen", "spicy"]))
    assert got == {"ramen": 0.6, "spicy": 0.1}  # "ram" is a word, but "ramen" was just said


def test_common_words_are_not_completed():
    assert dict(words([("the", 0.9)], heard=["theremin"])) == {"the": 0.9}  # a most-common word stays itself
    assert dict(words([("i", 0.9)], heard=["iguana"])) == {"i": 0.9}


def test_punctuation_maps_and_junk_drops():
    got = dict(words([(".", 0.3), ("…", 0.1), ("J", 0.1), ('"', 0.1), ("two words", 0.1)]))
    assert got == {".": 0.3, "...": 0.1}


def test_continuation_finishes_a_word_the_model_is_still_spelling():
    vocab = ["sandwich", "filling", "tonight", "a", "it"]
    assert continuation("sand", [("wich", 0.8), (" because", 0.1)], whole_words=vocab, heard=[]) == "sandwich"
    assert continuation("ram", [("en", 0.9)], whole_words=vocab, heard=["ramen"]) == "ramen"
    assert continuation("like", [("it", 0.9)], whole_words=vocab, heard=[]) is None  # "likeit" isn't a word
    assert continuation("sand", [(" wich", 0.9)], whole_words=vocab, heard=[]) is None  # a leading space = new word
    assert (
        continuation("sand", [("wich", 0.2), ("because", 0.3)], whole_words=vocab, heard=[]) is None
    )  # not the favourite
    contractions = ["don't", "i've", "you'll"]
    assert continuation("don", [("'t", 0.9)], whole_words=contractions, heard=[]) == "don't"
    assert continuation("i", [("’ve", 0.9)], whole_words=contractions, heard=[]) == "i've"


def test_contraction_halves_are_never_words_of_their_own():
    assert words([("don", 0.4), ("'ll", 0.3), ("ll", 0.2), ("both", 0.1)]) == [("both", 0.1)]


def _logprobs_response(tops):
    return {
        "choices": [{"logprobs": {"content": [{"top_logprobs": [
            {"token": t, "logprob": math.log(p)} for t, p in tops
        ]}]}}],
        "usage": {"prompt_tokens": 60, "cost": 0.00001},
    }  # fmt: skip


async def test_suggest_asks_for_top_logprobs_and_maps_them():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json=_logprobs_response([("both", 0.6), ("ram", 0.3), ("them", 0.1)]))

    s = NextWordSuggester(api_key="k", transport=httpx.MockTransport(handler))
    usage = Usage()
    chat = [ChatLine("Mallory", "spicy ramen"), ChatLine("Hollings", "ramen or pizza?")]
    no_ram = [w for w in DICT if w != "ram"]  # (with a 9-word dictionary every word is "most common")
    got = await s.suggest(chat, "Jev", ["i", "like"], whole_words=no_ram, complete_from=["ramen"], usage=usage)
    assert got.options == [("both", pytest.approx(0.6)), ("ramen", pytest.approx(0.3)), ("them", pytest.approx(0.1))]
    assert got.completes is None
    assert seen["max_tokens"] == 1 and seen["top_logprobs"] == 20 and seen["logprobs"] is True
    assert seen["messages"][1]["content"].endswith("Hollings: ramen or pizza?\nJev: i like")
    assert usage.calls == 1


async def test_a_deeper_list_goes_only_to_providers_that_serve_one():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=_logprobs_response([("both", 1.0)]))

    s = NextWordSuggester(api_key="k", transport=httpx.MockTransport(handler))
    await s.suggest([ChatLine("A", "hi")], "Jev", [], whole_words=DICT, complete_from=[], top=200)
    await s.suggest([ChatLine("A", "hi")], "Jev", [], whole_words=DICT, complete_from=[])
    deep, usual = seen
    assert deep["top_logprobs"] == 200 and deep["provider"]["only"] == ["novita"]
    assert usual["top_logprobs"] == 20 and "only" not in usual["provider"]  # the constructor's default


async def test_a_failed_deeper_list_is_asked_again_at_the_usual_depth():
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append(body["top_logprobs"])
        if body["top_logprobs"] > 20:  # a provider that caps at 20 answers 200 with an error body
            return httpx.Response(200, json={"error": {"message": "Requested sample logprobs of 200"}})
        return httpx.Response(200, json=_logprobs_response([("both", 1.0)]))

    s = NextWordSuggester(api_key="k", transport=httpx.MockTransport(handler))
    got = await s.suggest([ChatLine("A", "hi")], "Jev", [], whole_words=DICT, complete_from=[], top=200)
    assert seen == [200, 20]
    assert got.options == [("both", pytest.approx(1.0))]


async def test_suggest_without_logprobs_is_an_error():
    s = NextWordSuggester(
        api_key="k", transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"choices": [{}]}))
    )
    with pytest.raises(JevError, match="no logprobs"):
        await s.suggest([ChatLine("A", "hi")], "Jev", [], whole_words=DICT, complete_from=[])
