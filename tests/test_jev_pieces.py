"""Tests for hollingsbot.jev.pieces: Jev writing from the LLM's raw next pieces."""

from __future__ import annotations

import pytest

from hollingsbot.jev import ChatLine, JevError, JevWriter, WriterConfig
from hollingsbot.jev.pieces import PieceWriter, clean_piece, join_piece, make_writer, prefixes
from hollingsbot.jev.writer import DEEP_TOP, STOP

CHAT = [ChatLine("Hollings", "jev whats your favorite thing about fall?")]


class FakeSuggester:
    """The LLM: ``menus`` maps the reply so far to its next tokens ("*" for anything else)."""

    def __init__(self, menus=None, fail=False):
        self.menus = menus or {}
        self.fail = fail
        self.calls: list[tuple[list[str], int | None]] = []

    async def top_tokens(self, chat, name, words, style="", usage=None, top=None):
        self.calls.append((list(words), top))
        if self.fail:
            raise JevError("suggester down")
        text = words[0] if words else ""
        return [(t, 0.1) for t in self.menus.get(text, self.menus.get("*", []))]


class ScriptedJev:
    """Jev: ``script`` maps the blank it's shown to the piece it wants (STOP when not listed).

    It asks to turn the page whenever the piece it wants isn't on this one.
    """

    model = "fake-jev"

    def __init__(self, script):
        self.script = script
        self.requests: list[dict] = []

    async def ask(self, state, questions, usage=None):
        self.requests.append(questions)
        nxt = questions["next"]
        want = self.script.get(nxt["instructions"]["jev_reply"], STOP)
        options = list(nxt["criteria"])
        answers = {"next": {"probabilities": {o: float(o == want) for o in options}}}
        if "other" in questions:
            here = want in options
            answers["other"] = {"probabilities": {"pick": 0.98 if here else 0.02, "other": 0.02 if here else 0.98}}
        return answers

    def menus(self):
        return [list(q["next"]["criteria"]) for q in self.requests]


def config(**overrides):
    base = {"units": "pieces", "stop": "choice", "min_words": 2, "max_words": 10, "temperature": 0.0, "shuffle": False}
    return WriterConfig(**{**base, **overrides})


def writer(jev, suggester, **overrides):
    return PieceWriter(jev, name="Jev", config=config(**overrides), suggester=suggester)


def test_join_piece():
    joins = prefixes(["change", "forward", "fall", "ramen"])
    assert join_piece("", "I", joins) == "I"
    assert join_piece("the chang", "e", joins) == "the change"
    assert join_piece("im looking for", "ward", joins) == "im looking forward"
    assert join_piece("excited for", "the", joins) == "excited for the"
    assert join_piece("i love ram", "en", joins) == "i love ramen"
    assert join_piece("lay", "...", joins) == "lay..."
    assert join_piece("white", "-light", joins) == "white-light"
    assert join_piece("Jev", "'s", joins) == "Jev's"


def test_clean_piece():
    assert clean_piece(" fall ") == "fall"
    for bad in ("", "  ", "\n", "<|eot_id|>", "�", STOP):
        assert clean_piece(bad) is None


async def test_writes_pieces_joined_until_stop():
    menus = {"": ["I", "The"], "I": ["love", "am"], "I love": ["fa", "the"], "I love fa": ["ll", "st"], "*": ["so"]}
    script = {"___": "I", "I___": "love", "I love___": "fa", "I love fa___": "ll"}
    jev = ScriptedJev(script)

    reply = await writer(jev, FakeSuggester(menus)).write(CHAT)

    assert reply.text == "I love fall"
    assert reply.words == ["I", "love", "fa", "ll"]
    assert reply.stop_reason == "chose_stop"
    menus_shown = jev.menus()
    assert all(STOP not in m for m in menus_shown[:2]) and all(STOP in m for m in menus_shown[2:])
    first = jev.requests[0]["next"]["instructions"]
    assert first["jev_reply"] == "___" and "piece" in first["question"]
    assert "other" not in jev.requests[0]  # one page: nothing to turn to


@pytest.mark.parametrize(
    ("mode", "third_menu", "fourth_menu"),
    [
        ("never", ["sat", "on"], ["on", STOP]),
        ("adjacent", ["the", "sat", "on"], ["the", "cat", "on", STOP]),
        ("off", ["the", "cat", "sat", "on"], ["the", "cat", "sat", "on", STOP]),
    ],
)
async def test_no_repeat_takes_used_pieces_off_the_menu(mode, third_menu, fourth_menu):
    script = {"___": "the", "the___": "cat", "the cat___": "sat"}
    jev = ScriptedJev(script)
    w = writer(jev, FakeSuggester({"*": ["the", "cat", "sat", "on"]}), min_words=3, no_repeat=mode)

    reply = await w.write(CHAT)

    assert reply.text == "the cat sat"
    menus_shown = jev.menus()
    assert menus_shown[2] == third_menu
    assert sorted(menus_shown[3]) == sorted(fourth_menu)  # STOP goes in at a random place


async def test_turns_to_the_next_page_when_its_piece_isnt_here():
    suggester = FakeSuggester({"*": ["red", "blue", "green", "gold"]})
    jev = ScriptedJev({"___": "gold"})
    w = writer(jev, suggester, llm_pages=2, page_size=2, min_words=1)

    reply = await w.write(CHAT)

    assert reply.text == "gold"
    assert reply.steps[0].page == 1 and reply.steps[0].source == "llm" and reply.steps[0].other == 0.0
    assert jev.menus()[:2] == [["red", "blue"], ["green", "gold"]]
    assert "other" in jev.requests[0] and "other" not in jev.requests[1]  # the last page asks nothing
    assert suggester.calls[0][1] == DEEP_TOP  # more than one LLM page: the deep list


async def test_own_words_are_the_menu_when_the_llm_is_down():
    jev = ScriptedJev({"___": "zebra"})
    w = writer(jev, FakeSuggester(fail=True), min_words=1)

    reply = await w.write(CHAT, learned=["zebra"])

    assert reply.text == "zebra"
    assert reply.steps[0].source == "own"


async def test_own_page_comes_after_the_llm_pages_without_what_was_shown():
    jev = ScriptedJev({"___": "zebra"})
    w = writer(jev, FakeSuggester({"*": ["the", "fall"]}), own_page=True, min_words=1)

    reply = await w.write(CHAT, learned=["zebra"])

    assert reply.text == "zebra" and reply.steps[0].page == 1 and reply.steps[0].source == "own"
    own_menu = jev.menus()[1]
    assert "zebra" in own_menu and "the" not in own_menu and "fall" not in own_menu


async def test_stops_at_max_pieces():
    jev = ScriptedJev({"___": "red", "red___": "blue", "red blue___": "gold"})
    w = writer(jev, FakeSuggester({"*": ["red", "blue", "gold", "green"]}), min_words=5, max_words=3)
    reply = await w.write(CHAT)
    assert reply.stop_reason == "max_words" and reply.words == ["red", "blue", "gold"]


def test_make_writer_follows_units():
    jev, suggester = ScriptedJev({}), FakeSuggester()
    assert type(make_writer(jev, name="Jev", config=WriterConfig(), suggester=None)) is JevWriter
    assert isinstance(make_writer(jev, name="Jev", config=config(), suggester=suggester), PieceWriter)
    with pytest.raises(ValueError, match="suggester"):
        make_writer(jev, name="Jev", config=config(), suggester=None)
    with pytest.raises(ValueError, match="units"):
        make_writer(jev, name="Jev", config=config(units="letters"), suggester=suggester)
    with pytest.raises(ValueError, match="no_repeat"):
        make_writer(jev, name="Jev", config=config(no_repeat="sometimes"), suggester=suggester)
