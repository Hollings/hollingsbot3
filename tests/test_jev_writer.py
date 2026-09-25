"""Tests for JevWriter's word-by-word loop, against a scripted fake Decisions client."""

from __future__ import annotations

import asyncio
import random
from typing import TYPE_CHECKING

import pytest

from hollingsbot.jev.client import JevError
from hollingsbot.jev.suggest import Suggestion
from hollingsbot.jev.text import PUNCTUATION, words_in
from hollingsbot.jev.writer import BLANK, STOP, ChatLine, JevWriter, Reply, WriterConfig

if TYPE_CHECKING:
    from hollingsbot.jev.client import Usage

VOCAB = ["the", "a", "i", "is", "like", "pizza", "you", "yes", "no", "dog", "cat", "blue"]
CHAT = [ChatLine("Mallory", "hey"), ChatLine("Hollings", "Jev whats your favorite food")]


class ScriptedJev:
    """Answers like a Jev that wants to say ``script`` word by word, then send it.

    Requests come in three kinds: buckets (``b0``...) and the runoff (``next`` +
    ``fit*``) read the reply from the state's blank; the stop check (``send`` +
    ``sense``) reads it from its own instructions, since its state is the chat only.
    """

    model = "fake-jev"

    def __init__(self, script, *, fluency=0.8, sense=0.9, send_mid=0.02, hang_on_call=None):
        self.script = script
        self.fluency = fluency
        self.sense = sense
        self.send_mid = send_mid
        self.hang_on_call = hang_on_call
        self.requests: list[tuple[dict, dict]] = []

    def _wanted(self, typed: str) -> str | None:
        done = words_in(typed)
        return self.script[len(done)] if len(done) < len(self.script) else None

    async def ask(self, state, questions, usage: Usage | None = None):
        self.requests.append((state, questions))
        if self.hang_on_call is not None and len(self.requests) >= self.hang_on_call:
            await asyncio.Event().wait()
        if usage is not None:
            usage.record({"input_tokens": 100, "cost": 0.001}, 0.01)
        next_instructions = questions.get("next", {}).get("instructions")
        if isinstance(next_instructions, dict):  # single-call mode: the blank rides in the question
            typed = next_instructions["jev_reply"].replace(BLANK, "")
        elif "jev_reply" in state:
            typed = state["jev_reply"].replace(BLANK, "")
        else:
            typed = questions["send"]["instructions"]["text"]
        wanted = self._wanted(typed)
        answers = {}
        for key, q in questions.items():
            if key == "send":
                send = 1.0 if wanted is None else self.send_mid
                answers[key] = {"type": "choice", "probabilities": {"send": send, "keep typing": 1 - send}}
            elif key == "sense":
                answers[key] = {"type": "noul", "noul": self.sense}
            elif key == "other":  # "pick from this page, or type a word that isn't in it?"
                other = 0.05 if wanted in q["instructions"]["options"].split(", ") else 0.95
                answers[key] = {"type": "choice", "probabilities": {"pick": 1 - other, "other": other}}
            elif q["type"] == "choice":
                opts = list(q["criteria"])
                # With STOP on the menu, a Jev that has said everything picks it.
                target = STOP if wanted is None and STOP in opts else wanted
                if target in opts:
                    probs = {o: (0.9 if o == target else 0.1 / max(len(opts) - 1, 1)) for o in opts}
                else:
                    probs = {o: 1 / len(opts) for o in opts}
                answers[key] = {"type": "choice", "probabilities": probs}
            else:
                answers[key] = {"type": "noul", "noul": self.fluency}
        return answers


def writer_for(fake, **cfg) -> JevWriter:
    # Mechanics tests: greedy, no length floor, no style line (the defaults are tuned for Discord).
    base = {"bucket_size": 4, "vocab_size": len(VOCAB), "temperature": 0, "exempt_top": 3, "min_words": 0, "style": ""}
    return JevWriter(fake, config=WriterConfig(**{**base, **cfg}), vocab=VOCAB, rng=random.Random(0))


def test_default_style_names_the_bot():
    writer = JevWriter(ScriptedJev([]), name="Jevvy", vocab=VOCAB)
    assert "Jevvy writes long, chatty messages" in writer._blank_instructions
    assert "{name}" not in writer._send_instructions


async def test_writes_the_wanted_words_then_sends():
    fake = ScriptedJev(["i", "like", "pizza"])
    reply = await writer_for(fake).write(CHAT)
    assert reply.text == "i like pizza"
    assert reply.stop_reason == "sent"
    assert reply.final_check == (1.0, 0.9)
    assert [s.word for s in reply.steps] == ["i", "like", "pizza"]
    # word 1: buckets + runoff; words 2-3: buckets + stop check + runoff; then buckets + stop check
    assert reply.usage.calls == 2 + 3 + 3 + 2
    assert reply.model == "fake-jev"


async def test_word_questions_mark_the_next_word_with_a_blank():
    fake = ScriptedJev(["i", "like"])
    await writer_for(fake).write(CHAT)
    blanks = [state["jev_reply"] for state, qs in fake.requests if "send" not in qs]
    assert blanks[0] == BLANK
    assert f"i {BLANK}" in blanks
    assert fake.requests[0][0]["chat"] == [
        {"from": "Mallory", "text": "hey"},
        {"from": "Hollings", "text": "Jev whats your favorite food"},
    ]


async def test_stop_check_sees_the_chat_and_the_reply_but_no_blank():
    fake = ScriptedJev(["i", "like"])
    await writer_for(fake).write(CHAT)
    state, check = next((s, q) for s, q in fake.requests if "send" in q)
    assert set(state) == {"chat"}
    assert check["send"]["instructions"]["text"] == "i"
    assert set(check["send"]["criteria"]) == {"send", "keep typing"}


async def test_chat_words_get_their_own_first_bucket():
    fake = ScriptedJev(["yes"])
    await writer_for(fake).write(CHAT)
    _state, buckets = fake.requests[0]
    first = list(buckets["b0"]["criteria"])
    assert first[:4] == ["jev", "whats", "your", "favorite"]  # newest message first, capped at bucket_size
    offered = [w for q in buckets.values() for w in q["criteria"]]
    assert len(offered) == len(set(offered))  # nothing offered twice
    assert set(VOCAB) <= set(offered)


async def test_no_punctuation_or_stop_check_before_the_first_word():
    fake = ScriptedJev(["i", "like"])
    await writer_for(fake).write(CHAT)
    kinds = ["stop" if "send" in qs else "runoff" if "next" in qs else "buckets" for _, qs in fake.requests]
    assert kinds[:5] == ["buckets", "runoff", "buckets", "stop", "runoff"]
    assert not set(PUNCTUATION) & set(fake.requests[1][1]["next"]["criteria"])
    assert set(PUNCTUATION) <= set(fake.requests[4][1]["next"]["criteria"])


async def test_stops_at_max_words():
    fake = ScriptedJev(["i", "like", "pizza", "you", "dog"])
    reply = await writer_for(fake, max_words=2).write(CHAT)
    assert reply.words == ["i", "like"]
    assert reply.stop_reason == "max_words"


async def test_gives_up_when_even_the_best_word_reads_unnatural():
    fake = ScriptedJev(["i", "like", "pizza"], fluency=0.1)
    reply = await writer_for(fake, give_up=0.25).write(CHAT)
    assert reply.words == ["i"]  # the first word is never refused
    assert reply.stop_reason == "gave_up"


async def test_stops_when_the_reply_stops_making_sense():
    fake = ScriptedJev(["i", "like", "pizza", "you", "dog"], sense=0.1)
    reply = await writer_for(fake, sense_floor=0.3, sense_from=3).write(CHAT)
    assert reply.words == ["i", "like", "pizza"]
    assert reply.stop_reason == "lost_thread"


class FixedRng(random.Random):
    """Every dice roll comes up ``value``."""

    def __init__(self, value):
        super().__init__(0)
        self.value = value

    def random(self):
        return self.value


def test_send_below_threshold_is_sampled_only_when_sampling():
    def stop(send, draw, temperature, sense=0.9, n_words=5):
        config = WriterConfig(
            temperature=temperature, send_at=0.6, send_floor=0.3, send_power=2.0, sense_floor=0.3, min_words=0
        )
        writer = JevWriter(ScriptedJev([]), config=config, vocab=VOCAB, rng=FixedRng(draw))
        return writer._stop_reason(send, sense, n_words)

    assert stop(0.7, draw=0.99, temperature=0.7) == "sent"  # at or above send_at: always
    assert stop(0.5, draw=0.2, temperature=0.7) == "sent"  # 0.2 < 0.5**2
    assert stop(0.5, draw=0.3, temperature=0.7) is None
    assert stop(0.25, draw=0.0, temperature=0.7) is None  # below the floor: never
    assert stop(0.5, draw=0.0, temperature=0) is None  # greedy never rolls the dice
    assert stop(0.1, draw=0.99, temperature=0.7, sense=0.1, n_words=3) == "lost_thread"
    assert stop(0.1, draw=0.99, temperature=0.7, sense=0.1, n_words=2) is None  # too short to judge


def test_sample_stop_draws_send_like_a_word():
    def stop(send, draw, temperature=0.7):
        config = WriterConfig(temperature=temperature, stop="sample", min_words=0)
        return JevWriter(ScriptedJev([]), config=config, vocab=VOCAB, rng=FixedRng(draw))._stop_reason(send, 0.9, 5)

    # P(send)=0.3 at T=0.7 -> 0.3**(1/0.7) / (0.3**(1/0.7) + 0.7**(1/0.7)) = 0.23
    assert stop(0.3, draw=0.2) == "sent" and stop(0.3, draw=0.3) is None
    assert stop(0.1, draw=0.03) == "sent"  # no floor: even a low P(send) can end it
    assert stop(0.7, draw=0.8) is None  # no ceiling: 0.7 -> 0.77, not a forced send
    assert stop(0.5, draw=0.99, temperature=0) == "sent" and stop(0.49, draw=0.0, temperature=0) is None


async def test_stop_choice_puts_stop_on_the_menu_and_ends_when_picked():
    fake = ScriptedJev(["i", "like", "pizza"])
    reply = await single_writer(fake, stop="choice", min_words=2).write(CHAT)
    assert reply.text == "i like pizza" and reply.stop_reason == "chose_stop"
    assert reply.steps[-1].word == STOP and reply.words[-1] == "pizza"
    menus = [q["next"]["criteria"] for _, q in fake.requests]
    assert [STOP in m for m in menus] == [False, False, True, True]  # offered from min_words on
    assert not any("send" in q for _, q in fake.requests)  # no separate stop check
    last = fake.requests[-1][1]
    assert "is the message finished?" in last["next"]["instructions"]["question"]
    fit = next(q for k, q in last.items() if k.startswith("fit") and "next" not in q["instructions"])
    assert fit["instructions"] == {"text": "i like pizza", "question": "Is the end of `text` a natural place for Jev to stop and send the message?"}  # fmt: skip


def test_stop_modes_are_checked_and_the_tournament_keeps_its_stop_check():
    with pytest.raises(ValueError, match="stop mode"):
        JevWriter(ScriptedJev([]), vocab=VOCAB, config=WriterConfig(stop="whenever"))
    tournament = JevWriter(ScriptedJev([]), vocab=VOCAB, config=WriterConfig(vocab_size=5000, stop="choice"))
    assert tournament.stop_mode == "threshold"


async def test_min_words_keeps_typing_past_a_ready_reply():
    fake = ScriptedJev(["i", "like", "pizza", "you"], send_mid=0.95)  # Jev would send after every word
    reply = await writer_for(fake, min_words=3).write(CHAT)
    assert reply.words == ["i", "like", "pizza"]
    assert reply.stop_reason == "sent"


async def test_style_reaches_word_and_send_questions():
    fake = ScriptedJev(["i", "like"])
    await writer_for(fake, style="Jev writes long, chatty messages.").write(CHAT)
    buckets, stop = fake.requests[0][1], next(q for _, q in fake.requests if "send" in q)
    assert "Jev writes long, chatty messages." in buckets["b0"]["instructions"]
    assert "Jev writes long, chatty messages." in stop["send"]["instructions"]["question"]


def single_writer(fake, **cfg) -> JevWriter:
    base = {"vocab_size": len(VOCAB), "temperature": 0, "exempt_top": 3, "min_words": 0, "style": ""}
    return JevWriter(fake, config=WriterConfig(**{**base, **cfg}), vocab=VOCAB, rng=random.Random(0))


async def test_small_vocabulary_costs_one_call_per_word():
    fake = ScriptedJev(["i", "like", "pizza"])
    writer = single_writer(fake)
    assert writer.single_call
    reply = await writer.write(CHAT)
    assert reply.text == "i like pizza"
    assert reply.stop_reason == "sent"
    assert reply.usage.calls == 3 + 1  # one per word, plus the one whose stop check said send


async def test_single_call_request_carries_word_fit_and_stop_questions():
    fake = ScriptedJev(["i", "like"])
    await single_writer(fake).write(CHAT)
    first_state, first = fake.requests[0]
    second_state, second = fake.requests[1]
    assert set(first_state) == set(second_state) == {"chat"}  # no blank in the shared state
    assert first["next"]["instructions"]["jev_reply"] == BLANK
    assert second["next"]["instructions"]["jev_reply"] == f"i {BLANK}"
    assert "send" not in first and {"send", "sense"} <= set(second)
    assert "jev" in first["next"]["criteria"] and "pizza" in first["next"]["criteria"]  # chat words + vocab
    assert any(k.startswith("fit") for k in first)


async def test_fluency_checks_can_be_switched_off():
    fake = ScriptedJev(["i", "like"])
    reply = await single_writer(fake, fluency_check=False).write(CHAT)
    assert reply.text == "i like"
    assert not any(k.startswith("fit") for _, qs in fake.requests for k in qs)


def test_single_pool_keeps_born_words_then_chat_words_then_learned():
    writer = single_writer(ScriptedJev([]), vocab_size=3, bucket_size=8)
    pool = writer._single_pool(CHAT, learned=["sushi", "jev", "tacos", "the", "burrito"])
    assert pool[-3:] == ["the", "a", "i"]  # born words are always on the menu
    assert pool[:5] == ["jev", "whats", "your", "favorite", "food"]  # the chat, newest message first
    assert len(pool) == 8 and "sushi" not in pool  # bucket_size caps it: no room left for learned words


def test_learned_words_fill_leftover_room_in_rank_order():
    writer = single_writer(ScriptedJev([]), vocab_size=3, bucket_size=12)
    pool = writer._single_pool([ChatLine("A", "hey")], learned=["sushi", "the", "tacos", "burrito"])
    assert pool == ["hey", "sushi", "tacos", "burrito", "the", "a", "i"]


async def test_learned_words_reach_the_question():
    fake = ScriptedJev(["sushi"])
    await single_writer(fake).write(CHAT, learned=["sushi"])
    assert "sushi" in fake.requests[0][1]["next"]["criteria"]


def test_tournament_buckets_include_learned_words_once():
    writer = writer_for(ScriptedJev([]))  # bucket mode (vocab 12 >= bucket 4)
    buckets = writer._bucket_questions(CHAT, learned=["sushi", "jev"])
    offered = [w for q in buckets.values() for w in q["criteria"]]
    assert "sushi" in offered and offered.count("jev") == 1


def test_sense_floor_defaults_per_mode():
    small = JevWriter(ScriptedJev([]), vocab=VOCAB, config=WriterConfig(vocab_size=100))
    big = JevWriter(ScriptedJev([]), vocab=VOCAB, config=WriterConfig(vocab_size=5000))
    pinned = JevWriter(ScriptedJev([]), vocab=VOCAB, config=WriterConfig(vocab_size=100, sense_floor=0.4))
    assert (small.sense_floor, big.sense_floor, pinned.sense_floor) == (0.0, 0.2, 0.4)


class FakeSuggester:
    """Returns ``plan[i]`` on the i-th call (the last one repeats)."""

    def __init__(self, plan):
        self.plan = plan
        self.calls: list[list[str]] = []
        self.tops: list[int | None] = []

    async def suggest(self, chat, name, words, *, whole_words, complete_from, style="", usage=None, top=None):
        self.calls.append(list(words))
        self.tops.append(top)
        if usage is not None:
            usage.record({"cost": 0.00001}, 0.01)
        return self.plan[min(len(self.calls) - 1, len(self.plan) - 1)]


def suggesting_writer(fake, suggester, **cfg) -> JevWriter:
    base = {"vocab_size": len(VOCAB), "temperature": 0, "exempt_top": 3, "min_words": 0, "style": ""}
    return JevWriter(
        fake, config=WriterConfig(**{**base, **cfg}), vocab=VOCAB, rng=random.Random(0), suggester=suggester
    )


async def test_suggestions_are_jevs_menu():
    suggester = FakeSuggester([Suggestion([("i", 0.5), ("like", 0.3), ("zebra", 0.2)])])
    fake = ScriptedJev(["i", "like"])
    reply = await suggesting_writer(fake, suggester, shuffle=False).write(CHAT)
    assert reply.text == "i like"
    assert list(fake.requests[0][1]["next"]["criteria"]) == ["i", "like", "zebra"]
    assert reply.steps[0].suggested == 0.5
    assert (reply.steps[0].page, reply.steps[0].source) == (0, "llm")
    assert suggester.calls[:2] == [[], ["i"]]
    assert suggester.tops[0] is None  # one page: the suggester's usual 20 tokens
    assert "other" not in fake.requests[0][1]  # nothing to turn to, so nothing to ask


async def test_suggestions_are_shuffled_so_their_rank_cannot_leak():
    ranked = [(w, 1 / (i + 2)) for i, w in enumerate(VOCAB)]
    fake = ScriptedJev(["i"])
    await suggesting_writer(fake, FakeSuggester([Suggestion(ranked)])).write(CHAT)
    shown = list(fake.requests[0][1]["next"]["criteria"])
    assert sorted(shown) == sorted(VOCAB) and shown != VOCAB


def paging_writer(fake, suggester, **cfg) -> JevWriter:
    return suggesting_writer(fake, suggester, shuffle=False, page_size=2, **cfg)


RANKED = Suggestion([("the", 0.4), ("a", 0.3), ("i", 0.2), ("is", 0.1)])


async def test_jev_turns_the_page_when_its_word_is_not_on_it():
    suggester = FakeSuggester([RANKED])
    fake = ScriptedJev(["i"])
    reply = await paging_writer(fake, suggester, llm_pages=2).write(CHAT)
    assert reply.text == "i"
    first, second = (q for _, q in fake.requests[:2])
    assert list(first["next"]["criteria"]) == ["the", "a"] and "other" in first
    assert list(second["next"]["criteria"]) == ["i", "is"] and "other" not in second  # the last page
    assert (reply.steps[0].page, reply.steps[0].source, reply.steps[0].other) == (1, "llm", 0.0)
    assert suggester.tops[0] == 200  # a deeper list, to fill the pages


async def test_jev_stays_on_the_page_that_has_its_word():
    fake = ScriptedJev(["a"])
    reply = await paging_writer(fake, FakeSuggester([RANKED]), llm_pages=2).write(CHAT)
    assert len(fake.requests) == 2  # one per word (the word, then the send), no page turns
    assert (reply.steps[0].page, reply.steps[0].other) == (0, pytest.approx(0.05))


async def test_own_page_comes_last_without_the_words_jev_passed():
    fake = ScriptedJev(["pizza"])
    reply = await paging_writer(fake, FakeSuggester([RANKED]), llm_pages=2, own_page=True).write(CHAT)
    assert reply.text == "pizza"
    menus = [list(q["next"]["criteria"]) for _, q in fake.requests[:3]]
    assert menus[:2] == [["the", "a"], ["i", "is"]]
    assert "pizza" in menus[2] and not {"the", "a", "i", "is"} & set(menus[2])
    assert "other" not in fake.requests[2][1]
    assert (reply.steps[0].page, reply.steps[0].source) == (2, "own")


async def test_only_the_first_page_carries_the_stop_check():
    fake = ScriptedJev(["the", "is"])
    await paging_writer(fake, FakeSuggester([RANKED]), llm_pages=2).write(CHAT)
    second_word = [q for _, q in fake.requests if q["next"]["instructions"]["jev_reply"] == "the ___"]
    assert ["send" in q for q in second_word] == [True, False]


def test_page_turn_is_sampled_with_jevs_probability():
    def turns(other, draw, temperature=0.7, power=1.0):
        cfg = WriterConfig(temperature=temperature, page_power=power)
        return JevWriter(ScriptedJev([]), config=cfg, vocab=VOCAB, rng=FixedRng(draw))._turns_page(other)

    assert turns(0.3, draw=0.2) and not turns(0.3, draw=0.4)
    assert not turns(0.3, draw=0.2, power=2.0)  # 0.3**2 = 0.09
    assert turns(0.5, draw=0.99, temperature=0) and not turns(0.49, draw=0.0, temperature=0)


async def test_known_only_keeps_jev_to_words_it_knows():
    suggester = FakeSuggester([Suggestion([("zebra", 0.9), ("i", 0.1)])])
    fake = ScriptedJev(["i"])
    await suggesting_writer(fake, suggester, known_only=True).write(CHAT)
    assert list(fake.requests[0][1]["next"]["criteria"]) == ["i"]  # zebra: never born with, heard or learned


async def test_known_only_falls_back_to_known_words_when_nothing_survives():
    suggester = FakeSuggester([Suggestion([("zebra", 1.0)])])
    fake = ScriptedJev(["i"])
    await suggesting_writer(fake, suggester, known_only=True).write(CHAT)
    menu = list(fake.requests[0][1]["next"]["criteria"])
    assert "zebra" not in menu and "i" in menu and "jev" in menu  # the static menu: chat words + born words


async def test_a_word_the_model_is_still_spelling_gets_finished():
    suggester = FakeSuggester(
        [
            Suggestion([("sand", 1.0)]),
            Suggestion([("wich", 1.0)], completes="sandwich"),
            Suggestion([("yes", 1.0)]),
        ]
    )
    fake = ScriptedJev(["sand", "yes"])
    seen = []

    async def on_word(text):
        seen.append(text)

    reply = await suggesting_writer(fake, suggester).write(CHAT, on_word=on_word)
    assert reply.text == "sandwich yes"
    assert seen[:2] == ["sand", "sandwich"]  # the live message is corrected in place
    assert [s.word for s in reply.steps] == ["sandwich", "yes"]


async def test_a_failing_suggester_falls_back_to_jevs_own_menu():
    class Broken:
        async def suggest(self, *args, **kwargs):
            raise JevError("provider down")

    fake = ScriptedJev(["i", "like"])
    reply = await suggesting_writer(fake, Broken()).write(CHAT)
    assert reply.text == "i like"
    assert "jev" in fake.requests[0][1]["next"]["criteria"]  # the static menu (chat + born words)


def test_fallback_menu_fits_one_question_even_with_a_big_born_vocabulary():
    writer = suggesting_writer(ScriptedJev([]), FakeSuggester([]), vocab_size=12, bucket_size=6)
    pool = writer._single_pool(CHAT, learned=["sushi"])
    assert len(pool) == 6 and pool[0] == "jev"  # chat words first, capped at bucket_size


def test_big_vocabulary_uses_the_tournament():
    assert not JevWriter(ScriptedJev([]), vocab=VOCAB, config=WriterConfig(vocab_size=5000)).single_call


async def test_on_word_sees_the_text_grow():
    fake = ScriptedJev(["i", "like", "pizza"])
    seen = []

    async def on_word(text):
        seen.append(text)

    await writer_for(fake).write(CHAT, on_word=on_word)
    assert seen == ["i", "i like", "i like pizza"]


async def test_draft_keeps_progress_when_cancelled():
    # requests: buckets, runoff | buckets, stop, runoff | buckets (hangs), stop (hangs)
    fake = ScriptedJev(["i", "like", "pizza"], hang_on_call=6)
    draft = Reply()
    task = asyncio.create_task(writer_for(fake).write(CHAT, draft=draft))
    while len(fake.requests) < 6:
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert draft.text == "i like"
    assert draft.usage.calls == 5
    assert draft.stop_reason == "writing"
