# Jev in Discord

Jev is TypeSafe's "System One" decision model, served through OpenRouter's
Decisions API. It cannot generate text: every request is a `state` plus typed
questions (`choice` over up to 255 options, `noul` yes/no, `score`), and every
answer is a probability distribution. JevBot makes it chat anyway, one word at
a time, in the channels listed in `JEV_BOT_CHANNELS`.

Code: `src/hollingsbot/jev/` (writer, decoding rules, client, ledger) and
`src/hollingsbot/cogs/chat_bots/jev_bot.py` (Discord side). Try it from a
terminal with `python scripts/jev_try.py "Jev, what's your favorite food?"`
(add `--trace` to see every step's scores, `--samples` for the prompt set the
numbers below come from).

## How a reply is written

Per word, two rounds of requests (~0.5 s each):

1. **Buckets, and concurrently a stop check.** The vocabulary is the words of
   the last few chat messages (their own bucket, so topic words are always on
   the menu) plus the 5000 most common English words (`vocab_en.txt`, from
   `wordfreq`), cut into buckets of 250. Every bucket is a Choice
   (*"which word goes in the blank?"*, with the reply so far ending in `___`),
   all in one request. The stop check is a separate request: a Choice between
   *send* and *keep typing*, and a Noul *does this make sense as a reply?*
2. **Runoff.** Each bucket's leaders (until 60% of its probability, at most 4)
   meet in one Choice, asked with one Noul per finalist: *is `next` a natural
   next word right after `text`?* Score = runoff probability x fluency^2 x a
   repetition penalty, then temperature 0.7 + nucleus 0.6 sampling.

The reply ends when P(send) >= 0.6, or with probability P(send)^2 when it is
between 0.2 and 0.6 (Jev's calibrated stop probability, sampled like an LLM's
end token); when the reply stops making sense (sense < 0.3, from word 3); when
even the picked word is rated unnatural (< 0.25); or at 25 words.

Code-side decoding rules, the same knobs any LLM sampler has: no word twice in
a row, no repeated three-word run, no punctuation first or twice in a row, and a
repetition penalty (content words x0.3 per earlier use; common words x0.5 per
use in the last 6 words).

Cost: billed per input token ($0.042/M), and the 5000-word bucket request is
~35k tokens, so about $0.0017 per word. Short answers cost $0.003, a typical
reply ~$0.013, the 25-word cap ~$0.045. `jev_replies` logs every reply with its
cost and per-word trace; `JEV_DAILY_BUDGET_USD` stops Jev for the rest of the
UTC day (it reacts with a zzz instead).

## The bake-off (2026-09-25)

The same prompts through each method, sampled (T=0.7) unless noted:

| method | how | typical output |
|---|---|---|
| letters | one Choice per character (a-z, space, punctuation, end) | `'pasacn n r   c  opiip t '`; greedy: `'aaaaaaaaaaaa...'` |
| parallel letters | k character positions per request, each question isolated | `'i  y   nnn p nl. yn.o ni  ..ktsunonp.n'` |
| chunks | one Choice over ~250 common letter chunks (th, ing, er...) | `'likeusthatjustpjustghlohave...'` |
| words, small list | one Choice over ~200 common words + the chat's words | `'the capital of france is the capital france capital france...'` |
| **words, tournament** | the method above | `'paris'`, `"hey mallory you breathe deep slowly out longer than in. count 4"` |

Jev judges meaning, not spelling: after "i love pi", `z` is not in its top six
letters. Word-level choice works because "which word comes next" is a
judgment; the small word list fails because the word it wants is usually not on
the menu, so it loops.

## Findings that shaped the tournament

- **Bucket winners are sensible, their order is not.** Plain tournament output
  was on topic but scrambled (`'my is favorite food pizza'`, `"i'm am"`):
  Jev knows what the answer is about, not which word comes next. A per-finalist
  naturalness Noul fixed most of the order.
- **Ask "is `next` natural after `text`", not "is `text` fluent".** Judging
  the whole reply means one early slip drags every later candidate down equally,
  and then choice probability alone free-associates into lists
  (`...seattle st washington... port shore roof...`).
- **Never show the stop question a trailing blank.** With the reply shown as
  `"... ___"`, "is this complete?" scored a finished explanation 0.09; the same
  text without the blank scored 0.57. Jev reads the blank literally as "another
  word is coming". The stop check therefore has its own request whose state is
  the chat alone.
- **"Send vs keep typing" beats "is it complete?"** A Choice between the two
  actions separated good endings (0.65-0.93) from rambles (0.15-0.42) a little
  better than the absolute Noul, and mid-sentence it sits below 0.2.
- **Nucleus sampling matters.** Runoffs have ~80 options with a long flat tail;
  temperature alone kept picking 1-4% words ("supreme jersey"). top_p 0.6 fixed
  it; 0.8 was noticeably more unhinged.
- **Cost structure:** the `state` is billed once per request (extra questions
  cost only their own ~14+ tokens), so Nouls are nearly free and chat context
  is cheap; the option lists are what cost money.
- Fan-out is free on time: 30 questions in one request answer as fast as one.

Known quirks, left as charm: it echoes the question back sometimes ("hey bro
dude what's your favorite food"), swears, and occasionally announces a joke and
then sends.
