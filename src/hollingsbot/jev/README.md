# Jev in Discord

Jev is TypeSafe's "System One" decision model, served through OpenRouter's
Decisions API. It cannot generate text: every request is a `state` plus typed
questions (`choice` over up to 255 options, `noul` yes/no, `score`), and every
answer is a probability distribution. JevBot makes it chat anyway, one word at
a time, in the channels listed in `JEV_BOT_CHANNELS`.

Code: `src/hollingsbot/jev/` (writer, decoding rules, lexicon, client, ledger)
and `src/hollingsbot/cogs/chat_bots/jev_bot.py` (Discord side). Try it from a
terminal with `python scripts/jev_try.py "Jev, what's your favorite food?"`
(add `--trace` to see every step's scores, `--samples` for the prompt set the
numbers below come from).

## Opt-in: an LLM proposes, Jev chooses (`JEV_SUGGEST_MODEL=on`)

Off by default since 2026-09-25: it reads well, but every word then comes from
another model and Jev is only its filter.

Jev's weakness is word order; a small LLM's is nothing Jev cares about. So
per word, `llama-3.1-8b-instruct` (OpenRouter, ~0.4 s, max_tokens 1,
top_logprobs 20) is asked to continue the chat log, and its top next tokens
become Jev's menu for the next word (`suggest.py`). Jev then decides in one
request: which word, a naturalness Noul per option, the send/sense check.

- **Strict (default):** Jev only says proposals it knows: the 1000 most
  common words it is born with, the chat's words, and everything it has
  learned. Grammar is always available; "pizza" or "ramen" only after someone
  says them. `JEV_KNOWN_ONLY=0` lets it say any proposal (more on-topic, and
  learning stops mattering). `JEV_SUGGEST_MODEL=off` = Jev alone (below).
- **Word pieces:** LLMs predict tokens, not words. A 3+ letter piece that
  begins a rare word someone said becomes it ("ram" -> "ramen"); when the
  model's favourite next token glues onto the last word to spell a real word
  ("sand"+"wich", "don"+"'t") the last word is finished in place and the model
  asked again; contraction halves ("ll", "don") are never words of their own.
  (Assistant prefill would give exact word boundaries, but this provider's
  logprobs don't match the prefilled position.)
- If the LLM call fails, that word comes from Jev's own menu instead.

~20-25 words and ~$0.002 per reply, ~0.3 s per word. Examples (strict, born
1000): *"yeah well like i think probably ramen like because it is more simple
than pizza and also like i like some different things"*; free choice: *"why
did the computer get cold because it lost windows"*, *"hey mallory im sure you
will do great and just be yourself remember just to dress professional and
bring copies"*. Born with only 500 words, strict mode collapses into "something
great big be ever food thing"; with an LLM prior blended in
(`suggest_weight` 0.5) it reads most fluently but least like Jev.

## Default: a lobotomized parrot that learns (one request per word)

Jev is born knowing only the 100 most common English words (almost all glue:
the, is, you, like, good). Every word a human says in its channel is learned
for good (`jev_replies`' sibling table `jev_lexicon`: uses, first/last heard,
who taught it); `!jev` shows the count, the newest words and their teachers.

Each reply draws on a menu of at most 250 words, so it fits one Choice and a
word costs **one request** (~0.4 s): the 100 born words always, then the chat's
own words (newest message first), then learned words ranked by uses x
0.5^(days since last heard / 7). Words nobody says fade out of reach but are
never deleted. The same request carries a naturalness Noul per option and the
send/sense stop check; its state is the chat alone and the reply-with-blank
rides in the Choice's own instructions. Calls per reply = words + 1; a reply
costs ~$0.003-0.006.

Measured (8 prompts, style line on; `JEV_VOCAB_SIZE`, `JEV_FLUENCY_CHECK`):

| born vocabulary | naturalness check | mean words | $/reply | calls/word |
|---|---|---|---|---|
| 5000 (tournament, below) | on | ~20 | 0.035 | 3 |
| 200 | on | 10-16 | 0.006-0.010 | 1 |
| 200 | off | 9-13 | 0.001 | 1 |
| **100 (default)** | **on** | **17** | **0.006** | **1** |
| 100 | off | 13 | 0.0008 | 1 |

In a scripted conversation it picked up "ramen", "pizza", "spicy", "garlic"
from the chat and answered "do you like ramen or pizza more" with *"well i
like ramen but more pizza than ramen"*. The sense cutoff is off in this mode
(it killed such replies at word 4: a 100-word vocabulary reads as half-nonsense
even when it is going somewhere); hiding Jev's own earlier replies from it made
it parrot other people's sentences verbatim, so they stay in the context.

## The big-vocabulary tournament (`JEV_VOCAB_SIZE` >= 250)

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

Both questions also carry a style line, by default *"Jev writes long, chatty
messages, a few sentences at a time."* (see Length below).

From word 8 on, the reply ends when P(send) >= 0.6, or with probability
P(send)^2 when it is between 0.2 and 0.6 (Jev's calibrated stop probability,
sampled like an LLM's end token). At any length it ends when the reply stops
making sense (sense < 0.2, from word 3), when even the picked word is rated
unnatural (< 0.25), or at 40 words.

Code-side decoding rules, the same knobs any LLM sampler has: no word twice in
a row, no repeated three-word run, no punctuation first or twice in a row, and a
repetition penalty (content words x0.3 per earlier use; common words x0.5 per
use in the last 6 words).

Cost: billed per input token ($0.042/M), and the 5000-word bucket request is
~35k tokens, so about $0.0017 per word, i.e. cost is linear in reply length.
With the defaults a reply averages ~20 words and ~$0.035; the 40-word cap is
~$0.07. `jev_replies` logs every reply with its cost and per-word trace;
`JEV_DAILY_BUDGET_USD` stops Jev for the rest of the UTC day (it reacts with a
zzz instead).

## Length

Left to itself Jev answers the question and sends: the first live replies were
"hi bro", "hey dog", "pizza" (1.7 words, $0.004). Levers compared on the same 8
prompts (seed 3, cap 50):

| config | mean words | $/reply | effect |
|---|---|---|---|
| none | 7.2 | 0.014 | "hi", "pizza", "yes" |
| min 12 words | 15.5 | 0.028 | pads after the answer: "pizza is amazing really incredibly best ever literally world's earth universe planet" |
| stricter send (0.85 / floor 0.4 / power 3) | 10.9 | 0.020 | short answers stay short, rambles get longer |
| style line | 16.9 | 0.031 | "pizza is my favorite food because it's delicious. especially cheese" |
| style + min 10 | 24.8 | 0.044 | longest; rambles hit the cap |

The style line works by changing what Jev says (reasons, follow-ups), not just
when it stops. Across seeds it varied 8-17 words because the sense check cut
drifting replies at 0.3; the shipped default (style + min 8 + sense 0.2, cap
40) measured 17.0 and 21.5 words on two seeds.

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
