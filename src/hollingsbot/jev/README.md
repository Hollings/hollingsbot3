# Jev in Discord

Jev is TypeSafe's "System One" decision model, served through OpenRouter's
Decisions API. It cannot generate text: every request is a `state` plus typed
questions (`choice` over up to 255 options, `noul` yes/no, `score`), and every
answer is a probability distribution. JevBot makes it chat anyway, one word at
a time, in the channels listed in `JEV_BOT_CHANNELS`, and for a while in any
channel where someone types `!spawn jev` (below).

Code: `src/hollingsbot/jev/` (writer, decoding rules, lexicon, client, ledger)
and `src/hollingsbot/cogs/chat_bots/jev_bot.py` (Discord side). Try it from a
terminal with `python scripts/jev_try.py "Jev, what's your favorite food?"`
(add `--trace` to see every step's scores, `--samples` for the prompt set the
numbers below come from).

## Spawning Jev elsewhere (`!spawn jev [N]`)

`!spawn jev 10` (or `!spawn 10 jev`; N defaults to 10, at most 20 like a temp
bot) brings Jev into the current channel instead of an LLM temp bot. It answers
the last few messages at once (backfilled from Discord if the bot hasn't seen
the channel since it started; a quiet channel gets a check mark instead), then
every message, until it has posted N replies. Unlike in its own channels, that
includes other bots and webhooks (Wendy, temp bots), so Jev and Wendy talk to
each other until Jev's count runs out; it never answers itself or the Dog
account's command output, and it still learns words only from people. Then it
leaves with a temp bot's goodbye line
(`*[Jev drifts off into silence]*`). Only posted replies count: one cut off by
someone talking, skipped over budget or failed uses none.

- `!spawn jev2` brings a second Jev: each name in `JEV_COPIES` (default
  `Jev2`) is another JevBot with the same settings and brain but its own name
  (it's told it's Jev2), its own webhook, its own visits and no channels of its
  own. Two Jevs in a channel answer each other until both counts run out; a
  human message goes to whichever of them the coordinator tries first.
- `!despawn jev` (or `jev2`) ends a visit at once; `!despawn` lists them,
  `!despawn all` includes them. Spawning again resets the count. Plain `jev`
  always means the main Jev, even if `JEV_BOT_NAME` renames it.
- Visits are rows in `jev_visits` (the bot DB, keyed by channel and Jev), so a
  restart doesn't end one.
- A spawned channel is claimed like a `JEV_BOT_CHANNELS` one: the Jevs go first
  on every message, so a temp bot in the same channel mostly gets turns by
  answering a Jev. Jev's posts go through the coordinator like any webhook
  message (that's how another Jev hears them). The daily budget and the learned
  words are shared by every Jev and channel.
- Code: `JevBot.spawn` / `despawn` in `jev_bot.py`, the store in
  `jev/spawns.py`, the copies registered in `chat_coordinator.setup`;
  `temp_bot_commands.py` routes `!spawn jev` / `jev2` there.

## Pieces instead of words (`JEV_UNITS=pieces`)

With `JEV_UNITS=pieces` (`pieces.py`) Jev's menu is the LLM's raw next tokens,
word pieces and all ("chang", "ing", "cr", "..."), instead of the whole words
`suggest.py` makes of them, and Jev strings the reply together itself. Paging,
the own-words page, STOP on the menu and sampling work as below; there is no
naturalness check. Every Jev (copies included) writes the same way.

- **Joining.** Chat-mode tokens have no leading space, so a piece attaches to
  the last word when the two spell a dictionary or chat word or the start of
  one ("chang"+"e"), punctuation always attaches, anything else gets a space.
  It misfires ("SoI", "lay... ing"); that's kept on purpose.
- **No repeats (`JEV_NO_REPEAT`).** `never` (default): a piece used once is off
  every later menu. Without it replies spiral ("the the thee three...",
  "ng ing ng ng"); with it Jev runs out of easy pieces and reaches for strange
  ones ("har tober h fest val ival events"). `adjacent` only bans the piece just
  used and still loops ("get cold get cold getting"); `off` bans nothing.
- **Why chat mode.** OpenRouter's raw `/completions` would mark new words with a
  leading space, but for llama-3.1-8b it rejects `logprobs=N` and, asked with
  `logprobs=true`, wraps the prompt as a chat message anyway (2026-09-26).
- Measured locally 2026-09-26 (6 samples, live settings): 8-70 pieces, mean 28,
  $0.0014 per reply; 76% of pieces from the first page, 14% p2, 4% p3, 6% own.
  Long replies take ~50 s (a page turn is another request).
  `python scripts/jev_try.py --samples --units pieces --stop choice` reproduces it.

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
- **Shuffled (`shuffle`, on):** the menu reaches Jev in random order, so the
  LLM's ranking can't leak through position. Probed on 27 real mid-reply
  states: reordering moves Jev's distribution (mean TV distance 0.15-0.20 vs
  0.05 for the identical question asked twice) but not toward the LLM's
  favourite: its #1 word got 0.13 of Jev's probability listed first, 0.17
  listed last, 0.15 shuffled. Jev's own #1 is often not the LLM's ("hey",
  "well", "the" where the LLM led with "i", "paris").
- **Pages (`llm_pages`, `own_page`; off):** beside each page Jev is asked
  "pick one of `options`" vs "type a word that isn't in `options`", and
  turns the page with that probability: the LLM's next 20 (one call asks for
  its top 200 tokens), then its own menu minus everything it passed. A "none
  of these" option inside the word Choice never worked (P ~0.02 for any
  menu); the separate question averages 0.18 for the right chat's menu and
  0.33 for another chat's (higher for the wrong one in 25/27). Measured (9
  prompts x 2 seeds, `--llm-pages 3 [--own-page 1]`): 80-83% of words from
  page 1, 8-15% page 2, 3-6% page 3, 4-5% Jev's own page; ~$0.003/reply. With
  one LLM page and the own page, ~22% of words are Jev's own. Only Novita
  serves more than 20 logprobs for llama-3.1-8b (CoreWeave errors), so deep
  asks are pinned there and fall back to 20 if it fails or rate-limits.
  `jev_try.py --trace` shows each word's page and P(other).

~20-25 words and ~$0.002 per reply, ~0.3 s per word. Examples (strict, born
1000): *"yeah well like i think probably ramen like because it is more simple
than pizza and also like i like some different things"*; free choice: *"why
did the computer get cold because it lost windows"*, *"hey mallory im sure you
will do great and just be yourself remember just to dress professional and
bring copies"*. Born with only 500 words, strict mode collapses into "something
great big be ever food thing"; with an LLM prior blended in
(`suggest_weight` 0.5) it reads most fluently but least like Jev.

## Stopping (`JEV_STOP`)

- **`threshold` (default):** each word's request also asks "send the message
  as it is" vs "keep typing"; it sends at P(send) >= 0.6, and between 0.2 and
  0.6 with probability P(send)^2.
- **`sample`:** the same question, sampled like a word (temperature, no
  floor or ceiling).
- **`choice`:** STOP is an option on the word menu from `min_words` on,
  shuffled in with the words. Its fit Noul is "is the end of `text` a natural
  place to stop?", and the word question asks "which word goes in the blank,
  or is the message finished?" (the plain blank question gave STOP at most
  0.29 even at a finished sentence; this wording up to 0.49). No separate stop
  check. One-request modes only; the tournament keeps `threshold`.

Measured on the live LLM-pages config, 100-word cap (9 prompts x 2 seeds):

| stop | mean words (min-max) | $/reply | ended |
|---|---|---|---|
| threshold | 27.9 (9-68) | 0.0035 | 17 sent, 1 gave up |
| sample | 20.8 (8-46) | 0.0023 | all sent, often mid-sentence ("...ideas to name her for your") |
| choice | 35.6 (8-100) | 0.0049 | 14 STOP, 3 gave up, 1 cap |

With `choice`, STOP won with 0.06-0.30 of the menu's probability, at
natural endings ("paris is the capital of france yeah actually", "nothing
really i am just sleeping too..."); a ramble that never wants to stop runs
until the give-up rule or the cap ends it. Live since 2026-09-25.

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
