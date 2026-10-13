# Codebase review and repair

Reviewed October 12, 2026 (America/Los_Angeles), against Git baseline `4852f15`.

This review examined the production bot, cogs, database transactions, Celery
workers, provider adapters, tool execution, summarization, image processing,
configuration, tests, and development checks. The emphasis was reproducible
failures, concurrency, resource ownership, and removing duplicated work. This
report replaces the older review, which referenced removed modules and included
unsupported claims such as JSON parsing permitting code injection.

## Findings repaired

| Area | Failure or inefficiency | Applied fix |
| --- | --- | --- |
| Token accounting | Concurrent token deductions could both spend the same balance; redemption could debit tokens without granting credits. | Acquire a SQLite write transaction before checking balances; perform token redemption and credit grant in one transaction. Test simultaneous spend and rollback on credit-write failure. |
| Image accounting | Accounting cogs could use a different default database or stale budget. Concurrent image requests could all pass affordability checks. Partial batches were delivered without being charged. | Share the image cog's tracker and configured DB; serialize generation per user within the bot process; charge only successfully delivered prompt variants. |
| Price and budget configuration | NaN, infinity, negative prices, and malformed reloads could produce invalid accounting or partially replace live settings. `!models` failed on omitted prices. | Validate finite amounts and generator fields before changing live state; retain previous configuration on failure; use the default price when the per-model price is absent. Apply reduced budget caps immediately and avoid promising impossible accrual waits. |
| Celery integration | Publishing could block Discord's event loop. Duplicated polling paths handled cancellation inconsistently, including cancellation during submission. | Consolidate publishing, polling, result retrieval, timeout handling, and revocation in `utils/celery_utils.py`; run blocking broker calls in threads. |
| Worker resource lifecycle | Image generators were closed on a different event loop; cached SDK clients survived worker loops. Factory/output/disk errors could leave prompt status incorrect. | Generate and close on the same loop, close cached text/SVG clients before loop teardown, validate output, and record terminal failures for all these error paths. |
| Text generation | A broad `TypeError` fallback could retry a programming error as another paid generation and bypass normal failure handling. | Remove the retry; preserve the original failure and normal logging/cleanup. Enforce the text timeout for plain text tasks too. |
| Temporary bot replies | Cancellation could lose a reply reservation, refund a delivered reply, or continue into another bot's response. | Propagate cancellation and give each request ownership of its refund. Cover cancellation during preparation, before the child starts, during generation, and after delivery. |
| Bot shutdown/retry | Startup retries reused a closed Discord client across event loops; background workers and HTTP clients lacked coordinated cleanup. | Retry on one loop, reopen the client, stop restart/summary/generation tasks, and close owned clients. Scheduled restart exits after cleanup. |
| Image child tasks | Cancelling during emoji lookup could leave image generation running. Early rejected requests still started emoji generation. | Start optional emoji work after validation, and cancel/await both child tasks when the request ends. |
| Chat history | Concurrent events could prepare responses before the previous response was registered. A slow backfill could restore messages after `!clear`. | Serialize preparation through response registration, release the lock before awaiting the reply, recheck the clear point after backfill, and ignore delayed events older than the clear point. |
| Tool context | A mutable global context could mix identities/channel context between concurrent tool calls. | Use task-local `ContextVar` storage and copy values at the boundary. |
| URL previews | User URLs and redirects could reach private network addresses; a missing or misleading Content-Length bypassed download limits. | Validate HTTP(S) destinations and DNS results at connection time, validate every redirect, bound decoded streamed bytes, and apply a deadline to the entire chain. |
| Preview performance | HTML parsing and image decoding ran on the event loop; duplicate URLs were fetched repeatedly. | Move CPU work into threads and deduplicate links before fetching. Match Twitter/X by hostname instead of arbitrary substrings. |
| Image generation overhead | Plain text-to-image requests downloaded unrelated edit/history images. | Fetch editing inputs only for edit/outpaint modes. |
| Outpaint/image edges | Mask files closed before Replicate uploaded them; very narrow or one-pixel images could be resized to zero pixels. | Keep mask handles open through both single and multi-image requests and clamp dimensions to at least one pixel. |
| SVG repair | Whole-document HTML unescaping turned escaped text into markup and broke quoted attributes. | Decode named entities individually while retaining XML escaping. |
| Summaries | Triggers arriving during a summary run could be discarded; empty provider output could mark a group summarized. Context retrieval loaded the entire summary archive. | Coalesce pending triggers, reject empty summaries, move worker DB calls to threads, and query only the most recent context groups. Remove the unused Python filtering helper. |
| Jev state | Failed schema transactions could mark initialization complete; reply reservations could race; low temperatures could underflow every sampling weight. | Cache schema readiness after commit, reserve replies under a write transaction, and normalize scores before exponentiation. |
| Smaller lifecycle issues | SQLite connections were not always closed, failed starboard sends prevented retries, and overlapping streak events could duplicate announcements or erase a new streak. | Close/reset connections explicitly, release failed starboard reservations, and remove completed streak state before awaiting delivery. Reject nonpositive message chunk limits. |
| Development checks | The development extra did not install the security scanner required by `make check`. | Include `bandit[toml]` in the development dependencies and installation target. |

## Verification

The baseline suite passed **517 tests**. The expanded suite contains **590
passing tests**, including regression tests for transaction rollback and races,
task cancellation, startup retry, loop-bound cleanup, malformed configuration,
partial image delivery, private-address redirects, bounded streaming, mask
lifetimes, SVG escaping, and low-temperature sampling.

- Ruff lint and formatting checks pass for `src`, `tests`, and `scripts`.
- Bandit reports no findings using the existing project configuration. Its
  configured rule exclusions still apply; this is not a complete security audit.
- Mypy reports **107 errors in 29 files**, versus **113 errors in 29 files** on
  the unchanged baseline using the same installed dependencies. Comparing
  diagnostics with line numbers normalized found no newly introduced errors.
  `make check` therefore still fails at its type-check stage.
- Combined statement/branch coverage is approximately **48%**. Accounting,
  decoding, summary storage, and Jev behavior have substantially better coverage
  than several legacy bot/provider and Discord presentation paths.
- Git whitespace validation passes when treating the checkout's CRLF line
  endings correctly (`git -c core.whitespace=cr-at-eol diff --check`).

Checks were run on Windows with Python 3.13.2 and the existing local environment.
Provider requests, Discord delivery, and Celery broker behavior were mocked in
tests; no paid provider calls, live Discord posts, or deployment were performed.
Python 3.10/Linux container behavior still needs deployment-environment testing.

## Remaining constraints and follow-up work

1. **Type-check backlog:** Most diagnostics involve Discord channel/user unions,
   dynamically registered cog attributes, Pillow annotations, and payload types.
   They were retained rather than hidden with broad ignores or casts. A focused
   type-contract cleanup can make `make check` fully green.
2. **Live integration coverage:** Exercise reconnect/restart, worker revocation,
   image fallback, webhook delivery, and shutdown against a staging Discord bot
   and Redis/Celery deployment before production rollout. Coverage does not yet
   establish every provider path or platform combination.
3. **Accounting boundaries:** Per-user image serialization is scoped to one bot
   process. Multiple bot replicas would need a persistent reservation mechanism.
   A process crash between Discord delivery and billing can still leave an
   uncharged image; solving that requires durable delivery/accounting state.
4. **External cancellation:** Revocation is best effort. A provider request
   already accepted remotely may still complete and incur cost even after the
   local task or worker is stopped.
5. **Scale and retention:** Recent summary context is now bounded, but discovering
   unsummarized groups still examines historical records. Large archives would
   benefit from incremental checkpoints and an explicit retention policy for
   generated files, logs, and archived messages. This review does not delete user
   history or choose a retention policy.

All changes are left in the working tree for review. The pre-existing change to
`docker-compose.orange-pi.yml` was preserved. No commit, push, or deployment was
made.
