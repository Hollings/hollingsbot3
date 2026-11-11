# Stateful workflow review

Use this inventory to scope reviews to the workflows a diff changes. Most flows
need a clear owner and a few invariants, not a state-machine library.

| Workflow | Source of truth / owner | Review boundary |
| --- | --- | --- |
| Orange Pi deployment | Deployment script and last healthy revision | Build failure must leave the previous code running; failed startup must retry on the next timer tick even if checkout already matches remote. Record success after readiness. |
| Restart command | Restart owner and persisted cooldown | Concurrent calls cannot schedule overlapping restarts. Fallback must restart the supervisor rather than merely kill its child. |
| Channel clear | SQLite channel clear watermark | Cutoff only advances. In-memory history, backfill, recall, and summarization must respect it. Late provider output cannot restore cleared context. |
| Chat reply | Coordinator and active generation registration | Ownership lasts through delivery, including typing delays. Cancel/clear must stop unsent output. A visible partial reply must not trigger another bot's fallback. |
| Temporary bot reply | Reply reservation, active webhook ownership, persisted remaining count | Opening and normal replies obey the same cancellation/refund rules. Reaper cannot delete an in-flight webhook. Failed cleanup remains retryable. |
| Jev visit | Persisted visit and current visit identity | Completion from an older visit cannot consume a replacement visit's replies or delete it. |
| Image request | SQLite budget reservations, request owner, separately identified provider attempts, delivery accounting | Admission reserves the whole batch atomically; no user lock spans provider I/O. Delivery consumes its reservation and debits in one transaction. Failure/cancellation releases only that request's remaining holds. Expired holds cannot authorize a late send. Replay cannot debit a delivered variant twice. |
| Summary generation | Cached messages, clear cutoff, conditional summary insertion | Check cutoff before provider call and when saving its result. Group boundaries restart after clear. Legacy recall summaries without provenance must not reintroduce cleared context. |
| Starboard | In-flight protection and post log | Preserve overlap protection and retry after failed send. Durable replay protection is a future improvement if duplicates after restart become a problem. |

## Questions for an affected workflow

1. Who owns the operation from admission through its last external effect?
2. Which state survives restart, and which fields can disagree?
3. At each await or durable write, what happens on cancellation, crash, or timeout?
4. Can a retry, duplicate event, or stale completion repeat an external effect?
5. Does failure remain eligible for recovery, or silently become success/terminal?
6. Does a test reproduce the sequence and assert user-visible output/accounting?

High-value examples are: failed deploy then another timer tick at the same SHA;
clear at a newer message then an older message and restart; clear while a provider
is running; cancel after provider completion but before Discord send; a failed
SVG generation causing neither a success reaction nor a quota debit; and a
delivered first image followed by failure on the next variant.

Use fake external clients and temporary databases. Use events/barriers for races
rather than sleeps. Existing pytest coverage under `tests/` runs in CI and the
pre-commit/pre-push hooks. Linux CI also exercises shell deployment tests; local
Windows runs may need Git Bash for these. An ordinary review should inspect only
affected flows; a periodic broad audit can use parallel reviewers when requested.

## Scope and limits

Image requests from the same user may run concurrently, subject to worker capacity.
A slow provider previously held the user's lock across generation and delivery,
silently blocking their next request even after redeeming credits. Short SQLite
transactions now reserve available funds before dispatch; holds are shown in
`!usage`, and only acknowledged deliveries become charges. A batch is admitted
all-or-nothing, while each delivered variant settles independently.

Reservations expire after 15 minutes, longer than the worker timeout. Startup
recovery and subsequent budget reads/admissions reclaim expired holds and mark
unfinished jobs as outcome unknown. A crash between claiming a request and
reserving funds spends nothing; persisted request IDs prevent replay. A crash
after Discord accepts a send but before settlement remains an unknown outcome,
with no automatic resend or debit. Worker capacity still limits simultaneous
provider calls; it is independent of budget admission.

The opt-in `$$$$` art-direction pass runs after request claiming and budget
reservation, before image dispatch. It uses one bounded text call for the batch;
invalid output or timeout falls back to the original prompts without retrying.
Cancellation propagates to the text call and releases the request's holds;
replayed Discord events cannot repeat the preprocessing call. Original prompts
remain in the request record and captions. The rewritten prompts are sent only
to the image provider, and are not yet persisted for exact reproduction.

Jev Builder is disabled and excluded from this remediation. Do not enable it as
part of a main-bot deployment. Its data is preserved for a future explicit resume.

Do not claim exactly-once delivery across Discord and SQLite: Discord may accept
a send before the client observes its result. Keep unknown outcomes distinct from
definite failure, and do not blindly issue a paid fallback on an unknown outcome.

The dormant legacy tool parser, offline experiments, and simple request-local
pipelines do not warrant a generic lifecycle framework. Broaden their design only
when they become active workflows or a concrete failure warrants it.
