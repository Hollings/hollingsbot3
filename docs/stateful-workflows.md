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
| Image request | Request owner, separately identified provider attempts, delivery accounting | A file written by a worker is not a delivered image. Error placeholders cannot count as success. Timeout means outcome unknown unless cancellation is confirmed. Replay cannot debit a delivered variant twice. |
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

Jev Builder is disabled and excluded from this remediation. Do not enable it as
part of a main-bot deployment. Its data is preserved for a future explicit resume.

Do not claim exactly-once delivery across Discord and SQLite: Discord may accept
a send before the client observes its result. Keep unknown outcomes distinct from
definite failure, and do not blindly issue a paid fallback on an unknown outcome.

The dormant legacy tool parser, offline experiments, and simple request-local
pipelines do not warrant a generic lifecycle framework. Broaden their design only
when they become active workflows or a concrete failure warrants it.
