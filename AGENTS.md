# Review stateful workflows

When a change touches retries, cancellation, persistence, quotas, deployment,
or an external send, review the affected workflow using
[docs/stateful-workflows.md](docs/stateful-workflows.md).

Identify the owner and source of truth, the success/terminal condition, and what
happens if execution stops between two steps. Consider concurrent requests,
replayed events, late results, and restart recovery. Report a concrete failing
sequence rather than merely labeling code a state machine.

Add a targeted regression test for a demonstrated bug, using fake providers,
barriers, clocks, or temporary databases. Test the observable outcome across the
failure boundary, not just that a status field changed. Avoid framework rewrites
and blanket state-machine lint rules. Keep the workflow inventory current when
its contracts change. Run the relevant tests and the main test suite before push;
the production timer can deploy a push before GitHub CI finishes.

Do not resume Jev Builder or restore its public routes without an explicit user
request. Its containers and public endpoints are disabled.
