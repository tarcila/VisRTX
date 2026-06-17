# tsdFlow Phase 2 — Async Evaluator Design

**Date:** 2026-06-17
**Status:** Approved design, pending implementation plan
**Builds on:** Phase 1 (`docs/superpowers/specs/2026-06-16-tsd-dataflow-pipeline-app-design.md`, headless `tsd_graph` engine — committed, 16 tests green)

## Summary

Make the `tsd_graph` `Evaluator` asynchronous: `pull` schedules the lazy-pull
`ensure()` DFS on a single background worker, keeps the calling thread free, and
supports cooperative cancellation with **edit-cancels-in-flight** semantics,
completion-gated result visibility, a `Computing` state, per-pull `EvalReport`
accumulation, and error isolation. Still headless — no UI, no ANARI device, no
CUDA. Validated with synthetic slow/cancellable nodes.

## Decisions (locked)

| Topic | Decision |
|-------|----------|
| Worker | One background worker via `tsd::core::TaskQueue` (header-only, in `tsd_core`; keeps `tsd_graph` core-only). One task per pull runs the whole `ensure()` DFS — not one task per node. |
| Concurrency contract | **Edits cancel in-flight, then serialize.** A new `pullAsync`/`cancel()` sets an atomic cancel flag; the running task checks it between nodes and bails; the next task starts only after (FIFO single worker). The worker is the **sole** mutator of node cache/state, so **no snapshot** is needed. |
| Mutation rule | The owning thread mutates the `Graph` (param set, connect, delete) only while no pull is in flight: call `cancel()` then `waitIdle()`, mutate, then re-pull. Documented contract; `waitIdle()` makes it enforceable. |
| Async API | `pullAsync(id, onComplete?)` → `PullHandle`; `isReady(handle)` / `result(handle)` poll on the main thread; optional `onComplete(bool)` callback fires from the worker. |
| Sync compat | `bool pull(NodeId)` retained = `pullAsync` + `waitIdle` + `result`. All Phase 1 tests pass unchanged. |
| Cancellation surface | `EvalContext::cancelled()` for cooperative bail inside long node `evaluate()`; `ensure()` checks cancellation before evaluating each node. |
| Debounce | Out of scope for the engine — a caller/UI concern (Phase 4). The engine only provides cheap cancel + re-pull. |
| Publish | Completion-gated: results are read only after `isReady(handle)`; the worker is the sole writer, the main thread the sole reader-after-ready. No mid-flight concurrent reads of cache in Phase 2 (state reads mid-flight are advisory only). |

## Architecture

### Evaluator changes (`tsd/src/tsd/graph/Evaluator.hpp/.cpp`)

New state:
- `tsd::core::TaskQueue m_worker{N}` — one worker thread.
- `std::atomic<uint64_t> m_epoch{0}` — incremented per `pullAsync`; identifies the
  latest requested pull.
- `std::atomic<bool> m_cancel{false}` — set by a new pull or `cancel()`; checked by
  the running task.
- per-handle completion record: `std::map<uint64_t, {bool done; bool ok;}>` (or a
  small ring) guarded by a mutex, written by the worker, read by `isReady`/`result`.
- `tsd::core::Future m_lastFuture` — the in-flight task's future, used by `waitIdle`.

New API:
```cpp
struct PullHandle { uint64_t id{0}; };

PullHandle pullAsync(NodeId id, std::function<void(bool)> onComplete = {});
bool isReady(PullHandle h) const;   // task finished or was superseded
bool result(PullHandle h) const;    // success (valid once isReady); false if superseded/failed
void cancel();                      // request cancel of any in-flight pull
void waitIdle();                    // block until the worker is idle (safe to mutate/destroy)

bool pull(NodeId id);               // blocking: pullAsync + waitIdle + result (Phase 1 API)
```

`pullAsync(id, cb)`:
1. `uint64_t e = ++m_epoch;`
2. `m_cancel = true;` (ask any running task to bail)
3. enqueue one task on `m_worker` capturing `e`:
   - at task start: `m_cancel = false;` (this is now the desired work; safe because
     the prior task has fully returned — FIFO single worker)
   - `m_report.clear();`
   - `bool ok = ensure(id, e);`
   - record `{done:true, ok: (e==m_epoch) ? ok : false}` for handle `e`
   - if `cb` and not superseded: `cb(ok)`
4. return `PullHandle{e}`.

`ensure(id, epoch)` (Phase 1 body + cancellation):
- at entry and before each child/own `evaluate()`:
  `if (m_cancel.load() || epoch != m_epoch.load()) return false;` (cancelled/superseded)
- otherwise unchanged: dirty/version/param-hash recompute decision, sets
  `Computing`, runs `evaluate()`, records consumed versions, bumps `outputVersion`,
  stamps output versions, sets `Clean`. On node error → `Error`, returns false.

`EvalContext::cancelled()`:
```cpp
bool EvalContext::cancelled() const { return m_eval.cancelRequested(); }
```
where `Evaluator::cancelRequested()` returns `m_cancel.load()`. A long node loops
`if (ctx.cancelled()) return;` and leaves no partial output (its cache was cleared
at recompute start; an incomplete `setOutput` is discarded because `ensure` returns
false before finalizing versions — see error isolation).

### Why no snapshot / no per-node tasks

- Single worker + FIFO + cancel-before-enqueue means at most one task touches node
  state at a time, and the main thread only reads after `isReady`. The live `Graph`
  is therefore never concurrently mutated by two threads.
- One task per pull (not per node) keeps the DFS recursion intact from Phase 1 and
  avoids a scheduler; concurrency across independent nodes is deferred (future work,
  noted in Phase 1).

## Error handling

| Case | Behavior |
|------|----------|
| Node `evaluate()` fails | `ensure` returns false (Phase 1) → task records `ok=false`; downstream not published; last good cache untouched. |
| Cancelled mid-DFS | `ensure` returns false without finalizing the in-flight node's version; node left non-`Clean`; `result(handle)` is false; a later pull re-runs it. Not an error. |
| Superseded pull | A handle whose epoch != `m_epoch` reports `result=false`; its `onComplete` (if any) is suppressed. |
| Destruction while in flight | `~Evaluator` calls `cancel()` then relies on `TaskQueue`'s destructor (joins the worker) so no task outlives the Evaluator. `waitIdle()` available for explicit teardown. |
| Mutating the Graph during in-flight pull | Contract violation; callers must `cancel()`+`waitIdle()` first. Documented; `waitIdle()` provided. |

## Testing (headless, no CUDA)

New tests in `tsd/tests/`:

1. **Async completion** (`test_graph_AsyncEval.cpp`): `pullAsync` a 2-node chain;
   poll `isReady` in a loop until true; `result` is true; `output` matches the
   synchronous result; an `onComplete` callback fires exactly once with `true`.
2. **Cooperative cancellation** (`test_graph_AsyncCancel.cpp`): a `SlowNode` whose
   `evaluate()` spins on `ctx.cancelled()` (bounded by a max-iteration guard so the
   test can't hang). Start a pull; call `cancel()`; the pull's `result` is false and
   the node did not finalize. Then a fresh `pull()` (blocking) completes normally —
   proving the worker recovered.
3. **Edit-cancels-in-flight** (`test_graph_AsyncSupersede.cpp`): start `pullAsync` of
   a slow chain; immediately `pullAsync` again after a param edit (with
   `cancel()`+`waitIdle()` around the edit); the first handle reports superseded
   (`result=false`), the second produces the updated value. Use an eval counter to
   confirm the superseded run did not publish stale output.
4. **Error isolation** (`test_graph_AsyncError.cpp`): a node that throws/sets error;
   `pull` returns false, `result(handle)` false, sibling branch still resolves, and a
   subsequent corrected pull succeeds.
5. **Sync compat**: the existing 16 Phase 1 tests remain green (the retained blocking
   `pull()`), re-run as the suite gate.

A `SlowNode` test helper (busy-wait with a bounded iteration cap that checks
`ctx.cancelled()`) lives in the test files — no real sleep/clock in the engine
(keeps it deterministic and avoids `Date.now`-style nondeterminism).

## Out of scope for Phase 2

- Concurrent multi-node evaluation / thread pool (single worker only).
- contentTag-based version short-circuit (Phase 2+ optimization, noted in Phase 1).
- Cache eviction / budget (Phase 2+ item, still deferred).
- Real CUDA work and kernel-level cancellation (Phase 4; Phase 2 cancellation is
  cooperative at node granularity, which is the correct model for when kernels
  arrive — already established in Phase 1 spec).
- UI debounce, the render bridge, viewports (Phase 3+).

## As-built deviations (post-implementation, 2026-06-17)

- **Cancellation uses an epoch counter, not a bool.** TDD exposed a race in the
  originally-specified `bool m_cancel` (worker reset it to false at task start,
  stomping an owner-thread `cancel()` issued before the worker woke). The shipped
  design uses `std::atomic<uint64_t> m_cancelEpoch`: `cancel()` stores the current
  `m_epoch`; `cancelRequested() == (m_cancelEpoch >= m_epoch)`; the worker NEVER
  writes `m_cancelEpoch`. Supersession is handled solely by the existing
  `epoch != m_epoch` check. Reviewed and verified correct in all interleavings.
- **Completion publication ordering:** the worker stores `m_doneOk` *before*
  `m_doneEpoch` (both seq-cst), so any reader gated on `m_doneEpoch >= h.id` sees a
  tear-free `(epoch, ok)` pair plus all prior non-atomic graph-state writes.
- **Unconnected required input is NOT an eval-time error.** `EvalContext::input()`
  on an unconnected port returns an invalid `Value` (the node sees
  `in.valid()==false`); the missing-required-input → `Error` transition happens at
  topology-change time via `revalidateRequiredInputs` (Phase 1), not during a pull.
  The async error-isolation test drives the real path (connect, then `removeNode`
  the producer).

## Phasing note

This is Phase 2 of the 5-phase delivery from the Phase 1 spec. It depends only on
the committed Phase 1 engine and stays headless. Phase 3 (render bridge + viewports)
builds on the async evaluator next.
