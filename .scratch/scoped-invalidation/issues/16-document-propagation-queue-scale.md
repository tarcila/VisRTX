# 16: Assess and document propagation-queue deduplication scale

**What to build:** Document the current PropagationQueue deduplication cost and
trade-off. This ticket is documentation only: do not change the algorithm or
add instrumentation/benchmarks. State that scanning pending edges is linear
per insertion and can make distinct-edge insertion quadratic. Do not claim
queues stay small without evidence; future optimization requires representative
measurements.

**Blocked by:** None

**Status:** resolved

- [x] Comments accurately describe per-insertion and aggregate distinct-edge insertion complexity
- [x] Comments distinguish deduplication savings from cost, and note that no small-queue bound or measured performance guarantee is established
- [x] Queue algorithm, synchronization, reference lifetimes and behavior are unchanged
- [x] Documentation/format validation passes; no unmeasured optimization is introduced

## Comments

Triage approved by user: accept documentation only. Record honest complexity
and avoid both blind optimization and unsupported claims about scene sizes.

Final Standards review: optional performance/documentation nit. PropagationQueue::push scans the queued edges linearly; no benchmark or correctness defect was reported.

### 2026-10-03 iteration — resolved
Added a comment beside the deduplication scan in `devices/rtx/device/utility/ChangeKind.h`:
worst-case O(p) comparisons per insertion and O(n^2) total for n distinct edges
queued from empty without draining. Documented saved entries/propagation work,
no established small-queue bound or measured guarantee; future optimization
requires representative measurements. No runtime changes.
Verified: `clang-format --dry-run --Werror --lines=174:181 devices/rtx/device/utility/ChangeKind.h` → passed.
Verified: `diff -u <(jj file show -r @- devices/rtx/device/utility/ChangeKind.h | grep -v '^ *//') <(grep -v '^ *//' devices/rtx/device/utility/ChangeKind.h)` → identical non-comment content.
Verified: `jj diff -- devices/rtx/device/utility/ChangeKind.h .scratch/scoped-invalidation/issues/16-document-propagation-queue-scale.md` → reviewed comment accuracy against `push`, `take`, and `DeviceGlobalState::enqueuePropagation`/`drainPropagations`; Markdown status, checkboxes and iteration record validated. Builds, executable tests, instrumentation and benchmarks were not run (documentation-only scope).
Holes filed: none
