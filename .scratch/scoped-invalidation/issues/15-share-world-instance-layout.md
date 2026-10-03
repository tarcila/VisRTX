# 15: Consider sharing the World's surface-instance layout calculation

**What to build:** Share only the triangle/curve/user surface-instance ordering
and index calculation consumed by the World's OptiX instance builder, surface
GPU-data builder and light GPU-data builder. Keep their distinct payload
construction separate. Preserve synthesized Geometry Light lookup and existing
layout assertions; this is not a redesign of the builders.

**Blocked by:** None

**Status:** resolved

- [x] The three builders use one source of truth for geometry-type ordering and surface-instance indices
- [x] Builder-specific traversal, SBT and GPU-data payload construction remain separate and behaviorally unchanged
- [x] Mixed triangle/curve/user layouts, empty groups and shared Groups preserve correct Geometry Light lookup
- [x] Existing layout tripwires remain valid; mixed-layout and emissive-instance regressions pass
- [x] Formatting, build/install and full RTX API suite pass

## Comments

Triage approved by user: accept the narrowly scoped ordering/index extraction.
Do not unify the heavy payload builders or do unrelated layout refactors. No
current correctness failure is alleged; verification must preserve behavior.

Final Standards review: optional Message Chain/duplicated layout heuristic. Multiple World builders independently navigate Instance → Group and mirror the layout cursor, with an assertion acting as a tripwire. No current correctness defect or hard standards violation reported.

### 2026-10-03 iteration — resolved
Extracted a local per-transform triangle/curve/user index helper used by all three builders; traversal, SBT offsets and payloads remain distinct.
Preserved the light-layout assertion and added matching size tripwires to the OptiX and surface-data builders.
Expanded TestEmissiveInstanceAttribute with mixed Groups, a curve-only predecessor, shared Groups, transform arrays and interleaved empty Groups.
Six Geometry Lights and attribute/constant emission parity pass on quality and interactive (both mixed relative errors zero).
Verified: `clang-format --dry-run --Werror devices/rtx/device/world/World.cpp devices/rtx/apps/tests/api/TestEmissiveInstanceAttribute.cpp` → passed.
Verified: `just brd` → build/install passed (initial test compile error corrected before verification).
Verified: `ctest --test-dir _out/_cmake -C RelWithDebInfo -R 'TestEmissiveInstanceAttribute|TestEmissiveIncremental|TestEmissiveRecommit|TestScopedInvalidation' --output-on-failure` → 4/4 passed.
Verified: `ctest --test-dir _out/_cmake -C RelWithDebInfo -R '^TestEmissiveInstanceAttribute$' -V` → passed, including mixed-layout parity.
Verified: `ctest --test-dir _out/_cmake -C RelWithDebInfo --output-on-failure` → 56/56 passed (all 51 RTX API tests plus 5 unit tests).
Logs: `.scratch/scoped-invalidation/logs/15-{build-install,focused,mixed-layout,all-tests,format}.log`.
Holes filed: none
