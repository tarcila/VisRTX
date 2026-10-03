# 20: Consider naming individual scoped-Pipeline test scenarios

**What to build:** Name the phases of the shared/independent/manual Frame
Pipeline regression within its existing test function, using lightweight
sub-steps or local helpers as appropriate. Keep one execution sequence:
separate test cases must not destroy the cross-Frame temporal assertions.

**Blocked by:** None

**Status:** resolved

- [x] Scenario phases have clear names while remaining in one ordered test function
- [x] Build-count, shared-frame reset, independent-renderer and manual-mode assertions retain their original meaning and coverage
- [x] No production behavior or fixture lifetime changes are introduced
- [x] Formatting, build/install and focused Pipeline/scoped-invalidation regressions pass

## Comments

Triage approved by user: accept naming/structure only. Preserve the existing
single execution sequence rather than splitting stateful phases into independent
tests. No optional production refactor belongs in this ticket.

Final Standards review: optional test-readability heuristic. testScopedPipeline interleaves several renderers, Frames and rebuild cycles in a long function. Not a documented-standard breach or correctness defect; the review's Feature Envy label is weak for an integration test, but the scenario-naming suggestion is independently actionable.

### 2026-10-03 iteration — resolved
Added local scenario/phase labels in `devices/rtx/apps/tests/api/TestMdlPipelineRebuild.cpp`, included in failure diagnostics.
Kept `testScopedPipeline()` as one ordered function with unchanged assertions, sample/build-count expressions, cross-Frame render order and resource lifetime; no production changes.
Verified: `clang-format -i devices/rtx/apps/tests/api/TestMdlPipelineRebuild.cpp && clang-format --dry-run --Werror devices/rtx/apps/tests/api/TestMdlPipelineRebuild.cpp` → passed.
Verified: `just brd` → build and install passed; log `.scratch/scoped-invalidation/logs/20-build-install.log`.
Verified: `ctest --test-dir _out/_cmake -C RelWithDebInfo -R 'Test(MdlPipelineRebuild|ScopedInvalidation|EmissiveIncremental|EmissiveRecommit|EvictGPUArray)$' --output-on-failure` → 5/5 passed (7.05 s); log `.scratch/scoped-invalidation/logs/20-focused-tests.log`.
Verified: `ctest --test-dir _out/_cmake -C RelWithDebInfo --output-on-failure` → 56/56 passed (104.94 s); log `.scratch/scoped-invalidation/logs/20-full-tests.log`.
Holes filed: none
