# 11: Renderer pipeline rebuilds reset their dependent Frames

**Milestone:** scoped-invalidation

**Status:** resolved

**Blocked by:** 03, 10

**Files:** `devices/rtx/device/renderer/Renderer.h`, `devices/rtx/device/renderer/Renderer.cpp`, `devices/rtx/device/frame/Frame.h`, `devices/rtx/device/frame/Frame.cu`, `devices/rtx/device/optix_visrtx.h`, `devices/rtx/device/optix_visrtx.cpp`, `devices/rtx/apps/tests/api/TestScopedInvalidation.cpp`, `devices/rtx/apps/tests/api/TestMdlPipelineRebuild.cpp`

Part of 08. Keep the MDL material-library trigger device-global, but emit and
consume Renderer Pipeline changes per renderer, so only Frames whose renderer
actually rebuilt reset. Read ADRs 0002 and 0009, the frozen spec, and hole 09.
Blocked by 10 to serialize edits of Renderer/Frame and the shared test file.

## Implementation findings

- `Renderer::pipeline()` and `sbt()` lazily call `initOptixPipeline()` based on
  the MDL material registry timestamp. Initialization currently records that
  timestamp and rebuilds every callable table using `state.mdl` when present.
  Do not equate a matte-only world with an MDL-independent renderer without
  checking the actual pipeline dependency.
- In `Frame::renderFrame()`, `checkAccumulationReset()`, the sample-limit early
  return, `newFrame()` and `upload()` all happen before lazy `pipeline()`/`sbt()`
  initialization at `optixLaunch()`. A rebuild signal must arrive before reset
  decisions, including when the Frame previously reached its sample limit.
- Typed propagation must still happen during the commit-buffer flush; do not
  introduce ad hoc app-thread propagation or a global Frame reset timestamp.
  Shared-renderer Frames must independently consume the rebuild stamp.
- Existing `TestMdlPipelineRebuild` swaps Worlds, which already causes a reset;
  it does not isolate a library-triggered pipeline reset of an unchanged Frame.
  Retain its pipeline/module teardown and rendered-image regression checks.

## Acceptance criteria

- [x] An MDL library update rebuilds each renderer that uses that library and
      resets its Frames on their next render, without a one-render delay.
- [x] Two Frames sharing the rebuilt renderer reset regardless of render order.
- [x] Frames whose renderer did not rebuild continue accumulating; use actual
      pipeline dependency/rebuild evidence, not an assumed material distinction.
- [x] A Frame stopped at sampleLimit sees a necessary pipeline reset and renders.
- [x] Manual accumulationVersion mode retains application-owned reset semantics
      when no Frame buffer reallocation occurs.
- [x] `TestMdlPipelineRebuild`, focused scoped/MDL tests, formatter and all RTX
      API tests pass after `just brd`, using `_out/_cmake` and RelWithDebInfo;
      bounded logs go under `.scratch/scoped-invalidation/logs/`.

## Comments

### 2026-10-03 split context
Created from 08 before implementation. Pipeline dependency policy must be read
from current Renderer code; if the spec requires an impossible non-rebuilding
consumer, follow the hole workflow rather than inventing a new dependency model.

### 2026-10-03 iteration — resolved
Moved Pipeline preparation into `DeviceGlobalState::flushCommitBuffer(renderer)`, after pending finalization and before Frame reset/sample-limit decisions; pipeline/SBT accessors are now side-effect-free. Successful builds emit a per-renderer Pipeline stamp and notify dependent Frames through the existing queued typed propagation. The nextFrameReset query also prepares its current renderer. Ticket 10's storage/post-only behavior and manual reset policy remain unchanged.
All current renderer pipelines include the device MDL callable table, even with matte-only Worlds; retained this actual dependency rather than inventing material-based exclusions. Added `numPipelineBuilds` evidence and unchanged-World MDL regressions for both shared-Frame render orders, capped Frames, independently prepared renderers, and manual history. On the reverse cycle C absorbs the library first, then continues accumulating without a rebuild/reset when A/B's renderer rebuilds. Existing teardown and increasing rendered-pool checks remain intact. Self-reviewed against spec/style; no independent reviewer tool was available. Ticket 17's two pending review checks are covered here; its status is left for triage.
Verified: `cmake --build _out/_cmake --config RelWithDebInfo --target TestMdlPipelineRebuild && (cd _out/_cmake && ctest -C RelWithDebInfo -R '^TestMdlPipelineRebuild$' --output-on-failure)` → build passes, regression fails against old installed library (`logs/11-test-build.log`, `logs/11-red.log`).
Verified: `just brd` → build+install rc=0 (`logs/11-build-install-final.log`); `(cd _out/_cmake && ctest -C RelWithDebInfo -R 'TestScopedInvalidation|TestMdl' --output-on-failure)` → 3/3 pass (`logs/11-focused-final.log`).
Verified: `(cd _out/_cmake && ctest -C RelWithDebInfo --output-on-failure)` → 56/56 pass, 104.93s (`logs/11-all-tests-final.log`).
Verified: `clang-format --dry-run --Werror devices/rtx/device/renderer/{Renderer.h,Renderer.cpp} devices/rtx/device/frame/Frame.cu devices/rtx/device/optix_visrtx.{h,cpp} devices/rtx/apps/tests/api/TestMdlPipelineRebuild.cpp` → rc=0 (`logs/11-format.log`). All logs remain workspace-local; only bounded summaries printed.
Holes filed: none
