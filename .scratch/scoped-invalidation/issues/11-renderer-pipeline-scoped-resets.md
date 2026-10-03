# 11: Renderer pipeline rebuilds reset their dependent Frames

**Milestone:** scoped-invalidation

**Status:** ready-for-agent

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

- [ ] An MDL library update rebuilds each renderer that uses that library and
      resets its Frames on their next render, without a one-render delay.
- [ ] Two Frames sharing the rebuilt renderer reset regardless of render order.
- [ ] Frames whose renderer did not rebuild continue accumulating; use actual
      pipeline dependency/rebuild evidence, not an assumed material distinction.
- [ ] A Frame stopped at sampleLimit sees a necessary pipeline reset and renders.
- [ ] Manual accumulationVersion mode retains application-owned reset semantics
      when no Frame buffer reallocation occurs.
- [ ] `TestMdlPipelineRebuild`, focused scoped/MDL tests, formatter and all RTX
      API tests pass after `just brd`, using `_out/_cmake` and RelWithDebInfo;
      bounded logs go under `.scratch/scoped-invalidation/logs/`.

## Comments

### 2026-10-03 split context
Created from 08 before implementation. Pipeline dependency policy must be read
from current Renderer code; if the spec requires an impossible non-rebuilding
consumer, follow the hole workflow rather than inventing a new dependency model.
