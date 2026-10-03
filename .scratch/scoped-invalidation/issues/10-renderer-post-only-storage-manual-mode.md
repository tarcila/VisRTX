# 10: Renderer post-only changes preserve accumulation storage

**Milestone:** scoped-invalidation

**Status:** resolved

**Blocked by:** 02, 03

**Files:** `devices/rtx/device/renderer/*`, `devices/rtx/device/frame/Frame.h`, `devices/rtx/device/frame/Frame.cu`, `devices/rtx/apps/tests/api/TestScopedInvalidation.cpp`

Part of 08. Implement the `denoise*` and `sampleLimit` carve-outs and finish
manual-mode buffer-reset verification. Read the frozen spec and hole 09.

## Implementation findings

- `Renderer::markFinalized()` currently emits Image unconditionally. Classify
  post-only commits without suppressing image changes in the same commit,
  subtype parameters, or background-array notifications. Classification beyond
  this carve-out is out of scope.
- `Frame::renderFrame()` detects denoise settings changes and calls `finalize()`.
  `finalize()` compares an incomplete tuple of buffer addresses and forces a
  reset even for a pixel-output allocation caused solely by toggling denoise.
  `DeviceBuffer::reserve()` preserves storage only when capacity suffices;
  growing it discards the old contents. Albedo/normal accumulators are currently
  allocated only when requested by channels or the denoiser, and discarded when
  no longer requested. Enabling guides mid-accumulation cannot read uninitialized
  history or silently lose previous samples. Preserve meaningful accumulation
  storage across all denoise settings (including guide modes), not just frameID.
- Distinguish post-processing scratch/output changes from an authored Frame
  size/channel layout change. Genuine Frame buffer reallocations/layout changes
  always reset, including manual mode; mere post-only changes must not reset.
  Do not rely solely on pointer inequality: allocators can reuse addresses.
- `testManualResize()` already verifies an unchanged version suppresses camera
  changes and a growing size resets A only. Extend rather than duplicate it.
- Pipeline invalidation is the separate ticket 11, not this ticket.

## Acceptance criteria

- [x] Shared renderer `denoise` on/off and `denoiseStart`/`denoiseMode` changes
      reset neither Frame, with float and packed color outputs covered.
- [x] Denoise guide mode changes preserve valid color/albedo/normal history;
      validate output values/finite pixels as well as `numSamples`.
- [x] `sampleLimit` changes reset neither Frame; lowering stops accumulation,
      increasing resumes from the existing count.
- [x] A mixed post-only plus image-affecting renderer commit still resets both
      dependent Frames, including representative subtype parameters.
- [x] Manual-mode authored size growth/shrink and channel changes reset A only;
      unchanged version otherwise suppresses dependency resets.
- [x] Formatter, `just brd`, scoped/denoiser regressions and all RTX API tests
      pass using `_out/_cmake`, configuration RelWithDebInfo; logs are bounded
      and stored in `.scratch/scoped-invalidation/logs/`.

## Comments

### 2026-10-03 split context
Created from 08 before starting implementation. Existing manual resize coverage
is partial evidence, not completion of the post-only storage requirements.

### 2026-10-03 iteration — resolved
Renderer compares effective base image settings after subtype overrides, with subtype change flags and background-array data stamps; post-only commits no longer emit Image, while mixed commits and identical background unmaps still do.
Frame retains and collects color/albedo/normal history from the first sample (24 extra guide bytes/pixel when no guides were previously requested); scratch/output allocations do not reset. Capacity growth and authored size/channel changes do reset, regardless of manual mode or reused allocator addresses. Capped Frames re-resolve changed post settings without tracing extra samples.
Extended manual growth/shrink/all-channel coverage, shared float/packed denoise/limit tests, fast/quality mixed commits and background notifications. Guide tests compare late-enabled guides against an always-enabled reference, check nonempty finite color, exact matte albedo and normalized finite normals. Self-reviewed against spec/style; no independent reviewer tool was available. Pipeline ordering and array-owner audit remain untouched for 11/12.
Verified: `cmake --build _out/_cmake --config RelWithDebInfo --target TestScopedInvalidation && (cd _out/_cmake && ctest -C RelWithDebInfo -R '^TestScopedInvalidation$' --output-on-failure)` → old installed library failed 41 assertions (`logs/10-red.log`).
Verified: `just brd` → build+install rc=0 (`logs/10-build-install-final.log`); `(cd _out/_cmake && ctest -C RelWithDebInfo -R '^TestScopedInvalidation$' --output-on-failure)` → scoped/denoiser regressions pass 1/1 (`logs/10-focused-final.log`). No separately named denoiser CTest exists; the new scoped cases exercise all guide modes.
Verified: `(cd _out/_cmake && ctest -C RelWithDebInfo --output-on-failure)` → 56/56 pass (`logs/10-all-tests.log`).
Verified: `clang-format --dry-run --Werror devices/rtx/device/renderer/{Renderer.h,Renderer.cpp,Interactive.cpp,Quality.cpp,Fast.cpp,Debug.cpp} devices/rtx/device/frame/{Frame.h,Frame.cu} devices/rtx/apps/tests/api/TestScopedInvalidation.cpp` → rc=0 (`logs/10-format.log`). Logs stay workspace-local and bounded.
Holes filed: none
