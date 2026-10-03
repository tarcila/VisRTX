# 10: Renderer post-only changes preserve accumulation storage

**Milestone:** scoped-invalidation

**Status:** ready-for-agent

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

- [ ] Shared renderer `denoise` on/off and `denoiseStart`/`denoiseMode` changes
      reset neither Frame, with float and packed color outputs covered.
- [ ] Denoise guide mode changes preserve valid color/albedo/normal history;
      validate output values/finite pixels as well as `numSamples`.
- [ ] `sampleLimit` changes reset neither Frame; lowering stops accumulation,
      increasing resumes from the existing count.
- [ ] A mixed post-only plus image-affecting renderer commit still resets both
      dependent Frames, including representative subtype parameters.
- [ ] Manual-mode authored size growth/shrink and channel changes reset A only;
      unchanged version otherwise suppresses dependency resets.
- [ ] Formatter, `just brd`, scoped/denoiser regressions and all RTX API tests
      pass using `_out/_cmake`, configuration RelWithDebInfo; logs are bounded
      and stored in `.scratch/scoped-invalidation/logs/`.

## Comments

### 2026-10-03 split context
Created from 08 before starting implementation. Existing manual resize coverage
is partial evidence, not completion of the post-only storage requirements.
