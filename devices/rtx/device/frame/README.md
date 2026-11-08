# Frame result publication and failure

Readiness is completion, not success. An ordinary `anariFrameReady(...,
ANARI_WAIT)` returns true after the render's host continuation, including the
application completion callback, has finished. `ANARI_NO_WAIT` eventually
returns true for failed attempts too.

A Frame owns its device `RenderError` record. Each Launch resets that record;
medium evaluators call `recordRenderError()` with a category and the Volume and
Instance identity. The first failing evaluation wins. The record is copied to
pinned host storage in the render stream after the Launch. A normal host task
waits for the GPU event and publishes the result before calling application
code. Application callbacks never run in a CUDA stream host function.

On failure (including render preflight failure):

- The Frame reports `ANARI_SEVERITY_ERROR`, with the Frame as source and the
  cause/medium identity in the message, before its completion callback runs.
- Its published `numSamples` (`ANARI_INT32`) becomes zero. All channels and
  accumulated samples are invalidated, and denoiser state is discarded. Other
  Frames' already-published channels and counts are not invalidated.
- Every enabled CPU or CUDA channel maps to null, with width and height zero
  and type `ANARI_UNKNOWN`. This is VisRTX's error-path policy, not a new ANARI
  status API. Previously copied application data cannot be revoked.
- The next successful render starts fresh even in manual `accumulationVersion`
  mode. Correcting/committing parameters alone does not restore the old image.

## Callback reentrancy

Mapping/unmapping this Frame's channels and reading its published `numSamples`
are supported from its failure status callback and completion callback. Failure
is published before either callback: these reads return null/zero/UNKNOWN and
zero samples immediately, without waiting for the same callback, flushing
pending commits, touching CUDA, or recursively reporting the failure. Successful
completion callbacks may also map their newly published channels.

These reads use an internal callback-thread routing path in `VisRTXDevice` to
avoid the per-object lock an ordinary WAIT may be holding on another thread.
GPU/output publication is distinct from completion of the callback: while the
callback is executing, an ordinary WAIT still waits and NO_WAIT is still false.

Waiting for this same Frame's full completion, starting another render of it,
or mutating it from either callback is outside the supported reentrant-access
contract. Defer those operations until callback return. Likewise, wait for all
Frames using a resource before editing that resource; there is no concurrent
scene-mutation snapshot guarantee.

## Controlled test fixture

Only builds with `VISRTX_BUILD_TESTS=ON` compile the GPU fault fixture. Setting
`VISRTX_TEST_FRAME_FAILURE=1` at submission injects a majorant error through a
CUDA kernel after the ordinary OptiX Launch, using the same device record and
stream-ordered transfer as a medium evaluation. The test enables it only around
the affected Frame's submission. This is not an ANARI parameter or a production
injection API. `TestFrameFailure` observes outcomes solely through public ANARI
calls and application callbacks; no private layout or record reads are used by
the test.
