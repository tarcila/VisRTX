# Progressive sample addressing (ticket 03)

## Production contract

`gpu/sampleOrdinal.h` is the shared host/device boundary:
`sampleOrdinal(completedSamples, iteration)` returns
`uint32(completedSamples) + uint32(iteration)`. Both Quality's raygen loop and
`gpu/renderer/raygen_helpers.h` call it. Quality passes that same `sampleIdx`
to the camera, light sampling, hemisphere sampling and material QMC values.
The frame counter already counts samples, not launches; multiplying it by spp
again skipped ordinals at positive spp > 1.

Supported inputs are nonnegative completedSamples, positive effective spp,
0 <= iteration < spp, and a representable signed frame update
(completedSamples + spp <= INT_MAX). The sampler index has period 2^32, but
that does not widen the existing signed frame counter or promise unbounded
progression. The eight-sample batch starting at INT_MAX-8 ends at ordinal
INT_MAX-1, then updates the frame counter to INT_MAX without overflow.
Non-positive pixelSamples are **not** repaired or qualified here.

Reset starts the underlying ordinal at zero. Checkerboarding still forces
spp=1 and increments the completed count only after tile 3: every pixel gets
ordinal n during cycle n. Frame counters, reset triggers, accumulation divisors,
pixel mapping and initial checkerboard display replication are unchanged.

The first accumulated sample still overrides pixel jitter with the pixel
center; the underlying ordinal is nevertheless zero and aperture dimensions
2/3 are still sampled. Ray-buffer cameras bypass jitter/aperture. Debug still
calls `makePrimaryRay(ss, 0u, true)` and Test still calls
`makePrimaryRay(ss, 0u)`; their behavior is unchanged.

The fix intentionally changes old positive-spp>1 patterns. Changing batching
no longer changes the underlying QMC ordinal prefix or pixel seed. This is
**not** a byte-identical rendered-image promise across different batching:
PCG initialization and floating-point accumulation may differ. No draw-order,
public scramble control, volume transport, CDF-tie or BSDF-physics change is
included in ticket 03.

## Tests at production seams

- `rtx::SampleOrdinal` calls the production helper with spp 1,2,3,4,8, a literal
  24-ordinal prefix over multiple launches and reset, literal four-tile traces,
  and the literal eight ordinals near INT_MAX. Ordinal zero is checked without
  invoking the camera. The same executable calls the real `makePrimaryRay`:
  zero normally uses Sobol jitter, the explicit center override is observable,
  aperture origin is preserved, and a ray-buffer ignores jitter/aperture.
  This host test uses CUDA vector/toolkit headers but does not initialize a GPU.
- `TestSampleAddressing` loads the device and renders a full-coverage orthographic
  quad with linear vertex-color gradients. Mapped albedo observes actual camera
  jitter without stochastic transport noise. Expected prefixes enumerate indices
  from zero, **not** via the ordinal helper or a copy of the raygen formula.
  Quality and Interactive (the shared raygen path) run spp 1,2,3,4,8 across three
  launches; checkerboarding requests spp=8 and checks forced effective spp=1
  across three complete four-tile cycles. Every pixel/tile and `numSamples` is
  checked after every launch, not just the final image. Fresh and explicit
  accumulationVersion-reset runs compare both albedo and color exactly under
  identical batching. Denoising and firefly filtering are disabled.
- The API fixture preserves the existing mapped albedo representation: its
  pre-launch completedSamples+1 divisor, unlike color compositing's divisor,
  and zero/unupdated sparse tiles. It also observes the initial fresh image,
  then latches the public accumulationVersion restart before recording the
  progressive trace. It checks that the fresh image equals reset ordinal zero;
  it does not redesign the frame's initial version-latch lifecycle.
- Static callsite audit: only Quality/shared progressive raygens use the new
  helper; Quality's camera/surface sites reuse `sampleIdx`. Debug/Test camera
  calls and Frame/Renderer accounting are unchanged. The API oracle also
  exercises the real callsites: restoring the old multiplication in **both**
  raygens makes spp 2,3,4,8 fail on launch 1 in both Quality and Interactive,
  while spp=1 and checkerboarding still pass. For spp=2, pixel (0,0), red gives
  albedo x=0.0861252174 versus expected 0.069246456.

TDD sequence: the first production-helper test failed to compile because
`gpu/sampleOrdinal.h` did not exist; adding the minimal helper and wiring both
raygens made it pass. Camera-preservation coverage followed. The API regression
was qualified by the old-callsite mutation above, then restoring helper use
made it green. Initial fixture development exposed an incorrectly initialized
camera union (fixed in the test) and the existing albedo/version-latch
conventions (accounted for in the test, not changed in production).

## Verification (2026-10-31)

Workspace: `/home/tarcila/Code/ANARI/visrtx/sobol-improvements-03`, initially
parented at integration tip b3f54004. Fresh **local** builds below; no origin or
installed device binaries. Environment matches ticket 01's README: GCC 16.2.0,
CMake 4.4.4, Ninja 1.13.2, CUDA 13.4.92, driver 615.78.08, RTX 5880 Ada,
OptiX 9.1, ANARI 0.16.0, MDL package 2026.0.2 (SDK metadata 2025.0.3).
C++/CUDA17, default minimum compute_75 PTX, existing RTX arithmetic flags.

```sh
cmake -S . -B _out/native -G Ninja -DVISRTX_BUILD_GL_DEVICE=OFF -DVISRTX_BUILD_TESTS=ON -DVISRTX_ENABLE_MDL_SUPPORT=OFF -DOPTIX_FETCH_VERSION=9.1 -DCMAKE_BUILD_TYPE=Release
cmake --build _out/native -j 8
cmake -S . -B _out/mdl -G Ninja -DVISRTX_BUILD_GL_DEVICE=OFF -DVISRTX_BUILD_TESTS=ON -DVISRTX_ENABLE_MDL_SUPPORT=ON -DVISRTX_USE_MDL_FOR_PHYSICALLY_BASED=ON -DOPTIX_FETCH_VERSION=9.1 -DCMAKE_BUILD_TYPE=Release
cmake --build _out/mdl -j 8

native='^(rtx::SampleOrdinal|rtx::Sobol.*|TestSampleAddressing|TestPbrSpecularDefault|TestPowerLightPick|TestPowerLightPickEnv|TestAlphaCutoffMode|TestRayBufferCamera|TestEmissiveInteractive)$'
mdl='^(rtx::SampleOrdinal|rtx::Sobol.*|TestSampleAddressing|TestPbrSpecularDefault|TestPowerLightPick|TestPowerLightPickEnv|TestAlphaCutoffMode|TestRayBufferCamera|TestEmissiveInteractive|TestEmissiveMaterialParity)$'
LD_LIBRARY_PATH="$PWD/_out/native:${LD_LIBRARY_PATH:-}" ctest --test-dir _out/native -R "$native" --output-on-failure
LD_LIBRARY_PATH="$PWD/_out/mdl:${LD_LIBRARY_PATH:-}" ctest --test-dir _out/mdl -R "$mdl" --output-on-failure
```

Results: native **10/10 passed**, MDL-backed PBR **11/11 passed**. Both full
builds compiled affected RTX PTX. SobolCUDA's exact host/device sampler parity
also passed. Existing camera, native/MDL-backed PBR, light, alpha and emission
regressions passed; the new API camera fixture uses Matte in both configurations.
This is ticket-03 addressing verification, not ticket-06 integrated draw-order
or full callable branch qualification.

Loader provenance was verified for both configurations:

```sh
LD_DEBUG=libs LD_LIBRARY_PATH="$PWD/_out/native:${LD_LIBRARY_PATH:-}" _out/native/TestSampleAddressing > /tmp/sobol-03-native-loader.log 2>&1
LD_DEBUG=libs LD_LIBRARY_PATH="$PWD/_out/mdl:${LD_LIBRARY_PATH:-}" _out/mdl/TestSampleAddressing > /tmp/sobol-03-mdl-loader.log 2>&1
grep 'calling init:.*libanari_library_visrtx' /tmp/sobol-03-*-loader.log
```

`calling init:` identified the absolute workspace-local
`_out/native/libanari_library_visrtx.so` and `_out/mdl/libanari_library_visrtx.so`.
Both loader runs also passed the API checks. Changed C++/CUDA files were formatted
with repository clang-format and checked with `clang-format --dry-run --Werror`.
No ticket-03 device/tooling gate was missing.
