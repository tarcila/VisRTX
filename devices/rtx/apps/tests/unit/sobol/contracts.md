# Sampler misuse contracts (ticket 02)

## Production seams and policy

Tests call `sobolUInt` / `owenSobol` from `gpu/owenSobol.h` and
`NextRayQmc` / `nextRayUniform` / real PCG from `gpu/shadingState.h` and
`gpu/pcg.h`. No replacement sampler, mock random stream, callable-layout
change, or new production testing seam is used. `contracts.h` is test-only
behavioral checking shared between the host and CUDA executables.

Unsupported dimensions fail in `sobolUInt` before even an index-zero request
can return. The internal sampler-only failure helper uses unconditional host
`std::abort()` and device `trap;`, with no diagnostic assertion required.
Both assertion-enabled and NDEBUG builds therefore enforce the same policy.

`NextRayQmc` retains `float u[4]`, signed `int n`, and pointer-based callable
arguments. Non-null overlays require `0<=n<=4` and `0<=cursor<=n`. A supplied
request advances only the cursor; exhaustion takes one PCG draw and leaves the
cursor at n. Null overlays require cursor=0, never inspect n, take one PCG draw
and preserve cursor=0. A valid PCG pointer is required only for fallback draws.
Invalid input is a programmer error, not frame recovery or random replacement.

## Negative-process gate (Linux/POSIX)

The host parent execs each case in a fresh child and requires **SIGABRT** via
`waitpid`/`WTERMSIG`. A nonzero exit, unrelated crash/signal, or timeout fails.
Core dumps are disabled in children. Each child has a 10-second deadline;
the host CTest has a 120-second deadline.

The CUDA parent never initializes CUDA; each case execs a disposable child.
Every child first checks device availability, allocates memory, launches the
ordinary production-helper checks, synchronizes successfully, copies back a
success value and verifies it. Only then does it launch the negative case.
Prelaunch failure is not an expected trap. The reviewed Linux CUDA execution
error allowlist is:

- `cudaErrorAssert` (710): device assertion;
- `cudaErrorIllegalInstruction` (715): illegal instruction / trap;
- `cudaErrorLaunchFailure` (719): launch execution failure / trap.

These are deliberately the small assert/trap/launch-execution set allowed by
the spec, not all CUDA errors. In particular illegal-address, unknown,
availability and initialization errors fail closed and require investigation.
Successful synchronization also fails. After an allowed synchronization error
the child prints its enum and returns the explicit expected-error status **42**
from main, without any more CUDA calls. The parent requires normal exit with
exactly 42, not a signal or arbitrary nonzero status. No poisoned context is
reused. A separate baseline child must exit normally with 0. Each CUDA child
has a 20-second deadline; each CUDA CTest has a 240-second deadline.

Both modes observed `cudaErrorLaunchFailure (719)` for **all 46** negative
cases, with explicit expected-error completion; no additional enums were
accepted empirically. Baseline runtime and driver API versions were 13040.
The harness currently registers on UNIX; this records actual Linux coverage,
not qualification of another platform's abort status or CUDA errors.

## Coverage and TDD evidence

Each host/CUDA mode covers dimensions 16 and UINT32_MAX through both raw and
scrambled entry points, with indices 0 and 1 (8 cases); null cursor -1/1/4/5
(4 cases); and all invalid combinations of signed n=-1..5 and cursor=-1..5
(34 cases). This includes negative/oversized n, negative cursor, cursor>n,
n=0 with cursor>0, cursor at/exceeding capacity. Values are runtime arguments,
not unsigned casts. All valid n=0..4 and starting cursors 0..n are tested
with eight successive requests, covering supplied ordering, exhaustion,
zero-length and saturation. Every request checks the returned float, cursor,
PCG state and stream increment against an independent clone of the **real PCG
seam** advanced once only when fallback is expected. Supplied requests also
succeed with a null PCG pointer, proving they need no PCG access. Null-QMC
cursor=0 is checked for eight exact single-draw transitions.

Vertical red/green slices, before changing each production behavior:

1. Dimension tests failed in both modes: `expected SIGABRT: raw16` (old row-zero
   alias returned normally). Adding the unconditional dimension guard made all
   eight dimension cases pass.
2. Non-null bounds tests then failed in both modes at `qmc:-1:-1`. Validating
   signed n/cursor before access/fallback made the full bounds matrix pass.
3. Null-QMC misuse then failed in both modes at `null:-1`. Requiring cursor=0
   on the null path made it pass. The retained-behavior PCG checks and CUDA
   gates subsequently qualified the same implementation.

The first host test build needed an explicit `<initializer_list>` include;
that compile failure was repaired before observing the behavioral red above.

## Callsite audit

All production dimension requests were audited in `gpu/owenSobol.h`,
`gpu/sampleLight.h`, `renderer/Quality_ptx.cu` and the camera/hemi wrapper callers:

- Camera wrapper requests 0..3; hemisphere wrapper and direct hemisphere
  requests use 4..5.
- Analytic-light / HDRI lambdas use named constants 6..9; light category and
  instance selection use 10..11. The lambdas do not receive arbitrary dimensions
  from external data.
- Quality constructs `NextRayQmc{}` with n=4, loops d=0..3 and requests 12+d,
  i.e. 12..15. No out-of-range dimension or n producer was found.
- Native PBR and MDL nextRay callables initialize a signed cursor=0 and use
  `nextRayUniform`; no caller independently increments/resets that cursor.
  Their branch/request ordering is deliberately unchanged (ticket 04).
- `materialNextRay` forwards `const NextRayQmc *` as the fourth direct-call
  argument. Interactive uses its nullptr default; Quality passes the n=4
  overlay. Matte, native PBR and MDL retain the matching four-argument signature.
  Matte returns a dead ray without consuming the overlay.

Full native and MDL-backed-PBR builds compiled affected Quality, Interactive,
Matte, native-PBR and MDL PTX. TestPbrSpecularDefault and
TestEmissiveInteractive execute Quality/Interactive PBR wrappers (native in
the native build, MDL in the MDL-backed build); TestPbrSpecularDefault also
covers the explicit physicallyBasedMDL wrapper. No-coat diffuse/specular
material cases are exercised, not a claim of all coat/fifth-draw branch coverage.
TestAlphaCutoffMode executes Quality with Matte and PBR. TestPowerLightPick
and TestPowerLightPickEnv cover light/environment regressions, and
TestEmissiveMaterialParity covers the MDL emission integration. Exact PCG
fallback consumption is established at the approved production helper seam;
these image tests alone do not pin every light/hemisphere fallback callsite or
MDL xi assignment order. Those draw-role gates remain tickets 04/06.

## Reproduce verification

Run from the `sobol-improvements-02` workspace, parent `5c582cdd`, on 2026-10-31.
These are fresh **local** builds, not old ticket-01 binaries. Toolchain:
GCC 16.2.0, CMake 4.4.4, Ninja 1.13.2, CUDA nvcc/runtime 13.4.92,
NVIDIA driver 615.78.08, RTX 5880 Ada Generation; OptiX headers 9.1,
ANARI SDK 0.16.0, mdl-sdk-2026.0.2 (SDK CMake metadata 2025.0.3).
The environment/packages and arithmetic settings match the foundation
[README](README.md). CUDA17, compute_75 PTX; CUDA contract tests use
`--use_fast_math --expt-relaxed-constexpr`. Both production helper translation
units are compiled twice: Release flags plus `-UNDEBUG` for Assert, and
explicit NDEBUG for NDEBUG. Neither test relies on C assert for its checks.

```sh
cmake -S . -B _out/native -G Ninja -DVISRTX_BUILD_GL_DEVICE=OFF -DVISRTX_BUILD_TESTS=ON -DVISRTX_ENABLE_MDL_SUPPORT=OFF -DOPTIX_FETCH_VERSION=9.1 -DCMAKE_BUILD_TYPE=Release
cmake --build _out/native -j 8
cmake -S . -B _out/mdl -G Ninja -DVISRTX_BUILD_GL_DEVICE=OFF -DVISRTX_BUILD_TESTS=ON -DVISRTX_ENABLE_MDL_SUPPORT=ON -DVISRTX_USE_MDL_FOR_PHYSICALLY_BASED=ON -DOPTIX_FETCH_VERSION=9.1 -DCMAKE_BUILD_TYPE=Release
cmake --build _out/mdl -j 8
native='^(rtx::Sobol.*|rtx::SamplerContracts.*|TestPbrSpecularDefault|TestPowerLightPick|TestPowerLightPickEnv|TestAlphaCutoffMode|TestRayBufferCamera|TestEmissiveInteractive)$'
mdl='^(rtx::Sobol.*|rtx::SamplerContracts.*|TestPbrSpecularDefault|TestPowerLightPick|TestPowerLightPickEnv|TestAlphaCutoffMode|TestRayBufferCamera|TestEmissiveInteractive|TestEmissiveMaterialParity)$'
LD_LIBRARY_PATH="$PWD/_out/native:${LD_LIBRARY_PATH:-}" ctest --test-dir _out/native -R "$native" --output-on-failure
LD_LIBRARY_PATH="$PWD/_out/mdl:${LD_LIBRARY_PATH:-}" ctest --test-dir _out/mdl -R "$mdl" --output-on-failure
```

Results: native **12/12 passed**; MDL-backed PBR **13/13 passed**, including all
four contract executables in each configuration. Existing host/CUDA Sobol
reference/parity tests pass unchanged. New negative tests use no old table
values. Format all changed C++/CUDA headers/sources with repository
`clang-format -i`, then verify with `clang-format --dry-run --Werror`.

Loader checks for each build:

```sh
LD_DEBUG=libs LD_LIBRARY_PATH="$PWD/_out/native:${LD_LIBRARY_PATH:-}" _out/native/TestRayBufferCamera > /tmp/sobol02-native-loader.log 2>&1
LD_DEBUG=libs LD_LIBRARY_PATH="$PWD/_out/mdl:${LD_LIBRARY_PATH:-}" _out/mdl/TestRayBufferCamera > /tmp/sobol02-mdl-loader.log 2>&1
grep 'calling init:.*libanari_library_visrtx' /tmp/sobol02-*-loader.log
```

Both reported the absolute workspace-local `_out/native/` / `_out/mdl/`
`libanari_library_visrtx.so`, not an installed or ticket-01 device.
No ticket-02 gate was waived or unavailable. This is not ticket-06 integrated
qualification; addressing, draw-role ordering, seed control, reset/checkerboard
API regressions and complete integrated branch coverage remain their own
scopes. No volume, CDF-tie, non-positive-spp or transport-physics change.
