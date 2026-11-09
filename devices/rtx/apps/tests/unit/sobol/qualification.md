# Integrated Sobol–PCG qualification (tickets 06, 43)

## Final-tree cleanup requalification (ticket 43)

**Requalified 2026-10-31**, after the parent-adjudicated final-review cleanup,
from `/home/tarcila/Code/ANARI/visrtx/sobol-improvements-43`. Base/sole parent:
`aa5e654aec37dba43adde7753fe609748f43c356`, also the integration tip at handoff.
The ticket-06 record below remains historical evidence; this section records the
post-cleanup tree. **Both final reviewers were INCOMPLETE.** These passing gates
are not final code-review approval. No reviewer was redispatched or delegated;
missing authored Python/docs, fixture and test-body inspection remains missing.

### Finding dispositions and retained judgments

- Standards 1: project `gpu/gpu_decl.h` precedes `<cstdlib>` in
  `samplerContracts.h`; fail-fast behavior is unchanged.
- Standards 2: `sobolDirections.h` now has `#pragma once` and its own note about
  inclusion inside `visrtx`, availability of uint32_t and host/CUDA constant
  storage. All 512 table words are unchanged. The `__device__ __constant__`
  storage declaration is not a function qualifier; no function macro substitutes
  for it.
- Standards 3: LightDrawOrder, SamplerContractsCUDA and the shared Sobol kernel
  use `VISRTX_DEVICE` / `VISRTX_GLOBAL`. A small test-only `cudaResources.h`
  owns buffers via unique_ptr and owns the texture/array as one lifetime.
  Light test errors now unwind those owners; Sobol partial allocations unwind.
  This deliberately does not use production DeviceBuffer: these standalone
  tests require explicit CUDA allocation/copy error discrimination, while that
  wrapper's copies ignore CUDA return codes and its allocator adds device
  infrastructure dependencies. Allocation is centralized, not a production
  allocator redesign. The negative child relinquishes ownership *before* its
  trap launch, making no destructor CUDA call after the poisoned context;
  process exit reclaims it. Baseline and pre-negative paths retain normal RAII.
- Standards 4: shared implementation moved, not copied, to guarded
  `sobol/testSobol.h` in a test namespace. Thin `.cpp`/`.cu` mains call `run()`;
  CUDA selects the existing device parity section. Independent recurrence,
  pinned FastOwen excerpt, vectors, endpoint/strata checks and production seams
  are unchanged. No behavior repair or new seam required new red/green tests;
  this is the expressly authorized TDD review-stage refactoring.
- Standards 5 rejected as adjudicated: keep specified unconditional host abort /
  CUDA trap, including NDEBUG. No exception-policy change or spec edit.
- Heuristics: `contractProcess.h` shares the existing mode grammar, all 46
  invalid inputs and bounded fork/exec/wait transport. Host still requires
  SIGABRT; CUDA still requires normal expected-error exit 42, ordinary baseline
  execution and the unchanged reviewed enum allowlist. Timeouts remain 10/20 s
  per child and 120/240 s per host/CUDA CTest. Common registration is factored
  only for the four contract targets; CUDA arithmetic flags stay explicit.
  Retain separate main drivers and success/error predicates because their
  process/context contracts differ. Other unit registrations are outside this
  bounded duplication and remain explicit. Retain the specified sampleOrdinal
  shared seam, not a Middle Man removal.
- Minor scope observations: retain required license installation and functional
  includes. Also retain the preexisting firefly-comment wrap and sampleLight
  include sort: formatting the pre-feature versions with repository
  clang-format independently reproduces both exact hunks. Reverting them would
  immediately violate required canonical formatting; no extra production
  cosmetic changes were introduced in this pass.

### Fresh local builds and complete gates

Same dependency environment/effective options as ticket 06 below: GCC 16.2.0,
CMake 4.4.4, CUDA 13.4.92, driver 615.78.08 (API 13040), RTX 5880 Ada,
OptiX 9.1, MDL package 2026.0.2, C++/CUDA17, compute_75 and Release.
Both `_out` directories were configured fresh in this workspace; both full
libraries/PTX and all test executables were built locally, then rebuilt after
the final ownership declarations. Exact commands (workspace root):

```sh
cmake -S . -B _out/native -G Ninja -DVISRTX_BUILD_GL_DEVICE=OFF -DVISRTX_BUILD_TESTS=ON -DVISRTX_ENABLE_MDL_SUPPORT=OFF -DOPTIX_FETCH_VERSION=9.1 -DCMAKE_BUILD_TYPE=Release
cmake -S . -B _out/mdl -G Ninja -DVISRTX_BUILD_GL_DEVICE=OFF -DVISRTX_BUILD_TESTS=ON -DVISRTX_ENABLE_MDL_SUPPORT=ON -DVISRTX_USE_MDL_FOR_PHYSICALLY_BASED=ON -DOPTIX_FETCH_VERSION=9.1 -DCMAKE_BUILD_TYPE=Release
cmake --build _out/native -j 8
cmake --build _out/mdl -j 8
native='^(rtx::(Sobol.*|SamplerContracts.*|SampleOrdinal|DrawOrder|LightDrawOrder)|TestPbrSpecularDefault|TestPowerLightPick|TestPowerLightPickEnv|TestAlphaCutoffMode|TestRayBufferCamera|TestEmissiveInteractive|TestSampleAddressing|TestMaterialSampling)$'
mdl='^(rtx::(Sobol.*|SamplerContracts.*|SampleOrdinal|DrawOrder|LightDrawOrder)|TestPbrSpecularDefault|TestPowerLightPick|TestPowerLightPickEnv|TestAlphaCutoffMode|TestRayBufferCamera|TestEmissiveInteractive|TestEmissiveMaterialParity|TestSampleAddressing|TestMaterialSampling)$'
LD_DEBUG=libs LD_LIBRARY_PATH="$PWD/_out/native:${LD_LIBRARY_PATH:-}" ctest --test-dir _out/native -R "$native" -V > /tmp/sobol43-final-native.log 2>&1
LD_DEBUG=libs LD_LIBRARY_PATH="$PWD/_out/mdl:${LD_LIBRARY_PATH:-}" ctest --test-dir _out/mdl -R "$mdl" -V > /tmp/sobol43-final-mdl.log 2>&1
clang-format --dry-run --Werror devices/rtx/device/gpu/{samplerContracts.h,sobolDirections.h,gpu_util.h,sampleLight.h} devices/rtx/apps/tests/unit/{test_Sobol.cpp,test_SobolCUDA.cu,test_LightDrawOrder.cu,test_SamplerContracts.cpp,test_SamplerContractsCUDA.cu,sobol/testSobol.h,sobol/contractProcess.h,sobol/cudaResources.h}
```

Final **native 17/17 passed (50.26 s), MDL 18/18 passed (121.49 s)**, no skips
or waived gates; formatting passed. Initial post-cleanup runs also passed
17/17 and 18/18. Each final configuration reports 1764 exact host/device
comparisons, successful baselines and 92 negative CUDA children (46 per mode),
all `cudaErrorLaunchFailure (719)`, with normal explicit expected-error exit.
Host SIGABRT gates passed in Assert and NDEBUG. All callable matrix radiance
observations exactly match the ticket-06 table below, including MDL-backed PBR,
explicit physicallyBasedMDL and raw MDL, Quality and Interactive, both launches.
The coverage distinctions/limitations in that record remain unchanged.

Loader traces from these same *full* runs identify `calling init:` for all
8 native API processes (IDs 19,20,21,22,23,33,39,49) as
`/home/tarcila/Code/ANARI/visrtx/sobol-improvements-43/_out/native/libanari_library_visrtx.so`,
and all 9 MDL API processes (20,21,22,23,24,32,38,65,67) as the same workspace's
`_out/mdl/libanari_library_visrtx.so`. No installed/origin library counted.

Byte comparison against `jj file show -r aa5e654a` confirms joe-kuo.txt,
vectors.txt, seed-vectors.txt, FastOwenScrambler.reference.txt and LICENSE.pbrt
are unchanged. Regex extraction and little-endian uint32 serialization confirm
all 512 production words unchanged, SHA-256
`357f257f40c9f9b7efb784bdf355b543f456984987e8519686fc978cf319be96`;
Joe–Kuo fixture hash remains
`d155655216d8f60bac3d2b34f2926cf03b64e288c7df74d6c25c13cccce3b26f`.
No frozen spec, attribution, dimension map, ordinal or sampler behavior changed.

---

## Verdict and source baseline

**Qualified on the Linux/CUDA/OptiX configuration below, 2026-10-31.** All
required deterministic host, CUDA misuse/parity, PTX and native/MDL callable/API
gates passed, without skips or waivers. This is correctness/integration
qualification, **not** a benchmark, convergence result, multidimensional
discrepancy bound, or proof of statistical independence.

Workspace: `/home/tarcila/Code/ANARI/visrtx/sobol-improvements-06`.
Its parent equals integration tip `bfd0c6556cce1f6edb2c8560a5f223c21919193d`.
The five resolved prerequisite tickets and their committed records were read
before qualification; their coverage was rerun together, not accepted merely
from the earlier isolated runs:

| Ticket | Integrated commit | Durable prerequisite evidence |
|---|---|---|
| 01, reference table | `5c582cdd` | [README.md](README.md): pinned sources, licenses, extraction, both checksums, independent parameter recurrence and FastOwen comparison |
| 02, misuse | `b3f54004` | [contracts.md](contracts.md): signed bounds, abort/trap protocol, PCG/cursor state, finite child deadlines |
| 03, ordinals | `55e607e1` | [addressing.md](addressing.md): both production raygens, literal traces, camera/reset/checkerboard API oracle and earlier callsite mutation |
| 04, draw roles | `f3096750` | [draw-order.md](draw-order.md): production-expression branch/component checks, device fallback order and callable matrix |
| 05, internal seed | `bfd0c655` | [scramble.md](scramble.md): fixed key/depth vectors, invertibility rationale and sampler-level-only scope |

Ticket 06 changes no executable sampling behavior. The comment audit corrected
two stale Quality comments: surface NEE is not first-bounce-only QMC followed
by PCG, and separate dimensions do not prove decorrelation/independence.
No new production fix or new testing seam was needed. Earlier red/green and
mutation evidence remains in the prerequisite records; it is not presented as
new red/green work in this qualification.

## Exact environment and fresh builds

Both build directories were created here from scratch. No installed device,
origin volume-workspace build, or preceding ticket's library was reused.
Dependencies were available from the Nix development shell:

- GCC 16.2.0, CMake 4.4.4, Ninja 1.13.2; Python 3.14.8 for the existing
  production-expression test-build generator.
- CUDA nvcc/runtime 13.4.92, NVIDIA driver 615.78.08, runtime/driver API 13040;
  NVIDIA RTX 5880 Ada Generation (48 GiB).
- OptiX headers 9.1; ANARI SDK package `0.16.0-unstable-2026-10-23`
  (selected CMake package directory `anari-0.17.0`).
- MDL package `mdl-sdk-2026.0.2`, SDK CMake metadata version 2025.0.3;
  `MDL_SDK_ROOT=/nix/store/4b63f6kwzjdcd5c6bb6m3nsy6969fwnz-mdl-sdk-2026.0.2`.
- `MDL_SYSTEM_PATH=/data/mdl/vMaterials_2:/data/mdl/omni-core-materials/source/Base:/data/mdl/omni-core-materials/source/mdl`.
  The callable matrix's raw MDL fixture is inline code, not dependent on an
  external material asset.

Commands below are from the workspace root (each succeeded):

```sh
cmake -S . -B _out/native -G Ninja -DVISRTX_BUILD_GL_DEVICE=OFF -DVISRTX_BUILD_TESTS=ON -DVISRTX_ENABLE_MDL_SUPPORT=OFF -DOPTIX_FETCH_VERSION=9.1 -DCMAKE_BUILD_TYPE=Release
cmake --build _out/native -j 8
cmake -S . -B _out/mdl -G Ninja -DVISRTX_BUILD_GL_DEVICE=OFF -DVISRTX_BUILD_TESTS=ON -DVISRTX_ENABLE_MDL_SUPPORT=ON -DVISRTX_USE_MDL_FOR_PHYSICALLY_BASED=ON -DOPTIX_FETCH_VERSION=9.1 -DCMAKE_BUILD_TYPE=Release
cmake --build _out/mdl -j 8
```

Effective settings: RTX ON, tests ON, GL OFF, Release; MDL OFF in native,
MDL ON **and MDL-backed physicallyBased ON** in MDL. MaterialX, neural, NVTX
and HDRI-sampling debug OFF; NVML ON. C++/CUDA17, minimum arch 75, compute_75
PTX, existing `--use_fast_math --expt-relaxed-constexpr` arithmetic options.
Full builds compiled renderer/material PTX, including Quality, Interactive,
Matte, native PBR and (MDL build) MDL shader resources. Both full builds were
rerun successfully after the comment-only audit changes.

Misuse helper translation units are separately compiled in both configurations:
Assert uses Release `-DNDEBUG` followed by `-UNDEBUG`; NDEBUG explicitly defines
NDEBUG. This applies to host and CUDA, not just the test driver's assertions.
Neither test relies on `assert` to validate its results.

## Combined gates and results

Exact final commands, after comment changes and rebuilding both libraries:

```sh
native='^(rtx::(Sobol.*|SamplerContracts.*|SampleOrdinal|DrawOrder|LightDrawOrder)|TestPbrSpecularDefault|TestPowerLightPick|TestPowerLightPickEnv|TestAlphaCutoffMode|TestRayBufferCamera|TestEmissiveInteractive|TestSampleAddressing|TestMaterialSampling)$'
mdl='^(rtx::(Sobol.*|SamplerContracts.*|SampleOrdinal|DrawOrder|LightDrawOrder)|TestPbrSpecularDefault|TestPowerLightPick|TestPowerLightPickEnv|TestAlphaCutoffMode|TestRayBufferCamera|TestEmissiveInteractive|TestEmissiveMaterialParity|TestSampleAddressing|TestMaterialSampling)$'
LD_LIBRARY_PATH="$PWD/_out/native:${LD_LIBRARY_PATH:-}" ctest --test-dir _out/native -R "$native" -V > /tmp/sobol06-final-native.log 2>&1
LD_LIBRARY_PATH="$PWD/_out/mdl:${LD_LIBRARY_PATH:-}" ctest --test-dir _out/mdl -R "$mdl" -V > /tmp/sobol06-final-mdl.log 2>&1
```

Final native: **17/17 passed**, no skips (50.39 s). Final MDL-backed PBR:
**18/18 passed**, no skips (121.15 s). Durations describe test execution, not
performance measurements. Earlier pre-comment runs also passed 17/17 and 18/18.
Local full logs remain under `/tmp/sobol06-*` and each build's CTest temporary
logs; this document preserves the commands, actual observations and limitations
without depending on those transient logs for its acceptance summary.

| Gate (passed in both configurations unless noted) | Observed coverage |
|---|---|
| `rtx::Sobol` | All 512 production columns versus independently sourced Joe–Kuo recurrence; d=7/bit=4 `0x88000000`; binary-index vectors including high bits; pinned FastOwen operations, reversal, exact conversion endpoints/grid; all dimensions, three seeds, dyadic strata m=1..10; default/nonzero pixel and depth vectors |
| `rtx::SobolCUDA` | **1764 exact host/device comparisons** of keys, integer samples and float bits, including camera/hemi wrappers; successful device launch/sync/copy |
| `rtx::SamplerContractsAssert`, `rtx::SamplerContractsNDEBUG` | Valid signed n=0..4/cursors and null overlay, supplied ordering and exact real-PCG transitions; 46 negative children per mode, requiring POSIX SIGABRT, not arbitrary failure |
| `rtx::SamplerContractsCUDAAssert`, `rtx::SamplerContractsCUDANDEBUG` | Separate disposable process/context per misuse, ordinary success baseline before negatives; 46 negative children per mode all reported **cudaErrorLaunchFailure (719)** and normal expected-error exit 42 |
| `rtx::SampleOrdinal` | Literal spp=1/2/3/4/8 prefixes, resets/four-tile traces and INT_MAX-8..INT_MAX-1; ordinal zero separately from center-camera override, aperture and ray-buffer bypass |
| `rtx::DrawOrder` | Verbatim native shader body: no coat, selected coat, rejected-coat diffuse/specular/transmission, fifth PCG request and early returns; verbatim MDL xi x/y/z/w with partial/null overlays; PCG hemisphere/rectangle/HDRI expressions |
| `rtx::LightDrawOrder` | Full production CUDA rectangle/HDRI samplers, real texture/CDF, and PCG hemisphere; exact fallback order/state and retained QMC dimension mapping |
| `TestSampleAddressing` | Actual Quality/shared-Interactive raygens: spp=1/2/3/4/8, three launches and three four-tile cycles; forced checkerboard spp=1, fresh/reset repeatability, per-pixel albedo oracle, denoise/firefly disabled; identical batching only |
| `TestMaterialSampling` | Executed OptiX callable matrix detailed below, not merely signature compilation |
| Existing required API gates | `TestPbrSpecularDefault`, `TestPowerLightPick`, `TestPowerLightPickEnv`, `TestAlphaCutoffMode`, `TestRayBufferCamera`, `TestEmissiveInteractive` all passed in native and MDL; `TestEmissiveMaterialParity` passed in MDL |

CUDA allowlist remains the reviewed Linux execution-error set 710 (assert),
715 (illegal instruction), 719 (launch failure); only 719 was observed. No
prelaunch failure, timeout, unrelated crash, successful negative synchronization
or unknown enum was counted as an expected trap. Every child baseline succeeded;
poisoned contexts were never reused. Host child deadline 10 s / CTest 120 s;
CUDA child deadline 20 s / CTest 240 s. Sobol and draw-order CTests have 60 s
deadlines. This is Linux qualification, not a claim about other abort/error ABIs.

### Callable execution, not inferred ABI coverage

`TestMaterialSampling` uses a front-facing diffuse plane, hidden unit HDRI,
64 positive spp, 32x32 float output, no denoise/firefly filtering. Both launches
passed for **every** combination below. Mean red radiances in the final runs
(the two launches agreed) were:

| Build/material | Quality | Interactive |
|---|---:|---:|
| Native Matte | 0.499993162 | 0.500552112 |
| Native physicallyBased (native shader) | 0.500014328 | 0.500126390 |
| MDL Matte | 0.499993162 | 0.500552112 |
| MDL physicallyBased (MDL-backed PBR) | 0.500273009 | 0.500125931 |
| MDL explicit physicallyBasedMDL | 0.500273009 | 0.500125931 |
| MDL raw inline diffuse MDL | 0.500273009 | 0.500125931 |

The Lambertian expectation is 0.5 with test tolerance 0.04. This checks actual
continuation/MIS energy and callable execution. Quality passes the four-value
overlay; Interactive executes the nullptr-default PCG path. Matte's matching
fourth argument is executed but intentionally returns a dead continuation.

Precise native coat/specular/fifth-request and MDL xi assignment roles are
observed by the production-expression tests, not inferred from these means.
The image fixture exercises diffuse continuation, not every SDK-internal
branch; no internal trace of MDL scattering decisions is claimed. Device
light/hemisphere order is observed by `LightDrawOrder`, not guessed from an
image. These complementary boundaries are the approved section-7 seams.

## Loaded-library provenance for every API gate

After final rebuild, all eight native and nine MDL API gates were also rerun
with loader tracing. Both selections passed (8/8 and 9/9):

```sh
native_api='^(TestPbrSpecularDefault|TestPowerLightPick|TestPowerLightPickEnv|TestAlphaCutoffMode|TestRayBufferCamera|TestEmissiveInteractive|TestSampleAddressing|TestMaterialSampling)$'
mdl_api='^(TestPbrSpecularDefault|TestPowerLightPick|TestPowerLightPickEnv|TestAlphaCutoffMode|TestRayBufferCamera|TestEmissiveInteractive|TestEmissiveMaterialParity|TestSampleAddressing|TestMaterialSampling)$'
LD_DEBUG=libs LD_LIBRARY_PATH="$PWD/_out/native:${LD_LIBRARY_PATH:-}" ctest --test-dir _out/native -R "$native_api" -V > /tmp/sobol06-api-loader-native.log 2>&1
LD_DEBUG=libs LD_LIBRARY_PATH="$PWD/_out/mdl:${LD_LIBRARY_PATH:-}" ctest --test-dir _out/mdl -R "$mdl_api" -V > /tmp/sobol06-api-loader-mdl.log 2>&1
grep 'calling init:.*libanari_library_visrtx' /tmp/sobol06-api-loader-{native,mdl}.log
```

Every selected process reported `calling init:` with its correct absolute DSO:

- Native test IDs 19,20,21,22,23,33,39,49:
  `/home/tarcila/Code/ANARI/visrtx/sobol-improvements-06/_out/native/libanari_library_visrtx.so`
- MDL test IDs 20,21,22,23,24,32,38,65,67:
  `/home/tarcila/Code/ANARI/visrtx/sobol-improvements-06/_out/mdl/libanari_library_visrtx.so`

No installed or origin volume-workspace device was loaded in these traces.
Standalone loader-traced `TestMaterialSampling` runs also passed both matrices.

## Sampling-policy and callsite audit

Rechecked production `owenSobol`, camera/hemi wrappers, `owenPixelSeed`,
`owenBounceSeed`, `sampleOrdinal`, `NextRayQmc`, `nextRayUniform` and
`materialNextRay` callsites and sampling-policy comments under
`devices/rtx/device`. The map in `gpu/owenSobol.h` is authoritative:

| Dimensions (zero-based) | Role |
|---|---|
| 0–1 | Camera pixel jitter, except first accumulated sample's center override |
| 2–3 | Aperture; still sampled with the center-pixel exception |
| 4–5 | Ambient hemisphere and HDRI cosine strategy, intentionally shared |
| 6–7 | Chosen analytic-light coordinates OR HDRI CDF row/column |
| 8–9 | HDRI within-texel jitter |
| 10–11 | Ambient/light category and light-instance selection |
| 12–15 | First four sequential material requests at each surface vertex |

- `gpu/cameraCreateRay.h` uses only 0..3 and the two-argument default-zero
  pixel seed. Ray-buffer cameras bypass those draws. Debug/Test center behavior
  is unchanged. Both progressive raygens use `sampleOrdinal`; Quality reuses
  that exact ordinal for camera and surface requests.
- Quality uses the same default-zero pixel key, mixed with bounce/transparency
  depth per surface vertex. **All surface bounces**, not just the first, use
  these reseeded surface dimensions. Shared 4–5 is intentional, not accidental
  extra dimension allocation. Distinct dimensions/seeds do not prove independence.
- Analytic sphere/ring requests are separately sequenced 6 then 7; rectangle
  now explicitly draws edge1 then edge2. HDRI draws row, column, jitter u/v
  (6,7,8,9). Their fallback lambdas each take one PCG draw per request. Category
  and instance calls use only 10/11. No arbitrary external dimension producer
  exists.
- Quality's sole overlay producer initializes `NextRayQmc{}`, sets n=4,
  loops d=0..3 and requests 12+d. Native and MDL initialize signed cursor=0;
  only `nextRayUniform` advances it. Non-null bounds and null cursor contracts
  precede reads/fallback; exhaustion saturates and advances PCG once per request.
- Native coat selection is conditional; selected coat takes VNDF u then v and
  returns. Otherwise base selection precedes diffuse hemisphere u/v or specular
  VNDF u/v plus reflection/transmission selection. Early returns are retained;
  fifth request falls back to PCG. This is branch-dependent sequential assignment,
  not a universal per-dimension material semantic map. MDL xi x/y/z/w consume
  sequential requests. No multiple stateful draws remain in affected call
  argument lists; dimension-explicit pure Sobol calls are order-independent.
- Matte/native/MDL fourth-argument signatures still match the wrapper's direct
  call and SBT slot. Interactive still uses the nullptr default. There is no
  callable-layout or slot change.
- Geometry-light primitive/coordinate sampling remains PCG even for Quality
  QMC light selection. Volume transport/free-flight/phase and volume light
  requests remain PCG. Surface Russian roulette and stochastic coverage rejection
  remain PCG; coverage pass-through changes the subsequent vertex key, not the
  coverage RNG policy. Interactive's material and light/hemisphere fallbacks
  remain PCG. None was converted to new Sobol dimensions.
- FastOwen terminology denotes pbrt-style **FastOwenScrambler**, a hash-based
  fast Owen approximation, not pbrt's separate OwenScrambler or independent
  random permutations. No statistical/performance claim follows from the
  structural dyadic-stratum regression.

## Compatibility and deliberate exclusions

The complete table correction intentionally changes dimensions 7–15; fixing
positive-spp>1 addressing and explicitly selecting argument order also changes
old patterns. Old-image compatibility is **not** promised. Batching preserves
the underlying QMC ordinal prefix and key, not byte-identical images: PCG
initialization and floating-point accumulation can still differ. Reset starts
ordinal zero, whose camera pixel jitter is centered; checkerboarding retains
effective spp=1 and gives each pixel ordinal n in cycle n.

The float mapping remains `(scrambled >> 8) * 2^-24`, including legal zero,
maximum `1-2^-24`, and no one/clamp/exponential transform. Binary uint32 sampler
indices have period 2^32; supported positive-spp progression still requires
nonnegative completedSamples and no overflow of the existing signed frame
counter. No widened or infinite progression is claimed.

Seed zero preserves legacy **pixel keys**, not old images after other fixes.
For a fixed pixel distinct internal seeds give distinct keys via the bijective
finalizer; float collisions and arbitrary depth-tuple collisions remain possible.
Public full-render replicate control, camera/surface propagation, PCG replicate
policy and commit/reset semantics remain deferred. No new public parameter,
environment seed, mutable global or accumulation invalidation was introduced.

Explicitly untouched: inherited CDF tie behavior, non-positive pixelSamples,
volume transport and volume dimensions, BSDF physics/fixed-role redesign,
geometry-light QMC conversion, renderer-wide RNG replacement, lifecycle or
failed-frame redesign, broad finite-format proofs, historical principled-volume
documents, and GPU performance/convergence campaigns. None is repaired or
qualified by this record.

Formatting/scope verification:

```sh
clang-format --dry-run --Werror devices/rtx/device/renderer/Quality_ptx.cu
jj diff --name-only
```

Formatting passed. The diff contains only the Quality policy comments, this
qualification record and its README link: no scratch, build, benchmark, volume,
BSDF implementation or reference/spec changes. Integration bookmark is not moved
by this ticket; its tip must remain the code commit's parent at handoff.
