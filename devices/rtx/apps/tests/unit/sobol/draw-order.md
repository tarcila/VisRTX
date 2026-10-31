# Sequential material and light draw roles (ticket 04)

## Contract and scope

Every affected stateful pair is now drawn into named locals, u before v, before
calling a sampler or constructing a vector. MDL xi receives sequential x/y/z/w.
Native coat/base/reflection selections keep their original condition and place
in the control flow. No draw is hoisted across a branch or early return.
The four-element `NextRayQmc` layout, signed cursor, saturation and callable
fourth-argument types are unchanged. Interactive retains the nullptr default.

Dimensions 12–15 are the **first four sequential material requests**, including
categorical decisions; the fifth request uses the existing PCG stream. This is
branch-dependent consumption, not fixed semantic roles across materials.
Quality reseeds surface dimensions per bounce/transparency layer. Ambient and
HDRI cosine sampling intentionally share dimensions 4–5. Analytic coordinates
remain 6–7, HDRI row/column 6–7 and jitter 8–9. Geometry lights and volume draws
remain PCG; no transport equation, CDF tie, non-positive-spp, seed-control or
BSDF-physics change is included.

The selected explicit argument mapping can change old sample patterns. The old
C++ expression did not promise an evaluation order, so neither compiler-
independent old-image compatibility nor measured variance improvement is claimed.

## Tests observe production, not a replacement branch model

`rtx::DrawOrder` builds `shader_expressions.h` with `shader_expressions.py`.
Python 3 is a **test-build** dependency, not a device/runtime dependency. The
header is regenerated when any source supplying an expression changes:

- Native: the entire production `__direct_callable__nextRay` body, its pure
  physics helpers and clearcoat constant are compiled verbatim on the host.
  No branch or early return is reconstructed in the test. Only OptiX-only
  includes and unrelated entry points are omitted. `#line` preserves diagnostics.
- MDL: the actual shader statements from cursor initialization through
  `sample_data.xi` assignment are compiled verbatim. The wrapper supplies a
  float4 destination and returns it; it does not emulate MDL scattering.
- Hemisphere: both actual production overloads and the production basis helper.
- Rectangle: the actual uniform lambda and coordinate construction statements.
- HDRI: the actual lambda, CDF row/column selection, debug guard, jitter and UV
  expressions, with the production `inverseSampleCDF` and toolkit CUB LowerBound.

Host tests require the CUDA/OptiX/vector/CUB headers but do not initialize a GPU.
They use tagged supplied values, normal-incidence analytic direction references,
and controlled PCG state, never old direction-table vectors.

Coverage:

| Path | Observations |
|---|---|
| No coat, diffuse | base pick, hemisphere u/v; exactly 3 supplied values, no PCG |
| Coat selected | coat pick, VNDF u/v; exactly 3 supplied values, no base request |
| Coat rejected, diffuse | coat pick, base pick, hemisphere u/v; 4 supplied values |
| Coat rejected, specular reflection | coat pick, base pick, VNDF u/v; exactly one fifth-request PCG draw |
| Coat rejected, transmission | same pair, fifth PCG draw drives reflect/transmit; direction and continuation flag retained |
| No coat, specular | base pick, VNDF u/v, reflect pick fit in 4 supplied values |
| Invalid base local view | exits with zero requests |
| Selected coat invalid local view | exits after only coat pick |
| Zero base selection energy | exits with no base pick; with rejected coat, only coat pick consumed |
| Below-surface coat reflection | exits after coat pair |
| Below-surface base reflection | exits after reflection selection |
| MDL xi | x/y/z/w match tagged values; partial n=0..4 and nullptr verify exact PCG component/state ordering |
| PCG hemisphere | radius before azimuth, exactly 2 PCG advances |
| PCG rectangle | edge1 before edge2, exactly 2 PCG advances |
| PCG HDRI | row, column, jitter u/v, exactly 4 PCG advances |

Early-exit cases set n to precisely the permitted prefix, so an eager extra
request changes the controlled PCG state and fails. No-coat diffuse and selected
coat likewise use n=3 to detect a fourth request.

`rtx::LightDrawOrder` executes the **full production** rectangle and HDRI
samplers in a CUDA kernel (real device CDFs and a real float4 CUDA texture),
checking outgoing direction/position and exact PCG state. It also executes the
PCG hemisphere overload and checks the retained QMC light dimension mapping
and lack of PCG advancement. Allocation, launch, synchronization, copy or
texture errors fail, never skip; timeout is 60 seconds.

`TestMaterialSampling` executes Quality and Interactive with Matte and PBR;
the MDL configuration also executes explicit physicallyBasedMDL and inline raw
MDL diffuse material. A front-facing full-coverage plane under a hidden unit
HDRI must return Lambertian radiance approximately 0.5. This checks actual
OptiX callable execution, including PBR/MDL continuation's share of the MIS
partition; Matte intentionally returns a dead continuation with matching pdf=0.
This is an ABI/energy regression, not a statistical image-improvement claim.
Host tests qualify precise native branches/MDL component roles; API tests
qualify actual callable execution, not an internal trace of SDK decisions.
`TestSampleAddressing` separately retains fresh/reset/checkerboard and positive-
spp repeatability checks from ticket 03.

## TDD evidence

Each ordering fix was its own red/green slice. The repeated command was:

```sh
cmake --build _out/native --target rtx_test_DrawOrder -j 8
_out/native/rtx_test_DrawOrder
```

On GCC 16.2 the unfixed production expression failed, respectively:

1. `FAIL: no-coat diffuse assigns u before v`
2. `FAIL: coat VNDF assigns u before v`
3. `FAIL: base VNDF assigns u before v`
4. `FAIL: MDL xi x/y/z/w receive sequential supplied draws`
5. `FAIL: PCG-only hemisphere assigns radius before azimuth`
6. `FAIL: rectangle assigns first PCG draw to edge1, second to edge2`
7. `FAIL: HDRI draws row, column, then jitter u before v`

Each passed after only its corresponding named-local change, before the next
slice was added. An initial host compile required the missing GLM trigonometric
include in the harness; that setup failure is not counted as a behavioral red.
The full CUDA light test passed even before the light fixes on this nvcc (its
chosen old argument order happened to agree); the GCC expression tests exposed
the unspecified-order regression. Early exits, transmission, partial MDL
fallback, QMC light mapping and API ABI tests qualify retained behavior, not
additional physics changes.

## Static audit

Rechecked all `owenSobol`, camera/hemi wrappers, `NextRayQmc`, `nextRayUniform`
and `materialNextRay` callsites under `devices/rtx/device`:

- Camera wrapper requests 0–3; center override and ray-buffer bypass unchanged.
- Hemisphere calls request 4–5; simultaneous dimension-explicit calls are pure.
- Sphere/ring already draw 6 then 7 as separate statements; rectangle now does
  too. HDRI draws 6,7,8,9 in explicit order. No geometry-light draws converted.
- Quality category/instance requests are 10/11. Its sole overlay producer sets
  n=4 and fills 12+d for d=0..3, using the camera ordinal and bounce seed.
- Native and MDL start cursor=0. Matte takes the same fourth argument and
  ignores it. `gpu/evalShading.h` still passes state/ray/PCG/QMC in that order
  to the same SBT slot; no layout, signature or slot changes.
- Interactive still calls the three-argument wrapper, whose fourth argument
  defaults to nullptr. Exhaustion and null behavior remain the tested production
  `nextRayUniform` contract, not a new stream or a changed capacity.
- No multiple stateful draws remain in affected sampler argument lists.
  Geometry-light single-draw arguments and independent pure Sobol arguments
  are outside the ordering defect and unchanged.

## Verification environment and exact commands

2026-10-31, isolated workspace
`/home/tarcila/Code/ANARI/visrtx/sobol-improvements-04`, initially based on
integration tip `55e607e1`. Fresh ignored workspace-local builds, no origin or
installed device paths. GCC 16.2.0, CMake 4.4.4, Ninja 1.13.2, CUDA 13.4.92,
NVIDIA driver 615.78.08, RTX 5880 Ada Generation, OptiX headers 9.1,
ANARI SDK 0.16.0, mdl-sdk-2026.0.2 (SDK metadata 2025.0.3). As recorded in
[README.md](README.md), dependencies come from the Nix shell. CUDA C++17,
compute_75 PTX, `--use_fast_math --expt-relaxed-constexpr`.

```sh
cmake -S . -B _out/native -G Ninja -DVISRTX_BUILD_GL_DEVICE=OFF -DVISRTX_BUILD_TESTS=ON -DVISRTX_ENABLE_MDL_SUPPORT=OFF -DOPTIX_FETCH_VERSION=9.1 -DCMAKE_BUILD_TYPE=Release
cmake --build _out/native -j 8
cmake -S . -B _out/mdl -G Ninja -DVISRTX_BUILD_GL_DEVICE=OFF -DVISRTX_BUILD_TESTS=ON -DVISRTX_ENABLE_MDL_SUPPORT=ON -DVISRTX_USE_MDL_FOR_PHYSICALLY_BASED=ON -DOPTIX_FETCH_VERSION=9.1 -DCMAKE_BUILD_TYPE=Release
cmake --build _out/mdl -j 8

native='^(rtx::(DrawOrder|LightDrawOrder|Sobol.*|SamplerContracts.*|SampleOrdinal)|TestMaterialSampling|TestSampleAddressing|TestPbrSpecularDefault|TestPowerLightPick|TestPowerLightPickEnv|TestAlphaCutoffMode|TestRayBufferCamera|TestEmissiveInteractive)$'
mdl='^(rtx::(DrawOrder|LightDrawOrder|Sobol.*|SamplerContracts.*|SampleOrdinal)|TestMaterialSampling|TestSampleAddressing|TestPbrSpecularDefault|TestPowerLightPick|TestPowerLightPickEnv|TestAlphaCutoffMode|TestRayBufferCamera|TestEmissiveInteractive|TestEmissiveMaterialParity)$'
LD_LIBRARY_PATH="$PWD/_out/native:${LD_LIBRARY_PATH:-}" ctest --test-dir _out/native -R "$native" --output-on-failure
LD_LIBRARY_PATH="$PWD/_out/mdl:${LD_LIBRARY_PATH:-}" ctest --test-dir _out/mdl -R "$mdl" --output-on-failure
```

Full native and MDL builds passed, compiling the affected material and renderer
PTX. Final regression selections: **native 17/17 passed; MDL-backed PBR 18/18
passed**, no skipped or unavailable gates. This includes host/CUDA sampler
parity, assert/NDEBUG contract child-process gates, ordinal/addressing tests,
all new draw-order/runtime tests and the required existing integration tests.
See [contracts.md](contracts.md) for the abort/trap allowlist/protocol.

```sh
LD_DEBUG=libs LD_LIBRARY_PATH="$PWD/_out/native:${LD_LIBRARY_PATH:-}" _out/native/TestMaterialSampling > /tmp/sobol04-loader-native.log 2>&1
LD_DEBUG=libs LD_LIBRARY_PATH="$PWD/_out/mdl:${LD_LIBRARY_PATH:-}" _out/mdl/TestMaterialSampling > /tmp/sobol04-loader-mdl.log 2>&1
```

Loader `calling init:` identifies the absolute workspace-local
`_out/native/libanari_library_visrtx.so` and `_out/mdl/libanari_library_visrtx.so`.
Both runtime commands pass. Native mean radiance: Quality Matte 0.499993162,
PBR 0.500014328; Interactive Matte 0.500552112, PBR 0.500126390. MDL means:
Quality Matte 0.499993162, PBR/wrapper/raw MDL each 0.500273009; Interactive
Matte 0.500552112, PBR/wrapper/raw MDL each 0.500125931. Both launches pass
for each combination. The built libraries are loaded, not merely linked headers.

Formatted all changed C++/CUDA with repository clang-format; verified with:

```sh
clang-format --dry-run --Werror devices/rtx/device/gpu/gpu_util.h devices/rtx/device/gpu/owenSobol.h devices/rtx/device/gpu/sampleLight.h devices/rtx/device/gpu/shadingState.h devices/rtx/device/material/shaders/MDLShader_ptx.cu devices/rtx/device/material/shaders/PhysicallyBasedShader_ptx.cu devices/rtx/apps/tests/unit/test_DrawOrder.cpp devices/rtx/apps/tests/unit/test_LightDrawOrder.cu devices/rtx/apps/tests/api/TestMaterialSampling.cpp
```

This is ticket-04 evidence, not ticket-06 integrated qualification. No scramble
control, volume work, physics redesign, variance or performance claim is made.
