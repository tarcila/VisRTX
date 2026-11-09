# Internal scramble control (ticket 05)

## Contract and limits

The production seam is `owenPixelSeed(x, y, uint32_t scrambleSeed = 0)`.
It returns `legacyPixelSeed(x,y) XOR M(scrambleSeed)`, where M is the existing
pixel hash's finalizer, now shared privately in `detail`. XOR-right-shifts
are invertible (recover bits from high to low), and multiplication by an odd
integer is invertible modulo 2^32. Both constants, `0x7feb352d` and
`0x846ca68b`, are odd. Every operation fixes zero, so M is bijective and M(0)=0.
Thus zero preserves **pixel keys**, and distinct seeds for a fixed pixel yield
distinct keys. This is not a claim of independent samples: output floats can
collide. Bounce/transparency mixing is unchanged; arbitrary depth tuples are
not promised collision-free.

The camera (`gpu/cameraCreateRay.h`) and Quality surface
(`renderer/Quality_ptx.cu`) production callers still use two arguments and
therefore the zero default. No renderer parameter, environment control, mutable
seed, PCG change, or accumulation invalidation was added. Public full-render
replicate control remains deferred: camera/surface propagation, PCG replicate
policy, discovery/defaults, commit/reset behavior and full-render repeatability
must be specified together in future work. Internal alternate sampler scrambles
are not full-render independent replicates. Zero-seed compatibility does not
promise old images after the table, addressing and draw-order corrections.

## Tests and reference construction

The agreed seam is the actual corrected production pixel-seed/sampler helpers,
compiled for host and device, not a replacement sampler. TDD's first slice
added fixed pixel keys before production changes: building `rtx_test_Sobol`
failed with “too many arguments” at the new three-argument calls. Adding the
optional input and XOR-finalizer combination made `rtx::Sobol` pass. The key
cases include default and explicit zero, seeds 1/0x12345678/0xffffffff, pixels
(0,0), (1,2), (1920,1080), and (UINT32_MAX,UINT32_MAX), including uint32 wrap.
Repeated calls assert deterministic keys.

The subsequent retained-depth/sampler characterization adds 768 offline known
answers in `seed-vectors.txt`: pixel (1,2), all four seeds, bounce/transparency
(0,0)/(1,0)/(0,1)/(3,7), indices 0/16/UINT32_MAX, all 16 dimensions. Each row
asserts literal pixel and vertex keys and exact 24-bit float-grid output.
Selected nonzero seeds change the camera vector; the fixed depth keys differ
for the selected bounce and coverage cases. No assertion claims independence,
unique floats, or universal tuple separation. Repeated evaluation asserts
sample reproducibility.

`seed_reference.py` regenerates these offline without reading production code:
it reconstructs direction columns from the attributed Joe–Kuo parameter fixture,
uses arbitrary-precision Python with explicit uint32 masks, string bit reversal,
and the pinned pbrt FastOwen operations described in README.md. Ordinary test
execution reads the frozen fixture, not the generator or network data. SHA-256:

```
b7b95dab865b48826f5582e4aa0805ec8d548ebef4b41eeb9b27692f7f7a2d84  seed-vectors.txt
```

Both Sobol executables check these answers; CUDA additionally compares pixel
keys, vertex keys, raw/scrambled integers and exact float bits (including camera
and hemisphere wrappers) against host evaluation. Total parity cases increased
from 996 to 1764. CUDA failure is not skipped. Existing parameter/column,
pinned FastOwen, endpoint and dyadic-stratum tests remain enabled.

## Verification: 2026-10-31

Fresh local workspace `sobol-improvements-05`, parent/integration tip `f3096750`.
No preceding workspace's binaries were used. GCC 16.2.0, CMake 4.4.4, Ninja
1.13.2, nvcc/runtime 13.4.92 (runtime/driver API 13040), NVIDIA driver 615.78.08,
RTX 5880 Ada. OptiX 9.1, ANARI SDK 0.16.0, MDL package 2026.0.2 (SDK CMake
metadata 2025.0.3). Same Nix environment as the preceding tickets. C++/CUDA17,
compute_75 PTX; sampler CUDA uses `--use_fast_math --expt-relaxed-constexpr`.

Exact fresh configure and full build commands (all passed, including RTX PTX):

```sh
cmake -S . -B _out/native -G Ninja -DVISRTX_BUILD_GL_DEVICE=OFF -DVISRTX_BUILD_TESTS=ON -DVISRTX_ENABLE_MDL_SUPPORT=OFF -DOPTIX_FETCH_VERSION=9.1 -DCMAKE_BUILD_TYPE=Release
cmake --build _out/native -j 8
cmake -S . -B _out/mdl -G Ninja -DVISRTX_BUILD_GL_DEVICE=OFF -DVISRTX_BUILD_TESTS=ON -DVISRTX_ENABLE_MDL_SUPPORT=ON -DVISRTX_USE_MDL_FOR_PHYSICALLY_BASED=ON -DOPTIX_FETCH_VERSION=9.1 -DCMAKE_BUILD_TYPE=Release
cmake --build _out/mdl -j 8
```

```sh
native='^(rtx::(Sobol.*|SamplerContracts.*|SampleOrdinal|DrawOrder|LightDrawOrder)|TestPbrSpecularDefault|TestPowerLightPick|TestPowerLightPickEnv|TestAlphaCutoffMode|TestRayBufferCamera|TestEmissiveInteractive|TestSampleAddressing|TestMaterialSampling)$'
mdl='^(rtx::(Sobol.*|SamplerContracts.*|SampleOrdinal|DrawOrder|LightDrawOrder)|TestPbrSpecularDefault|TestPowerLightPick|TestPowerLightPickEnv|TestAlphaCutoffMode|TestRayBufferCamera|TestEmissiveInteractive|TestEmissiveMaterialParity|TestSampleAddressing|TestMaterialSampling)$'
LD_LIBRARY_PATH="$PWD/_out/native:${LD_LIBRARY_PATH:-}" ctest --test-dir _out/native -R "$native" --output-on-failure
LD_LIBRARY_PATH="$PWD/_out/mdl:${LD_LIBRARY_PATH:-}" ctest --test-dir _out/mdl -R "$mdl" --output-on-failure
```

Results: **native 17/17 passed**, **MDL-backed PBR 18/18 passed**. Includes host
SIGABRT and isolated CUDA negative tests under assertions and NDEBUG with the
existing finite timeouts/error allowlist (see contracts.md). Both CUDA contract
baselines succeeded; negative cases reported `cudaErrorLaunchFailure`
(719) in both build modes and configurations. Both Sobol CUDA runs passed all 1764 exact comparisons.

`TestMaterialSampling` executes Quality and Interactive with Matte and PBR in
both builds, plus explicit MDL in the MDL build. `rtx::DrawOrder` covers tagged
native no-coat, coat-selected, rejected-coat diffuse and specular fifth-PCG-draw
branches and MDL xi assignment via production expressions; `rtx::LightDrawOrder`
executes PCG hemisphere, ambient/HDRI and analytic-light fallbacks (see
[draw-order.md](draw-order.md) for limits). Image tests alone do not prove each
native branch or MDL xi component was taken. Existing ordinal/reset/checkerboard
and camera-sensitive API regressions passed. No new dimension, cursor or QMC
capacity callsite was added; the existing audit in contracts.md remains valid.
The only production pixel-seed callers are the two default-zero sites above.

Loader provenance (both passed):

```sh
LD_DEBUG=libs LD_LIBRARY_PATH="$PWD/_out/native:${LD_LIBRARY_PATH:-}" _out/native/TestMaterialSampling > /tmp/sobol05-loader-native.log 2>&1
LD_DEBUG=libs LD_LIBRARY_PATH="$PWD/_out/mdl:${LD_LIBRARY_PATH:-}" _out/mdl/TestMaterialSampling > /tmp/sobol05-loader-mdl.log 2>&1
grep 'calling init:.*libanari_library_visrtx' /tmp/sobol05-loader-{native,mdl}.log
```

The `calling init:` lines identify this workspace's absolute
`_out/native/libanari_library_visrtx.so` and `_out/mdl/libanari_library_visrtx.so`.

Reproduction/format checks (passed):

```sh
python3 devices/rtx/apps/tests/unit/sobol/seed_reference.py
sha256sum devices/rtx/apps/tests/unit/sobol/seed-vectors.txt
PYTHONPYCACHEPREFIX=/tmp/sobol05-pycache python3 -m py_compile devices/rtx/apps/tests/unit/sobol/seed_reference.py
clang-format --dry-run --Werror devices/rtx/device/gpu/owenSobol.h devices/rtx/apps/tests/unit/test_Sobol.cpp
```

No required ticket-05 tooling was unavailable or waived. This record does not
resolve ticket 06's consolidated qualification scope, nor claim volume/BSDF,
CDF-tie or non-positive-spp repairs, convergence improvement or performance.
