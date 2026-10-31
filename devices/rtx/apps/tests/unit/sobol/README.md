# Sobol reference foundation (ticket 01)

For the subsequent dimension/QMC misuse policy, isolated abort/trap tests,
callsite audit and ticket-02 verification, see [contracts.md](contracts.md).
For contiguous ordinals, camera/reset/checkerboard regressions and ticket-03
verification, see [addressing.md](addressing.md). For sequential native/MDL
material draws, PCG light/hemisphere fallbacks, production-expression tests and
native/MDL callable execution, see [draw-order.md](draw-order.md). The latter's
host shader-expression test adds a Python 3 test-build dependency.
For optional internal pixel-key scrambles, fixed depth/sample vectors and
ticket-05 verification, see [scramble.md](scramble.md).
For the combined host/CUDA/native/MDL gates, loaded-library evidence and final
sampling-policy audit, see [qualification.md](qualification.md) (ticket 06).

## Contract and compatibility

Tests call the production `gpu/owenSobol.h` seam, not a replacement sampler.
`sobolUInt(index, dimension)` is the **binary-index** XOR matrix product; it
neither scrambles nor converts the index to Gray code. There are 16 dimensions,
32 columns and a uint32 index period of 2^32. The table is immutable on the host
and in CUDA constant memory.

`owenScramble` retains the integer operations of pbrt's **FastOwenScrambler**,
a hash-based fast Owen approximation. It is not pbrt's separate OwenScrambler,
nor a proof of independent random permutations or independent dimensions.
Dimension-dependent seed mixing is unchanged. `sobolFloat` exposes the existing
conversion `(scrambled >> 8) * 2^-24` for direct endpoint testing: zero is legal,
one is impossible, and the maximum is exactly `1 - 2^-24`. No clamp or
exponential-input transformation is applied.

Replacing the full table intentionally changes sample patterns in dimensions
7–15, including high-bit samples; old images are not a compatibility promise.
This is a reference/structural correctness fix, **not** a measured convergence,
multidimensional discrepancy, or performance improvement. Default pixel keys
and bounce/transparency mixing are unchanged; ticket 05 adds an optional
internal scramble input at the pixel-seed seam only. Misuse guards and addressing
are covered by the linked ticket-02 and ticket-03 records; draw ordering is
covered by the linked ticket-04 record. Integrated qualification remains with
ticket 06.

## Primary sources and extraction

- Production matrix: [pbrt-v4 sobolmatrices.cpp](https://github.com/mmp/pbrt-v4/blob/b0bdf461ba5dd6504b70ebf0932acc0fe7aa744f/src/pbrt/util/sobolmatrices.cpp),
  revision `b0bdf461ba5dd6504b70ebf0932acc0fe7aa744f`.
  Extract `SobolMatrices32` rows 0..15, columns 0..31. **Source row stride is
  52**, not 32. The derived `gpu/sobolDirections.h` retains the pbrt Apache-2.0
  and Leonhard Gruenschloss MIT notices and Joe–Kuo publication attribution.
- Independent parameters: [joe-kuo/sobol_data new-joe-kuo-6.21201](https://github.com/joe-kuo/sobol_data/blob/835a7d7b1ee3bc83e575e302a985c66ec4b65249/new-joe-kuo-6.21201),
  revision `835a7d7b1ee3bc83e575e302a985c66ec4b65249`. `joe-kuo.txt` contains
  three LF-terminated attribution comments followed by the source's unchanged
  header and first 15 parameter rows (one-based dimensions 2..16), preserving
  whitespace and LF endings. These inputs were downloaded independently, never
  reverse-engineered from production columns. See `LICENSE.joe-kuo`.
- Integer scramble: [pbrt-v4 lowdiscrepancy.h](https://github.com/mmp/pbrt-v4/blob/b0bdf461ba5dd6504b70ebf0932acc0fe7aa744f/src/pbrt/util/lowdiscrepancy.h)
  at the same pbrt revision. `FastOwenScrambler.reference.txt` is its unmodified
  copyright header and FastOwenScrambler excerpt. See `LICENSE.pbrt` and
  `NOTICE`. Before freezing vectors, the production operation sequence was
  compared to this source: reverse bits; XOR multiply `0x3d20adea`; add seed;
  multiply `(seed >> 16) | 1`; XOR multiply `0x05526c56`; XOR multiply
  `0x53a22864`; reverse bits. All arithmetic is uint32. Tests also compile the
  pinned excerpt as an independent host oracle with bit-at-a-time reversal.
  Binary distributions install the upstream licenses and NOTICE.

### Required content digests (SHA-256)

- Exactly 512 uint32 words, dimension-major / column-minor, **little-endian
  bytes without any header**:
  `357f257f40c9f9b7efb784bdf355b543f456984987e8519686fc978cf319be96`
- Exact UTF-8 `joe-kuo.txt` fixture bytes, including attribution/header/LFs:
  `d155655216d8f60bac3d2b34f2926cf03b64e288c7df74d6c25c13cccce3b26f`
- Additional frozen-vector integrity digest (`vectors.txt`):
  `a25db35646e5f5efc17500bff4f6d957102d2c5ea78cd3baee8813919eb779f9`

`reference.py` verifies the full downloaded source-file hashes before extraction.
It independently regenerates all 512 columns from Joe–Kuo inputs and stops with
both values and zero-based dimension/column on any mismatch with pinned pbrt.
The references agreed in every column; no oracle adjustment was needed.

Dimension 0 uses `v[i] = 1u << (31-i)`. For other dimensions, degree `s`,
polynomial encoding `a` and all initial numerators `m` come from the fixture:
`v[i] = m[i] << (31-i)` for `i < s`. For `i >= s`, start with
`v[i-s] XOR (v[i-s] >> s)` and XOR `v[i-k]` for `k=1..s-1` when
bit `s-1-k` of `a` is set. Shifts are logical and operations uint32; bit zero
of `a` is its LSB. Dimension 7 is one-based dimension 8: degree 5, `a=2`,
`m={1,1,5,5,17}`. Its column 4 is `17 << 27 = 0x88000000`, not `0x28000000`.

`vectors.txt` covers indices 0,1,2,3,15,16,31,32,0x80000000,0xffffffff in every
dimension at pixel seeds 0,0x12345678,0xffffffff. Raw samples use pinned pbrt
columns; the separate arbitrary-precision Python scramble rendition masks to
uint32 and uses string-based reversal. Float expectations are exact grid
integers, checked separately from integer scramble expectations. Host tests
independently reconstruct every production column from primitive parameters.

### Header regeneration regression

`rtx::SobolTableHeader` compares the complete generated header byte-for-byte
against `gpu/sobolDirections.h`, including the upstream notice, `#pragma once`
and the namespace/storage comments. It uses the same formatter as `reference.py`
with the vendored notice and independently regenerated Joe–Kuo columns; no
network, CUDA compiler or GPU is needed to run the script directly:

```sh
python3 -B devices/rtx/apps/tests/unit/sobol/test_table_header.py
```

### Recreate fixtures (network required only for this maintenance step)

From the workspace root:

```sh
pbrt=b0bdf461ba5dd6504b70ebf0932acc0fe7aa744f
joe=835a7d7b1ee3bc83e575e302a985c66ec4b65249
curl -fL https://raw.githubusercontent.com/mmp/pbrt-v4/$pbrt/src/pbrt/util/sobolmatrices.cpp -o /tmp/sobol-01-pbrt.cpp
curl -fL https://raw.githubusercontent.com/mmp/pbrt-v4/$pbrt/src/pbrt/util/lowdiscrepancy.h -o /tmp/sobol-01-lowdiscrepancy.h
curl -fL https://raw.githubusercontent.com/mmp/pbrt-v4/$pbrt/LICENSE.txt -o /tmp/sobol-01-pbrt-license
curl -fL https://raw.githubusercontent.com/joe-kuo/sobol_data/$joe/new-joe-kuo-6.21201 -o /tmp/sobol-01-joe-kuo
curl -fL https://raw.githubusercontent.com/joe-kuo/sobol_data/$joe/LICENSE -o /tmp/sobol-01-joe-license
python3 devices/rtx/apps/tests/unit/sobol/reference.py \
  /tmp/sobol-01-pbrt.cpp /tmp/sobol-01-joe-kuo \
  --fast-owen-source /tmp/sobol-01-lowdiscrepancy.h \
  --pbrt-license /tmp/sobol-01-pbrt-license \
  --joe-license /tmp/sobol-01-joe-license \
  --table devices/rtx/device/gpu/sobolDirections.h
```

Ordinary builds/tests need **no downloads of reference data** and no pbrt runtime
or Python dependency for the sampler tests. As elsewhere in this project,
configure can fetch OptiX headers. The host sampler test needs CUDA toolkit
vector headers but does not initialize a GPU. The CUDA test requires a working
GPU and fails rather than skips when one is unavailable.

## Verification environment and commands

Run on 2026-10-31 in the isolated `sobol-improvements-01` workspace, parent
`5c7ea1a5`. The parent Sobol workspace had no build cache; its handoff identified
the origin volume build `_out/_cmake`, which was **not used**. Both configurations
below were fresh local builds, not installed or origin-volume devices.

Environment: Nix-provided GCC 16.2.0, CMake 4.4.4, Ninja 1.13.2, CUDA nvcc/runtime
13.4.92, NVIDIA driver 615.78.08 (CUDA driver API 13040), RTX 5880 Ada Generation.
OptiX headers 9.1, ANARI SDK 0.16.0, MDL package `mdl-sdk-2026.0.2` (CMake reports
SDK metadata version 2025.0.3). These packages were already available in the
shell's `NIXPKGS_CMAKE_PREFIX_PATH` / compiler include and library paths; no
origin build was added to any search path. Minimum arch was the CUDA-13 default
75, emitting compute_75 PTX, C++/CUDA17. CUDA sampler parity uses
`--use_fast_math --expt-relaxed-constexpr`, matching RTX arithmetic options.

Exact configure/build commands:

```sh
cmake -S . -B build-native -G Ninja -DVISRTX_BUILD_GL_DEVICE=OFF -DVISRTX_BUILD_TESTS=ON -DVISRTX_ENABLE_MDL_SUPPORT=OFF -DOPTIX_FETCH_VERSION=9.1 -DCMAKE_BUILD_TYPE=Release
cmake --build build-native -j 8
cmake -S . -B build-mdl -G Ninja -DVISRTX_BUILD_GL_DEVICE=OFF -DVISRTX_BUILD_TESTS=ON -DVISRTX_ENABLE_MDL_SUPPORT=ON -DVISRTX_USE_MDL_FOR_PHYSICALLY_BASED=ON -DOPTIX_FETCH_VERSION=9.1 -DCMAKE_BUILD_TYPE=Release
cmake --build build-mdl -j 8
```

These `build-*` names are not ignored by the repository. They were excluded
from local jj auto-tracking and untracked before committing. Future workspaces
can instead use the repository-ignored `_out/native` and `_out/mdl` paths.

```sh
native='^(rtx::Sobol.*|TestPbrSpecularDefault|TestPowerLightPick|TestPowerLightPickEnv|TestAlphaCutoffMode|TestRayBufferCamera|TestEmissiveInteractive)$'
mdl='^(rtx::Sobol.*|TestPbrSpecularDefault|TestPowerLightPick|TestPowerLightPickEnv|TestAlphaCutoffMode|TestRayBufferCamera|TestEmissiveInteractive|TestEmissiveMaterialParity)$'
LD_LIBRARY_PATH="$PWD/build-native:${LD_LIBRARY_PATH:-}" ctest --test-dir build-native -R "$native" --output-on-failure
LD_LIBRARY_PATH="$PWD/build-mdl:${LD_LIBRARY_PATH:-}" ctest --test-dir build-mdl -R "$mdl" --output-on-failure
```

Results: native 8/8 passed; MDL-backed PBR 9/9 passed. Full native and MDL builds
compiled the affected RTX PTX. Each sampler executable checks 512 oracle
columns, 480 vectors, conversion endpoints, bit reversal, camera/hemi component
mapping, and dyadic 1D strata for all dimensions, three fixed seeds and m=1..10.
CUDA additionally compares 996 cases (512 basis + 480 vectors + 4 endpoints)
against host results, with exact integer and float-bit equality including the
camera/hemi wrappers. Timeout is 60 seconds; CUDA availability/launch/sync/copy
failures fail the test, never count as expected success.

Loader provenance was checked with the following command for each build dir;
`calling init:` identified the absolute workspace-local
`build-native/libanari_library_visrtx.so` and `build-mdl/libanari_library_visrtx.so`:

```sh
LD_DEBUG=libs LD_LIBRARY_PATH="$PWD/build-native:${LD_LIBRARY_PATH:-}" build-native/TestRayBufferCamera > /tmp/sobol-01-native-loader.log 2>&1
LD_DEBUG=libs LD_LIBRARY_PATH="$PWD/build-mdl:${LD_LIBRARY_PATH:-}" build-mdl/TestRayBufferCamera > /tmp/sobol-01-mdl-loader.log 2>&1
grep 'calling init:.*libanari_library_visrtx' /tmp/sobol-01-*-loader.log
```

TDD evidence: the first oracle test failed on the old table with 266 column
mismatches plus the explicit dimension-7 regression (267 failures); replacing
the whole table made it pass. The separate direct-conversion slice first failed
to compile because `sobolFloat` did not exist; extracting the unchanged
conversion into that production helper made it pass. Subsequent reference,
structural and CUDA checks qualify retained behavior, not new scramble logic.
Formatting: run repository clang-format on both production headers and both
C++/CUDA test sources, then `clang-format --dry-run --Werror` on those files.

No required ticket-01 tooling was unavailable. This is not ticket-06 integrated
qualification: misuse trap/abort coverage, ordinal/reset/checkerboard tests,
sequential material draw-role coverage and controlled alternate pixel-key
scrambles remain with their respective tickets. No volume, CDF-tie,
non-positive-spp or BSDF-physics repair is claimed.
