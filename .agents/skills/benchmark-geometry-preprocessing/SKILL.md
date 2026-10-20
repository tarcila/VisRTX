---
name: benchmark-geometry-preprocessing
description: Run the RTX geometry-preprocessing benchmark suite and interpret the numbers. Use when measuring a geometry preparation, preprocessing, BVH-readiness or uniform-update change, comparing a candidate against a pinned control revision, or deciding whether a measured delta justifies promoting, reworking or dropping a change.
---

# Geometry preprocessing benchmark

Produces a paired control-vs-candidate latency comparison for RTX geometry
preparation, then a promote / rework / drop recommendation. The suite measures
**complete paths to GPU-ready, renderable data** — input upload, temporaries,
derived-data upload, scalar readback, registry publication, acceleration-structure
work — not isolated kernels. Accounting details and the historical results live
in `devices/rtx/docs/geometry-preprocessing-performance.md` and
`devices/rtx/docs/geometry-uniform-updates.md`.

Numbers from this suite are **pilots**. They carry scheduling and thermal noise
and no confidence intervals. Report ranges, and treat a single trial's ratio as
a hint, not a result.

## 1. Pin the control

Ask the user for the control revision when they have not named one, or read it
from the relevant doc's Measurement section. Check that revision out into a
**second working tree**, using whichever VCS the checkout uses — `git worktree
add` or `jj workspace add`. Never rewrite the current working tree to get a
control build: the candidate must stay buildable throughout.

Record the control revision id; every later claim is relative to it. Place both
working trees and both build directories wherever the host keeps scratch space,
and use those paths consistently in the commands below.

## 2. Build both sides matched

Mismatched MDL / MaterialX / OptiX / compiler / CUDA-architecture settings
invalidate the comparison. Configure **both** working trees identically:

```bash
cmake -S SOURCE -B BUILD -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DVISRTX_BUILD_GL_DEVICE=OFF -DVISRTX_ENABLE_MDL_SUPPORT=OFF \
  -DVISRTX_ENABLE_MATERIALX_SUPPORT=OFF -DVISRTX_ENABLE_NVML=ON \
  -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared
```

Disabling the optional components keeps the comparison narrow and the builds
short; what matters is that both sides agree. The CUDA architecture defaults to
`VISRTX_MIN_ARCH`-virtual (see `devices/rtx/CMakeLists.txt`); leave it alone
unless reproducing an existing table, which records the architecture it used.
Custom library name suffixes must also match on both sides.

Build `anari_library_visrtx` in the control. Add `-DVISRTX_BUILD_TESTS=ON` in the
candidate and build `BenchmarkGeometryPreprocessing`. The control does not need
the benchmark sources: one executable runs against either ABI-compatible library.

Confirm the loader picks the intended device library **and** a real CUDA driver:

```bash
LD_LIBRARY_PATH=CONTROL_BUILD ldd CANDIDATE_BUILD/BenchmarkGeometryPreprocessing
```

A development environment that resolves `libcuda` to a CUDA toolkit link-time
stub is not ready to benchmark: fix the environment, since measurements taken
against stubs are discarded rather than reported. The runner injects only the
device library under test and inherits everything else.

## 3. Run the suite

```bash
python3 devices/rtx/apps/tests/api/benchmark_preprocessing.py \
  CANDIDATE_BUILD/BenchmarkGeometryPreprocessing \
  CONTROL_BUILD CANDIDATE_BUILD RESULTS.json
```

Defaults: five cases, three fresh-process trials each, 21 iterations, alternating
control/candidate order by trial, first iteration excluded from warm statistics.
`--trials` and `--repeats` raise the budget; `--case NAME` narrows to one default
case; `--extra-case NAME:COUNT:GEOMETRIES:MODE:FRAMES` adds a workload the default
matrix misses. `--help` lists everything; the benchmark's own argument meanings are
in the docs' "Benchmark interface and accounting" section.

The runner writes incrementally and stops on the first failure, so a partial JSON
is a failed run, not a short one. Labels `base`/`branch` in the JSON mean control
and candidate.

Targeted-case guidance: the optimization's own case needs the most trials; the
other cases are **regression guards**. When a guard moves, re-run that case alone
with higher `--trials` and `--repeats` before calling it a regression.

## 4. Decide and propose

Report, per case: control median, candidate median, ratio, and the **per-trial
median range** on both sides. Then take exactly one position per case:

- **Overlapping trial ranges are noise.** Say so; do not report the ratio as an
  effect. Buy resolution with more trials, not with a stronger adjective.
- **A large win on the targeted case plus clean guards** supports promotion. State
  what the win does *not* cover: the untested geometry types, input residencies,
  renderer settings and threading in the suite's blind spots.
- **A single-digit-percent win** is below this suite's resolution and does not
  justify a semantic change (removing a wait, relaxing an error boundary). Keep
  the existing fences and say which cheaper target to attack instead.
- **A guard regression** blocks promotion until a dedicated re-run either clears
  it or bounds it, even when the targeted case wins big.

Back a headline win with a work-elimination check rather than timing alone —
otherwise a "win" can be skipped work. Count kernel instances in a CUDA trace of
each side:

```bash
nsys profile --trace=cuda --sample=none -o TRACE \
  env LD_LIBRARY_PATH=BUILD_DIR \
  BUILD_DIR/BenchmarkGeometryPreprocessing cylinder 32 0 6 1 64 replace 0 1
nsys stats --report cuda_api_sum,cuda_gpu_mem_size_sum TRACE.nsys-rep
```

Traces are local scratch, not repository artifacts.

Profiled totals cover whole-process time including initialization and teardown;
never compare them against the unprofiled table.

A timing result alone never promotes a change. Pair it with the affected CTests
and, for anything touching retirement or device memory, Compute Sanitizer.

## 5. Record the evidence

Each JSON under `devices/rtx/docs/` is frozen evidence for one pinned comparison
on one host, not a refreshable baseline. A new comparison gets a **new** file
plus a doc section; leave the existing files untouched.

When committing results, state in the doc section: both revision ids, GPU, driver,
CUDA and OptiX versions, build settings, trials × iterations, and the per-trial
ranges. Omit controlled-clock or confidence-interval language — this suite
establishes neither.
