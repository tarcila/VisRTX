# Quality volume baseline driver

> Sampling integration status: recorded images, convergence and timings predate
> the Halton-camera / compact Sobol-transport split. They are historical evidence,
> not qualification of the current sampling policy. Seed zero preserves the new
> Halton baseline, not the former all-Sobol images. See the [sampling contract](../../devices/rtx/apps/tests/unit/sobol/README.md)
> for the seed policy and outstanding rendering requalification. Optical/unit
> checks that do not consume camera or transport samples remain applicable.

This runner creates controlled scenes through public ANARI calls and writes linear
RGBA plus actual sample counts, CUDA-event duration, host/map time and GPU-wide
memory usage. It does not change renderer behavior or control GPU clocks.

Build with the ANARI SDK and CUDA toolkit on CMake's search path:

```sh
cmake -S tools/principled_volume -B _out/quality-baseline-build \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_QUALITY_BASELINE=ON
cmake --build _out/quality-baseline-build --target quality_baseline --parallel
```

Runner arguments:

```text
quality_baseline LIBRARY RENDERER SCENE ASSETS OUTPUT SIZE SPP LAUNCHES DEPTH RATE [MOTION] [RANDOM_SEED] [SHADOW_RATE] [FIELD_SAMPLING] [SOURCE_SAMPLING]
```

`SCENE` is `cloud`, `plume`, `support`, `overlap`, `receiver-matte` or
`receiver-native`, or `receiver-near`. The near fixture places a nonuniform
RGB emitter just outside the camera footprint above a Matte receiver, isolating
source-sampling variance without visible primary emission. It is a synthetic
benchmark, not an analytic golden. `ASSETS` is the canonical cell-v2 `datasets` directory. `LIBRARY`
is the ANARI library name (for this workspace, `visrtx_principle_volume`). The
runner writes every image at sizes <=128; larger runs write only their final image.
`MOTION` defaults to `static`; other modes are `camera`, `light`, `instance`,
`density`, `temperature`, and `material`. Motion fixtures require cloud/plume;
temperature/material require plume. Density/temperature edits change binding
multipliers, not voxel bytes. Wall timings include edit/preparation work.

`SHADOW_RATE` defaults to `RATE` for compatibility. Supply it explicitly to
measure camera sampling independently of shadow sampling. Both rates must be
finite and at least the renderer minima (0.001 camera, 0.0001 shadow). For example,
append `static 0 1` after `RATE=0.75` to use camera/shadow rates 0.75/1.
The archive driver likewise accepts `[volumeRate] [shadowRate]` after its
`controlled|viewer` argument; its defaults remain 1/1. It explicitly preserves
floating-point ANARI parameter types even for integer-looking CLI values.

`FIELD_SAMPLING` is `reference` (default) or `fast` (Quality only), selecting
Quality's opt-in `volumeFastSampling` parameter. The archive driver accepts the
same final argument after both rates. See
[the sampling contract](../../devices/rtx/docs/quality-volume-fast-sampling.md)
for eligibility, fallback, and the approximate-field semantics.

`SOURCE_SAMPLING` is `standard` (default) or `resample` (Quality only), selecting
`volumeSourceResampling`. The archive driver accepts the same final argument.
This is two-candidate source resampling with a corrected selection weight, not
temporal reuse. See [the estimator contract](../../devices/rtx/docs/quality-volume-source-resampling.md).

See [the approximation experiments](../../devices/rtx/docs/principled-volume-approximation-experiments.md)
for bounded speed/error results; the renderer default is unchanged.

`RANDOM_SEED` is Quality's public uint32 `randomSeed`, default0. No private seed
switch is used. Subsequent disjoint sample blocks are reconstructed from cumulative
images and public `numSamples`, including startup resets. Disjoint blocks alone
are not independent randomized replicates.

The measurement script takes the binary, library name, assets, output directory,
phase and library directory. Use a fresh output directory and preserve a frozen
library/binary when comparing changes. Example:

```sh
bash tools/principled_volume/run_quality_baseline.sh \
  _out/quality-baseline-build/quality_baseline visrtx_principle_volume \
  _out/principled_volume/cell-v2/datasets _out/principled_volume/quality-baseline \
  perf _out/_cmake/Release
```

For compiler diagnostics only, set `VISRTX_CAPTURE_VERBOSE=1` to forward all
ANARI status messages, including OptiX compiler output. Leave it unset for timing
runs. An optimized RelWithDebInfo device build supplies `-lineinfo` without using
the unoptimized Debug path.

`analyze_fp64.py ROOT` analyzes `compile-*/run.log` and `profile-*/` directories
containing Nsight Compute `details.csv`, SASS `source.csv`, and optional combined
CUDA/SASS `correlated.csv` exports. It distinguishes static warnings from executed
instructions and records the attribution/counter caveats. See the
[FP64 investigation](../../devices/rtx/docs/principled-volume-fp64-attribution.md).

Run the `convergence`, `depth` and `support` phases the same way. Optional `profile`
collects Nsight Compute counters/source sampling. `perf` needs Nsight Systems for
its two selected captures. No profiling command changes clock settings; coordinate
GPU-wide clock policy separately and preserve any existing external lock.

```sh
python3 tools/principled_volume/analyze_quality_baseline.py \
  _out/principled_volume/quality-baseline
```

The analysis checks finite images and actual accumulation increments, computes
held-out/block diagnostics, and builds independent overlap/receiver references.
Missing profiles are reported as missing rather than fabricated. Profiles' replay
durations are not used as warmed performance measurements. NumPy is required;
the optional visual montage is not part of the numeric acceptance logic.

For a frozen seed-randomized study, copy the library and runner to an evidence
folder, then use `run_quality_replicates.py OUTPUT LIBRARY_FILE RUNNER ASSETS`
(`--plan-only` writes/checks the plan without rendering). Analyze it with
`analyze_quality_replicates.py OUTPUT`. Seeds, work, reference-error treatment and
mean-consistency gates are frozen before rendering. `compare_quality_replicates.py`
measures exact-graph optimizations against those candidate runs and requires every
retained image to match byte-for-byte. See `quality-volume-convergence.md` and
`quality-volume-dynamic.md` under `devices/rtx/docs/` for the bounded campaigns.

See `devices/rtx/docs/quality-volume-baseline.md` for the frozen first campaign,
results, limitations and implementation priorities. These assets/settings are not
claimed to match the archived Blender EXRs.
