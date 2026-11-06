# FP64 diagnostics, executed work and continuation state

Investigation prompted by `compile-shaders.log`. Its original contents are left
untouched. Evidence is under `_out/principled_volume/fp64-attribution/`.
This is an attribution/optimization campaign, not a blanket precision reduction.

## 1. Optimized line-information capture

Built the existing RelWithDebInfo configuration, with `-lineinfo`, `NDEBUG` and
OptiX's default optimization level. This is not the unoptimized Debug/-O0 path.
The measurement driver gains opt-in `VISRTX_CAPTURE_VERBOSE` status logging.
Two headless 32-square captures force compilation with `OPTIX_CACHE_MAXSIZE=0`
for that command only; neither requires CUDA/OpenGL interoperability.

Both captures produce 15,647 FP64-warning occurrences, now with source locations
and none saying "No source location available". They compile the same module
set even though different renderers are selected. The pipeline summary still
says `debug information: no`: that summary field alone is not proof that usable
line information is absent. The warnings and Nsight CUDA/SASS correlation confirm
that source locations are available.

Largest static warning groups (including repeated diagnostics):

| Source | Occurrences |
| --- | ---: |
| `gpu/volumeOpticsReference.h` | 5,313 |
| `gpu/aggregateVolume.h` | 3,335 |
| `spatial_field/StoredFieldReferenceSampler.h` | 1,310 |
| `gpu/volumeSource.h` | 808 |
| `gpu/blackbody/volumeBlackbody.h` | 648 |
| `gpu/blackbody/blackbody.h` | 630 |
| `gpu/volumeTracking.h` | 525 |
| `gpu/renderer/volumePreview.h` | 427 |

These are not unique bugs, operations executed per frame, or timing percentages.
Locations are explicitly approximate. In particular, many warnings land on
inlined `promote()`/bit-expansion helpers; that does not prove the conversion
itself is the arithmetic hotspot. Shader modules include canonical/wide fallback
code even when a fast path is selected at runtime.

## 2. Runtime attribution

Nsight Compute 2026.3.1 on RTX 5880 Ada. Separate bounded clock leases, with
`--clock-control none` inside the lease. Captures use 640x640, one full sample,
depth 32, camera/shadow rates 1/1, no denoising/checkerboarding/firefly filtering,
and standard source sampling. Each profile captures the third `optixLaunch`.
Profiling replays are diagnostic, not unprofiled performance acceptance runs.

| Renderer / scene | Profile duration ms | FP64 pipe active % | DADD/DMUL/DFMA thread instructions, billions | L1 local requested load/store GB |
| --- | ---: | ---: | ---: | ---: |
| Interactive cloud | 28.81 | 0.95 | 0.038 | 27.29 / 24.88 |
| Interactive plume | 16.54 | 1.48 | 0.044 | 10.24 / 11.05 |
| Interactive subway | 24.24 | 2.17 | 0.062 | 4.68 / 5.02 |
| Quality cloud, reference fields | 22.14 | 62.98 | 1.718 | 7.31 / 6.48 |
| Quality cloud, fast fields | 16.71 | 65.24 | 1.381 | 6.76 / 5.44 |
| Quality plume, reference fields | 25.80 | 50.84 | 1.635 | 5.52 / 5.09 |
| Quality plume, fast fields | 20.74 | 49.50 | 1.191 | 4.94 / 3.97 |
| Quality subway, reference fields | 87.20 | 47.39 | 3.458 | 26.99 / 26.69 |
| Quality subway, fast fields | 73.29 | 43.29 | 2.550 | 25.37 / 23.60 |

### Counter interpretation

The aggregate `smsp__sass_thread_inst_executed_op_dadd/dmul/dfma_pred_on.sum`
metrics returned zero in these OptiX captures despite nonzero executed SASS
counts and FP64-pipe activity. Instruction counts above are instead summed from
per-address **Predicated-On Thread Instructions Executed** counters for DADD,
DMUL and DFMA, including rounding-mode modifiers. Addresses are deduplicated.
They are instruction counts, not FLOPs; conversions and comparisons are not
included in that arithmetic subtotal.

Local-memory byte counts are requests/accesses, not DRAM traffic and not solely
register-pressure spills. They include local state/stack use. OptiX continuation
state and callable ABI traffic must not be conflated with compiler spill reports.
Per-function PC-sampling shares likewise are not elapsed-time partitions.

`tools/principled_volume/analyze_fp64.py` records these caveats and produces
function, file and approximate line attribution from exported reports. It retains
the first CUDA mapping per address to avoid counting inlined source mappings twice.

### What is hot

- **Interactive:** the camera-segment continuation dominates cloud/plume PC
  samples. Its hot fragments and point samplers execute no DADD/DMUL/DFMA in
  these captures. Local accesses cluster around the surface-shadow trace and
  continuation boundaries, including `volumePreviewFloat.h`'s light connection.
  Further blanket FP64 removal is not the main opportunity in these workloads.
- **Quality:** the volume callable accounts for about 53–68% of sampled PCs
  with reference fields; the scalar reference sampler contributes about 23–27%.
  Fast sampling removes that sampler's FP64 arithmetic, leaving the volume
  callable dominant. Hot source mappings include inverse-exponential sampling,
  slab divisions, reference arithmetic/binding work and thermal evaluation.
- Both renderers retain significant state traffic. Static continuation-spill
  totals from compilation are not the bytes accessed on a particular path.

## 3. Targeted experiments

### Interactive multiplication-prefix lifetime — reverted

Compute `kernel * lightRadiance * volumeTransmission` before the surface trace,
keeping the original multiplication order. This avoids carrying three separate
RGB factors across the trace. All 42 retained HDR frames match exactly.
Cloud improves only about 1%; subway/plume are effectively neutral in paired
runs. This is insufficient to retain additional code on performance grounds.

### Separate FP32 and wide camera callables — reverted

Split the camera segment into separate continuation-callable entries, selecting
with the existing eligibility predicate. Preserve all numerical graphs and
fallback support. Public Interactive, typed-custom and policy checks pass;
42 retained HDR frames match exactly.

Subway/cloud/plume medians are essentially unchanged. Separate support improves
about 7%, but the main targets do not benefit enough to justify the extra SBT
entry/dispatch maintenance. This does not rule out a larger pipeline design;
it rejects this specific split. No extra callable or slot remains.

### Midpoint-grid inverse exponential — retained

Production tracking RNG `openUnit()` yields odd multiples of `2^-53`, from two
PCG words. Each complement `1-u` is another exactly representable member of that
finite grid. Therefore subtraction loses no information, including at the ends
of the interval. For that RNG only, use `-log(1-u)` instead of `-log1p(-u)`.

Generic RNGs retain `log1p`, since an arbitrary small open-unit input can round
`1-u` to one. This changes neither random draws nor the mathematical inverse CDF,
uses no FP32 arithmetic, and leaves optical bounds/source ownership unchanged.
The two library math functions need not be bit-identical for every input; controls
bound their relative difference by four binary64 epsilons rather than claiming
universal bitwise identity.

Six interleaved/reversed-order Release timing pairs at the same full-frame work
show median candidate/baseline ratios:

| Scene | Ratio | Time reduction |
| --- | ---: | ---: |
| Subway | 0.9873 | 1.3% |
| Cloud | 0.9658 | 3.4% |
| Plume | 0.9711 | 2.9% |
| Separate support | 0.9526 | 4.7% |
| Overlap | 0.9799 | 2.0% |

Individual outliers are retained, including one slower candidate overlap run and
one slower baseline plume run. No universal/large speedup is claimed.

Post-change profiles **do not show fewer FP64 instructions**: cloud/reference
increases from 1.718 to 1.735 billion and subway/fast from 2.550 to 2.579 billion.
Logical local-access counts are identical in those paired profiles. Profile
durations improve (22.14 -> 20.99 ms and 73.29 -> 71.56 ms), consistent with the
unprofiled evidence. This is a faster math-path choice, not an FP64-count or
spill-traffic reduction; the exact microarchitectural cause is not established.

## 4. Verification and limits

- CPU/CUDA/fast-math tracking controls check randomized midpoint inputs, endpoint
  and power-of-two transitions, exact complements, positive optical depths and
  relative agreement. Actual RNG state/draw consumption is compared. A generic
  RNG returning `2^-60` verifies that its `log1p` fallback remains nonzero.
- 560 retained Quality HDR frames match baseline exactly: four seed settings
  with reference fields and an additional fast-field run, across seven scenes.
  This is bounded replay evidence, not a promise for every seed/scene.
- Release build and 169/169 selected CTests pass in 614.69 s, excluding the MDL
  compile benchmark as in earlier campaigns. Python tooling: 26 tests run,
  two optional OpenVDB checks skipped, including three new attribution tests.
- Aggregate tracking CUDA fast-math, public Quality and public source-lighting
  memcheck/full-leak workloads report zero device errors/leaks.

Only the RNG-grid specialization and diagnostic tooling are retained. Both
Interactive candidates are fully reverted. The original log, user metadata
changes, clock policy, installed library and unrelated CUDA/OpenGL issue are not
modified by this work. Line-info/profile binaries are separate from the Release
performance baselines. The investigation supports targeted optimization, not
silencing warnings or weakening precision indiscriminately.
