# GPU geometry preprocessing: synchronization and performance investigation

## Objective and status

Compare complete paths to GPU-ready, renderable data, not isolated CPU loops
against isolated GPU kernels. Input upload, temporary allocation, derived-data
upload, scalar readback, registry publication, and acceleration-structure work
all belong in the accounting. GPU-produced input can avoid an additional host
round trip even when individual GPU stages have higher overhead.

Initial investigation completed: recovery fixes, dependency map, stream-ordering
probe, many-geometry benchmark, matched-build pilot, and one explanatory trace.
**No production synchronization has been removed.** Results below are exploratory,
not performance acceptance thresholds or proof of safety for an optimization.

Reviewed starting branch: `69db6cfc23897d9f5a1552fe4acf65b77753c78c`.
Base: `73c02f6acfbddc2706e6ac1564014c9f3659687d`.
Pilot branch includes the recovery fixes and benchmark changes accompanying this
report. Those fixes consume handled allocation errors and integrate isosurface
and neural finalization into the preparation-state lifecycle; successful-path
synchronization is unchanged.

## Dependency map

All paths below are relative to `devices/rtx/device/`.

| Boundary / source | Producer → consumer | Current ordering / host requirement | Candidate and constraint |
|---|---|---|---|
| `VisRTXDevice.cpp`, `initOptix()` | default stream ↔ work stream | `cudaStreamCreate`, not nonblocking; legacy NULL-stream ordering | This is an existing invariant, not permission to switch to nonblocking without replacing dependencies. |
| `array/Array.cpp`, `uploadArrayData()`; `utility/MemoryAllocation.cpp` | host source → device array → preprocessing | `cudaMemcpy` on legacy default stream; pageable H2D may return before DMA completes; subsequent blocking work-stream launch is ordered | `waitForPreparationInputs()` may be redundant as a *GPU ordering* fence under this invariant. It also detects errors; preserve failure behavior. Check all producers before changing it. |
| `geometry/Geometry.cpp`, `commitParameters()` | previous tracing → release/replacement of input references | host waits on work stream before subclasses replace refs | GPU event waits alone do not protect host-side retirement. Consider skipping the wait only when no input ownership changes, or use deferred reclamation. Do not remove globally. |
| `geometry/Preprocessing.cu`, `bounds()` / `areas()` | kernels → scalar D2H result → host validity/area/scale decisions | work-stream order plus explicit completion; host stack holds readback destination | A host completion boundary is required before reading results or returning. Batch results across geometries rather than blindly deleting waits. Pageable tiny D2H copies can already block at the copy call. |
| `geometry/Preprocessing.cu`, quad/curve generation | generated indices/radii → BLAS, registry and tangent consumers | same-stream ordering internally; `finishPreparation()` waits for execution | Quad tangent generation uses the default stream. Legacy ordering can carry that edge, but launch/execution failure must still prevent publication. Curve combines its generators before one final completion. |
| `geometry/SDF.cpp`, alignment copy | unaligned authored bytes → aligned device buffer → bounds | D2D copy and bounds kernel on the same work stream | No additional inter-stream edge needed between those two operations; buffer retirement and host summary still matter. |
| `geometry/ComputeTangent.cu` | input arrays/generated quad indices → default-stream tangent kernels → registry | default-stream kernels and Thrust; device-wide completion; converted buffers reset on failure | Existing broad waits can dominate some inputs. Do not attribute this cost to the new preprocessing waits. |
| `world/Group.cpp`, `rebuildSurfaceBVHs()` | lazy authored index uploads in build-input creation → work-stream BLAS build | default-stream wait after all build inputs have been assembled | One wait covers all pending default-stream producers, not just the first geometry used for reporting. Conditional redundancy under legacy blocking-stream semantics. |
| `optix_visrtx.cpp`, `buildOptixBVH()` | BLAS/TLAS build → bounds/compaction-size readback → compacted traversal handle | device-wide waits after build and compaction; synchronous readbacks | Already a strong completion boundary. Optimize separately from preprocessing; changing this affects host decisions and temporary storage lifetime. |
| `frame/Frame.cu`, publication after `upload()` | registry/frame default-stream copies → work-stream `optixLaunch()` | explicit default-stream completion | Legacy blocking-stream ordering already supplies a GPU producer edge. A stream event is needed if moving to nonblocking. Preserve launch/error and host staging lifetime semantics. |
| `array/Array.cpp`, `privatize()` | outstanding readers → D2D private copy → application frees shared input | host completion of readers and copy before last-public-release returns | Must not replace with a GPU-only wait. Application is allowed to free the old allocation immediately after release. Failure permanently disables data operations. |
| `utility/DeviceBuffer.h`, allocation/resizing/destruction | old allocation → new allocation / reclamation | synchronous allocator/free paths introduce further implicit completion | Count allocation frequency and time; reusable workspace can matter more than another short wait. Do not rely on incidental allocation as a documented producer edge. |

Reference semantics: NVIDIA CUDA Runtime API, “API synchronization behavior,”
and CUDA C++ Programming Guide, “Default Stream” / “Implicit Synchronization.”
The probe below tests the current legacy-default compilation and stream flags;
it is not a test of per-thread default streams or every copy/memory type.

## Ordering experiment

Build and run `BenchmarkCudaOrdering` (manual tool, not CTest). It queues a roughly
30-ms kernel in one stream, then records an event in the other stream. Waiting
for that consumer event reveals whether it inherits the producer dependency.
No events or streams are allocated during the timed interval.

Observed on RTX 5880 Ada, CUDA 13.2, driver 615.71.09:

| Work stream flags | Default producer → work consumer | Work producer → default consumer |
|---|---:|---:|
| blocking (`cudaStreamDefault`) | 31.548 ms | 31.262 ms |
| nonblocking | below printed resolution | below printed resolution |

Thus the current work stream does **not** run independently of legacy default
stream work. An explicit wait can be redundant for ordering without its deletion
eliminating the underlying dependency. Conversely, nonblocking streams would
expose races if introduced without explicit producer edges.

## Benchmark interface and accounting

```text
BenchmarkGeometryPreprocessing TYPE COUNT DEVICE REPEATS \
  [SHARED_SURFACES] [DISTINCT_GEOMETRIES] \
  [recommit|replace|unchanged] [FRAMES:0=bounds] [EMISSIVE:0|1]
```

- COUNT is primitives **per geometry**, not total primitives.
- Shared surfaces refer to each geometry; distinct geometries each own inputs and
  preparation state. The pilot uses overlapping placements to isolate update
  overhead, not a representative scene layout or ray workload.
- `recommit`: update a uniform attribute, retaining input arrays/workspace.
- `replace`: replace all inputs. Timing starts **before** allocation, host staging,
  application H2D copy for DEVICE=1, parameter installation, and release work.
- `unchanged`: no input changes after the initial iteration.
- FRAMES=0 ends at a completed bounds query / acceleration-structure readiness.
  FRAMES>0 submits all frames, waits for all, then queries bounds. Submission time
  includes deferred update work; it is not just API enqueue overhead.
- Render mode uses interactive, one sample per call, unlimited sample count.
  Completion callback counts reject skipped launches. Finite images, positive
  sample counts, and finite bounds are checked **outside** the timed interval;
  validation image downloads are not included in rendering latency.
- `setup_to_ready_ms` also includes initial input setup/staging (and host fixture
  generation), unlike `first_ms`. Device/context creation occurs before this
  timer. First-use figures are not pure initialization or steady-state metrics.
- First iteration is excluded from warm statistics. Every sample is printed.
- DEVICE=1 currently stages host-generated fixtures to device in the benchmark.
  It is **not** a claim that upstream GPU-generated data requires H2D upload.
  Add a pre-resident / GPU-producer mode before making that claim.
- Bounds/images are sanity checked; radiometric equivalence remains the job of
  the existing API test suite, not this performance tool.

## Matched-build pilot

Environment: RTX 5880 Ada Generation; driver 615.71.09; CUDA 13.2.86; C++17;
Release; OptiX 7.7; architecture `89-virtual`; MDL and MaterialX disabled in both
builds; shared cudart; native physically based materials; GL disabled.

Use the **same executable** with either ABI-compatible device library. Verify
loader selection with `ldd`. One base run resolved CUDA link-time driver stubs
instead of the real driver and was discarded. No failing or unsupported base/device-input run enters the table.

Three fresh-process trials per revision, 21 iterations each, alternating base /
branch order by trial. Table shows the median of each trial's warm-sample median.
All compared inputs are host-resident, emissive cylinders. Raw stdout, stderr,
commands, return codes, and per-trial medians: `preprocessing-pilot.json`.

| Workload | Base ms | Branch ms | Branch/base |
|---|---:|---:|---:|
| 64 geometries × 32 primitives, recommit, bounds | 0.9932 | 2.1376 | 2.15 |
| 64 × 32, replace, bounds | 2.1959 | 3.4517 | 1.57 |
| 1 × 262144, replace, bounds | 8.9448 | 6.6888 | 0.75 |
| 64 × 32, unchanged, one completed frame | 0.4128 | 0.4884 | 1.18 |
| 64 × 32, replace, two completed frames | 3.0801 | 4.5081 | 1.46 |

Interpretation: there is a real small-geometry overhead worth investigating,
but the GPU-driven path already wins the large replacement case **including**
transfers and acceleration-structure readiness. Do not choose a CPU fallback or
reject GPU preprocessing from the small-geometry result alone. Conversely, the
large-case win does not excuse a ~1.25-ms small-replacement regression.

These are pilots with visible scheduling/outlier noise, not confidence intervals.
Per-trial warm median ranges (base / branch, ms):

- small recommit: 0.9719–1.0040 / 2.0444–2.1383
- small replace: 2.1828–2.2239 / 3.4298–3.5109
- large replace: 8.2200–9.0990 / 6.3787–7.2037
- unchanged frame: 0.4116–0.4454 / 0.4359–0.5112
- two frames: 2.8402–3.2613 / 4.3659–4.7387

No GPU clocks were locked, thermal state was not controlled, and unrelated
system load was not excluded. Set acceptance budgets only after a noise pilot
on the intended benchmark host, before evaluating optimization candidates.

### Explanatory CUDA trace

Separate profiled run: matched branch, cylinder 32, host inputs, six iterations,
64 distinct geometries, replacement, bounds, emissive. Entire process totals
include initialization, first preparation, and teardown, **not just warm samples**.
Do not compare profiled timing directly to the unprofiled table.

- 1926 `cudaStreamSynchronize` calls: 1.985 ms total host API time; 0.275 µs median.
- 768 `cudaMemcpyAsync` calls: 8.416 ms total; 8.148 µs median. This workload has
  two tiny status/summary readbacks per geometry per generation.
- 2304 kernel launches: 5.021 ms API time; 1.985 µs median.
- 1529 mallocs and 1529 frees: 2.529 ms and 2.240 ms API time respectively.
- 1195 H2D transfers: 0.767 MB total; 792 D2H transfers: approximately 0.005 MB.

The tiny byte count is misleading: per-call launch/readback overhead is material.
GPU activity, host waits, and preceding work overlap; these totals are not an
additive model of wall time. Nsight can add instrumentation overhead.

The trace was taken on the matched branch build with the `nsys` invocation in the
skill below; an earlier trace from the development build is not used for the
matched results. Traces are local scratch, not durable repository artifacts.

## Reproduction

The pinned-control working tree, matched build settings, loader check, runner
invocation, trace commands, and the rules for turning new numbers into a
promote / rework / drop decision live in the project skill
[`.agents/skills/benchmark-geometry-preprocessing/SKILL.md`](../../../.agents/skills/benchmark-geometry-preprocessing/SKILL.md).
Build `BenchmarkCudaOrdering` alongside the preprocessing benchmark to repeat the
ordering experiment above.

The scratch paths, GPU, driver and architecture recorded in this report describe
the host that produced these numbers. They are provenance, not required setup.

## Isolated input-wait experiment

A separate working tree changes only the call inside
`waitForPreparationInputs()` from `cudaStreamSynchronize(nullptr)` to
`cudaGetLastError()`, retaining error reporting, work-stream result completion,
frame-publication synchronization, and all retirement waits. This is **not** a
production change, and does not establish equivalent asynchronous error detection.

With matched native-PBR/OptiX 7.7 builds, all 31 selected API tests pass. Compute
Sanitizer reports zero memory errors for device-array retirement and expanded,
unaligned SDF inputs. That does not establish correctness for per-thread default
streams, nonblocking work streams, or arbitrary producer streams.

A second paired pilot compares the unchanged GPU branch (label `base` in the JSON)
to this candidate (label `branch`), again three trials × 21 iterations:

| Workload | Unchanged branch ms | Input-wait candidate ms | Candidate/control |
|---|---:|---:|---:|
| small recommit | 2.0552 | 2.0261 | 0.986 |
| small replace | 3.4430 | 3.1901 | 0.927 |
| large replace | 6.8276 | 7.1374 | 1.045 |
| unchanged frame | 0.4783 | 0.4695 | 0.982 |
| two frames replace | 4.4925 | 4.3602 | 0.971 |

Raw results: `preprocessing-input-wait-experiment.json`. The largest apparent
improvement is about 7% in small replacements, much less than their full overhead
relative to the CPU base. The large-case result varies in the opposite direction.
Given uncontrolled noise and unresolved error-boundary semantics, this is **not a
sufficiently clear win to promote**. Keep the original fences and investigate
readback/launch batching next. Merely deleting waits is not the main solution
supported by this pilot.

## Verification and remaining decision gates

The production recovery fixes pass all 31 selected API tests. Real-OOM recovery
and isosurface recovery also pass Compute Sanitizer. For the real-OOM test use
`--report-api-errors no` to suppress deliberately generated allocation/last-error
API failures while retaining memory checking; without it the expected OOM is
reported as an API error. Optional neural support was not compiled/runtime tested.
Kernel-call interposition was removed from the fault test: CUDA 13 can launch via
internal runtime entry points, so public launch hooks did not measure all launches.
The tests instead assert real allocation failure, error isolation/recovery, bounds,
and completion callbacks; they do not claim complete kernel-launch accounting.

1. **Preserve the input-wait experiment as evidence, not an optimization.** Repeat
   under controlled conditions and resolve asynchronous error attribution before
   considering promotion. Do not generalize to nonblocking/per-thread streams.
2. **Batch scalar result readbacks.** The pilot suggests that hundreds of tiny
   readbacks/launches may be a larger target than already-short explicit waits.
   This needs a deliberate finalization/publication protocol, not deletion of
   `finishPreparation()` while its stack-local readback is still in flight.
3. **Retirement precision.** Skip waits only when no old inputs/storage can be
   destroyed; otherwise defer reclamation or retain a real host completion edge.
4. **Expand coverage before choosing an implementation:** observer-driven updates,
   buffer growth, pre-resident GPU inputs, pinned/pageable host inputs, shared
   geometry across worlds, non-emissive materials, triangle/quad/curve/SDF, and
   renderer/image-size sweeps. Current two-frame pilot shares one world/device
   and does not exercise concurrent host threads or establish overlap.
5. For each candidate: paired trials; unprofiled headline timings; CUDA trace;
   full affected API tests; allocation/copy/launch/completion failure cases;
   in-flight update/retirement cases; Compute Sanitizer. Keep only changes with
   an end-to-end win and no material large-input or steady-frame regression.

Open assessment work: peak memory, GPU-event stage timing, controlled clocks /
load, confidence intervals, image equivalence across revisions, and the expanded
matrix above. No claim of completed optimization or broad performance acceptance
is made by this initial investigation.
