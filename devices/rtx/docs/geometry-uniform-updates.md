# Uniform-only geometry updates

## Scope

Triangle, quad, sphere, curve, cone, cylinder, and SDF geometries opt into a
metadata-only finalization path. Changing `attribute0`–`attribute3` or `color`
republishes the geometry registry slot without regenerating tangents, indices,
radii, bounds, or area CDFs, and without invalidating the surface BLAS timestamp.
Normal object-finalization timestamps still advance, so automatic accumulation
reset remains intact. No stream or synchronization policy is changed; the
existing input-retirement check still runs on a parameter commit.

This is deliberately not general fine-grained invalidation. Any changed
non-uniform input takes the existing full preparation path, including shading
arrays and scalar/string parameters that could be narrowed further later.
Isosurfaces, optional neural geometry, and unknown geometries retain their
existing full-preparation behavior; they do not opt in.

## Reuse contract

`Geometry::beginPreparation(parameters)` is the opt-in seam. A subtype must list
**every non-uniform parameter it consumes**, including parameters read lazily
by tangent generation or `gpuData()`. The base adds primitive and vertex
attribute arrays and primitive IDs. When adding a subtype parameter, update its
list or keep that subtype on the default conservative path.

The helper compares committed parameter values, not live staging iterators.
For referenced arrays it also records object update/finalization timestamps,
data modification timestamp, and device backing pointer. Consequently:

- Changed handles, values, logical ranges, or backing pointers invalidate reuse.
- A uniform commit coalesced with an array unmap or privatization cannot hide it.
- Observer-only finalization always follows the full preparation path.
- Missing/invalid initial data and previously failed preparation cannot reuse
  uninitialized or failed derived state.
- Reuse becomes eligible only after a valid, successful geometry upload.

Only present parameters occupy snapshot storage. The cached `AnariAny` values
retain internal references to the current input objects until replacement or
geometry destruction; they do not copy authored data. This also prevents cached
parameter identity from becoming a dangling object pointer. The extra reference
lifetime is intentional; old snapshot references are dropped on the next full
preparation after the existing retirement check.

Uniform values are excluded because they do not change geometric area or bounds.
Geometry Light synthesis uses the material's emission estimate, while sampled
attribute-driven radiance reads the updated geometry registry data at rendering
time. Material changes continue to use their existing light-set invalidation.

## Correctness checks

`TestGeometryUniformUpdates` exercises seven types with host and shared-device
inputs. It verifies all five uniforms and their removal against the initial
full-preparation rendering, stable bounds, and no preparation/BVH-build diagnostic
messages. Debug rendering modulates attributes by geometric facing, so the initial
full-preparation image is the exact per-shape reference, with a separate nonzero
control. Scalar/string edits, array ranges, in-place host unmaps, and device-array
privatization act as positive controls requiring preparation and BVH rebuilding.

The test also stages an input removal **after** a uniform commit: that uncommitted
edit must not leak into the committed update. Emissive materials request area data
where supported. Existing numeric-failure tests additionally verify that a
uniform-only commit cannot rehabilitate a failed generation.

## Measurement

Control: `18e1c93ce559` (GPU preprocessing before this optimization).
Candidate: accompanying implementation. Both use matched Release/native-PBR,
CUDA 13.2, OptiX 7.7, shared cudart, `89-virtual`, RTX 5880 Ada, driver 615.71.09.
These are **GPU branch vs GPU branch** comparisons, not a new CPU baseline.

Reproduce with the project skill
[`.agents/skills/benchmark-geometry-preprocessing/SKILL.md`](../../../.agents/skills/benchmark-geometry-preprocessing/SKILL.md),
using a separate control workspace and library built from the pinned revision
above. Raw three-trial × 21-iteration paired results: `preprocessing-uniform-updates.json`.
Each entry retains commands, stdout/stderr, and warm-sample median; labels `base`
and `branch` mean control and optimized respectively.

| Workload | Control ms | Optimized ms |
|---|---:|---:|
| 64 × 32 primitives, uniform recommit, bounds-ready | 2.5237 | 0.0728 |
| 64 × 32, replace inputs | 4.0583 | 3.9914 |
| 1 × 262144, replace inputs | 8.8882 | 8.6679 |
| Unchanged completed frame | 0.9812 | 0.9521 |
| Two frames, replace inputs | 5.6716 | 6.9213 |

The targeted bounds-ready recommit result is approximately **35× faster**.
This query does not force registry publication or tracing. A separate matched
three-trial × 31-iteration render pilot includes registry upload and completed
interactive rendering (64 geometries × 32 primitives):

| Uniform recommit + completed rendering | Control ms | Optimized ms |
|---|---:|---:|
| One frame | 3.4393 | 0.9564 |
| Two frames sharing the device/world | 4.8932 | 1.9079 |

Raw commands, stdout/stderr and medians: `preprocessing-uniform-render.json`.
The roughly 3.6× / 2.6× gains are application-visible, not just host metadata
queueing. Both sides use the same executable with the respective device library;
frame completion callbacks reject skipped renders.

The remaining replacement/unchanged workloads are guards, not optimization targets. Their variation is substantial;
in particular the first two-frame pilot showed an apparent 22% regression and
must not be described as proving no regression.

A longer confirmation ran six paired trials × 101 iterations for that two-frame
replacement case: median of trial medians was 5.9140 ms control vs 6.1896 ms
optimized (approximately +4.7%). Individual trial medians ranged 5.4788–7.5070 ms
control and 5.3358–7.9087 ms optimized. Raw samples/commands are retained in
`preprocessing-uniform-confirmation.json`. There is no controlled-clock/load
claim or confidence interval; a small replacement-path overhead remains possible.
Do not infer a speedup for genuine input replacements from this optimization.

An independent CUDA trace with six iterations and 64 geometries confirms actual
work elimination rather than skipped rendering or relocated waits:

| Kernel instances, including initial preparation | Control | Optimized |
|---|---:|---:|
| `primitiveBounds` | 384 | 64 |
| `primitiveAreas` | 384 | 64 |
| Each scan/normalization/publication stage | 384 | 64 |
| `optixAccelBuild` | 12 | 2 |
| `optixAccelCompact` | 12 | 2 |

Thus only initial preparation remains in the optimized trace. Reproduce with the
skill's `nsys` invocation, using `recommit` instead of `replace` and each side's
library path in turn. The reports themselves were local scratch and are not kept.
Trace timing is not used for headline latency.

## Verification

The RTX Release build and all 41 selected geometry, tangent, radiometry, shared
input/lifecycle, normal-map, and failure-recovery CTests pass. Compute Sanitizer
reports zero errors for `TestGeometryUniformUpdates`, including host updates and
shared-device pointer retirement. Native-PBR and the existing MDL/MaterialX-enabled
development configuration were exercised; optional neural support was not built.

## Remaining work

Broader array/shading invalidation precision, avoiding retirement waits when no
ownership changes, batched readback/dispatch, and fine-grained area-vs-bounds
invalidation are separate improvements. This change does not repack or download
inputs, introduce CPU preprocessing, or weaken the guarantee that changed inputs
are prepared on the GPU before their derived data is published.
