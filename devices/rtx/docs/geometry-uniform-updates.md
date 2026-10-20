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

## Local performance checks

Use the [benchmark skill](../../../.agents/skills/benchmark-geometry-preprocessing/SKILL.md)
to compare against a revision before the optimization with matched builds.
Measure `small-recommit` for bounds/BVH readiness and add
`--extra-case uniform-render:32:64:recommit:1` for registry publication and a
completed render. Keep replacement and unchanged-frame cases as guards.
The [benchmark guide](geometry-preprocessing-performance.md) defines the timed
boundaries and local result interpretation; no historical baseline is required.

## Remaining work

Broader array/shading invalidation precision, avoiding retirement waits when no
ownership changes, batched readback/dispatch, and fine-grained area-vs-bounds
invalidation are separate improvements. This change does not repack or download
inputs, introduce CPU preprocessing, or weaken the guarantee that changed inputs
are prepared on the GPU before their derived data is published.
