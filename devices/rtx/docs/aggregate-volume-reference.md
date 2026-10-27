# Aggregate volume traversal and optical consumption

This record originally qualified the shared aggregate layer. Experimental
Interactive/default now consume it; see the [preview record](interactive-volume-preview.md).
The remaining renderers retain their legacy paths and principled omission policy.
`gpu/aggregateVolumeData.h` defines counted occurrence views;
`gpu/aggregateVolume.h` supplies CPU/CUDA intervals, optical aggregation and a
constant-segment integration primitive. A CUDA consumer exercises the actual
prepared World data. No principled subtype/extension advertisement is enabled.

## Complete prepared scenes

`World::referenceVolumes()` is an explicit internal preparation operation, called
after the existing commit flush. It rebuilds existing membership as necessary,
then publishes a flat GPU list of every eligible volume occurrence:

- direct world volumes and instanced groups;
- repeated volume entries and repeated instance handles;
- every transform-array placement;
- existing valid/visible filtering, without deduplicating object pointers or IDs.

Group retains original authored volume-array slots alongside its valid membership.
Each record has a `(placement, groupSlot)` identity within the publication, an
inverse affine transform computed in double, and a tagged optical payload. A
separate publication generation distinguishes rebuilt views even when memory
addresses are reused. Unchanged requests reuse the same publication rather than
free/reupload storage that another read-only frame could still consume. Keys are not public picking IDs or permanent authoring IDs.

Storage is O(scene occurrences), not O(active overlaps) per ray. Allocation and
count representability still follow ordinary implementation/resource limits;
there is no arbitrary overlap cap or partial-list overflow fallback. The producer
stream is joined before publication. Interactive/default request this list and
lazy TF references; legacy renderers do not. Group's slot bookkeeping remains
ordinary host membership metadata.

In strict mode (`referenceVolumes(false)`, the default), if a volume cannot
provide the required reference payload, the method
reports a warning and returns `complete=false`, null data and count zero. It does
**not** publish a prefix of the scene, reuse a last-good view, or globally invalidate
objects usable by legacy renderers. Empty eligible scenes instead return
`complete=true`, count zero. The consumer must check completeness at preparation;
this internal readiness result is not a new ANARI frame-failure API.

Interactive explicitly requests `referenceVolumes(true)`: unavailable references
are diagnosed and omitted from that renderer’s eligible set. This includes TF1D
providers without the typed reference contract, such as the unfinished custom
provider adapter. There is no silent legacy optical fallback. The cache keys the
omission policy, so a permissive view cannot satisfy a strict request. Intrinsic
validity and use by legacy renderers remain independent. Custom-provider opt-in is
still pending, not retired from the intended scope.

Views are borrowed until a relevant scene/object update. The cache checks the
World build stamp and retained volumes' finalized revisions before reusing a view.
Owner references keep revision checks safe after membership edits, but do not
retain allocations that an owner replaces during an update. The publication ID
is not a resource lease. Existing supported wait/update/prepare
ordering remains required; no device-wide asynchronous redesign is introduced.

## TF1D reference payload

`TransferFunction1D::referenceVolume()` lazily publishes a separate payload with:

- its field's typed reference sampler;
- a **volume-owned** canonical copy of its already-discretized RGBA TF;
- committed valueRange and unitDistance;
- matching global alpha-derived optical bounds.

Reference classification preserves normalized texture-center coordinates and
clamped endpoints, but uses controlled software interpolation instead of hardware
weight rounding. It consumes the reference scalar sampler, checks effective
field-domain membership before classification, and uses the shared TF conversion:
`A=t`, `S=0`, `j=t*color`, with the existing alpha epsilon/finite ceiling policy.
Bounds use the whole canonical alpha table, not a shared mutable field-opacity
grid. Different TF volumes sharing a field therefore keep independent reference
classification/bounds. Raw field extrema are not needed for this global TF-alpha
bound; typed finite sampling and a controlled classifier are required.

The cache resets on finalization, including observer-driven field changes. Failed
metadata readiness is cached until recommit/finalization; allocation/copy exceptions
do not mark an attempted build as a successful/settled cache. Reference construction
requires finite ordered valueRange and positive finite distance; unavailable
metadata does not alter legacy TF intrinsic validity.

**Interactive/default TF rendering now uses this emission–absorption model.**
Quality/Fast/Debug/Test still retain their legacy TF implementation. This is the
current incremental implementation state, not a new user-facing legacy-model
switch or a claim that the planned cross-renderer migration is complete.

## Intervals and world units

`worldDistanceRay` promotes the represented input ray, normalizes its world direction
in double and scales its limits accordingly. It preserves the physical segment
for nonunit input directions. The far limit may be positive infinity for distant
connections. Inputs require finite origin/direction and a nonzero direction.

The inverse instance transform is applied to origin and direction **without
renormalizing the object-space direction**. Slab parameters therefore remain world
path length. Instance scaling changes optical thickness; neither density nor
emission gets determinant/length compensation.

`nextSpan` replays the counted list to find the next strictly greater represented
endpoint. It merges volume-domain and reachable bound-field/ROI boundaries, processes
ties together and yields positive-length spans with an unchanged volume active set.
It skips vacuum gaps, not zero-extinction media. There is no epsilon jump, per-ray
hit array, fixed overlap limit, or arbitrary material chosen from traversal order.
Internal voxel/filter/classification discontinuities are not all enumerated.

Spans are half-open. Activity is selected from the span's beginning, not from a
rounded field coordinate. Field membership remains closed on nonempty domains.
This prevents a point rounded to a touching boundary from introducing an extra
active volume. It does not imply exact real geometry for every extreme finite
transform/ray: matrix inversion, slabs and final field positions remain numerical
approximations. The tests cover ordinary finite maps, shear/scale/reflection,
parallel/inside/boundary cases and thin representable intervals; they are not a
universal precision certificate. The multiword cursor proposal is not adopted.

## Shared consumer

`boundsForSpan` combines the stored absorption/scattering bounds in list order,
without sampling fields or sources. The renderer must choose any required positive
source-support proposal rate separately; a zero extinction bound is not proof of
zero emission.

`evaluate` replays the same active occurrences and sums absorption, scattering and
explicit/thermal source, using each occurrence's object coordinates. It adds the
RGB scattering kernels `sum(s_i * phase_i)` with both HG lobes, using the supplied
world-space scattering cosine. It does not average g or select a single medium.
Thermal evaluation is injected through the established provider seam; production
calibrated blackbody integration is not supplied by the test provider.

The matching sampled and bound reductions use the same order. Permuting authored
entries preserves the mathematical target; bitwise equality of arbitrary floating
reductions is not promised.

`integrateSegment` provides exponential transfer for constant aggregate coefficients
and a supplied constant source (j, or a future preview's j plus direct-scattering
estimate). It preserves the zero-extinction source limit, including numerical
underflow of `t*length`, and applies no opacity/contribution cutoff. This is **not**
full single/multiple-scattering rendering. Homogeneous optical coefficients do not
make illumination/visibility constant. Heterogeneous or lit preview cells require
spatial/angular refinement; stochastic quality transport requires its own candidate,
phase and source-ownership machinery. Evaluating one midpoint per whole span is
only an analytic-test consumer, not a qualified heterogeneous renderer.

## Verification

`volumeReference/testAggregate.h` and `test_AggregateVolume.cpp/.cu` exercise
**25 analytic cases** on host, CUDA and CUDA fast math. Independent host C++
long-double formulas check Beer attenuation, source integrals and two-lobe phase
kernels. Host oracles compile separately from device code. Cases include:

- overlaps, nesting, touching, gaps, clipped rays and inside starts;
- reversed/nonunit directions, parallel misses, face boundaries and a one-ULP miss;
- uniform/nonuniform/reflected/sheared placements with world-distance attenuation;
- duplicate payloads, insertion permutation and zero-extinction sources;
- mixed TF/principled coefficients and per-volume phase-kernel addition;
- field-ROI cuts without shrinking the Medium Domain or losing outside emission;
- thin representable spans and infinite far connections;
- the tiny optical-depth source limit, checked on CPU and CUDA;
- metadata-only bounds matching the sampled reduction.

`test_WorldVolumeReference.cu` creates/commits real ANARI objects, flushes using the
public World bounds query, then obtains the private prepared view and executes the
actual shared CUDA consumer. It checks 13 occurrences built from repeated group
volumes, repeated instances, transform arrays and direct placement; original slot
identity after visibility filtering; transform and opacity updates; publication
identity and stable reuse on unchanged requests; metadata failure without partial
output or intrinsic invalidation; and recovery. A **257-overlap** case verifies no small active-list cap. TF-only shared
fields retain separate classifications; a nonconstant TF and field map/unmap update
exercise fresh source evaluation without a volume recommit. Empty scenes distinguish
successful emptiness from unavailable preparation.

The four new CTest entries pass. The combined regression selection passed **26/26**
tests, including the previous 22 object/field/optics/Sobol/render-safety tests. Compute Sanitizer memcheck
with full leak checking passes the real-World consumer and fast-math analytic test
with zero errors and zero leaked allocations. This is not a convergence or GPU
performance campaign, and it does not qualify the as-yet-disabled renderer paths.

```
cmake --build _out/_cmake --config Release --target \
  rtx_test_AggregateVolume rtx_test_AggregateVolumeCUDA \
  rtx_test_AggregateVolumeCUDAFastMath rtx_test_WorldVolumeReference -j 6
LD_LIBRARY_PATH="$PWD/_out/_cmake/Release:${LD_LIBRARY_PATH:-}" \
  ctest --test-dir _out/_cmake -C Release \
  -R '^rtx::(AggregateVolume.*|WorldVolumeReference)$' --output-on-failure
LD_LIBRARY_PATH="$PWD/_out/_cmake/Release:${LD_LIBRARY_PATH:-}" \
  compute-sanitizer --tool memcheck --leak-check full --error-exitcode 1 \
  _out/_cmake/Release/rtx_test_WorldVolumeReference
compute-sanitizer --tool memcheck --leak-check full --error-exitcode 1 \
  _out/_cmake/Release/rtx_test_AggregateVolumeCUDAFastMath
```

## Remaining integration

Live renderer consumption remains outstanding. Preserve the accepted lighting,
background/coverage, AOV/denoiser, BSDF ownership and TF migration policies when
hooking up this layer; do not enable a partial shading model merely to display a
volume. Calibrated blackbody packaging, custom-provider opt-in and full-render
replicate seeding remain their respective work. Current principled omission and
unadvertised extension status stay in place until those paths are ready.
