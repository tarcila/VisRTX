# Concrete stored-field reference snapshots

This slice implements concrete producers/consumers for the
[reference binding contract](reference-field-bindings.md). It does **not** switch
renderer volume dispatch or advertise a new ANARI subtype. Subsequent experimental
Interactive integration and its public provider matrix are recorded in
[the preview implementation record](interactive-volume-preview.md); the ownership
and decoding contract below remains applicable.

## Ownership and publication

`SpatialField::referenceField()` is an internal, lazy query used **after** the
existing supported commit/finalization boundary. Each field has a stable identity
and a finalized revision. `markFinalized()` invalidates its cached reference
snapshot and advances that revision, including observer-driven array/axis updates
and invalid-to-valid recovery. Invalid fields return no snapshot. Custom fields do
not implicitly opt in through the base-class fallback.

A `StoredFieldReference` owns:

- canonical float32/packed RGB device data for dense fields, or an owned copy of
  the original sparse grid (no densification);
- three canonical 8192-entry mapping LUTs for rectilinear fields;
- global scalar/RGB raw hulls and the same finalized sample/range descriptor;
- a non-owning packed `StoredFieldReferenceGPUData` view for sampling.

The optional preview representation may separately expand a bounded scalar grid
into a CUDA array. Its linear texture is shadow-only; its point texture supplies
exact endpoints to the existing software preview interpolation. Canonical Quality
sampling and raw hulls still use the original representation, not either texture.
See [the clock-leased qualification](principled-volume-leased-optimizations.md).
The internal stored-view layout contract is4.

Construction publishes only a complete snapshot. CUDA copies are checked and the
producer default stream is explicitly joined before returning; consumption need
not rely on implicit ordering with a nonblocking renderer stream. CPU decoding and
LUT/range construction temporarily establish round-to-nearest and restore the
caller's thread-local rounding mode. The implementation source has strict host
arithmetic options; GPU decoding/interpolation uses the explicit reference
primitives under production fast-math flags.

Legacy rendering never requests these snapshots, so it incurs no extra range scan
or reference upload. There is no content-validity/NaN scan: the passes construct
raw ranges and converted data under the existing valid-input contract.

**Lifetime remains explicit:** returned snapshots/views are borrowed from their
field and invalidated on finalization/destruction. Consumers retain the field and
use the existing supported wait/update/prepare ordering. They must rebuild the
provider/view table after a field update, then compare old prepared dependency
keys against the **new** table. Do not dereference an old snapshot pointer to test
whether that pointer has become stale. These are single-generation resources, not
leases or a device-wide asynchronous retirement redesign.

`StoredFieldReferenceProvider` adapts a registry-indexed table of retained current
snapshots to the host preparation bridge. Missing entries return unavailable, not
zero data. `StoredFieldReferenceSampler` uses the corresponding GPU-view table.
The contract ID must change if decoding, layout or mapping semantics change.
The stored-view ABI is now version 2: an optional preview texture handle was
added without changing canonical reference decoding or bounds. An eligible
scalar snapshot can lazily own a bounded dense CUDA texture for FP32 shadow
integration; its canonical sparse grid is retained and reference sampling never
uses that hardware-filtered view. See the [cache follow-up](interactive-volume-cache-followup.md)
for eligibility, ownership, memory cost and quality evidence.

## Backend coverage

| Backend | Endpoint representation and decoding | Mapping |
| --- | --- | --- |
| structuredRegular | FIXED8/16, UFIXED8/16, FLOAT32, FLOAT64, FLOAT32_VEC3 converted/copied to canonical float/vec3 | Original origin and spacing, computed in double; cell/node offset and clamped outer extension |
| structuredRectilinear | Same dense formats | Normalized object coordinate → canonical inverse-axis LUT → sample index, retaining cell/node segment counts |
| nanovdb | Float, Vec3f, Fp4/8/16/N; sparse tree remains sparse | Grid double world-to-index mapping, retained min-relative cell/node transform and index-box clamping |
| nanovdbRectilinear | Same sparse formats | Retained grid-map/normalized-LUT warp and min-relative centering; no axis/model redesign |

### Dense formats

Integer normalization is explicit double division followed by RN32; saturation of
the signed minimum to -1 is format decoding, not an optical clamp. Float bits are
copied, including subnormals. FLOAT64 storage is rounded to the existing float32
sample-value contract; values whose decoded/effective samples are nonfinite do not
satisfy that contract merely because the stored double was finite.

A necessary narrow legacy fix accompanies this work: CUDA has no 64-bit floating
texture element descriptor. `uploadDoubleFieldArray` converts accepted FLOAT64
storage to float texels before legacy texture creation. Both dense finalizers use
it; host and device input uploads are tested through the legacy sampler too. Other
legacy sampling and renderer dispatch are unchanged.

### Sparse formats and ranges

The reference decoder reads the vendored NanoVDB packed codes/quantum/minimum and
uses controlled double multiply/add plus RN32. FpN widths 1,2,4,8,16 are supported.
Float/Vec3f endpoints and constant tiles/backgrounds are raw values.

A custom NanoVDB traversal operation selects controlled leaf decode or plain tile/
background loads in one traversal. It deliberately avoids a generic `getValue`
fallback that would compile the legacy fast-math compressed decoder into the
reference kernel. No per-sample type/content/revision validation is added; the
prepared descriptor establishes supported type and arity.

Global hull construction visits **all 512 represented values of each leaf**, every
nonchild internal/root tile (active or inactive), and root background. It does not
trust active-only grid statistics. Whole-tree ranges may include values outside
the sampled index/domain box; that is conservative but can be loose. A loose range
may fail an optical validity enclosure even when a tighter domain-specific range
could succeed. That is unavailable certificate metadata, not detected bad samples
or permission to silently omit a required volume. Tighter ranges are not part of
this slice.

## Interpolation and mapping boundaries

Nearest fetches return canonical endpoints. Linear fetches use seven controlled
float-stored convex lerps from `referenceFieldBinding.h`, so returned values remain
in the canonical endpoint hull. Coordinates are clamped **before** integer
conversion; neighbor increments avoid overflow at the represented upper index.
Raw field sampling still does not apply outside values: the binding performs
field-domain membership first, then either samples or uses the outside value.

Dense mapping and rectilinear LUT construction use double intermediates. The
8192-texel normalized texture-center convention is retained, but full float weights
replace hardware eight-fractional-bit filtering. The reference LUT builder avoids
repeated float coordinate addition and overflowing float endpoint differences.
These rounding changes are intentional; old-image bitwise identity is not claimed.

The existing NanoVDB rectilinear centering/LUT scale convention is retained rather
than corrected incidentally. In particular the two-voxel cell-centered fixture's
high boundary reaches the final half-cell through that legacy warp. The test records
this behavior explicitly; it does not infer a new general coordinate convention
from its result. Uniform/nonuniform mapping expectations are checked separately
from value-hull enclosure.

This is a value-enclosure design, not a universal proof of geometric accuracy for
arbitrary pathological transforms. Defined finite mapping/index arithmetic and
valid input metadata remain preconditions. Field/volume transforms, optical units,
ROI semantics and the intentional full-cell convention are not redefined.

## Verification

`apps/tests/unit/test_StoredFieldReference.cu` creates and commits fields using the
public ANARI API, flushes through the public WAIT property query, then inspects the
private snapshot seam and samples it with the **actual CUDA reference sampler**.
This is a white-box integration test, not a new public sampling property/channel.
It exercises:

- all four backends, scalar and RGB, dense integer/float/double formats;
- all sparse formats and explicitly generated FpN leaves of every supported width;
- nearest/linear and node/cell configurations;
- offset index boxes and nonidentity NanoVDB scale/translation;
- inactive leaf values, root background, and inactive tiles at all tree levels;
- nonuniform dense axes and observer-driven axis edits;
- array map/unmap updates without field recommit, ROI/mapping changes, invalid
  data removal/recovery, cache reuse and sample/range revision equality;
- device-backed dense input, exact subnormal samples, and FLOAT64 halfway,
  underflow, signed-zero and extreme finite decoding cases;
- real snapshot-provider input to optical-bound preparation, followed by CUDA
  binding/optical evaluation inside and outside the field ROI against that bound;
- host rounding-mode restoration.

Dense expectations use analytic values/normalized format rules. Sparse endpoint
expectations use NanoVDB's independent native decoder, with a small allowance for
its float operation rounding; midpoint expectations integrate those endpoints.
The test does not compare the controlled decoder to itself as the only oracle.
Existing legacy `StoredFieldSampler` checks additionally exercise the repaired
FLOAT64 texture upload for host and device sources.

The RTX library and targeted tests build in the current Release/CUDA13.4.92,
compute_89 configuration. The combined selection passed **20/20 tests**, including
prior reference optics/bindings, Sobol addressing/material execution, public RGB
fields, isosurface ramp and fog interaction regressions. No tests were skipped.
The stored-reference CUDA test checks **293 raw samples** plus three end-to-end
stored-binding optical evaluations in its final fixture set.
Compute Sanitizer reported **zero errors and zero leaked bytes/allocations**.
This is not GPU performance/default qualification.

Commands:

```
cmake --build _out/_cmake --config Release --target \
  anari_library_visrtx rtx_test_StoredFieldReference rtx_test_StoredFieldSampler \
  TestSpatialFieldRGB TestIsosurfaceRamp TestSampleAddressing \
  TestMaterialSampling TestRendererFogInteractions -j 6
LD_LIBRARY_PATH="$PWD/_out/_cmake/Release:${LD_LIBRARY_PATH:-}" \
  ctest --test-dir _out/_cmake -C Release \
  -R '^(rtx::(StoredFieldReference|StoredFieldSampler|ReferenceFieldBindings.*|VolumeOpticsReference.*|VolumeLogOracle|VolumeOptics.*|Sobol|SobolTableHeader|SampleOrdinal)|TestSpatialFieldRGB|TestIsosurfaceRamp|TestSampleAddressing|TestMaterialSampling|TestRendererFogInteractions)$' \
  --output-on-failure
LD_LIBRARY_PATH="$PWD/_out/_cmake/Release:${LD_LIBRARY_PATH:-}" \
  compute-sanitizer --tool memcheck --leak-check full --error-exitcode 1 \
  _out/_cmake/Release/rtx_test_StoredFieldReference
```

A loader-traced run identified the workspace's
`_out/_cmake/Release/libanari_library_visrtx_principle_volume.so`, not an installed
device. Test-kernel PTX compiled with production fast math retained controlled
f64 value arithmetic/non-FTZ narrowing; after using the traversal operation seam,
no `.ftz`, approximate log or approximate reciprocal remained. This verifies the
sampler seam. Subsequent public OptiX provider tests are described in the
[preview record](interactive-volume-preview.md).

## Remaining work

The [principledVolume object](principled-volume-object.md) now consumes these
snapshots and publishes prepared per-volume state. Prepared-scene/aggregate
traversal and experimental Interactive transport are integrated and separately
tested. Custom-provider opt-in/adapters and Quality transport remain later work. Existing custom fields retain legacy behavior; this slice does not silently
certify them or remove their planned support. Extension advertisement is unchanged.
No new optical clamp, content scan for validation, picking work, general callback/
interop framework or full-render statistical campaign is introduced.
