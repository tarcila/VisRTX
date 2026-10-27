# Reference raw-range and binding bridge

This slice connects typed raw ranges to the optical reference core without changing
renderer dispatch or the ANARI field finalizers. It adds:

- `device/gpu/referenceFieldBinding.h`: canonical raw hull construction/merge,
  controlled scalar/RGB multiplication and interpolation, domain-aware binding
  evaluation, and raw-to-effective range construction.
- `device/spatial_field/ReferenceFieldPreparation.h`: a host-only typed provider
  contract, revision checks, dependency deduplication and transactional preparation
  of bindings plus optical bounds.

**This bridge alone does not certify existing CUDA texture or NanoVDB samplers.**
The subsequent [stored-field reference snapshots](stored-field-reference.md) add
concrete controlled adapters, format decoding, field ownership and finalized
revisions. Legacy hardware samplers are not silently tagged as certified by their
texel min/max alone. Prepared-volume/registry and experimental Interactive
integration are now covered by the [object](principled-volume-object.md),
[aggregate](aggregate-volume-reference.md) and
[preview readiness](interactive-volume-readiness.md) records.

## Sample/range agreement

A provider supplies `describe(index, FieldSampleDescriptor&)` and
`range<T>(index, RawRangeCertificate<T>&)` during preparation. Successful describe
promises an available, compatible scalar or RGB sampler. The descriptor contains
its finalized field-bounds/ROI intersection and a key:

- stable field identity, not a reused registry index;
- finalized sampling/range revision;
- sampling contract/implementation identity.

A revision covers payload, mapping, interpolation, centering, domain/ROI, decoder
and sparse-background changes. A range certificate has the same key and encloses
**every actual returned sample over that entire field domain**, including its closed
boundary. The generic bridge does not infer compatibility with arbitrary compiled
code from a nonzero key: a provider's successful describe must promise it. Keys
establish matching versions, not correctness of a dishonest certificate.

`prepareVolumeBindings` checks every bound role's type and descriptor. It requests
raw ranges only for density, transmission color and transmission distance, and only
where the medium can sample that field. Source, temperature, intensity, tint,
albedo and phase remain sample-only roles; their effective-value contracts still
apply. A constant needs no provider or range. A disjoint/empty field domain uses
outside values and needs no range, but retains its typed dependency descriptor.

Duplicate bindings using one registry field index produce one dependency entry.
The fixed eleven-entry capacity derives from the eleven binding roles, not a
limit on field size, overlapping media or authored occurrences. The low-level
`prepareBinding` is an internal construction helper for that bounded role list;
callers use `prepareVolumeBindings`, not an unbounded sequence of helper calls.

A final metadata recheck rejects a changed key, type, domain or reused registry
slot before publication. Failures leave the previous output untouched and identify
the property/index where available. These internal construction results feed
existing diagnostics; they are not a new frame-failure API. A range crossing a
physical limit may be loose, not proof of invalid field contents.

**Lifetime boundary:** the prepared result copies bindings/bounds/dependency metadata;
it neither owns field allocations nor locks a racing producer. Owners must retain
the corresponding immutable sample payload through consumption using the existing
supported preparation/update lifecycle. Call `matchesPreparedFields` at publication
or reuse, not at every sample. Integrating real field finalizers/observer ownership
is still required; these tests do not claim that changing a revision integer alone
retains old allocations.

## Raw and effective bounds

`RawRange<T>` distinguishes empty from singleton zero. `canonicalRange` constructs
a global component hull from canonical finite float32 values; `mergeRanges` can
include tiles, backgrounds or independently reduced partitions. These are range
construction operations, **not extra data-validation scans**. They assume valid
values and do not promise NaN detection. A sparse adapter must actually enumerate
all relevant values/tiles/backgrounds before it can issue a certificate; a unit
range-merge test is not that enumeration implementation.

The controlled sampled multiplier is

```
effective = RN32(RN64(exactDouble(scale32) * exactDouble(raw32)))
```

The float32 product is exactly representable in binary64 before RN32. Evaluating
both raw endpoints and taking component minima/maxima therefore encloses this same
operation, including negative scales and negative raw values. There is no assumption
that raw d/C/D values themselves satisfy the physical effective-property ranges.
Subnormal inputs/conversion use the optical core's non-FTZ arithmetic.

For medium V and finalized field domain F:

- a constant's interval is its scale alone; outside is irrelevant;
- include scaled raw endpoints when nonempty V and F intersect as closed sets;
- include scaled outside when V is not wholly contained in F;
- empty F contributes outside everywhere;
- reaching only a face/edge/corner still includes raw boundary values, matching
  membership-before-filtering at possible endpoint samples.

A wholly contained medium does not include an unreachable invalid outside product.
Outside values are not feathered into the interpolation stencil. Medium evaluation
checks V separately; a source-only region outside F but inside V remains active.
Reference domain comparisons also use non-FTZ promotion.

`bindingRange` requires an available raw enclosure whenever V can sample F;
preparation enforces that precondition. It is not safe to call it with an empty raw
range for a reachable field and treat the result as a certificate.

Effective intervals pass directly to the reference core's `buildBounds`. Optical
bounds are stored in the prepared volume, never written into shared field state.
Two volumes with different scales/outside values can therefore share one raw
certificate without overwriting each other's derived bounds.

## Controlled interpolation helper

For float endpoints a,b and float w in [0,1], `canonicalLerp` uses

```
RN32(RN64(RN64(RN64(1-w)*a) + RN64(w*b)))
```

All operations/conversions use the optical reference primitives. This convex form
avoids overflowing float `b-a`. The double expression is monotone in a and b;
its error at equal float endpoints is much less than half a float ULP, so final
RN32 reproduces that endpoint. The returned value consequently stays within the
canonical endpoint hull. Successive float-stored lerps extend this argument to
trilinear interpolation. No optical clamp is introduced.

A backend using this helper must still preserve its coordinate mapping, edge
extension, nearest/linear choice and cell/node convention. This slice supplies
and tests the arithmetic; it does not replace legacy hardware filtering globally.

## Verification and limitations

`apps/tests/unit/volumeReference/testBindings.h` defines a concrete canonical-corner
provider. Range construction and its trilinear sampler use the same eight values
and metadata. It is deliberately a test provider, not a stub claimed to certify
all production stored formats.

The host/CUDA/CUDA-fast-math tests cover:

- 81 spatial samples, including boundaries, field outside, medium outside and
  interior trilinear values; independent analytic field/optical expectations;
- signed scalar and per-component RGB multipliers, full-domain raw hulls and
  reachable/unreachable outside contributions;
- controlled product/lerp extremes, subnormal products, tiny-domain comparisons,
  and equality with host arithmetic under production fast-math settings;
- sample-only source/phase capability acceptance with no range callback;
- mismatched range/sample revisions, wrong arity, unavailable range, changed
  contract/identity/domain metadata, and transactional publication behavior;
- changed field values plus coherent revision rebuilding larger optical bounds;
  removing one versus all uses of a shared field; separate per-volume scales;
- empty ROI, touching domains, zero-extinction source-only preparation,
  effective-product overflow rejection and raw/background range union.

The first run exposed an incorrectly authored test fixture (emission scale was
left at its API default zero); setting the intended scale to one fixed the
fixture, not the production default or an oracle tolerance.

Build/run:

```
cmake --build _out/_cmake --config Release --target \
  rtx_test_ReferenceFieldBindings rtx_test_ReferenceFieldBindingsCUDA \
  rtx_test_ReferenceFieldBindingsCUDAFastMath -j 4
ctest --test-dir _out/_cmake -C Release \
  -R '^rtx::(ReferenceFieldBindings.*|VolumeOpticsReference.*|VolumeLogOracle|VolumeOptics.*|Sobol|SobolTableHeader|SampleOrdinal)$' \
  --output-on-failure
compute-sanitizer --tool memcheck --error-exitcode 1 \
  _out/_cmake/Release/rtx_test_ReferenceFieldBindingsCUDAFastMath
```

The initial combined 13-test selection passed; fast-math CUDA memcheck reported
zero errors. The test-kernel PTX compiled at compute_89 with nvcc13.4.92/O3 retained
explicit f64 multiplication and non-FTZ float narrowing, with no `.ftz`, approximate
log/reciprocal or fused operations found. This is executable qualification of
these helpers, not end-to-end rendering or a performance measurement.

Concrete stored-field certificate producers and sampler views now use this
contract; see [their integration record](stored-field-reference.md) for supported
formats and actual field commit/update tests. This bridge's metadata-only tests
remain distinct evidence, not a substitute for those concrete adapter tests. No source-range requirement,
content validation, broad lifecycle redesign or new public channel is added.
