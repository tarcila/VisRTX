# Typed custom fields — experimental source-build contract

Explicit scalar/RGB opt-in is implemented for principled reference transport and
reference TF1D. Quality and Interactive's wide/reference path consume it. This is
**not a binary plugin ABI or arbitrary PTX loader**, and does not advertise the
principled/custom extensions as shipping-qualified.

## Interface and build integration

Keep existing host factory registration and legacy custom Init/SampleValue/
SampleNormal hooks. Registration alone does not enable typed sampling. An opted-in
field overrides the existing protected
`SpatialField::createReferenceField(optical_reference::FieldRevision) const` and
publishes a `StoredFieldReference::custom<T>()` generation.

Configure a single absolute manifest header:

```cmake
set(VISRTX_TYPED_CUSTOM_PROVIDER_HEADER
    "${CMAKE_CURRENT_SOURCE_DIR}/MyProviders.h" CACHE FILEPATH "" FORCE)
add_subdirectory(path/to/visrtx)
target_sources(anari_library_visrtx PRIVATE MyField.cpp)
# Add any shared provider dependencies/definitions to this interface target:
# target_link_libraries(visrtx_typed_custom_config INTERFACE my_provider_deps)
```

The header declares provider traits and
`VISRTX_TYPED_CUSTOM_PROVIDERS(X)` as `X(ProviderType)` entries. Host negotiation,
OptiX reference sampling and optional CUDA sampling receive the same manifest.
Provider IDs are nonzero and unique; duplicates fail compilation. Host/provider/
device code and dependencies must be rebuilt compatibly. Existing C++ classes,
GPU layouts and compiler options are not made binary-compatible by a version tag.
Provider host code belongs in the device build; this is not an exported standalone
binary-provider SDK.

A provider defines:

- `Value`: exactly `float` or `visrtx::vec3` (linear-sRGB/D65).
- `Payload`: trivially copyable; alignment no greater than cudaMalloc's256-byte
  guarantee. Larger/aligned resources can be referenced through leased pointers.
- `ID`, `REVISION`, `CUDA_SAMPLING`, `RESOURCE_LEASE` constants.
- Static `sampleOptix(const Payload &, const vec3 &) -> Value`.
- If CUDA is declared, static `sampleCuda` with the same signature.

Declarations must be visible to host signature/layout checks. OptiX-only bodies
or headers are guarded by `VISRTX_TYPED_CUSTOM_OPTIX`; optional CUDA code must not
require OptiX execution. Backend-specific helpers must also obey C++ ODR: do not
change a shared function's body by a macro while retaining the same symbol name.
The dispatch implementation encodes its backend in template symbol identity.

Samples depend only on frozen state and object-space position. Do not add nested
OptiX traces/direct calls or rely on mutable launch/payload/RNG state; the existing
direct-call depth remains unchanged. No initialization or gradient is required
by the typed point-sampling interface. Legacy normals remain legacy behavior.

Compiled examples:
- `apps/tests/unit/typedCustom/ExampleProvider.h`: standalone OptiX-only manifest.
- `apps/tests/unit/typedCustom/FixtureProviders.h`: scalar/RGB, CUDA and resource
  traits.
- `apps/tests/unit/typedCustom/FixtureFields.cpp`: snapshot publication, existing
  factory registration, observing an array and replacement resource generations.

Test factories are compiled only with `VISRTX_BUILD_TESTS`; their runtime names
are not added to subtype/extension discovery. Empty-manifest production builds
retain legacy behavior and have no implicitly eligible custom providers.

## Snapshot publication

`typedCustomDescriptor<Provider>()` describes the caller's expected version,
output, payload revision/size/alignment and capabilities. The custom factory
compares it against the compiled manifest before allocation or dispatch. V1 has
an exact semantic version and required prefix; extra trailing descriptor bytes
are ignored, not interpreted as extra capabilities. Unknown providers, versions,
capabilities, layout mismatches, wrong arity, absent range or required lease are
rejected. Base-field publication also checks finalized arity and revision.

The factory copies the payload into snapshot-owned device storage. Supply the
existing field revision key, finite effective domain (`bounds ∩ ROI`), a matching
componentwise **global raw range**, the consuming device stream, and any required
`std::shared_ptr<const void>` resource lease. Raw ranges describe the executed
sampler over that domain, including procedural/interpolation extrema. They do not
include binding scale or outside values. Empty domain is valid; an empty range
cannot describe a nonempty domain. A whole-domain support superset is sufficient.

Range honesty is a provider obligation, not something finite probing can prove.
Only structural/range metadata is checked. There are no validation-only content
scans, per-sample value checks or source/extinction clamps. Sampling/arithmetic
preconditions remain those of the reference evaluator.

A resource lease must transitively retain an **immutable allocation generation**.
Retaining a mutable field or array object is insufficient when its storage can be
replaced. Queue resource uploads on the supplied stream, or complete them before
publication. The factory joins that stream and its payload-copy producer before
publishing, including before plain-CUDA guide preparation. Custom retirement
joins the consuming stream before freeing payload or releasing the lease.
The stream/device must outlive the snapshot. This uses the supported serialized
prepare/update discipline; arbitrary concurrent host mutation and cross-device
resource sharing are not promised.

Use stable, deduplicated observers for provider dependencies. Array changes must
re-finalize the field without requiring a volume recommit. Parameter/domain/
resource changes publish a new field revision. A rejected typed generation does
not silently reuse the previous good snapshot, and does not by itself invalidate
an otherwise usable legacy field.

## Sampling, optional acceleration and properties

Binding membership is checked before invoking a provider. Outside its effective
domain, the binding owns `scale * outside`; outside the Medium Domain the volume
contributes zero. There is no scalar broadcast, channel reduction or coefficient
Jacobian factor.

The existing reference scalar/RGB callable slots dispatch typed custom samples;
legacy custom slots and dynamic MDL offsets are unchanged. Custom fields initially
use the wide/reference Interactive path, without stored-field preview textures or
shadow-cache eligibility. This is a fidelity fallback, not omission.

If any bound sampler lacks CUDA support, `SourceImportance` skips only the optional
pilot/CDF. Source eligibility/count/readiness and the full-support angular-distance
proposal remain intact. OptiX-only sources therefore still illuminate receivers,
including emission in vacuum. CUDA-capable providers may use the normal guide.

Field properties (with normal `ANARI_WAIT` preparation behavior):

| Property | Type | Meaning |
|---|---|---|
| `valueType` | `ANARI_DATA_TYPE` | Finalized scalar/RGB arity or invalid |
| `visrtx.typedSampling` | `ANARI_BOOL` | A matching reference snapshot is available |
| `visrtx.conservativeRanges` | `ANARI_BOOL` | The snapshot supplies the required raw-range contract |
| `visrtx.principledVolumeCompatible` | `ANARI_BOOL` | Field capability, not validity of every possible binding/scale |
| `visrtx.typedCudaSampling` | `ANARI_BOOL` | Reference CUDA point sampling available |
| `visrtx.typedContractVersion` | `ANARI_UINT32` | Semantic interface version1 when available, otherwise0 |

These queries can prepare a lazy snapshot. Internal stored-view layout/revision
contract4 (including the optional preview point-texture handle) is distinct
from the public semantic interface version1. Legacy-only
fields report no typed capability; required capability failures use existing
binding diagnostics and whole-volume omission, not a new frame-result interface.

## Qualification

- Pure host metadata negotiation: version/size, missing entry, capability/layout/
  arity mismatch, missing raw range and missing resource lease.
- Empty and standalone configured manifests: host checks plus actual CUDA and
  OptiX compilation, including an OptiX-only provider with no CUDA entry.
- Public controls: scalar/RGB optics, procedural scalar variation, poisoned-outside
  ROI checks, empty ROI with emission retained, signed raw values/scales, repeated
  bindings and unbind/update behavior, invalid-to-valid recovery, application
  handle release, CUDA/no-CUDA receiver illumination, leased allocation replacement,
  in-flight old generation across a WAIT flush, registry growth, array edits without
  field/volume recommits, legacy TF and init/value/normal consumers, and shared
  custom fields in mixed TF/principled volumes.
- An intermittent Interactive dispatch failure exposed macro-dependent backend
  symbol bodies. Explicit backend identities/routing fixed it; ten repeat runs and
  OptiX memcheck passed without changing the frozen analytic gates.
- Final selected regression (including the added OptiX-only scalar/TF and nonzero
  outside-value controls):169/169 passed in696.25 s. Explicit OptiX memcheck for
  custom fields, dynamic/seed updates, public Quality and Interactive reported
  zero errors/leaks. The independently reproduced multi-context sanitizer issue
  from the dynamic checkpoint remains a separate coverage limitation.

Evidence: `_out/principled_volume/typed-custom/` and
`/tmp/visrtx-typed-*-memcheck.log`. Matched stored cloud/plume512-spp HDR averages
remain bit-identical to the pre-custom checkpoint. The final frozen-library
paired capture measured28.50 ms cloud/28.26 ms plume at640-square/one sample,
with maxima29.66/31.16 ms; its frozen pre-custom controls measured28.34/27.81 ms.
Some intervening sessions showed48–65 ms stalls that also affected the frozen
pre-custom library. Those runs are retained, not discarded or attributed to this
feature without evidence. No universal frame-latency guarantee is claimed.
