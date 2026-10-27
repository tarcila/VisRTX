# principledVolume object and committed state

**Implemented:** direct factory creation, committed parameters, stored-field
bindings, optical-bound preparation, GPU payload publication and invalid/recovered
state handling. Experimental Interactive/default transport is now integrated; see
[the preview record](interactive-volume-preview.md) for current qualification.
Quality/Fast/Debug/Test still omit principled contributions. Extension/subtype
advertisement remains disabled.
This record does not supersede the intended optics in the root
[`principledVolume.md`](../../../principledVolume.md).

## Object and parameters

`volume/PrincipledVolume.h/.cpp` implements the exact `principledVolume` factory
name. `Volume::createInstance` constructs it; discovery metadata intentionally does
not list it yet. Required `bounds` must be FLOAT32_BOX3, finite and positive on
all axes. The eleven optical roles accept their specified scalar/RGB `P`, `P.field`
and `P.outside` types and retained defaults. Optional null field handles select
constants. Wrong types/arity are diagnosed rather than coerced. Common id, visible
and name parameter types are checked; unsupported TF value/valueRange/color/
opacity/unitDistance parameters are explicitly rejected.

Parsing happens under the SDK's committed snapshot, not a live parameter read from
rendering code. Every commit starts from the specified defaults, so unset parameters
restore defaults rather than retaining stale values. Observer-only finalization
uses those captured bindings and refreshes field metadata/resources; it does not
reparse uncommitted staging values.

Validation remains structural. The new object does not scan field contents or
check samples for NaNs. Extinction-bound construction can report an unavailable
certificate (including loose/unrepresentable effective ranges), which is not a
claim that all underlying samples are invalid. Other effective-value validity
remains the documented application/provider responsibility.

`valid` means the object has structurally compatible inputs and a prepared optical
state, **not** that a renderer supports it or that every sample has been validated.
The `bounds` property returns the exact prepared domain when valid. BaseDevice
retains ownership of WAIT flushing; this property adds no NO_WAIT global flush.

## Dependencies and publication

Each distinct spatial field gets exactly one stable retained ChangeObserverPtr
owner. Multiple roles reference that owner via a compact local index. Unchanged
owners survive commits; removing one role does not remove the remaining role's
subscription. Removing the last use drops it. This avoids the SDK's remove-all-
equal-observers behavior and occupied move-assignment hazard.

Finalization obtains current field-owned reference snapshots, prepares all bindings
and bounds together, and copies the matching GPU views into a volume-owned compact
table. It owns the bounds and prepared payload, not shared field classification.
A TF volume can classify the same scalar field without altering these optical
bounds. Stored providers are implemented; custom typed opt-in remains pending and
is diagnosed as unavailable, never reinterpreted as the legacy scalar ABI.

`gpu/principledVolumeData.h` defines the separate payload containing bindings,
optical bounds and the field-view table. `VolumeGPUData` carries a distinct
PRINCIPLED tag and pointer, not a TF1D-shaped surrogate. Constants need neither a
dummy field nor a nonempty view table. Finalization clears prior payloads and
publishes invalid/null state on failure, retaining dependency owners for recovery.
Successful uploads join the default producer stream before publication.

Host preparation establishes round-to-nearest and restores the caller's rounding
mode using `utility/ScopedReferenceRounding.h`, shared with field-snapshot
construction. The existing reference host gradual-underflow/environment assumptions
remain; this is not a universal floating-environment repair.

Field owners, view tables and payloads follow the existing supported prepare/update
lifecycle. A borrowed committed-state pointer is valid only until the next
finalization/destruction. This implementation does not add asynchronous generation
leases or make a new device-wide lifetime claim.

`PrincipledVolume::markFinalized` dirties the existing volume-BLAS/membership cache
for changes to bounds or validity even without a field change. Visibility uses the
existing Volume mechanism. World bounds may conservatively expand the authored
box through OptiX; the volume's own bounds property remains exact.

## Historical pre-renderer coexistence checkpoint

The following describes the initial object-only slice, not current Interactive
behavior. Its temporary Group warning has since moved to renderer-local Frame
preparation. Interactive now consumes the aggregate payload, while legacy paths
retain their type guards. TF1D uses emission–absorption in the new preview path;
other renderers retain legacy TF integration.

Intrinsic Group membership and bounds retain valid principled objects. Group
preparation emits a warning that their transport is not integrated and current
renderers omit their contribution. This is a temporary all-renderers diagnostic,
not the final renderer-specific eligibility policy.

The shared volume intersection entry rejects non-TF1D tags **before** accessing
TF payloads or field indices. Thus current Quality, Interactive/default, Fast,
Debug and Test cannot accidentally interpret the new pointer payload as a TF field.
The existing TF integration helpers remain legacy-only; they must not be fed a
principled hit when future transport is added. Current tests confirm omission
without globally invalidating the shared object and without changing the tested
legacy TF image. No principled picking/normal/depth behavior is promised here.

When renderer support is added, move the temporary diagnostic to the approved
renderer-specific preparation policy and consume the dedicated payload through
aggregate transport. Do not simply remove the type guard and call old TF helpers.
TF1D's scattering-to-emission/absorption rendering transition is still pending.

## Verification

`apps/tests/api/TestPrincipledVolume.cpp` tests public ANARI behavior:

- missing, wrong-type, nonfinite, degenerate and reversed bounds; exact committed
  object bounds and isolation of later uncommitted staging;
- all eleven roles' P/P.outside/P.field type checks, scalar/RGB arity, null/unset
  bindings, unsupported TF parameters, and common parameter types;
- field retention through deferred commit/public release;
- shared-field observer preservation after one role is removed, arity changes and
  observer-only recovery, empty ROI/outside-only operation, and consumer deletion;
- public render/map omission comparisons against empty and TF-only worlds for
  Quality/Fast/Debug/Test, warning presence and unchanged intrinsic validity;
  Interactive/default transport is now covered by `TestPrincipledInteractive`;
- world bounds/membership after domain, visibility, invalidity and recovery changes;
- no premature subtype discovery.

`apps/tests/unit/test_PrincipledVolumeState.cu` follows public creation/commit/update
with private payload inspection and **actual CUDA evaluation**:

- all defaults and explicit scalar/RGB/outside parameters in committed state;
- published host/device payload agreement, constants with no fields, domain clip;
- phase values, explicit source and thermal coupling with an injected analytic
  provider (not a claim that production Planck transport is integrated);
- compact deduplicated field views, source outside field domains at zero extinction,
  changed coefficients/bounds after field updates, null payload on failure and
  observer-only recovery;
- shared TF/principled raw data without classification interference;
- removing a final dependency, compact-index remapping and no further notification
  from the unused field; host rounding-mode restoration.

The initial public test incorrectly required exact equality of world bounds with
the authored box. Inspection confirmed that World uses OptiX-emitted acceleration
bounds; the test now checks finite enclosure there, while retaining exact equality
for the object's own property. No stochastic tolerance was fitted to an image.

The RTX library and the selected **22 tests passed**, including existing reference
optics/bindings/stored fields, Sobol/sample-addressing/material tests, public RGB
fields, isosurface and renderer-fog regressions. Compute Sanitizer memcheck with
full leak checking passed both new tests with **zero errors and zero leaks**.
This is constructor/state/safe-omission qualification, not principled rendering,
full provider coverage, convergence or performance qualification.

Representative commands from the workspace root:

```
cmake --build _out/_cmake --config Release --target \
  anari_library_visrtx TestPrincipledVolume rtx_test_PrincipledVolumeState -j 6
LD_LIBRARY_PATH="$PWD/_out/_cmake/Release:${LD_LIBRARY_PATH:-}" \
  ctest --test-dir _out/_cmake -C Release \
  -R '^(TestPrincipledVolume|rtx::(PrincipledVolumeState|StoredFieldReference|StoredFieldSampler|ReferenceFieldBindings.*|VolumeOpticsReference.*|VolumeLogOracle|VolumeOptics.*|Sobol|SobolTableHeader|SampleOrdinal)|TestSpatialFieldRGB|TestIsosurfaceRamp|TestSampleAddressing|TestMaterialSampling|TestRendererFogInteractions)$' \
  --output-on-failure
LD_LIBRARY_PATH="$PWD/_out/_cmake/Release:${LD_LIBRARY_PATH:-}" \
  compute-sanitizer --tool memcheck --leak-check full --error-exitcode 1 \
  _out/_cmake/Release/TestPrincipledVolume
LD_LIBRARY_PATH="$PWD/_out/_cmake/Release:${LD_LIBRARY_PATH:-}" \
  compute-sanitizer --tool memcheck --leak-check full --error-exitcode 1 \
  _out/_cmake/Release/rtx_test_PrincipledVolumeState
```

## Next boundary

The [aggregate reference layer](aggregate-volume-reference.md) now consumes this
state across counted occurrences and transformed overlaps. Live renderer transport,
production calibrated blackbody integration/packaging, custom-provider opt-in,
accepted BSDF/output policies and the live TF migration still require their planned work.
Do not advertise the extension or treat successful creation as a rendered feature.
No subsequent implementation slice or campaign is automatically authorized.
