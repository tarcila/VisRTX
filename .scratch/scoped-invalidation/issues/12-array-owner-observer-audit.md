# 12: Audit and complete array-to-owner invalidation edges

**Milestone:** scoped-invalidation

**Status:** resolved

**Blocked by:** 03, 07, 11

**Files:** `devices/rtx/device/geometry/*`, `devices/rtx/device/spatial_field/*`, `devices/rtx/device/light/*`, `devices/rtx/apps/tests/api/TestScopedInvalidation.cpp`

Part of 08. Audit actual array relationships for Geometry, SpatialField and
Light, observing every referenced array that can affect the object. Arrays
notify repacking owners, which re-finalize and emit their own typed changes.
Read the frozen spec and hole 09. The edge from SpatialField to Isosurface is
also a repacking dependency, not a structural parent that can skip finalize.
Blocked by 11 only to serialize the shared regression test edits.

## Implementation findings

The examples in 08's original prose are stale: Cone, Curve and Cylinder already
use ChangeObserverPtr for their index/position/radius/caps arrays. Isosurface
already observes its SpatialField and isovalue array. Structured regular,
structured rectilinear, NanoVDB regular/rectilinear fields and HDRI radiance
also have observed data/axis/radiance arrays. Verify constructor binding,
replacement and actual unmap behavior, not just member declarations.

Actual plain retained references found in the audit:

- GeometryAttributes stores primitive and subtype vertex/face-varying attribute
  and color arrays as IntrusivePtr. Trace every `commitAttributes()` caller and
  initialize owners correctly for each live attribute set.
- Geometry::m_primitiveId is plain IntrusivePtr.
- Triangle vertex normals, face-varying normals, vertex tangents and
  face-varying tangents are plain IntrusivePtr. Tangent helper signatures take
  IntrusivePtr; adapt only as needed for observer-backed storage.
- Quad vertex normals are plain IntrusivePtr.
- Optional Neural geometry retains its layer arrays in a vector of
  IntrusivePtr. Ensure observed membership refresh removes stale edges rather
  than adding duplicates. Feature-gated verification must be recorded honestly.
- UniformAttributes declares array members, but current Geometry commit/GPU
  packing uses only its optional values. Establish whether these are live
  relationships before mechanically converting declarations.

Preserve the repacking behavior needed by Isosurface and other data consumers.
Do not remove hole 09's default deferred bridge as incidental cleanup, nor
change the spec's per-type conservative Geometry BLAS classification.

## Acceptance criteria

- [x] Audit records all live Geometry/SpatialField/Light array relationships,
      including constructor owner binding and array replacement/removal.
- [x] Complete missing observed edges for live attribute/color, primitive ID,
      Triangle/Quad normal and tangent arrays, and feature-gated Neural layers.
- [x] Changed and identical-byte cylinder radius unmaps re-finalize the owner
      and reset only its Frame; representative cone/curve behavior is covered.
- [x] Representative attribute/ID, normal/tangent, field/isovalue and HDRI
      array unmaps reset only reachable Frames; shared dependencies reset both.
- [x] Replacing/removing a referenced array detaches the old observer; edits to
      the old array no longer reset the former owner's Frame.
- [x] Formatter, `just brd`, focused scoped/array regressions and all RTX API
      tests pass using `_out/_cmake`, RelWithDebInfo, with bounded logs under
      `.scratch/scoped-invalidation/logs/`; optional unavailable configurations
      are explicitly recorded rather than claimed tested.

## Comments

### 2026-10-03 split context
Created from 08 before implementation. Tickets 03 and 07 already test several
array-data propagation cases; extend that harness and avoid duplicating their
completed structural-edge changes. No source code changed in the split.

### 2026-10-03 iteration — resolved
Completed live Geometry attribute/color, primitive-ID, Triangle normal/tangent, Quad normal and optional Neural layer observations. A new geometry-local retained observer pointer counts aliased bindings: helium registrations are set-like, and the added alias regression initially failed when removing one binding detached another. All Geometry Array1D bindings now share that accounting, including existing index/position/radius/caps/SDF/isovalue edges; counts outlive pointer destruction. Neural clears old layer membership on each commit and supports vector relocation without duplicate registrations. No compatibility bridge or Geometry BLAS classification changes.
Expanded two-world tests cover every primitive/vertex attribute/color slot on sphere/triangle/quad/cylinder/cone/curve, face-varying triangle attributes/normals/tangents, ID, changed and identical radius unmaps, replacement/removal, aliased bindings and shared geometry (B-first). Field→Isosurface and isovalue edits exercise conservative geometry rebuilds; HDRI changed/identical radiance edits exercise only reachable Frames without BLAS work. Self-reviewed against spec/style; no independent reviewer tool was available.

Audit inventory (live authored relationships and ownership):
- Geometry base: primitive.attribute0–3/color and primitive.id; all use geometry-local observed ownership. UniformAttributes' array declarations have no writers/readers anywhere in geometry: only optional scalar vec4 values are committed/packed, so the dead declarations remain untouched.
- Sphere: primitive.index, vertex.position/radius and vertex.attribute0–3/color. Cone/Cylinder: primitive.index, vertex.position/cap, vertex.radius (Cone) or primitive.radius (Cylinder), plus vertex attributes/color. Curve: primitive.index, vertex.position/radius and vertex attributes/color. Triangle/Quad: primitive.index, vertex.position/normal and vertex attributes/color; Triangle also faceVarying attributes/color/normal and both tangent arrays. All constructor-bound `this` owners remain intact (new sets bind through member initialization). SDF: primitive.sdf/neighbor; Isosurface: field/isovalue plus inherited primitive attributes/ID. Undersized isosurface primitive arrays explicitly detach when ignored. UnknownGeometry has no additional arrays. Neural: layer_0…layer_(2*n_layers-1), rebuilt observed membership, with retained GPU-data reads.
- StructuredRegularField and NvdbRegularField: data. StructuredRectilinearField and NvdbRectilinearField: data plus coordsX/Y/Z. All existing observer pointers bind `this` in constructors and assign current getParamObject results on commit, including null on parameter removal; validation paths explicitly null rejected arrays. Their data/coordinate unmaps re-finalize fields before publishing Slot/bounds kinds. Existing ticket 07 regressions cover regular data and rectilinear coordinate changes. CustomField/UnknownSpatialField and field-sampler implementation objects add no authored array relationships.
- HDRI is the only Light subtype with an authored array: radiance, bound to `this`, refreshed on commit, null on removal/rejection; pointer replacement invalidates its radiance timestamp. Other Light types and internal GeometryLight do not own authored arrays. New tests verify old radiance arrays detach.
- Remaining fallback consumer for ticket 09: SpatialField → Isosurface uses queued default Object::propagateChangeFrom to re-finalize derived bricks/isovalue pointers and GPU geometry. Preserve it explicitly before removing the bridge. Array notifications use helium directly (arrays are not visrtx::Object), so Geometry/Field/HDRI/Sampler/TF/Instance/ObjectArray owner repacking does not use that queued fallback. Sampler → Material intentionally bypasses the fallback through Sampler::observersRefinalizeOnChange(); Matte/PBR/MDL sampler bindings must keep repacking. Surface, Group, Instance, World, Frame and TransferFunction1D have explicit typed receivers already. Observer inventory retained in `logs/12-observer-inventory.log`.

Verified: `cmake --build _out/_cmake --config RelWithDebInfo --target TestScopedInvalidation && (cd _out/_cmake && ctest -C RelWithDebInfo -R '^TestScopedInvalidation$' --output-on-failure)` → initial regression failed against old installed library (`logs/12-red.log`); subsequently corrected test sampleLimit saturation and required Cone radius setup. Added alias test then failed 12 assertions with ordinary ChangeObserverPtr storage, passed after counted geometry ownership.
Verified: `just brd` → build+install rc=0 (`logs/12-build-install-final.log`); `(cd _out/_cmake && ctest -C RelWithDebInfo -R 'TestScopedInvalidation|TestIsosurface|TestEvictGPUArray' --output-on-failure)` → 3/3 pass (`logs/12-focused.log`).
Verified: `(cd _out/_cmake && ctest -C RelWithDebInfo --output-on-failure)` → 56/56 pass, 103.36s (`logs/12-all-tests.log`).
Verified: `clang-format --dry-run --Werror devices/rtx/device/geometry/{Geometry.h,Geometry.cpp,Cone.h,Cylinder.h,Curve.h,Sphere.h,Triangle.h,Triangle.cpp,Quad.h,Isosurface.h,Isosurface.cpp,SDF.h,Neural.h,Neural.cpp} devices/rtx/apps/tests/api/TestScopedInvalidation.cpp` → rc=0 (`logs/12-format.log`).
Verified: Neural.cpp syntax-only compilation using Geometry.cpp's RelWithDebInfo compile_commands entry, replacing the source, removing `-o <object>`/`-c`, adding `-DVISRTX_USE_NEURAL -fsyntax-only` → rc=0; exact expanded compiler command in `logs/12-neural-syntax.log`. Cached configuration has VISRTX_ENABLE_NEURAL=OFF: full Neural-enabled build/runtime was not tested or claimed. NanoVDB runtime configurations likewise were audited statically, not separately rendered here.
Holes filed: none
