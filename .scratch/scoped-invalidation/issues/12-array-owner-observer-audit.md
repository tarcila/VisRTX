# 12: Audit and complete array-to-owner invalidation edges

**Milestone:** scoped-invalidation

**Status:** ready-for-agent

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

- [ ] Audit records all live Geometry/SpatialField/Light array relationships,
      including constructor owner binding and array replacement/removal.
- [ ] Complete missing observed edges for live attribute/color, primitive ID,
      Triangle/Quad normal and tangent arrays, and feature-gated Neural layers.
- [ ] Changed and identical-byte cylinder radius unmaps re-finalize the owner
      and reset only its Frame; representative cone/curve behavior is covered.
- [ ] Representative attribute/ID, normal/tangent, field/isovalue and HDRI
      array unmaps reset only reachable Frames; shared dependencies reset both.
- [ ] Replacing/removing a referenced array detaches the old observer; edits to
      the old array no longer reset the former owner's Frame.
- [ ] Formatter, `just brd`, focused scoped/array regressions and all RTX API
      tests pass using `_out/_cmake`, RelWithDebInfo, with bounded logs under
      `.scratch/scoped-invalidation/logs/`; optional unavailable configurations
      are explicitly recorded rather than claimed tested.

## Comments

### 2026-10-03 split context
Created from 08 before implementation. Tickets 03 and 07 already test several
array-data propagation cases; extend that harness and avoid duplicating their
completed structural-edge changes. No source code changed in the split.
