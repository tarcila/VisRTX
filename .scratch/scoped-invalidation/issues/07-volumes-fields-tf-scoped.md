# 07: Volumes, spatial fields and transfer functions are scoped

**What to build:** Volume content invalidates only the Frames and Groups that
reach it, and it rebuilds BVHs only when the volume's bounds change.

- Add Spatial Field → Volume (transfer function) and Volume → Group links.
- A field or transfer-function change that alters the volume's bounds emits
  BLAS. Other changes, such as data values or a color or opacity map, emit
  Slot and Image only.
- Decide during implementation which field parameters affect bounds, and
  record the decision in the spec's translation table.

**Blocked by:** 04 (Each Group rebuilds its BLAS only for its own changes).

**Status:** resolved

- [x] Editing the color or opacity map of a transfer function in world A: A
      resets, B does not, no BVH builds
- [x] Unmapping the data array of a field in world A without changing its
      bounds: A resets, no BVH builds
- [x] Changing the bounds of a field in world A (origin, spacing or
      dimensions): group A rebuilds its volume BVH, world A rebuilds its TLAS
- [x] All existing RTX API tests pass

## Comments

### 2026-10-03 iteration — resolved
SpatialField now emits Slot on every finalization and BLAS only when its evaluated bounds or validity change. TF volumes consume queued field changes without re-finalizing: refresh opacity bounds and their GPU slot while retaining the TF texture. Volume bounds/validity are checked independently, so equal-bounds field swaps and data-only edits are not structural. Existing Volume → Group links from 04 remain unchanged.
Bounds classification (recorded here instead of editing the frozen spec): regular fields use origin, spacing, dimensions and dataCentering; rectilinear fields use coordinate endpoints; NanoVDB fields use the grid's evaluated world bounds; custom fields use their virtual bounds(). Any parameter or array edit is BLAS only if that evaluated AABB or validity changes. ROI/filter, interior rectilinear coordinates, data values and TF color/opacity/valueRange/unitDistance remain Slot/Image when bounds stay fixed.
Expanded regressions cover color/opacity map commits and unmaps, changed and identical field-data unmaps, filter/ROI, origin/spacing/centering/dimensions, invalid↔valid fields, equal-bounds handle replacement, shared fields rendered B-first, and rectilinear interior versus endpoint edits. Empty volume sets clear the BVH without an OptiX build; the test checks unchanged BLAS build count but rebuilt TLAS for that transition. Self-reviewed spec/style; independent review tools were unavailable. Hole 09's default bridge remains for unmigrated repacking consumers (including isosurface, audited by 08), not TF volumes.
Verified: `cmake --build _out/_cmake --config RelWithDebInfo --target TestScopedInvalidation && (cd _out/_cmake && ctest -C RelWithDebInfo -R '^TestScopedInvalidation$' --output-on-failure)` → red against the previous installed library: data-only/filter edits rebuilt BLAS/TLAS; later invalid-field case aborted. Log `logs/07-red.log`.
Verified: `just brd` → rc=0, build+install (`logs/07-build-install-final.log`); `(cd _out/_cmake && ctest -C RelWithDebInfo --output-on-failure)` → 56/56 pass, 102.91s (`logs/07-all-tests.log`).
Verified: `clang-format --dry-run --Werror devices/rtx/device/spatial_field/SpatialField.h devices/rtx/device/volume/{Volume.cpp,Volume.h,TransferFunction1D.cpp,TransferFunction1D.h} devices/rtx/apps/tests/api/TestScopedInvalidation.cpp && clang-format --dry-run --Werror --lines=49:63 devices/rtx/device/spatial_field/SpatialField.cpp` → rc=0; SpatialField.cpp check limited to modified hook to leave pre-existing formatting untouched.
Holes filed: none
