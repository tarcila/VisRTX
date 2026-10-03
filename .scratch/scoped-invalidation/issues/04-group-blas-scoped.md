# 04: Each Group rebuilds its BLAS only for its own changes

**What to build:** A Group rebuilds a BLAS only when a surface or volume it
contains emits the BLAS kind. Slot-only changes, such as a material
parameter, never trigger a BVH build.

- Geometry → Surface → Group carries BLAS for changes to the primitive set:
  positions, primitive count, validity.
- A Group consumes its own BLAS stamp. The device-wide surface and volume BLAS
  stamps are removed. World's "did any BLAS change" check reads its Groups'
  stamps.
- A change in a geometry's or surface's validity emits BLAS on its link.

**Blocked by:** 03 (World contents reach the Frame).

**Status:** resolved

- [x] Changing a material's color: no BLAS or TLAS build anywhere
- [x] Changing a geometry's vertices in group A: only group A rebuilds a BLAS
- [x] Geometry in group A becomes invalid: group A rebuilds, Frame A resets,
      Frame B neither rebuilds nor resets
- [x] A group shared by two instances is rebuilt once per change, not once per
      instance
- [x] All existing RTX API tests pass

## Comments

### 2026-10-03 iteration — resolved
Replaced global surface/volume BLAS triggers with typed Geometry → Surface → Group propagation and Group-local surface/volume stamps. Worlds check their Groups' aggregate BLAS stamps independently, including after another World or a bounds query builds the shared Group. Surface/volume validity and visibility transitions emit BLAS; invalid authored contents remain observed.
Migrated existing volume visibility/field-handle triggers and forwarded field BLAS stamps through TF finalization, preserving field-driven repacking and conservative field classification for 07. Global TLAS and light-set triggers remain for 05/06. Fixed flush convergence to drain commit-only repacking as well as propagation, matching hole 09's documented bridge; the volume regression exposed the missing commit-buffer emptiness check.
Added counter regressions for material color, isolated vertices, valid↔invalid geometry, surface visibility, shared instances/Worlds and pre-render bounds queries, zero-instance validity, TF-only color edits and isolated field bounds edits. Self-reviewed the diff against spec/style; no independent review tool was available.
Verified: `cmake --build _out/_cmake --config RelWithDebInfo --target TestScopedInvalidation && (cd _out/_cmake && ctest -C RelWithDebInfo -R '^TestScopedInvalidation$' --output-on-failure)` → regression red against previous installed library (10 assertions failed); log `logs/04-red.log`.
Verified: `just brd` → rc=0, build+install; `(cd _out/_cmake && ctest -C RelWithDebInfo --output-on-failure)` → final run 56/56 pass (111.85s), `logs/04-all-tests-final.log`. Previous run had a transient `TestMaterialXMultiAndConcurrent` SIGSEGV; `(cd _out/_cmake && ctest -C RelWithDebInfo -R '^TestMaterialXMultiAndConcurrent$' --output-on-failure)` → 1/1 pass; final full rerun also passes it. Logs under `.scratch/scoped-invalidation/logs/04-*`.
Verified: `clang-format --dry-run --Werror devices/rtx/device/{geometry/Geometry.cpp,surface/Surface.cpp,surface/Surface.h,world/Group.cpp,world/Group.h,world/World.cpp,volume/Volume.cpp,volume/Volume.h,volume/TransferFunction1D.cpp,volume/TransferFunction1D.h,optix_visrtx.h,optix_visrtx.cpp} devices/rtx/apps/tests/api/TestScopedInvalidation.cpp && clang-format --dry-run --Werror --lines=49:54 devices/rtx/device/spatial_field/SpatialField.cpp` → rc=0 (field check restricted to modified hook; pre-existing formatting elsewhere unchanged).
Holes filed: none
