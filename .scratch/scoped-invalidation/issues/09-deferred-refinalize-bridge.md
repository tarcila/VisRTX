# 09: Remove the deferred re-finalization compatibility bridge after the observer audit

**Kind:** extends

**Status:** resolved

**Blocked by:** 12 (Audit and complete array-to-owner invalidation edges)

**What to build:** Deferred cleanup, not an immediate behavior change. Replace
remaining intentional data-repacking uses of the default propagation fallback
with explicit handling, then remove the compatibility fallback without losing
notifications or derived-data refreshes.

Ticket 01 says two things that pull against each other: device behaviour
doesn't change yet, and a notified parent runs its queued propagation during
the commit-buffer flush "without re-finalizing". Until an edge type turns the
child's kinds into the parent's own kinds (tickets 02-07), a parent that
skipped finalize would never rebuild. Example: a Group would not rebuild its
BLAS after a geometry change.

**Assumption taken:** the default `visrtx::Object::propagateChangeFrom` does
`markUpdated()` and `commitBuffer.addObjectToFinalize(this)`, and it runs only
from `DeviceGlobalState::flushCommitBuffer()`. That function alternates commit
flush and propagation drain until both are empty. So the parent still
re-finalizes, but only from the flush, never from the app thread at notify
time. Existing tests pass unchanged (56/56).

## Acceptance criteria

- [x] Audit all callers of the default `Object::propagateChangeFrom`, using
      ticket 12's findings. Distinguish structural propagation from actual
      data-repacking dependencies.
- [x] Preserve explicit re-finalization for SpatialField → Isosurface and any
      other consumer that must rebuild derived data. Do not globally enable
      re-finalization for SpatialField observers: TransferFunction1D has typed
      propagation that preserves its transfer-function texture.
- [x] Remove the compatibility fallback only after every live dependency has
      an explicit structural or repacking policy; choose a safe default that
      cannot silently drop required updates.
- [x] Remove stale migration comments and confirm structural links do not
      re-finalize their parents.
- [x] Build/install and run scoped-invalidation, field/isovalue regressions
      and the full RTX API suite. Isosurface field-data updates must remain
      visible, and unrelated Frames must preserve accumulation.

## Comments

### 2026-10-03 triage — deferred cleanup approved
The user accepts this as temporary compatibility and approves deferring removal
until it is safe. Structural receivers Surface, Group, Instance, World, Frame
and TransferFunction1D already have typed propagation overrides. However,
SpatialField → Isosurface still uses the default deferred re-finalization to
refresh derived geometry state. Consequently the fallback cannot simply become
a no-op now. Ticket 12 audits those repacking relationships; this cleanup is
blocked by that audit and then makes remaining policies explicit. No source
code changed during triage.

### 2026-10-03 iteration — resolved
Audited the live observer inventory against 12; SpatialField → Isosurface was the only queued receiver still relying on the compatibility bridge. Added its explicit deferred repacking override and removed the bridge/migration comments. The default now throws `std::runtime_error` for an unclassified observer policy rather than silently dropping updates or hiding a missing structural translation through blanket re-finalization.
Added an end-to-end isosurface depth regression: doubling the field via unmap, without object recommits, moves the visible plane from z=3.5 to z=1.75. Added TF texture-build debug messages/assertions and independent-world sampler→material recommit/unmap regressions. Existing accumulation/BVH regressions remain green. Self-reviewed against spec/style; no independent reviewer tool was available.

Audit/policy inventory (source search retained in `logs/09-observer-inventory.log`):
- Geometry/Material → Surface, Surface/Volume/Light → Group, Group → Instance, Instance/internal zero Instance → World, and World/Camera/Renderer → Frame already have explicit structural receivers. Their propagation hooks only stamp/notify/update validity, never queue or invoke parent finalize. World membership commits still explicitly finalize their internal objects; that is not structural propagation.
- SpatialField → TransferFunction1D retains its typed refresh of field/grid-derived opacity bounds, validity and GPU slot; it does not finalize or recreate the unchanged TF texture. Field bounds/data/filter/ROI/spacing/centering/dimensions/validity/rectilinear edits assert no TF texture build; authored TF map commits/unmaps and field-handle swaps assert texture creation, so the callback counter is positively exercised.
- SpatialField → Isosurface explicitly queues finalize only for its current field; obsolete queued edges are ignored after replacement. Finalize refreshes derived bricks, isovalue pointers and the geometry GPU slot, then emits Geometry's conservative BLAS/Image kinds. Array→isovalue behavior remains helium-driven.
- Array/ObjectArray notifications do not pass through `visrtx::Object::notifyChangeObserver`: Geometry attributes/IDs/subtype arrays (including optional Neural), Field data/coordinates, HDRI radiance, Sampler images, TF color/opacity, Instance transform/ID, Renderer background and Group/World membership retain helium owner repacking. `Sampler::observersRefinalizeOnChange()` remains true for Matte/PBR/MDL-wrapper sampler bindings; it bypasses queued default propagation and repacks materials. No other authored Object observer relationship uses the default hook.
- No spec/addenda changes and no unrelated cleanup. Cached Neural support remains OFF; this cleanup does not modify feature-gated array behavior and does not claim a Neural-enabled runtime test.

Verified: `just brd` → build+install rc=0 (`logs/09-build-install-final.log`).
Verified: `(cd _out/_cmake && ctest -C RelWithDebInfo -R 'TestScopedInvalidation|TestIsosurface|TestEvictGPUArray|TestEmissiveIncremental|TestEmissiveRecommit' --output-on-failure)` → 5/5 pass, 4.69s (`logs/09-focused-final.log`).
Verified: `(cd _out/_cmake && ctest -C RelWithDebInfo --output-on-failure)` → 56/56 pass, 103.40s (`logs/09-all-tests-final.log`).
Verified: `clang-format --dry-run --Werror devices/rtx/device/Object.{h,cpp} devices/rtx/device/geometry/Isosurface.{h,cpp} devices/rtx/device/volume/TransferFunction1D.cpp devices/rtx/apps/tests/api/{TestScopedInvalidation,TestIsosurfaceRamp}.cpp` → rc=0 (`logs/09-format.log`).
Holes filed: none
