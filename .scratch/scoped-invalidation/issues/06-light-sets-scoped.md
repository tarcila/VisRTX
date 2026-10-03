# 06: Light sets rebuild only for their own changes

**What to build:** A Group's and a World's light sets, including Geometry
Lights synthesized from Emissive Surfaces, rebuild only when a light or
emissive material they reach changes.

- Light → Group carries LightSet.
- Material → Surface carries LightSet when a material's emission changes from
  sampleable to not (or back), or when its average changes. Surface → Group
  and Instance → World carry LightSet through.
- The device-wide light-set stamp is removed. A light-set change builds no
  BVH.

**Blocked by:** 03 (World contents reach the Frame).

**Status:** resolved

- [x] Changing the intensity of a light in world A: A's light set rebuilds, no
      BVH builds, A resets, B does not
- [x] Toggling emission on a material used only in world A: A's light set
      gains or loses the Geometry Light, B is untouched
- [x] All `TestEmissive*` and `TestPowerLightPick*` tests pass

## Comments

### 2026-10-03 iteration — resolved
Replaced the last device-wide ObjectUpdates stamp with LightSet emission and propagation along Light/Material → Surface/Group → Instance → World. Group and World consume local LightSet stamps; geometry/instance structural changes still refresh World light data through the existing local TLAS path. Geometry Lights do not emit changes during synthesis. Hole 09's deferred bridge remains unchanged for unmigrated edges.
Added a World light-set debug counter and regressions for explicit/zero-instance Worlds: isolated light intensity, emission enable/average/disable, non-emission edits, unreachable lights, and a shared Group consumed once by two Worlds. Tests assert Geometry Light membership, accumulation isolation, and no BLAS/TLAS work. Self-reviewed against spec and style; no independent reviewer tool was available.
Verified: `cmake --build _out/_cmake --config RelWithDebInfo --target TestScopedInvalidation && (cd _out/_cmake && ctest -C RelWithDebInfo -R '^TestScopedInvalidation$' --output-on-failure)` → red against prior installed library (26 assertions failed, including unrelated Group light-set rebuilds); `logs/06-red.log`.
Verified: `just brd` → build+install rc=0; `(cd _out/_cmake && ctest -C RelWithDebInfo -R 'TestScopedInvalidation|TestEmissive|TestPowerLightPick' --output-on-failure)` → 17/17 pass (45.65s); `(cd _out/_cmake && ctest -C RelWithDebInfo --output-on-failure)` → 56/56 pass (103.00s). Logs under `.scratch/scoped-invalidation/logs/06-{build,focused-tests,all-tests}.log`.
Verified: `clang-format --dry-run --Werror devices/rtx/device/{material/Material.cpp,light/Light.cpp,light/GeometryLight.cpp,light/GeometryLight.h,surface/Surface.cpp,world/Group.cpp,world/Instance.cpp,world/World.cpp,world/World.h,optix_visrtx.h} devices/rtx/apps/tests/api/TestScopedInvalidation.cpp` → rc=0. `rg -n 'lastLightSetChange|deviceState\(\)->objectUpdates|state.objectUpdates' devices/rtx/device` → no matches (rc=1).
Holes filed: none
