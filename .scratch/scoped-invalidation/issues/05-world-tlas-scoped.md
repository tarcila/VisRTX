# 05: Each World rebuilds its TLAS only for its own changes

**What to build:** A World rebuilds its TLAS only when one of its Instances
emits TLAS. That happens on a transform, group handle or validity change, or
when the instance's group rebuilt a BLAS.

- The Group → Instance link translates BLAS to TLAS. Instance → World carries
  TLAS.
- The device-wide TLAS stamp is removed.

**Blocked by:** 04 (Each Group rebuilds its BLAS only for its own changes).

**Status:** resolved

- [x] Changing the transform of an instance in world A: world A rebuilds its
      TLAS only (no BLAS); world B rebuilds nothing
- [x] Adding or removing an instance in world A's instance array: only world A
      rebuilds its TLAS
- [x] A BLAS rebuild in a group shared by both worlds rebuilds both worlds'
      TLASes
- [x] All existing RTX API tests pass

## Comments

### 2026-10-03 iteration — resolved
Replaced the global TLAS stamp and per-render Group BLAS scan with Group BLAS → Instance TLAS → World TLAS propagation. Worlds refresh their valid-instance partition lazily on TLAS changes, retain observation of invalid authored Instances, and observe their internal zero Instance. Structural links do not re-finalize their parent (hole 09's bridge remains for unmigrated edges).
Separated light rebuilding from TLAS rebuilding while preserving the device-wide LightSet trigger and Group light migration for 06. Added regressions for scalar/array transforms, instance-array insertion/removal, unreachable Instances, valid↔invalid Instances, Group-handle replacement and light-only TLAS stability; existing shared-Group tests cover independent World consumption. Self-reviewed against spec and style; independent review tools unavailable.
Verified: `cmake --build _out/_cmake --config RelWithDebInfo --target TestScopedInvalidation && (cd _out/_cmake && ctest -C RelWithDebInfo -R '^TestScopedInvalidation$' --output-on-failure)` → regression red against previous installed library (8 assertion failures, then SIGSEGV on invalid Instance traversal); `logs/05-red.log`.
Verified: `just brd && (cd _out/_cmake && ctest -C RelWithDebInfo --output-on-failure)` → final build+install rc=0, 56/56 tests pass (106.43s); logs `.scratch/scoped-invalidation/logs/05-{build-final,all-tests-final}.log`. Earlier focused run passed 1/1 and full run passed 56/56.
Verified: `clang-format --dry-run --Werror devices/rtx/device/world/{World.cpp,World.h,Instance.cpp} devices/rtx/device/optix_visrtx.h devices/rtx/apps/tests/api/TestScopedInvalidation.cpp` → rc=0.
Holes filed: none
