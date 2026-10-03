# 03: World contents reach the Frame (Image kind)

**What to build:** A change inside a world resets only the Frames that
transitively reach it, i.e. the Frames it is a Frame Dependency of.

- Add the missing upward links, each carrying at least the Image kind:
  - Surface → Geometry and Material
  - Group → Surface
  - Instance → Group
  - World → Instance (and World → its zero-instance surfaces, volumes and
    lights)
  - Frame → World
- Remove the temporary world-content stamp from ticket 02.
- Deferred Upload stops triggering resets; it is transport only. An array's
  unmap is the change.
- BVH and light-set rebuild scoping is not part of this ticket; they stay on
  today's device-wide triggers.

**Blocked by:** 02 (Frame resets only for its own camera, renderer and
parameters).

**Status:** resolved

- [x] Changing the color of a material used only in world A: A resets, B does
      not
- [x] Unmapping the vertex array of a geometry used only in world A: A resets,
      B does not
- [x] Changing the transform of an instance used only in world A: A resets, B
      does not
- [x] A geometry shared by both worlds changes: both A and B reset
- [x] Committing a surface that no world reaches: neither Frame resets
- [x] All `TestEmissive*`, `TestMdl*` and `TestEvictGPUArray` tests pass

## Comments

### 2026-10-03 iteration — resolved
Wired Geometry/Material → Surface → Group → Instance → World → Frame Image propagation during commit flush, without structural re-finalization. Groups observe all authored surfaces/volumes/lights (including invalid ones); Worlds observe authored instances and zero-instance contents. Deduplicated observer lists are refreshed on membership re-finalization and stale queued edges are ignored.
Replaced ticket 02's device-wide world-content stamp with per-object Image emission and removed the Frame's temporary consumed stamp. Deferred uploads remain transport only; Frame reads only its Image stamp. Array/ObjectArray → owner and Sampler/Field data-repacking notifications retain helium behavior; hole 09's default deferred bridge remains for unmigrated edges. Global BVH/TLAS/light-set triggers remain for tickets 04–07.
Added explicit and zero-instance two-world regressions with shared camera/renderer: isolated material, light, vertex edits and identical-byte unmaps; instance transforms; shared geometry; unreachable surfaces; replacement removes an old geometry edge. The old installed library failed 19 assertions before installation; the updated library passes. Self-reviewed against the spec/style; independent reviewer tools were unavailable.
Verified: `cmake --build _out/_cmake --config RelWithDebInfo --target TestScopedInvalidation && (cd _out/_cmake && ctest -C RelWithDebInfo -R '^TestScopedInvalidation$' --output-on-failure)` → regression red against old installed library (19 failures).
Verified: `just brd` → rc=0, build and install; `(cd _out/_cmake && ctest -C RelWithDebInfo -R 'TestScopedInvalidation|TestEmissive|TestMdl|TestEvictGPUArray' --output-on-failure)` → 18/18 pass; `(cd _out/_cmake && ctest -C RelWithDebInfo --output-on-failure)` → 56/56 pass (105.01s). Logs: `.scratch/scoped-invalidation/logs/03-{build,red,focused-tests,all-tests}.log`.
Verified: `clang-format --dry-run --Werror devices/rtx/device/{Object.h,Object.cpp,optix_visrtx.h,frame/Frame.h,frame/Frame.cu,surface/Surface.h,surface/Surface.cpp,world/Group.h,world/Group.cpp,world/Instance.h,world/Instance.cpp,world/World.h,world/World.cpp} devices/rtx/apps/tests/api/TestScopedInvalidation.cpp` → rc=0.
Holes filed: none
