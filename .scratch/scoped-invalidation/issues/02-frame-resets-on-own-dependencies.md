# 02: Frame resets only for its own camera, renderer and parameters

**What to build:** Fixes the bug in `_tmp/HANDOFF.md`. Committing an unrelated
object (a new camera, or another Frame sharing camera, renderer and world) no
longer causes an Accumulation Reset on an accumulating Frame.

- A Frame resets when its own image-affecting parameters change, or when its
  camera or renderer emits a change:
  - Image-affecting: `size`, `channel.*`, the camera, renderer or world
    handles, `accumulation`.
  - Not image-affecting: `frameCompletionCallback*`.
- Changes inside the world still reset the Frame through a temporary,
  device-wide "world content changed" stamp. Ticket 03 removes it. It must
  fire only on finalizes of world content, never on Frame or Camera commits.
- Remove the unused per-Frame "last changed" fields.

**Blocked by:** 01 (Change Kind stamps and test harness).

**Status:** resolved

- [x] Committing an unrelated new camera: neither Frame resets
- [x] Committing a new Frame C that shares A's camera, renderer and world, and
      rendering it once: A does not reset
- [x] Re-committing A's camera with a changed value: A resets, B does not
- [x] Re-committing A's camera with an unchanged value: A does not reset
- [x] A commit to A that changes only the completion callback: A does not reset
- [x] Changing A's `size`: A resets
- [x] Editing a world's content still resets the Frames that use it (temporary
      behaviour)
- [x] All existing RTX API tests pass

## Comments

### 2026-10-03 iteration — resolved
Frame now observes Camera/Renderer and consumes typed stamps through flush-time propagation without re-finalizing; cameras emit Image only when effective base/subtype values change. Own image parameters are compared independently of callback settings, and buffer-layout/reallocation changes force a reset even in manual mode. Removed dead Frame fields and device-wide finalization/upload reset triggers.
Added a temporary `DeviceGlobalState::lastWorldContentChange`, bumped only by world-content Object finalizes (not Camera/Renderer/Frame); ticket 03 must remove it and extend Frame propagation to World. Renderer post-only carve-outs and Pipeline rebuild emission remain ticket 08; hole 09's default bridge remains unchanged for other structural edges.
Regressions failed before the fix and pass now: unrelated camera/new shared Frame, independent and shared cameras, subtype/no-op edits, callback-only commits, renderer changes, normal/manual resize, and world material edits. Baseline needs only one warm-up round. Self-reviewed against the ticket and STYLEGUIDE; no independent subagent tool was available.
Verified: `just brd` → rc=0 (build and install); `(cd _out/_cmake && ctest -C RelWithDebInfo --output-on-failure)` → 56/56 pass, including TestScopedInvalidation (105.10s). Logs: `.scratch/scoped-invalidation/logs/02-{build,all-tests}.log`.
Verified: `clang-format --dry-run --Werror devices/rtx/device/{Object.h,Object.cpp,optix_visrtx.h,optix_visrtx.cpp,camera/Camera.h,camera/Camera.cpp,camera/Perspective.cpp,camera/Orthographic.cpp,renderer/Renderer.h,renderer/Renderer.cpp,frame/Frame.h,frame/Frame.cu} devices/rtx/apps/tests/api/TestScopedInvalidation.cpp` → rc=0.
Holes filed: none
