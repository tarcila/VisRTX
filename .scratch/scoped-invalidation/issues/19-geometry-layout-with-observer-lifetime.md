# 19: Reconcile Geometry class layout with observer-map lifetime

**What to build:** Make Geometry's declaration follow the documented public/protected/private ordering while preserving alias-safe observer-registration lifetime. This is a standards-review follow-up, not permission for a mechanical member reorder that changes destruction safety.

**Blocked by:** None

**Status:** resolved

- [x] Public interface appears before protected hooks and private implementation details, consistent with the class-layout standard
- [x] Observer-count storage remains alive while every GeometryArrayPtr detaches; preserve this through an explicit lifetime design if declaration order changes
- [x] Array alias replacement/removal and geometry destruction regressions pass
- [x] Build/install, formatter and RTX API suite pass

## Comments

Triage: user approved the required fix, with observer lifetime preservation
mandatory. Optional review improvements remain deferred.

Final Standards review reported Geometry's private observer helpers/count map preceding the public constructor as a class-layout breach. Confirmed against STYLEGUIDE section 5. Important qualification: the count map currently precedes owner-backed array members, so reverse member destruction keeps it alive during their destructor callbacks. Simply moving that map below the array members would invalidate this lifetime guarantee. The reviewer did not account for this constraint; investigate an explicit safe layout rather than following its reorder suggestion literally.

### 2026-10-03 iteration — resolved
Moved Geometry's private observer helpers/count map below its public interface and protected hooks.
Added an explicit destructor that clears all six base observer bindings while the count map is alive; derived bindings have already detached, and subsequent base member destructors see empty pointers.
Extended the six geometry-subtype regressions with alias replacement, direct alias removal, destruction with all base bindings populated/aliased with derived bindings, and a surviving owner sharing the same array. No optional refactors or spec/addenda edits.
Verified: `clang-format --dry-run --Werror devices/rtx/device/geometry/Geometry.h devices/rtx/device/geometry/Geometry.cpp devices/rtx/apps/tests/api/TestScopedInvalidation.cpp` → exit 0; `just brd` → build/install passed; `ctest --test-dir _out/_cmake -C RelWithDebInfo -R TestScopedInvalidation --output-on-failure` → 1/1 passed; `ctest --test-dir _out/_cmake -C RelWithDebInfo --output-on-failure` → 56/56 passed (102.14 s). Final logs: `.scratch/scoped-invalidation/logs/19-{format,build-install-final,focused-final,all-tests-final}.log`.
Holes filed: none
