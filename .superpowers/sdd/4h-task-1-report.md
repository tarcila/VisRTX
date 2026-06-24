# Task 1 Report: BoundingBox cylinder wireframe

## Status: DONE

## Commit
`c4ca0737` — `feat(graph_nodes): BoundingBox emits a cylinder wireframe (non-occluding)`

## TDD RED/GREEN

**RED** — after test edits only, build succeeded, test `39: tsd::nodes::Surface` FAILED:
```
REQUIRE( s->geomSubtype == Token("cylinder") )  # was Token("triangle")
REQUIRE( r->primSubtype == Token("cylinder") )   # was Token("triangle")
2 failed assertions
```

**GREEN** — after implementation, `39: tsd::nodes::Surface` PASSED (0.93 sec, 100%).

## Files Changed

- `tsd/src/tsd/graph_nodes/BoundingBox.cpp`
  - Added `#include <algorithm>` and `#include <cmath>`
  - Replaced 36-index triangle table + `AnyArray(36)` + `geomSubtype("triangle")` block
  - New: 12-edge `static const int edge[12][2]`, `AnyArray(24)`, `geomSubtype("cylinder")`, `radius` scalar
  - Used `tsd::core::math::dot` (fully qualified — no `using namespace` in scope)

- `tsd/tests/test_nodes_Surface.cpp`
  - SCENARIO title updated to "cylinder wireframe renderable"
  - THEN block: asserts `geomSubtype == Token("cylinder")`, `size() == 24`, `radius > 0.f` via `foundPos`/`foundRadius` sentinels
  - DisplaySurface WHEN: `primSubtype == Token("cylinder")`

## Self-Review

- `dot` is qualified as `tsd::core::math::dot` — correct, since only `float3` is aliased in scope.
- `static const int edge[12][2]` avoids stack churn on repeated calls.
- Radius formula `max(0.004 * sqrt(dot(d,d)), 1e-4)` — 0.4% of diagonal, floored at 0.1 mm. Reasonable heuristic for a visualization wireframe.
- No bridge changes needed: `DisplaySurface` passes `geomSubtype` through to `primSubtype` unchanged.

## Concerns

None.
