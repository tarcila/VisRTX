## Task 5 Report: GraphViewport ImGuizmo overlay + app wiring

### What was integrated

**GraphViewport.hpp** — extended ctor with `tsd::graph::Graph*`, `tsd::graph::NodeId*`, `bool* graphDirty`; added `#include "tsd/graph/Graph.hpp"` + `#include "ImGuizmo.h"`; added `drawGizmo(const ImVec2&, const ImVec2&)` private declaration; added five new members (`m_graph`, `m_selected`, `m_graphDirty`, `m_gizmoOp = TRANSLATE`, `m_gizmoMode = WORLD`).

**GraphViewport.cpp** — ctor init list gains `m_graph(graph)`, `m_selected(selected)`, `m_graphDirty(graphDirty)`; added `#include "tsd/graph_nodes/TransformableNode.hpp"` and `#include <algorithm>`; `buildUI()` calls `drawGizmo(pos, imgSize)` after the image blit and gates `handleNavigation()` on `!gizmoActive`; `drawGizmo` implemented (see deviations for projection).

**tsdFlow.cpp** — added `#include "tsd/graph_nodes/DisplayTransform.hpp"`; viewport pool construction updated to pass `&m_graph, &m_selected, &m_graphDirty`; `syncDisplays()` appends transform sync: `for (const auto &dt : tsd::graph_nodes::collectDisplayTransforms(m_graph)) m_bridge->setDisplayTransform(dt.node, dt.xfm);`.

### linalg / ImGuizmo specifics

- **`linalg::lookat_matrix(eye, at, up)`** — confirmed by BaseViewport.cpp; used for `view`. The `view` type is `linalg::aliases::float4x4` (same as `tsd::math::mat4`), column-major; `&view[0].x` gives 16 contiguous floats for ImGuizmo.
- **`linalg::length(at - eye)`** — confirmed used in Manipulator.cpp and procedural code.
- **`linalg::perspective_matrix` does NOT exist** in this codebase's linalg. BaseViewport.cpp builds the matrix manually; `drawGizmo` follows the same pattern: manual column-major perspective construction with `oneOverTanFov = 1/tan(kFovy/2)`, standard OpenGL depth encoding, `fovy = π/3 ≈ 1.04719755f`.
- **ImGuizmo matrix layout**: `&view[0].x`, `&proj[0].x`, `&m[0].x` — 16 contiguous column-major floats; matches BaseViewport's `Manipulate` call form exactly.

### tsdFlow build result

All 3 translation units rebuilt + linked cleanly after both the initial edit and after clang-format.

### Full ctest summary

```
100% tests passed, 0 tests failed out of 63
Total Test time (real) = 65.33 sec
```

Phase 4h tests confirmed passing: `tsd::nodes::DisplayTransform` (#38), `tsd::rendering::BridgeTransform` (#46), `tsd::nodes::Surface` (#40).

### .envrc uncommitted

Confirmed: `jj status` shows `.envrc` not listed (untracked); only the 3 implementation files were committed.

### Commit

SHA `6af16ae8`, subject: `feat(app): GraphViewport ImGuizmo overlay for the selected display's transform`

### Deviations from brief

1. **No `linalg::perspective_matrix`** — the function doesn't exist in this linalg. Replaced with manual column-major perspective construction matching BaseViewport.cpp's approach. The formula is identical (oneOverTanFov / aspect, oneOverTanFov, OpenGL depth row, translation row).
2. **No gizmo op/mode key toggles** — brief marks these as optional ("include if quick, else default TRANSLATE/WORLD is fine for v1"). Defaulting to TRANSLATE/WORLD as specified.

### Manual checklist (GUI, not CI-tested)

- [ ] `tsdFlow` launches; the bounding box is a wireframe and the volume is visible through it.
- [ ] Select `DisplayVolume`/`DisplaySurface` → Inspector shows Translate/Rotate/Scale fields (+ "Transform" header, "Reset"); editing moves/rotates/scales the rendered object live.
- [ ] ImGuizmo overlay appears on the selected display only in viewports it's masked into; dragging moves the object smoothly (no stutter/rebuild), and Inspector fields track it.
- [ ] "Reset" restores identity; camera orbit/pan/zoom still work when not interacting with the gizmo.

### Self-review

- `drawGizmo` returns early (false) for all 5 invalid-state cases before touching ImGuizmo state — no BeginFrame/SetRect pollution on non-gizmo frames.
- `*m_graphDirty = true` is set only when `Manipulate` returns true (drag changed the matrix), not on every frame — correct.
- Navigation gating: `if (!gizmoActive) handleNavigation()` — clean, no InvisibleButton changes needed.
- The `syncDisplays()` transform sync runs after the mask loop, so each display's layer exists before its transform is set; consistent with the brief's ordering requirement.
- `linalg::length` is protected against zero (via `std::max(..., 1e-4f)`) so near/far are never degenerate.

### Concerns

None. The implementation is minimal, build-verified, and all 63 tests pass.
