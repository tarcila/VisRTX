## Task 5: `GraphViewport` gizmo + app wiring (+ suite gate)

**Files:** Modify `tsd/src/tsd/ui/imgui/windows/GraphViewport.hpp`, `GraphViewport.cpp`, `tsd/apps/interactive/tsdFlow/tsdFlow.cpp`.

**Interfaces:** Consumes `ITransformableNode` + `collectDisplayTransforms` (Task 2), `setDisplayTransform` (Task 3). `GraphViewport` ctor gains `tsd::graph::Graph*`, `tsd::graph::NodeId* selected`, `bool* graphDirty`.

No automated test (GUI). Deliverable: `tsdFlow` builds; full suite green; manual checklist.

- [ ] **Step 1: Extend `GraphViewport.hpp`.** Add includes `#include "tsd/graph/Graph.hpp"` and `#include "ImGuizmo.h"`. Change the ctor signature to:
```cpp
  GraphViewport(Application *app,
      tsd::rendering::GraphRenderBridge *bridge,
      int viewportIndex,
      anari::Device device,
      tsd::graph::Graph *graph,
      tsd::graph::NodeId *selected,
      bool *graphDirty,
      const char *name = "Viewport");
```
Add members:
```cpp
  tsd::graph::Graph *m_graph{nullptr};
  tsd::graph::NodeId *m_selected{nullptr};
  bool *m_graphDirty{nullptr};
  ImGuizmo::OPERATION m_gizmoOp{ImGuizmo::TRANSLATE};
  ImGuizmo::MODE m_gizmoMode{ImGuizmo::WORLD};
```

- [ ] **Step 2: Store the new ctor args** in `GraphViewport.cpp`'s ctor init list (`m_graph(graph), m_selected(selected), m_graphDirty(graphDirty)`), matching the new signature.

- [ ] **Step 3: Add a `drawGizmo()` helper + call it.** In `GraphViewport::buildUI`, after the `ImGui::Image(...)` blit (where `pos = ImGui::GetCursorScreenPos()` before the image and `imgSize` are known) and **before** `handleNavigation()`, call `drawGizmo(pos, imgSize)`, and gate navigation:
```cpp
  const bool gizmoActive = drawGizmo(pos, imgSize);
  if (!gizmoActive)
    handleNavigation();
```
Implement `drawGizmo` (returns true if the gizmo is using/hovered, so navigation is suppressed):

```cpp
bool GraphViewport::drawGizmo(const ImVec2 &imgPos, const ImVec2 &imgSize)
{
  if (!m_selected || *m_selected == tsd::graph::INVALID_NODE || !m_graph)
    return false;
  auto *gn = m_graph->node(*m_selected);
  if (!gn || !gn->impl)
    return false;
  auto *itf = dynamic_cast<tsd::graph_nodes::ITransformableNode *>(gn->impl.get());
  if (!itf)
    return false;
  // Only show the gizmo if this display is masked into this viewport.
  const int mask = gn->impl->parameters().getOr<int>(Token("viewportMask"), 0);
  if (!((mask >> m_viewportIndex) & 1))
    return false;

  using tsd::math::float3;
  using tsd::math::mat4;
  const float3 eye = m_manip.eye(), at = m_manip.at(), up = m_manip.up();
  const mat4 view = linalg::lookat_matrix(eye, at, up);
  const float aspect = float(m_size.x) / float(m_size.y);
  constexpr float kFovy = 1.04719755f; // π/3 — the ANARI/VisRTX perspective default
  const float focusDist = linalg::length(at - eye);
  const mat4 proj = linalg::perspective_matrix(
      kFovy, aspect, std::max(0.01f * focusDist, 1e-3f), 100.f * focusDist + 10.f);

  ImGuizmo::BeginFrame();
  ImGuizmo::SetOrthographic(false);
  ImGuizmo::SetDrawlist();
  ImGuizmo::SetRect(imgPos.x, imgPos.y, imgSize.x, imgSize.y);

  mat4 m = itf->transform();
  if (ImGuizmo::Manipulate(
          &view[0].x, &proj[0].x, m_gizmoOp, m_gizmoMode, &m[0].x)) {
    itf->transform() = m; // root has no parent → manipulated matrix IS the transform
    *m_graphDirty = true;
  }
  return ImGuizmo::IsUsing() || ImGuizmo::IsOver();
}
```
Declare `bool drawGizmo(const ImVec2 &, const ImVec2 &);` in the header's private section. (Add `#include <algorithm>` for `std::max` and confirm `linalg::lookat_matrix`/`perspective_matrix`/`length` are the right names against the linalg the project uses — `BaseViewport.cpp` is the reference; adjust to match and report. Optional: key toggles for `m_gizmoOp`/`m_gizmoMode` à la `BaseViewport` — include if quick, else default TRANSLATE/WORLD is fine for v1.)

- [ ] **Step 4: Wire the app.** In `tsd/apps/interactive/tsdFlow/tsdFlow.cpp`, update the viewport-pool construction to pass the new args:
```cpp
      auto *vp = new ui::GraphViewport(
          this, m_bridge.get(), i, m_device, &m_graph, &m_selected, &m_graphDirty, nm);
```
And in `syncDisplays()`, after the existing mask loop, add the transform sync:
```cpp
    for (const auto &dt : tsd::graph_nodes::collectDisplayTransforms(m_graph))
      m_bridge->setDisplayTransform(dt.node, dt.xfm);
```
(Add `#include "tsd/graph_nodes/DisplayTransform.hpp"` to tsdFlow.cpp. The mask loop runs first so each display's layer exists before its transform is set; combined with the bridge's `if (d.layer)` guard this is safe.)

- [ ] **Step 5: Build the app** — `cmake --build _out/_cmake --config RelWithDebInfo --target tsdFlow 2>&1 | tail -15` → compiles + links. Fix linalg name mismatches per Step 3's note; report.

- [ ] **Step 6: Full suite gate**

```bash
cmake --build _out/_cmake --config RelWithDebInfo --target tsdTests --parallel
ctest --test-dir _out/_cmake/tsd/tests -C RelWithDebInfo --output-on-failure
```
Expected: all green — prior suite + the 3 new/changed Phase 4h tests (`[nodes-surface]` updated, `tsd::nodes::DisplayTransform`, `tsd::rendering::BridgeTransform`). Report the summary line.

- [ ] **Step 7: Confirm `.envrc` uncommitted** — `jj status` shows `.envrc` untracked, nothing else from this task after the commit. NEVER commit `.envrc`.

- [ ] **Step 8: Commit**

```bash
clang-format -i tsd/src/tsd/ui/imgui/windows/GraphViewport.hpp tsd/src/tsd/ui/imgui/windows/GraphViewport.cpp tsd/apps/interactive/tsdFlow/tsdFlow.cpp
jj commit tsd/src/tsd/ui/imgui/windows/GraphViewport.hpp tsd/src/tsd/ui/imgui/windows/GraphViewport.cpp tsd/apps/interactive/tsdFlow/tsdFlow.cpp -m "feat(app): GraphViewport ImGuizmo overlay for the selected display's transform"
```

- [ ] **Step 9: Record the manual test checklist** (GUI not CI-tested) in the report:
  - `tsdFlow` launches; the bounding box is a **wireframe** and the volume is visible through it.
  - Select `DisplayVolume`/`DisplaySurface` → the Inspector shows **Translate/Rotate/Scale** fields (+ "Transform" header, "Reset"); editing them moves/rotates/scales the rendered object live.
  - The `ImGuizmo` overlay appears on the selected display **only in viewports it's masked into**; dragging it moves the object **smoothly** (no stutter/rebuild), and the Inspector fields track it.
  - "Reset" restores identity; camera orbit/pan/zoom still work when not interacting with the gizmo.

---

## Phase 4h completion checklist

- [ ] `BoundingBox` wireframe (cylinder edges) + `[nodes-surface]` test updated (Task 1)
- [ ] `ITransformableNode` + display nodes + `collectDisplayTransforms` + test (Task 2)
- [ ] `GraphRenderBridge::setDisplayTransform` + layer-root application + test (Task 3)
- [ ] Inspector additive transform TRS section, no `markDirty` (Task 4)
- [ ] `GraphViewport` gizmo + app wiring + transform sync (Task 5)
- [ ] full suite green; `.envrc` uncommitted; manual checklist recorded

## Out of scope (per spec)

Data-resampling `Transform` node; transform on non-display nodes; gizmo snapping/extra modes; per-viewport independent transforms; transform persistence (Phase 5 must serialize the typed `transform()` state explicitly, like TF control points).

## Self-review notes

- **Spec coverage:** Component 1 → Task 1; Component 2 data model + helper → Task 2; bridge application → Task 3; Inspector → Task 4; gizmo + app wiring + flow → Task 5. The Q5 meta-review decision (transform = `ITransformableNode` typed state, off the hash) is realized in Task 2 and relied on by 3/4/5.
- **Type consistency:** `ITransformableNode::transform()→mat4&`, `DisplayTransform{NodeId,mat4 xfm}`, `collectDisplayTransforms(Graph&)`, `setDisplayTransform(NodeId,const mat4&)`, `Display.transform`, and the new `GraphViewport` ctor args are used identically across tasks and match the verified API reference.
- **Tested seam:** the device-free silent-failure-prone logic (`collectDisplayTransforms`, the `BoundingBox` geometry) is unit-tested; the bridge layer-root application gets a VisRTX membership/transform test (Task 3); Inspector + gizmo are build-verified + manual (Task 5) — consistent with prior phases.
- **Flagged for the implementer (adjust minimally, report):** exact `Layer::root()`/`LayerNodeData::setAsTransform`/`getTransform` spelling + const-ness (Task 3 test); `&m[0].x` 16-float column-major layout for ImGuizmo; linalg `lookat_matrix`/`perspective_matrix`/`length` names (match `BaseViewport.cpp`); whether `getTransform()` is readable from a `const Layer*` in the Task 3 test (fall back to build+manual if not, and report); `<cmath>`/`<algorithm>` includes.
