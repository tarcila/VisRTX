## Task 4: Inspector transform TRS section

**Files:** Modify `tsd/src/tsd/ui/imgui/windows/Inspector.cpp`.

**Interfaces:** Consumes `ITransformableNode` (Task 2). No new produced interface.

No automated test (GUI). Deliverable: `tsd_ui_imgui` compiles + links.

- [ ] **Step 1: Add includes** to `Inspector.cpp`:
```cpp
#include "tsd/graph_nodes/TransformableNode.hpp"
#include "ImGuizmo.h"
```
(ImGuizmo is already linked for the UI target; `tsd/core/TSDMath.hpp` arrives transitively.)

- [ ] **Step 2: Append an additive transform section in `buildUI`.** The current `buildUI` ends with the `if (ITransferFunctionNode) {...} else { drawParameters(*m_selected); }` block. **After** that block (so it renders in addition to params/TF — a display still shows its `viewportMask`), add:

```cpp
  if (auto *it = dynamic_cast<tsd::graph_nodes::ITransformableNode *>(gn->impl.get())) {
    ImGui::Separator();
    ImGui::TextUnformatted("Transform");
    tsd::core::math::mat4 &m = it->transform();
    float t[3], r[3], s[3];
    ImGuizmo::DecomposeMatrixToComponents(&m[0].x, t, r, s);
    bool changed = false;
    changed |= ImGui::DragFloat3("Translate", t, 0.01f);
    changed |= ImGui::DragFloat3("Rotate", r, 0.5f);
    changed |= ImGui::DragFloat3("Scale", s, 0.01f);
    if (changed) {
      ImGuizmo::RecomposeMatrixFromComponents(t, r, s, &m[0].x);
      *m_graphDirty = true; // NOTE: NO m_graph->markDirty — transform is render-routing, not node data
    }
    if (ImGui::Button("Reset##transform")) {
      m = tsd::core::math::IDENTITY_MAT4;
      *m_graphDirty = true;
    }
  }
```
(`&m[0].x` is the 16 contiguous column-major floats ImGuizmo expects — `mat4` is `float4x4`, column-major. `m` is a reference into the node, so writing it edits the node's transform directly. **Do not** call `markDirty` here.)

- [ ] **Step 3: Build `tsd_ui_imgui`** — `cmake --build _out/_cmake --config RelWithDebInfo --target tsd_ui_imgui 2>&1 | tail -15` → compiles + links warning-free.

- [ ] **Step 4: Commit**

```bash
clang-format -i tsd/src/tsd/ui/imgui/windows/Inspector.cpp
jj commit tsd/src/tsd/ui/imgui/windows/Inspector.cpp -m "feat(ui): Inspector transform TRS section for ITransformableNode"
```

---

