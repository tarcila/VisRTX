# tsdFlow Layer Debug Panel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a read-only, hidden-by-default `LayerDebug` window that shows the `GraphRenderBridge`'s realized layer forest per enabled viewport, with a greyed object-param pane for the selected node.

**Architecture:** One new `Window` subclass reading the bridge's already-built per-viewport layers (after the app's per-frame `update()`). It renders a stacked `CollapsingHeader` per enabled viewport with a read-only selectable tree; selection is a per-frame-re-resolved key `(viewport, layer, nodeIndex)` (never a persisted `Object*`, because the render scene is rebuilt each frame). The selected object's params render via the existing `buildUI_object` widget wrapped in `ImGui::BeginDisabled` (read-only). The bridge gains one inline accessor, `renderScene()`.

**Tech Stack:** C++17, Dear ImGui, `tsd::scene::Layer` traversal, `tsd::rendering::GraphRenderBridge`, `tsd::ui::buildUI_object`.

## Global Constraints

- **Version control is jj, not git.** Commit with **explicit file paths** (`jj commit <path>... -m "..."`); never bare `jj commit`, never raw `git` (sandboxed).
- **No `Co-Authored-By` lines.**
- **Never run clang-format on any `CMakeLists.txt`.**
- **Read-only:** the panel must not mutate the bridge's render scene or the app selection. `buildUI_object` is only ever called inside `ImGui::BeginDisabled(true)` / `ImGui::EndDisabled()`.
- **Selection is re-resolved each frame** from the key `(m_selViewport, m_selLayer, m_selNodeIndex)`; never store a raw `Object*`/`Layer*` across frames.
- Configure: `cmake _out/_cmake`. Build: `cmake --build _out/_cmake --parallel`. Test: `ctest --test-dir _out/_cmake -C RelWithDebInfo --output-on-failure` (the `-C RelWithDebInfo` selector is REQUIRED; suite is 68 tests and must stay green).
- **Testing note:** the only non-GUI addition is the one-line `renderScene()` getter; the window is pure ImGui rendering with no logic seam the unit harness can drive (consistent with the Phase 4i GUI specs). No new unit test is written; verification is build-green + full-suite-green + a reasoned manual smoke walk. Do not add a hollow test or a VisRTX-heavy test for a trivial getter.

---

### Task 1: LayerDebug window + bridge accessor + app wiring

**Files:**
- Modify: `tsd/src/tsd/rendering/bridge/GraphRenderBridge.hpp` (add `renderScene()` accessor)
- Create: `tsd/src/tsd/ui/imgui/windows/LayerDebug.hpp`
- Create: `tsd/src/tsd/ui/imgui/windows/LayerDebug.cpp`
- Modify: `tsd/src/tsd/ui/imgui/CMakeLists.txt` (add the new source)
- Modify: `tsd/apps/interactive/tsdFlow/tsdFlow.cpp` (construct + hide the window)

**Interfaces:**
- Consumes (existing): `GraphRenderBridge::numViewports() const`, `GraphRenderBridge::layersForViewport(int) const → std::vector<const tsd::scene::Layer*>`; `tsd::scene::Layer::root() const → LayerNodeRef` and `traverse_const(LayerNodeRef, ConstVisitorEntryFunction&&) const` where the entry function is `[&](auto &node, int level) -> bool` (return true to descend); `LayerNodeData` (via `node->`): `bool isObject() const`, `Object *getObject() const`, `anari::DataType type() const`, `const std::string &name() const`, and `node.index() → size_t`; `tsd::scene::Object::subtype()` (has `.c_str()`); `tsd::ui::buildUI_object(tsd::scene::Object&, tsd::scene::Scene&, bool useTableForParameters, int level=0)` from `tsd/ui/imgui/tsd_ui_imgui.h`; `Window(Application*, const char*)`, `Window::hide()`, `INDENT_AMOUNT`. `anari::toString(anari::DataType) → const char*`.
- Produces: `tsd::rendering::GraphRenderBridge::renderScene() → tsd::scene::Scene&`; `tsd::ui::imgui::LayerDebug` window class.

**Context:** `GraphRenderBridge` builds a `tsd::scene::Layer` per enabled display inside its private `m_renderScene`, regenerated whenever a source node re-evaluates. The app (`tsdFlow.cpp`) calls `m_bridge->update()` each frame before building UI. The window only reads those layers. `getObject() const` returns a non-const `Object*`, so const traversal suffices for the read-only pane.

- [ ] **Step 1: Add the `renderScene()` accessor to the bridge**

In `tsd/src/tsd/rendering/bridge/GraphRenderBridge.hpp`, in the `public:` section just after the `anari::World world(int viewport) const;` line, add:

```cpp
  // Read access to the generated render scene (for the layer debug panel).
  // The scene is rebuilt each frame from the graph; do not retain references.
  tsd::scene::Scene &renderScene()
  {
    return m_renderScene;
  }
```

(`m_renderScene` and `tsd::scene::Scene` are already members/includes of this header.)

- [ ] **Step 2: Create the window header**

Create `tsd/src/tsd/ui/imgui/windows/LayerDebug.hpp`:

```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/rendering/bridge/GraphRenderBridge.hpp"
#include "tsd/ui/imgui/windows/Window.h"
// std
#include <cstddef>

namespace tsd::ui::imgui {

// Read-only, hidden-by-default debug view of the GraphRenderBridge's realized
// per-viewport layers, with a greyed param pane for the selected object.
struct LayerDebug : public Window
{
  LayerDebug(Application *app,
      tsd::rendering::GraphRenderBridge *bridge,
      const char *name = "Layer Debug");
  void buildUI() override;

 private:
  tsd::rendering::GraphRenderBridge *m_bridge{nullptr};
  // Selection key, re-resolved each frame (never a persisted pointer).
  int m_selViewport{-1};
  int m_selLayer{-1};
  std::size_t m_selNodeIndex{~std::size_t(0)};
};

} // namespace tsd::ui::imgui
```

- [ ] **Step 3: Create the window implementation**

Create `tsd/src/tsd/ui/imgui/windows/LayerDebug.cpp`:

```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/ui/imgui/windows/LayerDebug.hpp"
#include "tsd/scene/Layer.hpp"
#include "tsd/scene/LayerNodeData.hpp"
#include "tsd/scene/objects/Object.hpp"
#include "tsd/ui/imgui/tsd_ui_imgui.h"
// imgui
#include "imgui.h"
// anari
#include <anari/anari_cpp.hpp>
// std
#include <cstdio>
#include <string>

namespace tsd::ui::imgui {

namespace {

std::string stripAnariPrefix(const char *s)
{
  std::string t(s ? s : "");
  const std::string p = "ANARI_";
  if (t.rfind(p, 0) == 0)
    t = t.substr(p.size());
  return t;
}

std::string nodeLabel(const tsd::scene::LayerNodeData &n)
{
  if (n.isObject()) {
    auto *o = n.getObject();
    const std::string sub = o ? std::string(o->subtype().c_str()) : std::string();
    const std::string ty = stripAnariPrefix(anari::toString(n.type()));
    return sub.empty() ? ty : (ty + " : " + sub);
  }
  const std::string &nm = n.name();
  return (nm.empty() ? std::string("node") : nm) + " (transform)";
}

} // namespace

LayerDebug::LayerDebug(
    Application *app, tsd::rendering::GraphRenderBridge *bridge, const char *name)
    : Window(app, name), m_bridge(bridge)
{}

void LayerDebug::buildUI()
{
  if (!m_bridge) {
    ImGui::TextDisabled("No bridge");
    return;
  }

  tsd::scene::Object *selObj = nullptr;

  for (int i = 0; i < m_bridge->numViewports(); ++i) {
    const auto layers = m_bridge->layersForViewport(i);
    if (layers.empty())
      continue;

    char hdr[32];
    std::snprintf(hdr, sizeof(hdr), "Viewport %d", i + 1);
    if (!ImGui::CollapsingHeader(hdr, ImGuiTreeNodeFlags_DefaultOpen))
      continue;

    ImGui::PushID(i);
    for (int L = 0; L < int(layers.size()); ++L) {
      const tsd::scene::Layer *layer = layers[std::size_t(L)];
      if (!layer)
        continue;
      ImGui::PushID(L);
      layer->traverse_const(
          layer->root(), [&](auto &node, int level) -> bool {
            if (level > 0) {
              const std::string label = nodeLabel(*node);
              const bool isSel = (i == m_selViewport && L == m_selLayer
                  && node.index() == m_selNodeIndex);
              ImGui::Indent(INDENT_AMOUNT * level);
              ImGui::PushID(int(node.index()));
              if (ImGui::Selectable(label.c_str(), isSel)) {
                m_selViewport = i;
                m_selLayer = L;
                m_selNodeIndex = node.index();
              }
              ImGui::PopID();
              ImGui::Unindent(INDENT_AMOUNT * level);
              if (isSel && node->isObject())
                selObj = node->getObject();
            }
            return true; // descend into children
          });
      ImGui::PopID();
    }
    ImGui::PopID();
  }

  ImGui::Separator();
  if (selObj) {
    ImGui::BeginDisabled(true);
    tsd::ui::buildUI_object(*selObj, m_bridge->renderScene(), /*useTable=*/true);
    ImGui::EndDisabled();
  } else {
    ImGui::TextDisabled("No object selected");
  }
}

} // namespace tsd::ui::imgui
```

- [ ] **Step 4: Add the source to the UI library CMake**

In `tsd/src/tsd/ui/imgui/CMakeLists.txt`, add `windows/LayerDebug.cpp` to the source list, on the line after the existing `windows/LayerTree.cpp` entry. (Do not reformat the file.) If the sources are listed without a `windows/` prefix, match the prefix style already used for the other `windows/*.cpp` entries in that file.

- [ ] **Step 5: Construct the window in tsdFlow, hidden by default**

In `tsd/apps/interactive/tsdFlow/tsdFlow.cpp`, add the include near the other window includes (after `#include "tsd/ui/imgui/windows/GraphViewport.hpp"`):

```cpp
#include "tsd/ui/imgui/windows/LayerDebug.hpp"
```

In `setupWindows()`, immediately before `windows.emplace_back(new ui::Log(this));`, add:

```cpp
    auto *layerDebug = new ui::LayerDebug(this, m_bridge.get(), "Layer Debug");
    layerDebug->hide(); // debug tool: off by default; toggle via the View menu
    windows.emplace_back(layerDebug);
```

(The base `Window`'s `visiblePtr()` is auto-listed in the View menu, so this provides the toggle with no extra code.)

- [ ] **Step 6: Configure and build**

Run: `cmake _out/_cmake && cmake --build _out/_cmake --parallel`
Expected: configures and builds clean (`tsd_ui_imgui` picks up `LayerDebug.cpp`; `tsdFlow` relinks). If `tsd::scene::Object::subtype()` or the `Object.hpp` include path differs, adjust the include to the header that declares `tsd::scene::Object` (the same one `node->getObject()`'s return type comes from) — do not change behavior.

- [ ] **Step 7: Run the full test suite**

Run: `ctest --test-dir _out/_cmake -C RelWithDebInfo --output-on-failure`
Expected: all pass (68), no regressions. (No test exercises this GUI code; this confirms nothing else broke and the bridge header change compiles against all consumers.)

- [ ] **Step 8: Walk the manual smoke checklist against the code**

Confirm the control flow supports each (the human's on-screen acceptance criteria):
- The View menu lists "Layer Debug" and it starts hidden (`hide()` in `setupWindows`; auto-listed via `visiblePtr()`).
- Toggling it visible shows one `CollapsingHeader` per viewport whose `layersForViewport(i)` is non-empty, each with an indented read-only tree (`Surface : ...`, `... (transform)`).
- Clicking an object row highlights it and the pane below shows that object's params **greyed/non-interactive** (`BeginDisabled` wrapping `buildUI_object`); a transform row selects but the pane shows "No object selected" (transforms have no object).
- Because selection is re-resolved each frame from `(viewport, layer, nodeIndex)` and `selObj` is fetched fresh during that frame's traversal, re-evaluating a node (which rebuilds layers) cannot dangle — at worst the key stops resolving and the pane shows the empty state.

- [ ] **Step 9: Commit**

```bash
jj commit tsd/src/tsd/rendering/bridge/GraphRenderBridge.hpp tsd/src/tsd/ui/imgui/windows/LayerDebug.hpp tsd/src/tsd/ui/imgui/windows/LayerDebug.cpp tsd/src/tsd/ui/imgui/CMakeLists.txt tsd/apps/interactive/tsdFlow/tsdFlow.cpp -m "feat(tsdflow): read-only per-viewport layer debug panel"
```

---

## Self-Review

**Spec coverage:**
- New read-only `LayerDebug` window, hidden by default, View-menu toggle via `visiblePtr()` → Steps 2, 3, 5. ✓
- Stacked `CollapsingHeader` per enabled (non-empty-layers) viewport; read-only selectable tree with `Type : subtype` / `name (transform)` labels → Step 3. ✓
- Selection re-resolved per frame via `(viewport, layer, nodeIndex)`, no persisted `Object*` → Step 3 + Global Constraints. ✓
- Companion pane via `buildUI_object` inside `BeginDisabled` (read-only, no read-only flag exists) → Step 3. ✓
- Bridge `renderScene()` accessor → Step 1. (Spec also floated a non-const `layersForViewport` overload; dropped — `getObject() const` returns `Object*`, so const traversal suffices. Refinement, less change, same behavior.) ✓
- Testing = build + suite + smoke; no hollow/heavy test → Steps 6–8 + Global Constraints. (Spec's optional accessor test dropped as low-value given the getter is one line and the non-const overload was removed.) ✓

**Placeholder scan:** none — every code step carries complete code. Step 4/Step 6 contain conditional "adjust if the prefix/header differs" guidance, but each names the exact concrete action and fallback, not a TODO.

**Type consistency:** `renderScene()` returns `tsd::scene::Scene&`, consumed by `buildUI_object(Object&, Scene&, bool)` in Step 3; selection fields `m_selViewport`(int)/`m_selLayer`(int)/`m_selNodeIndex`(size_t) declared in Step 2 and used identically in Step 3; `nodeLabel`/`stripAnariPrefix` defined and used in Step 3; `layersForViewport(int) const` and `numViewports() const` match the existing bridge API; visitor lambda is `[&](auto&, int) -> bool` matching `traverse_const`.
