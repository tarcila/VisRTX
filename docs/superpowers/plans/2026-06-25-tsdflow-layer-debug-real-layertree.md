# tsdFlow Layer Debug via real LayerTree — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show the actual tsd layer structure in tsdFlow's debug panel by reusing the real `LayerTree` widget, parameterized to render an arbitrary scene read-only, pointed at the bridge's render scene.

**Architecture:** Two tasks. (1) Parameterize `LayerTree` with a scene override + a read-only mode (defaults preserve current behavior exactly, so tsdViewer is untouched). (2) Rewrite `LayerDebug` as a thin composer that owns a read-only `LayerTree` on `GraphRenderBridge::renderScene()` and renders a greyed object-param pane below it; delete the old custom tree.

**Tech Stack:** C++17, Dear ImGui, `tsd::scene::Layer`/`LayerNodeData`, `tsd::ui::buildUI_object`, `GraphRenderBridge`.

## Global Constraints

- **jj, not git.** Commit with explicit file paths; never bare `jj commit`, never raw `git`.
- **No `Co-Authored-By` lines.**
- **Never run clang-format on any `CMakeLists.txt`.**
- **Behavior-preservation invariant (critical):** with the new `LayerTree` parameters at their defaults (`sceneOverride == nullptr`, `readOnly == false`), `LayerTree` must behave byte-for-byte as today — tsdViewer's "Layers" window is unchanged. In non-read-only mode the new dispatch helpers call the exact same `appContext()` methods, and the new `!m_readOnly &&` guards are always true.
- **Read-only means no mutation:** in read-only mode no node/layer add/remove/clear, no drag/drop, no cut/copy/paste, no scene context menus; selection is local single-select and must not touch `appContext()` selection.
- Configure: `cmake _out/_cmake`. Build: `cmake --build _out/_cmake --parallel`. Test: `ctest --test-dir _out/_cmake -C RelWithDebInfo --output-on-failure` (`-C RelWithDebInfo` REQUIRED; suite is 68, must stay green).
- **Testing:** GUI rendering over an existing scene; no pure-logic seam (consistent with prior tsdFlow GUI work). No new unit test; verify build-green + suite-green + a reasoned manual smoke walk (including that tsdViewer's LayerTree still edits/selects normally).

---

### Task 1: Parameterize `LayerTree` (scene override + read-only)

**Files:**
- Modify: `tsd/src/tsd/ui/imgui/windows/LayerTree.h`
- Modify: `tsd/src/tsd/ui/imgui/windows/LayerTree.cpp`

**Interfaces:**
- Consumes (existing): `appContext()->tsd.scene` (`tsd::scene::Scene`), `appContext()->tsd.sceneLoadComplete`, the `appContext()` selection API (`isSelected/getFirstSelected/getSelectedNodes/setSelected/addToSelection/removeFromSelection/clearSelected`); `tsd::scene::Scene::{numberOfLayers, layer(idx), layers()}`; `tsd::scene::Layer::at(size_t)→LayerNodeRef`; `LayerNodeRef::{valid(), index()}` and `(*ref)->{isObject(),getObject()}`; `TSD_INVALID_INDEX` (from `tsd/core/FlatMap.hpp`, already included).
- Produces: `LayerTree(Application*, const char* = "Layers", tsd::scene::Scene* sceneOverride = nullptr, bool readOnly = false)`; `tsd::scene::Object *LayerTree::readOnlySelectedObject() const`.

**Context:** `LayerTree` (`buildUI` calls `buildUI_layerHeader`, `buildUI_tree`, `buildUI_handleSelection`, then four `*SceneMenu` helpers) is welded to the app scene + global selection. We add a scene source indirection and a read-only mode so it can render the bridge's render scene without editing it or polluting app selection.

- [ ] **Step 1: Header — constructor, members, method declarations**

In `tsd/src/tsd/ui/imgui/windows/LayerTree.h`:

Replace the constructor declaration:
```cpp
  LayerTree(Application *app, const char *name = "Layers");
```
with:
```cpp
  LayerTree(Application *app,
      const char *name = "Layers",
      tsd::scene::Scene *sceneOverride = nullptr, // null => appContext()->tsd.scene
      bool readOnly = false);

  // Valid only in read-only mode: the locally-selected node's object, or null.
  tsd::scene::Object *readOnlySelectedObject() const;
```

In the `private:` section, add these method declarations after `void buildUI_setActiveLayersSceneMenus();`:
```cpp
  // Scene source: the override when set, else the app scene.
  tsd::scene::Scene &activeScene();
  // Selection dispatch: app selection normally; a local single-select in
  // read-only mode (so the render-scene view never touches app selection).
  bool isSel(const tsd::scene::LayerNodeRef &r);
  tsd::scene::LayerNodeRef firstSel();
  std::vector<tsd::scene::LayerNodeRef> selNodes();
  void setSel(const tsd::scene::LayerNodeRef &r);
  void setSel(const std::vector<tsd::scene::LayerNodeRef> &v);
  void addSel(const tsd::scene::LayerNodeRef &r);
  void removeSel(const tsd::scene::LayerNodeRef &r);
  void clearSel();
```

In the `// Data //` section, add after `tsd::scene::LayerNodeRef m_anchorNode;`:
```cpp
  tsd::scene::Scene *m_sceneOverride{nullptr};
  bool m_readOnly{false};
  size_t m_roSelectedIndex{TSD_INVALID_INDEX}; // read-only local selection
```

Add the include for `Object`/`Scene`/`LayerNodeRef` types if not transitively present — `Window.h` + `tsd/core/FlatMap.hpp` are included; the types resolve via the existing includes in the `.cpp`. If the header fails to compile on the new `tsd::scene::*` references, add `#include "tsd/scene/Scene.hpp"` to `LayerTree.h`.

- [ ] **Step 2: Constructor + activeScene() + selection dispatch + readOnlySelectedObject()**

In `tsd/src/tsd/ui/imgui/windows/LayerTree.cpp`, replace the constructor:
```cpp
LayerTree::LayerTree(Application *app, const char *name) : Window(app, name) {}
```
with:
```cpp
LayerTree::LayerTree(Application *app,
    const char *name,
    tsd::scene::Scene *sceneOverride,
    bool readOnly)
    : Window(app, name), m_sceneOverride(sceneOverride), m_readOnly(readOnly)
{
  if (m_readOnly)
    m_enableAddRemove = false; // also disables layer new/delete + node add/remove
}

tsd::scene::Scene &LayerTree::activeScene()
{
  return m_sceneOverride ? *m_sceneOverride : appContext()->tsd.scene;
}

bool LayerTree::isSel(const tsd::scene::LayerNodeRef &r)
{
  if (m_readOnly)
    return r.valid() && r.index() == m_roSelectedIndex;
  return appContext()->isSelected(r);
}

tsd::scene::LayerNodeRef LayerTree::firstSel()
{
  if (m_readOnly) {
    if (m_roSelectedIndex == TSD_INVALID_INDEX
        || m_layerIdx >= int(activeScene().numberOfLayers()))
      return {};
    return activeScene().layer(m_layerIdx)->at(m_roSelectedIndex);
  }
  return appContext()->getFirstSelected();
}

std::vector<tsd::scene::LayerNodeRef> LayerTree::selNodes()
{
  if (m_readOnly) {
    std::vector<tsd::scene::LayerNodeRef> v;
    auto r = firstSel();
    if (r.valid())
      v.push_back(r);
    return v;
  }
  return appContext()->getSelectedNodes();
}

void LayerTree::setSel(const tsd::scene::LayerNodeRef &r)
{
  if (m_readOnly)
    m_roSelectedIndex = r.valid() ? r.index() : TSD_INVALID_INDEX;
  else
    appContext()->setSelected(r);
}

void LayerTree::setSel(const std::vector<tsd::scene::LayerNodeRef> &v)
{
  if (m_readOnly)
    m_roSelectedIndex = (!v.empty() && v.front().valid())
        ? v.front().index()
        : TSD_INVALID_INDEX;
  else
    appContext()->setSelected(v);
}

void LayerTree::addSel(const tsd::scene::LayerNodeRef &r)
{
  if (m_readOnly)
    m_roSelectedIndex = r.valid() ? r.index() : TSD_INVALID_INDEX;
  else
    appContext()->addToSelection(r);
}

void LayerTree::removeSel(const tsd::scene::LayerNodeRef &r)
{
  if (m_readOnly) {
    if (r.valid() && r.index() == m_roSelectedIndex)
      m_roSelectedIndex = TSD_INVALID_INDEX;
  } else
    appContext()->removeFromSelection(r);
}

void LayerTree::clearSel()
{
  if (m_readOnly)
    m_roSelectedIndex = TSD_INVALID_INDEX;
  else
    appContext()->clearSelected();
}

tsd::scene::Object *LayerTree::readOnlySelectedObject() const
{
  if (!m_readOnly || !m_sceneOverride
      || m_roSelectedIndex == TSD_INVALID_INDEX)
    return nullptr;
  if (m_layerIdx < 0 || m_layerIdx >= int(m_sceneOverride->numberOfLayers()))
    return nullptr;
  auto *layer = m_sceneOverride->layer(m_layerIdx);
  if (!layer)
    return nullptr;
  auto ref = layer->at(m_roSelectedIndex);
  if (!ref.valid() || !(*ref)->isObject())
    return nullptr;
  return (*ref)->getObject();
}
```

- [ ] **Step 3: Route the scene-load guard and all scene access through `activeScene()`**

In `buildUI()`, change the guard:
```cpp
  if (!appContext()->tsd.sceneLoadComplete) {
```
to (an override scene is always ready):
```cpp
  if (!m_sceneOverride && !appContext()->tsd.sceneLoadComplete) {
```

Then replace **every** remaining occurrence of `appContext()->tsd.scene` in `LayerTree.cpp` with `activeScene()`. Specifically: the `auto &scene = appContext()->tsd.scene;` lines (in `buildUI_layerHeader`, `buildUI_tree`, `buildUI_handleSelection`, `buildUI_objectSceneMenu`, `buildUI_newLayerSceneMenu`, `buildUI_setActiveLayersSceneMenus`) become `auto &scene = activeScene();`, and the standalone uses (e.g. `appContext()->tsd.scene.removeAllObjects();`, `appContext()->tsd.scene.signalLayerStructureChanged(&layer);`) become `activeScene().removeAllObjects();` / `activeScene().signalLayerStructureChanged(&layer);`. Leave `appContext()->tsd.sceneLoadComplete` and `appContext()->tsd.stashedSelection` as-is (those are not the scene object).

- [ ] **Step 4: Early-return the editing/menu helpers in read-only mode**

Add `if (m_readOnly) return;` as the **first line** of each of these five methods: `buildUI_handleSelection`, `buildUI_activateObjectSceneMenu`, `buildUI_objectSceneMenu`, `buildUI_newLayerSceneMenu`, `buildUI_setActiveLayersSceneMenus`. (They are all clipboard/delete/add/menu mutation; read-only disables them wholesale.)

- [ ] **Step 5: Route `buildUI_tree` display + click selection through the dispatch helpers**

In `buildUI_tree`, the display-highlight reads:
```cpp
      auto selectedNodeRef = appContext()->getFirstSelected();
      auto currentNodeRef = layer.at(node.index());

      // Check if this node is in the selection set
      const bool isSelectedNode = appContext()->isSelected(currentNodeRef);
```
become:
```cpp
      auto selectedNodeRef = firstSel();
      auto currentNodeRef = layer.at(node.index());

      // Check if this node is in the selection set
      const bool isSelectedNode = isSel(currentNodeRef);
```
and:
```cpp
        const auto &selectedNodes = appContext()->getSelectedNodes();
```
becomes:
```cpp
        const auto selectedNodes = selNodes();
```

Replace the entire click-selection block (`if (ImGui::IsItemClicked() && m_menuNode == TSD_INVALID_INDEX) { ... }`) with (modifier multi-select gated off in read-only; all selection via helpers):
```cpp
      if (ImGui::IsItemClicked() && m_menuNode == TSD_INVALID_INDEX) {
        auto clickedNode = layer.at(node.index());

        ImGuiIO &io = ImGui::GetIO();
        bool ctrlPressed = io.KeyCtrl;
        bool shiftPressed = io.KeyShift;
        bool isAlreadySelected = isSel(clickedNode);

        if (!m_readOnly && ctrlPressed) {
          if (isSel(clickedNode)) {
            removeSel(clickedNode);
          } else {
            addSel(clickedNode);
          }
          m_anchorNode = clickedNode;
        } else if (!m_readOnly && shiftPressed) {
          if (m_anchorNode.valid()) {
            auto rangeNodes =
                computeSelectionRange(layer, m_anchorNode, clickedNode);
            if (!rangeNodes.empty()) {
              setSel(rangeNodes);
            }
          } else {
            addSel(clickedNode);
            m_anchorNode = clickedNode;
          }
        } else if (!isAlreadySelected) {
          setSel(clickedNode);
          m_anchorNode = clickedNode;
        }
      }
```

- [ ] **Step 6: Gate drag-and-drop in `buildUI_tree` behind `!m_readOnly` (keep release-selection)**

Replace the drag-source / deferred-selection block. Change the opening of the drag source so it short-circuits in read-only (then the `else` — the release-selection — runs), and route its selection through helpers:
```cpp
      // Drag and drop source
      if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_None)) {
```
becomes:
```cpp
      // Drag and drop source (disabled in read-only mode)
      if (!m_readOnly && ImGui::BeginDragDropSource(ImGuiDragDropFlags_None)) {
```
and inside the `else { ... }` deferred-selection branch:
```cpp
          bool isAlreadySelected = appContext()->isSelected(clickedNode);
          ...
            appContext()->setSelected(clickedNode);
```
become:
```cpp
          bool isAlreadySelected = isSel(clickedNode);
          ...
            setSel(clickedNode);
```

Wrap the **drag-and-drop target** block (`if (ImGui::BeginDragDropTarget()) { ... }` inside the node lambda) in `if (!m_readOnly) { ... }`.

Wrap the **panel-level drop target** block (the `if (ImGui::BeginDragDropTarget()) { ... }` after `ImGui::EndTable();`) and the **deferred drop handling** block (`if (dragAndDropTarget.valid() && !droppedNodes.empty()) { ... }`) together in `if (!m_readOnly) { ... }`.

- [ ] **Step 7: Gate the "clear" button + reset read-only selection on layer change**

In `buildUI_layerHeader`, wrap the "clear" button in `!m_readOnly`:
```cpp
  if (ImGui::Button("clear")) {
    appContext()->clearSelected();
    appContext()->tsd.scene.removeAllObjects();
  }
```
becomes:
```cpp
  if (!m_readOnly && ImGui::Button("clear")) {
    clearSel();
    activeScene().removeAllObjects();
  }
```
And capture the layer-dropdown change to drop a stale read-only selection. Change:
```cpp
  ImGui::Combo("##layer",
      &m_layerIdx,
      UI_layerName_callback,
      (void *)&layers,
      layers.size());
```
to:
```cpp
  if (ImGui::Combo("##layer",
          &m_layerIdx,
          UI_layerName_callback,
          (void *)&layers,
          layers.size())
      && m_readOnly) {
    m_roSelectedIndex = TSD_INVALID_INDEX;
  }
```

- [ ] **Step 8: Configure and build**

Run: `cmake _out/_cmake && cmake --build _out/_cmake --parallel`
Expected: builds clean. If a `tsd::scene::*` type is unresolved in `LayerTree.h`, add `#include "tsd/scene/Scene.hpp"` there. Watch for any remaining `appContext()->tsd.scene` (must all be `activeScene()` except `sceneLoadComplete`/`stashedSelection`).

- [ ] **Step 9: Run the full suite**

Run: `ctest --test-dir _out/_cmake -C RelWithDebInfo --output-on-failure`
Expected: 68 pass, no regressions.

- [ ] **Step 10: Verify the behavior-preservation invariant by reading the diff**

Confirm that with defaults (`m_sceneOverride==nullptr`, `m_readOnly==false`): `activeScene()` returns `appContext()->tsd.scene`; every dispatch helper calls the identical `appContext()` method it replaced; every `!m_readOnly &&` guard is true; the five early-returns don't fire. So tsdViewer's LayerTree path is unchanged. (No code change to behavior when read-only is off.)

- [ ] **Step 11: Commit**

```bash
jj commit tsd/src/tsd/ui/imgui/windows/LayerTree.h tsd/src/tsd/ui/imgui/windows/LayerTree.cpp -m "feat(ui): LayerTree scene-override + read-only mode (defaults unchanged)"
```

---

### Task 2: `LayerDebug` becomes a composer over the real LayerTree

**Files:**
- Modify: `tsd/src/tsd/ui/imgui/windows/LayerDebug.hpp` (rewrite)
- Modify: `tsd/src/tsd/ui/imgui/windows/LayerDebug.cpp` (rewrite — delete the custom tree)

**Interfaces:**
- Consumes: `LayerTree(Application*, const char*, tsd::scene::Scene*, bool)` and `LayerTree::readOnlySelectedObject()` (Task 1); `GraphRenderBridge::renderScene()`; `tsd::ui::buildUI_object`.
- Produces: unchanged `LayerDebug(Application*, GraphRenderBridge*, const char*)` ctor (tsdFlow wiring stays as-is).

**Context:** `LayerDebug` is registered in `tsdFlow.cpp` (hidden, View-menu toggle). Its current `.cpp` renders a custom tree; replace that with an owned read-only `LayerTree` on the render scene + a greyed object pane. The ctor signature must not change (tsdFlow already constructs it).

- [ ] **Step 1: Rewrite the header**

Replace `tsd/src/tsd/ui/imgui/windows/LayerDebug.hpp` with:
```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/rendering/bridge/GraphRenderBridge.hpp"
#include "tsd/ui/imgui/windows/LayerTree.hpp"
#include "tsd/ui/imgui/windows/Window.h"
// std
#include <memory>

namespace tsd::ui::imgui {

// Hidden-by-default debug panel: the real LayerTree, read-only, over the
// bridge's render scene, plus a greyed param pane for the selected object.
struct LayerDebug : public Window
{
  LayerDebug(Application *app,
      tsd::rendering::GraphRenderBridge *bridge,
      const char *name = "Layer Debug");
  void buildUI() override;

 private:
  tsd::rendering::GraphRenderBridge *m_bridge{nullptr};
  std::unique_ptr<LayerTree> m_tree;
};

} // namespace tsd::ui::imgui
```
Note: `LayerTree` is declared in `LayerTree.h` (not `.hpp`). Use the correct existing path in the include: `#include "tsd/ui/imgui/windows/LayerTree.h"`. (Adjust the include above to `LayerTree.h`.)

- [ ] **Step 2: Rewrite the implementation**

Replace `tsd/src/tsd/ui/imgui/windows/LayerDebug.cpp` with:
```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/ui/imgui/windows/LayerDebug.hpp"
#include "tsd/scene/Object.hpp"
#include "tsd/ui/imgui/tsd_ui_imgui.h"
// imgui
#include "imgui.h"

namespace tsd::ui::imgui {

LayerDebug::LayerDebug(
    Application *app, tsd::rendering::GraphRenderBridge *bridge, const char *name)
    : Window(app, name), m_bridge(bridge)
{
  if (m_bridge) {
    m_tree = std::make_unique<LayerTree>(
        app, "Render Layers", &m_bridge->renderScene(), /*readOnly=*/true);
  }
}

void LayerDebug::buildUI()
{
  if (!m_bridge || !m_tree) {
    ImGui::TextDisabled("No bridge");
    return;
  }

  m_tree->buildUI(); // the real LayerTree, read-only, over the render scene

  ImGui::Separator();
  if (auto *o = m_tree->readOnlySelectedObject()) {
    ImGui::BeginDisabled(true);
    tsd::ui::buildUI_object(*o, m_bridge->renderScene(), /*useTable=*/true);
    ImGui::EndDisabled();
  } else {
    ImGui::TextDisabled("No object selected");
  }
}

} // namespace tsd::ui::imgui
```
(Use `#include "tsd/scene/Object.hpp"` — the path confirmed correct in the shipped version. The CMake source list already contains `windows/LayerDebug.cpp`; no CMake change.)

- [ ] **Step 3: Build**

Run: `cmake --build _out/_cmake --parallel`
Expected: builds clean. `tsdFlow.cpp` is unchanged (the `LayerDebug(this, m_bridge.get(), "Layer Debug")` construction + `hide()` still apply).

- [ ] **Step 4: Run the full suite**

Run: `ctest --test-dir _out/_cmake -C RelWithDebInfo --output-on-failure`
Expected: 68 pass, no regressions.

- [ ] **Step 5: Walk the manual smoke checklist**

Confirm the control flow supports each:
- View menu → "Layer Debug" shows the **real LayerTree** rows over the render scene; the layer dropdown lists the render layers; expand/collapse and the genuine `[S]/[V]/[T]` rows render.
- Read-only: "new"/"delete"/"clear" do nothing or are absent, no drag/drop, no node add/remove, no context menus, Delete/Ctrl-C/V/X inert; clicking a node highlights it but does **not** change the app's global selection (the main Inspector/selection is unaffected).
- Selecting an object row shows its params greyed/read-only in the pane; a transform row → "No object selected".
- tsdViewer (separate app) "Layers" window still fully edits — unaffected by defaults.

- [ ] **Step 6: Commit**

```bash
jj commit tsd/src/tsd/ui/imgui/windows/LayerDebug.hpp tsd/src/tsd/ui/imgui/windows/LayerDebug.cpp -m "feat(tsdflow): layer debug panel embeds the real read-only LayerTree on the render scene"
```

---

## Self-Review

**Spec coverage:**
- LayerTree scene-override + read-only, defaults preserve behavior → Task 1 Steps 1–3, 10. ✓
- Read-only gating (menus/clipboard early-return; drag/drop guarded; mutation buttons gated) → Task 1 Steps 4, 6, 7. ✓
- Local single-select, no app-selection pollution → Task 1 Steps 2, 5. ✓
- `readOnlySelectedObject()` exposure → Task 1 Step 2. ✓
- LayerDebug composer: real LayerTree on `renderScene()` + greyed pane; custom tree deleted → Task 2 Steps 1, 2. ✓
- Native layer dropdown; per-viewport grouping dropped → inherent in using LayerTree (Task 2). ✓
- Layer-naming nicety → **deferred** (the bridge's layer-creation site is not specified here; `<unnamed layer>` in the dropdown is acceptable for v1; noted as follow-up rather than left as a placeholder). ✓ (documented divergence)
- Testing = build + suite + smoke; tsdViewer-unaffected check → Steps 8–10 (T1), 3–5 (T2). ✓

**Placeholder scan:** none — every code step carries complete code. Step 3/Step 8 mechanical-replace instructions name the exact transformation and the exact exceptions (`sceneLoadComplete`, `stashedSelection`); not TODOs.

**Type consistency:** `activeScene()` returns `tsd::scene::Scene&` used as `auto &scene = activeScene();`; dispatch helpers' signatures in Step 1 match their definitions in Step 2 and call sites in Steps 5–7; `setSel` has both ref and vector overloads (vector used only in the non-read-only shift branch); `readOnlySelectedObject()` declared (Step 1) / defined (Step 2) / consumed (Task 2 Step 2); `LayerDebug` ctor signature unchanged so tsdFlow wiring stays valid; include path `LayerTree.h` (not `.hpp`) and `tsd/scene/Object.hpp` noted explicitly.
