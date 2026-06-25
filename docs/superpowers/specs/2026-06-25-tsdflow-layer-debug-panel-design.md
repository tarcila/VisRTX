# tsdFlow: Per-Viewport Layer Debug Panel — Design

**Date:** 2026-06-25
**Status:** Approved (pending user spec review)
**Phase:** Intermediary step 1 of 3 (layer debug panel → per-viewport menus → lights)

## Problem

There is no way to see what the `GraphRenderBridge` actually realized per viewport — the layer/scene-graph it built from the graph. A read-only debug panel that, per enabled viewport, shows the realized layer forest and lets you inspect a selected object's live render parameters would make the bridge's output observable while debugging the rest of tsdFlow.

## Scope

This is the first of three independent intermediary specs (this one; then per-viewport viewport menus; then light node types). It covers **only** the layer debug panel. It is a debug/inspection tool: **read-only**, **hidden by default**.

## Why not reuse `LayerTree` / `ObjectEditor`

- `LayerTree` is a full editor bound to `appContext()->tsd.scene` (selection, add/remove, drag/drop). The bridge's layers live in its private, **per-frame-regenerated** `m_renderScene` and must not be edited. Wrong tool.
- `ObjectEditor` is hardwired to `appContext()->getFirstSelected()` + the app scene. Also wrong binding.
- The reusable piece is the widget both wrap: `tsd::ui::buildUI_object(Object&, Scene&, useTableForParameters, level)` (declared in `tsd/src/tsd/ui/imgui/tsd_ui_imgui.h`). It has **no read-only flag**, so the panel renders it inside `ImGui::BeginDisabled(true)` / `ImGui::EndDisabled()` — every param widget shows its live value but is greyed and non-interactive, so nothing mutates the render object.

## Component: `LayerDebug` window

**Files:** `tsd/src/tsd/ui/imgui/windows/LayerDebug.hpp` + `.cpp` (new), constructed in `tsd/apps/interactive/tsdFlow/tsdFlow.cpp`.

### Construction & visibility

```cpp
LayerDebug(Application *app, tsd::rendering::GraphRenderBridge *bridge,
    const char *name = "Layer Debug");
```

Registered in `setupWindows()` and immediately `->hide()` (matching how the viewport pool starts with only viewport 1 visible). The base `Window`'s `visiblePtr()` is auto-listed in the View menu (`uiMainMenuBar_View`), which provides the toggle for free — "togglable through a menu entry, disabled by default" needs no extra menu code.

### Layout (`buildUI`)

One window. For each viewport `i` in `[0, bridge->numViewports())` whose layer list is non-empty (an "enabled" viewport), a stacked `ImGui::CollapsingHeader("Viewport N")` containing a read-only tree of that viewport's layer forest. Below the headers, a separator and a read-only **object pane** for the current selection.

```
[Layer Debug]                         (hidden by default)
 v Viewport 1
     root (transform)
       Surface : cylinder      <- click to select (highlighted)
       Volume  : transferFunction1D
 v Viewport 2
     root (transform)
       Surface : triangle
 ----------------------------------------------------------
 Object: Surface                       (read-only / greyed)
   vertex.position  [24]
   radius = 0.04
```

### Tree rendering

For each viewport's layers (see bridge API below), walk the forest and render one selectable row per node, indented by traversal level:

- Object node (`node->isObject()` / `node->getObject() != nullptr`): label from the object's type + subtype (e.g. `Surface : cylinder`), via the `tsd::scene::Object` API (`type()`, `subtype()`); fall back to the node `name()` if no subtype.
- Transform node: `name()` + ` (transform)`.

Rows are `ImGui::Selectable`; the root (level 0) may be shown or skipped consistent with `LayerTree`'s convention. No drag/drop, no rename, no add/remove, no mutation of the app selection state.

### Selection & the object pane (per-frame re-resolution — critical)

The bridge **rebuilds layer objects every time a source node re-evaluates**, so a stored `Object*` would dangle. Selection is therefore a stable *key*, re-resolved each frame:

- State: `int m_selViewport{-1}; size_t m_selNodeIndex{TSD_INVALID_INDEX};`
- At the top of `buildUI`, clear a frame-local `tsd::scene::Object *selObj = nullptr;`.
- While walking viewport `m_selViewport`'s layers, when a node's index equals `m_selNodeIndex` and it is an object node, set `selObj = node->getObject()` (valid for the remainder of this frame only) and highlight that row.
- Clicking a row sets `m_selViewport`/`m_selNodeIndex` to that node.
- After the headers: if `selObj`, render the pane:
  ```cpp
  ImGui::Separator();
  ImGui::BeginDisabled(true);
  tsd::ui::buildUI_object(*selObj, bridge->renderScene(), /*useTable=*/true);
  ImGui::EndDisabled();
  ```
  else `ImGui::TextDisabled("No object selected")`.

If the key no longer resolves (the node was rebuilt away), `selObj` stays null and the pane shows the empty state — no crash, no dangling pointer.

## Bridge API additions

`GraphRenderBridge` (`tsd/src/tsd/rendering/bridge/GraphRenderBridge.hpp/.cpp`) gains two read accessors; no behavior change:

- `std::vector<tsd::scene::Layer *> layersForViewport(int i);` — non-const overload (the existing `const` one stays). Needed because `buildUI_object` and `getObject()` require non-const access for the pane.
- `tsd::scene::Scene &renderScene();` — returns `m_renderScene`, so the pane can pass the owning scene to `buildUI_object`. (The panel only ever calls `buildUI_object` under `BeginDisabled`, so it does not mutate the scene; the accessor is non-const solely to satisfy the widget signature.)

## Data flow & boundaries

- Self-contained: a new window reading the bridge's already-built per-viewport layers after `update()` (which the app calls each frame before UI). No graph, evaluator, or node changes.
- The window depends only on `GraphRenderBridge` (the two new accessors), `tsd::scene::Layer`/`LayerNodeData` traversal, and `buildUI_object`.

## Testing

The panel is ImGui rendering over the bridge's layers — no pure-logic seam the unit harness drives, consistent with the other tsdFlow GUI specs. Verification:

- **Build green + full suite green, no regressions** (currently 68 tests).
- A trivial bridge accessor test (`test_bridge_*`): after building the demo graph + `update()`, assert the non-const `layersForViewport(i)` returns the same count as the existing const overload for each viewport, and `renderScene().numberOfLayers() > 0`. (Cheap, guards the new accessors; the panel rendering itself has no assertable seam.)
- **Manual smoke checklist:** the View menu lists "Layer Debug" and it starts hidden; toggling it shows a section per enabled viewport with the realized layer tree; clicking an object row shows its params greyed/read-only in the pane; params are non-interactive; re-evaluating a node (e.g. editing `isovalue`/`isovalue`-like params) does not crash the selection.

## Out of Scope

- Editing render-scene objects (read-only by decision; editing is ephemeral and confusing).
- Mapping a layer object back to its source graph node / driving the graph Inspector.
- One-window-per-viewport variant (single stacked window chosen).
- Per-node parameter expansion in the tree itself (params live in the pane).
- Persistence of the panel's open/closed state beyond the existing imgui-ini behavior.
