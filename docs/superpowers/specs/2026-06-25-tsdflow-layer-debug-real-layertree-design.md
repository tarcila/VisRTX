# tsdFlow: Layer Debug Panel via the real LayerTree — Design (revision)

**Date:** 2026-06-25
**Status:** Approved (pending user spec review)
**Phase:** Intermediary step 1 of 3 (revision)
**Supersedes:** `2026-06-25-tsdflow-layer-debug-panel-design.md` (custom read-only tree). The custom tree shipped (`229e2394`) but did not show the *actual* tsd layer structure; this revision replaces its internals with the genuine `LayerTree` widget.

## Problem

The shipped `LayerDebug` panel renders a simplified custom tree (type:subtype labels). The user wants the **actual tsd layer structure** — the genuine `LayerTree` visualization (the real forest: transforms, object refs, enabled state, expand/collapse) — pointed at the bridge's render scene, rather than a custom reimplementation.

## Approach

Parameterize the existing `LayerTree` window so it can render an **arbitrary scene, read-only**, and embed it (pointed at `GraphRenderBridge::renderScene()`) inside the `LayerDebug` panel. `LayerTree`'s defaults are preserved exactly, so tsdViewer's editable LayerTree is unchanged.

The actual layers in tsdFlow live in the bridge's `m_renderScene` (its own `tsd::scene::Scene`, regenerated each frame from the graph). The app's `appContext()->tsd.scene` is not the populated scene here.

## Component 1: `LayerTree` parameterization

**File:** `tsd/src/tsd/ui/imgui/windows/LayerTree.{h,cpp}`.

### New constructor + state

```cpp
LayerTree(Application *app,
    const char *name = "Layers",
    tsd::scene::Scene *sceneOverride = nullptr, // null => appContext()->tsd.scene
    bool readOnly = false);
```

New members: `tsd::scene::Scene *m_sceneOverride{nullptr}; bool m_readOnly{false};` and, for read-only selection that must not touch the app's global selection, a local `tsd::scene::LayerNodeRef m_roSelected{};`.

### Scene source indirection

Add a private helper and route **every** `appContext()->tsd.scene` access through it:

```cpp
tsd::scene::Scene &scene() { return m_sceneOverride ? *m_sceneOverride : appContext()->tsd.scene; }
```

Replace the ~8 `appContext()->tsd.scene` / `auto &scene = appContext()->tsd.scene;` sites with `scene()`. When `m_sceneOverride` is set, also skip the `appContext()->tsd.sceneLoadComplete` early-return guard (that flag pertains to the app scene; an override scene is always "ready").

### Read-only gating

When `m_readOnly`:
- Force `m_enableAddRemove = false` (already gates the new/delete layer buttons and node add/remove). Additionally gate the **"clear"** button (`removeAllObjects`) and any other mutation behind `!m_readOnly`.
- Early-return the editing/menu helpers: `buildUI_activateObjectSceneMenu`, `buildUI_objectSceneMenu`, `buildUI_newLayerSceneMenu`, `buildUI_setActiveLayersSceneMenus`, and the drag/drop + delete-key handling in `buildUI_handleSelection` / the drop processing. (Tree display, layer dropdown, expand/collapse, and single-select remain.)
- Route selection through the local ref instead of `appContext()`:
  - read: `isSelected(ref)` → `m_readOnly ? (ref == m_roSelected) : appContext()->isSelected(ref)`; `getFirstSelected()`/`getSelectedNodes()` → the local ref;
  - write: `setSelected`/`addToSelection`/`removeFromSelection`/`clearSelected` → set/clear `m_roSelected` (single-select only; range/multi-select disabled in read-only).
  Wrap each of the ~8 selection sites in an `if (m_readOnly) {local} else {appContext}` branch.

### Expose the read-only selection

So the embedding panel can show the selected object's params:

```cpp
tsd::scene::Object *readOnlySelectedObject() const; // m_roSelected valid && isObject ? getObject() : nullptr
```

### Invariant

With `sceneOverride == nullptr && readOnly == false` (the defaults), behavior is byte-for-byte the current LayerTree — tsdViewer is unaffected.

## Component 2: `LayerDebug` becomes a thin composer

**File:** `tsd/src/tsd/ui/imgui/windows/LayerDebug.{hpp,cpp}` (rewrite; remove the custom-tree + traversal code).

`LayerDebug` owns a `LayerTree` instance constructed read-only on the render scene and renders the greyed object pane below it:

```cpp
struct LayerDebug : public Window {
  LayerDebug(Application *app, tsd::rendering::GraphRenderBridge *bridge,
      const char *name = "Layer Debug");
  void buildUI() override;
 private:
  tsd::rendering::GraphRenderBridge *m_bridge{nullptr};
  std::unique_ptr<LayerTree> m_tree; // LayerTree(app, "Render Layers", &bridge->renderScene(), /*readOnly=*/true)
};
```

`buildUI()`:
```cpp
m_tree->buildUI();                       // the real LayerTree, read-only, on renderScene()
ImGui::Separator();
if (auto *o = m_tree->readOnlySelectedObject()) {
  ImGui::BeginDisabled(true);
  tsd::ui::buildUI_object(*o, m_bridge->renderScene(), /*useTable=*/true);
  ImGui::EndDisabled();
} else {
  ImGui::TextDisabled("No object selected");
}
```

The embedded `LayerTree` is **not** registered in the app's window array (it is a member, its `buildUI` is called directly), so it is not separately listed/shown. `LayerDebug` remains the registered window: hidden by default (`hide()`), toggled from the View menu. The per-viewport `CollapsingHeader` grouping is dropped — the render scene's layers are browsed via LayerTree's native layer dropdown.

The lifetime concern from the prior design is gone: `LayerTree` reads `scene()` fresh each frame and stores only a `LayerNodeRef` (an `ObjectPool` handle, not a raw pointer); after a rebuild it either resolves or doesn't.

## Bridge

`GraphRenderBridge::renderScene()` (added in the shipped version) is reused unchanged. Optional readability nicety, **in scope**: name each render layer after its source node (so LayerTree's dropdown reads e.g. `BoundingBox` instead of `<unnamed layer>`) — set the layer name when the bridge creates/realizes a display's layer. If trivial at the layer-creation site, do it; otherwise defer (note it).

## Data flow & boundaries

- `LayerTree` gains a scene source + read-only behavior; its editable default path is untouched (the binding invariant above is the contract).
- `LayerDebug` depends only on `LayerTree` + `GraphRenderBridge::renderScene()` + `buildUI_object`.
- No graph/evaluator/node changes.

## Testing

GUI rendering over an existing scene; no pure-logic seam (consistent with prior tsdFlow GUI specs). Verification:
- **Build green + full suite green** (currently 68), no regressions.
- **Manual smoke checklist:**
  - tsdViewer's "Layers" window still adds/removes/drags/renames and selects through the app exactly as before (defaults unchanged).
  - tsdFlow View menu → "Layer Debug" shows the **real LayerTree** over the render scene; the layer dropdown lists the render layers; expand/collapse and the genuine row rendering work.
  - It is read-only: no new/delete/clear buttons act, no drag/drop, no node add/remove, no scene context menus; selecting a node does **not** change the app's global selection.
  - Selecting an object row shows its params greyed/read-only in the pane below; selecting a transform shows "No object selected".

## Out of Scope

- Editing the render scene (read-only by decision).
- Per-viewport grouping (dropped; native layer dropdown instead).
- Multi-select / range-select in read-only mode (single-select only).
- Any change to tsdViewer's LayerTree behavior.
