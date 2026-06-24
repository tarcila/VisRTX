# tsdFlow Phase 4h — Spatial Display & Manipulation Design

**Status:** approved (brainstorm) + meta-reviewed, pending implementation plan
**Date:** 2026-06-24
**Depends on:** Phase 4a (catalog: BoundingBox/DisplayVolume/DisplaySurface), Phase 3 (GraphRenderBridge), Phase 4d (Inspector + ITransferFunctionNode pattern), Phase 4c/4f/4g (GraphViewport, viewportMask, collectDisplayMasks pattern)
**Sibling (next):** Spec B — isosurface compute node (separate spec).

## Goal

Two cohesive spatial features: (1) make the `BoundingBox` a **wireframe** so it no longer
occludes the volume, and (2) make displayed spatial data **transformable** — a render-time
instance transform on each display node, editable numerically in the Inspector and
interactively via an `ImGuizmo` overlay in the viewport.

## Decisions (from brainstorm + meta-review)

| # | Decision |
|---|----------|
| Q1 | `BoundingBox` always emits a **wireframe** (12 edges as ANARI `cylinder` geometry) — no solid/wireframe choice. |
| Q2 | The transform is a **render-time instance transform** (not a data-resampling node). |
| Q3 | Edit surface = **Inspector TRS fields + viewport `ImGuizmo`** (reuse the `BaseViewport` gizmo pattern). |
| Q4 | The transform is the display's **layer-root transform**, applied by the bridge each `update()` **without rebuilding objects**. |
| Q5 (meta-review) | The transform is **typed node state behind an `ITransformableNode` interface — NOT a `ParameterList` param** — so it stays out of `ParameterList::hash()`; editing it never bumps the param hash, so the Evaluator does **not** recompute the node and the bridge does **not** rebuild the layer (this is what makes dragging smooth). Mirrors the `ITransferFunctionNode` pattern. (A `transform` param would re-eval + rebuild the volume every drag frame — the blocker this avoids.) |

No new third-party dependency. One justified `GraphRenderBridge` addition (`setDisplayTransform`).

## Architecture

```
#1 Wireframe bounds                    #2 Transformable spatial data
───────────────────                    ─────────────────────────────
BoundingBox.cpp emits a SurfaceData:   DisplayVolume/DisplaySurface implement ITransformableNode
  geomSubtype = Token("cylinder")        (a mat4 member, default identity) — typed state, NOT a param
  prim.arrays {vertex.position: 24}            │ read by
  prim.scalars {radius: r}              collectDisplayTransforms(Graph&) → [{NodeId, mat4}]   [tested, tsd_graph_nodes]
  (was a solid 12-triangle box)               │ app syncDisplays()
DisplaySurface maps geomSubtype→        bridge.setDisplayTransform(node, mat4) → stored in Display;
  primSubtype; bridge builds it           applied to d.layer->root()->setAsTransform(m) each update()
  unchanged                               (no clearLayerObjects/buildVolume → smooth; picked up by the
                                          existing setIncludedLayers repopulate)
                                               ▲ edited by
                                        Inspector (dynamic_cast<ITransformableNode*> → TRS, decompose/recompose)
                                        + GraphViewport ImGuizmo (selected display masked into this viewport)
```

## Component 1: wireframe `BoundingBox`

Localized change to `BoundingBox::evaluate` (`tsd/src/tsd/graph_nodes/BoundingBox.cpp`). Note the
node outputs a **`SurfaceData`** (field `geomSubtype`), which `DisplaySurface` later maps to a
`Renderable` (`primSubtype`); the bridge then builds it. The node, not the bridge, fully controls
the geometry.

- Compute the 8 corners from the field's `origin`/`spacing`/`dims` (unchanged), then emit the
  **12 box edges** as ANARI `cylinder` geometry on the `SurfaceData`:
  - `s->geomSubtype = Token("cylinder")` (was `Token("triangle")`).
  - `s->prim.arrays = {(Token("vertex.position"), AnyArray(ANARI_FLOAT32_VEC3, 24))}` — 12 edges ×
    2 endpoints, consecutive pairs (VisRTX `cylinder` with no `primitive.index` builds one cylinder
    per consecutive position pair — confirmed `devices/rtx/.../Cylinder.cpp`).
  - `s->prim.scalars = {(Token("radius"), r)}` — global radius.
- **Radius** auto-scales from the box extent: `r = max(kRadiusFrac * length(hi - lo), 1e-4f)` with
  `kRadiusFrac = 0.004f`, so the outline reads thin at any zoom.
- The 12 edges are the 12 corner-pairs (4 bottom, 4 top, 4 verticals). The old 36-vertex triangle
  array is removed.
- Output remains `Kind::Surface` on `portSurface()`; `DisplaySurface` copies `geomSubtype →
  primSubtype` and `prim → prim` as today; the bridge's `buildSurface` does
  `createObject<Geometry>(r.primSubtype)` + `applyParams(prim)` — so the cylinder + its params flow
  through with **no bridge change**.

**Testing:** **extend the existing `tsd/tests/test_nodes_Surface.cpp`** (it already exercises
`BoundingBox → DisplaySurface` asserting `geomSubtype == Token("triangle")` and 36 vertices): change
the assertions to `geomSubtype == Token("cylinder")`, `vertex.position` count `== 24`, and add a
`radius` scalar present-and-`> 0` check (and the downstream `DisplaySurface` `primSubtype ==
Token("cylinder")`). Do **not** add a separate BoundingBox test file (the coverage already exists
there). Device-free.

## Component 2: transformable spatial data

### Typed node state (NOT a param) — `ITransformableNode`

New public header `tsd/src/tsd/graph_nodes/TransformableNode.hpp`:

```cpp
struct ITransformableNode
{
  virtual ~ITransformableNode() = default;
  virtual tsd::core::math::mat4 &transform() = 0; // local instance transform (identity default)
};
```

- `DisplayVolume` and `DisplaySurface` implement it with a `mat4 m_transform{tsd::core::math::IDENTITY_MAT4}`
  member and `mat4 &transform() override { return m_transform; }`. (The concrete node structs are in
  anonymous namespaces; they already implement the polymorphic `Node` interface, so adding a second
  base is fine; UI reaches them via `dynamic_cast<ITransformableNode*>(graph.node(id)->impl.get())`,
  exactly like `ITransferFunctionNode`.)
- **Why typed state, not a `ParameterList` param:** `ParameterList::hash()` mixes every param's
  bytes, so a `transform` *param* edit would bump the hash → the Evaluator recomputes the node →
  `outputVersion++` → `rebuildLayer` does a full `clearLayerObjects` + `buildVolume`/`buildSurface`
  every drag frame (re-copying the field). Keeping the transform off the `ParameterList` keeps it
  off the eval hash → no re-eval, no rebuild → smooth dragging. `evaluate()` does not read it; the
  data path is untouched.

### Helper (UI-free, tested) + bridge

```cpp
// tsd/src/tsd/graph_nodes/DisplayTransform.hpp
struct DisplayTransform { tsd::graph::NodeId node; tsd::core::math::mat4 xfm; };
// Every node implementing ITransformableNode and its transform (identity if it doesn't / for
// non-transformable nodes, it's simply skipped). Graph& non-const (node()->impl is non-const).
std::vector<DisplayTransform> collectDisplayTransforms(tsd::graph::Graph &g);
```
Implementation follows the *iteration structure* of `collectDisplayMasks` (loop `g.nodeIds()`,
inspect each node's `impl`), but **selects via `dynamic_cast<ITransformableNode*>(gn->impl.get())`**
rather than `collectDisplayMasks`'s `typeInfo().name` string compare; for each hit push
`{id, itf->transform()}`.

- `GraphRenderBridge`: add `void setDisplayTransform(tsd::graph::NodeId, const tsd::math::mat4 &);`
  and a `tsd::math::mat4 transform{IDENTITY_MAT4}` field on the per-display `Display` record. In
  `update()` — **unconditionally for every display that has a layer, not inside `rebuildLayer`'s
  post-realize path** (a transform change does not bump the node version, so `rebuildLayer`
  early-returns) — set the display's **layer-root transform**: guard `if (d.layer)` then
  `d.layer->root()->setAsTransform(d.transform)` (TSD's `Layer` root is always a transform node and
  survives `clearLayerObjects`, which only drops level-1 children; each display owns its own layer,
  so the root transform scopes to exactly that display — this is the same `setAsTransform` mechanism
  `BaseViewport`'s gizmo uses; we set the **root node itself**, never an object node). The new
  transform is then applied to the ANARI instances by the **existing** `update()` →
  `setIncludedLayers` → repopulate path (`RenderIndexAllLayers` re-reads layer-node transforms via
  `syncLayerTransforms`), so no extra signal/rebuild is needed. (Confirm the exact `root()` /
  `LayerNodeData::setAsTransform` accessor against `Layer.hpp`/`LayerNodeData.hpp` during
  implementation.)
- The app's `syncDisplays()` calls `collectDisplayTransforms(m_graph)` and
  `bridge.setDisplayTransform(node, xfm)` for each — **after** the existing mask sync (so every
  display's layer exists before its transform is set; combined with the `if (d.layer)` guard this
  avoids any null-layer deref).

### Inspector (numeric TRS)

In `Inspector::buildUI`, add an `ITransformableNode` transform section (via `dynamic_cast` on the
selected node's `impl`). **It must render IN ADDITION to the node's normal parameters, not replace
them** — a display node also has a `viewportMask` param that must still show. The current control
flow is `if (ITransferFunctionNode) {TF editor} else {drawParameters}`; the transform section is
**additive** (render it, then still call `drawParameters` for non-transform params like
`viewportMask`), unlike the TF branch which substitutes. Render a transform section:
- `float t[3], r[3], s[3]; ImGuizmo::DecomposeMatrixToComponents(&m[0].x, t, r, s);`
- three `ImGui::DragFloat3` rows ("Translate" / "Rotate" / "Scale"); a "Reset" button → identity;
- on change: `ImGuizmo::RecomposeMatrixFromComponents(t, r, s, &m[0].x); itf->transform() = m; *m_graphDirty = true;`
  (write directly to the interface; **do NOT call `m_graph->markDirty(...)`** — the sibling
  `ITransferFunctionNode` branch *does* call `markDirty` (that's correct for TF data), but the
  transform is render-routing, not node data, so dirtying it would force the very re-eval/rebuild we
  are avoiding. Set only `*m_graphDirty = true`.)
- Add `#include "ImGuizmo.h"` to `Inspector.cpp` (currently only `imgui.h`; the lib is already linked
  for the UI target).

### Viewport gizmo (`GraphViewport`)

Port the `BaseViewport` `ImGuizmo` pattern into `GraphViewport` (a standalone `Window`, so port not
inherit):

- Extend the `GraphViewport` ctor to also receive `tsd::graph::Graph *graph`, the app's
  `tsd::graph::NodeId *selected`, and `bool *graphDirty` (same pointers the editor/inspector hold;
  the app already owns them).
- Each frame, if `*selected` is a node implementing `ITransformableNode` **and** is a display whose
  `viewportMask` param includes this viewport's bit (`(getOr<int>("viewportMask") >> m_viewportIndex) & 1`):
  - read `mat4 m = itf->transform();`
  - build `view` from the viewport's `Manipulator` (`linalg::lookat_matrix(eye, at, up)` — `m_manip`
    exposes eye/at/up, as `BaseViewport` uses) and `proj` from a **known perspective**: fovy =
    **`π/3`** (the ANARI perspective default — `GraphViewport`'s `anari::Camera` is opaque and never
    has `fovy` set, so it must NOT be read back; this differs from `BaseViewport`, which reads fovy
    from a TSD scene-camera object), aspect = `m_size.x / m_size.y`, near/far estimated as
    `BaseViewport` does but using the **transform's translation (`m[3]`) as the focus point**
    (GraphViewport has no scene-camera/world-transform distance source; the gizmo tolerates loose
    near/far);
  - `ImGuizmo::BeginFrame(); ImGuizmo::SetOrthographic(false); ImGuizmo::SetDrawlist();
    ImGuizmo::SetRect(imageMin.x, imageMin.y, float(m_size.x), float(m_size.y));` (image rect = the
    `ImGui::Image` blit rect, i.e. `GetCursorScreenPos`/`m_size`);
  - `ImGuizmo::Manipulate(&view[0].x, &proj[0].x, m_gizmoOp, m_gizmoMode, &m[0].x);`
  - if `ImGuizmo::IsUsing()`: `itf->transform() = m; *m_graphDirty = true;` (the bridge applies this
    matrix to the display's layer **root**, which has no parent, so `parentWorld` is identity and the
    manipulated matrix **is** the transform directly — no parent-inverse composition like
    `BaseViewport` does);
  - while `ImGuizmo::IsUsing() || ImGuizmo::IsOver()`, **suppress the viewport's camera-drag input**
    (`GraphViewport` claims the mouse via an `InvisibleButton`; gate its navigation handling on these).
- Op/mode state (`m_gizmoOp` TRANSLATE/ROTATE/SCALE, `m_gizmoMode` WORLD/LOCAL) toggled by the same
  keys `BaseViewport` uses.

### Re-render flow

Editing the transform (Inspector or gizmo) writes `itf->transform()` + sets `*m_graphDirty`. The
app's frame hook runs `syncDisplays()` (pushes the new `mat4` via `setDisplayTransform`) then
`bridge.update()` (sets `root()`'s transform; the existing `setIncludedLayers` repopulate re-reads
it). The node is **not** dirtied and **not** re-evaluated — the data path is untouched, so dragging
a volume is smooth.

## Testing

**Headless Catch2 (`tsd/tests`):**
- **Extend `test_nodes_Surface.cpp`** (Component 1): assert the `BoundingBox` `SurfaceData` has
  `geomSubtype == Token("cylinder")`, `vertex.position` count `== 24`, `radius` scalar present and
  `> 0`; and the downstream `DisplaySurface` `Renderable` `primSubtype == Token("cylinder")`.
- `test_nodes_DisplayTransform.cpp` (new): `collectDisplayTransforms` on the demo returns the 2
  display nodes with identity by default; after writing a known `mat4` via the `ITransformableNode`
  interface on one (`dynamic_cast` the node's `impl`), the helper reports that matrix for it and
  identity for the other; non-display/non-transformable nodes are excluded. Device-free.

**Build-verified + manual (GUI):** wireframe bounds (volume visible through it); Inspector TRS fields
move/rotate/scale the selected display live; the `ImGuizmo` overlay appears for the selected display
in viewports it's masked into and dragging it updates the render **smoothly** (no per-frame volume
rebuild); "Reset" restores identity.

## Scope

**In scope (4h):** wireframe `BoundingBox` (cylinder edges); `ITransformableNode` + `transform()` on
the two display nodes; `collectDisplayTransforms` helper + test; the `test_nodes_Surface.cpp`
extension; `GraphRenderBridge::setDisplayTransform` + layer-root transform application in `update()`;
app `syncDisplays` transform sync; Inspector `ITransformableNode` TRS branch; `GraphViewport`
`ImGuizmo` overlay (ctor extension + selection/mask gating + own-projection).

**Out of scope (deferred):** a data-resampling `Transform` processor node; transform on
non-display nodes; gizmo snapping/extra modes beyond `BaseViewport`'s; per-viewport independent
transforms (the transform is per display, shared across the viewports it appears in); transform
**persistence** — like the TF control points, the transform is typed node state outside the
`ParameterList`, so Phase 5 must serialize it explicitly (not free via param serialization).

## File plan (indicative)

| File | New/Mod | Responsibility |
|------|---------|----------------|
| `tsd/src/tsd/graph_nodes/BoundingBox.cpp` | Mod | emit 12 cylinder edges on the `SurfaceData` (`geomSubtype`/`prim`) instead of a solid box |
| `tsd/src/tsd/graph_nodes/TransformableNode.hpp` | New | `ITransformableNode` interface |
| `tsd/src/tsd/graph_nodes/DisplayVolume.cpp`, `DisplaySurface.cpp` | Mod | implement `ITransformableNode` (`mat4 m_transform` + `transform()`) |
| `tsd/src/tsd/graph_nodes/DisplayTransform.hpp/.cpp` | New | `DisplayTransform` + `collectDisplayTransforms` (via `dynamic_cast`) |
| `tsd/src/tsd/graph_nodes/CMakeLists.txt` | Mod | add `DisplayTransform.cpp` |
| `tsd/src/tsd/rendering/bridge/GraphRenderBridge.hpp/.cpp` | Mod | `setDisplayTransform` + `Display.transform` + apply `root()->setAsTransform` in `update()` |
| `tsd/src/tsd/ui/imgui/windows/Inspector.cpp` | Mod | `ITransformableNode` branch → TRS fields (ImGuizmo decompose/recompose); `#include "ImGuizmo.h"` |
| `tsd/src/tsd/ui/imgui/windows/GraphViewport.hpp/.cpp` | Mod | ctor extension; own view/proj; `ImGuizmo` overlay for the selected, masked, transformable display |
| `tsd/apps/interactive/tsdFlow/tsdFlow.cpp` | Mod | pass graph/selected/graphDirty to `GraphViewport`; sync transforms in `syncDisplays` |
| `tsd/tests/test_nodes_Surface.cpp` | Mod | cylinder/24/radius assertions |
| `tsd/tests/test_nodes_DisplayTransform.cpp` | New | `collectDisplayTransforms` unit test |
| `tsd/tests/CMakeLists.txt` | Mod | register the new test |

## Self-review notes

- Each brainstorm decision maps to a component; Q5 (meta-review) changes the transform from a
  `ParameterList` param to `ITransformableNode` typed state — the fix for the per-drag-frame rebuild
  blocker — mirroring the `ITransferFunctionNode` precedent.
- Tested seam: the device-free, silent-failure-prone logic (`collectDisplayTransforms`, the
  `BoundingBox` geometry output) is unit-tested; the bridge layer-root application, Inspector fields,
  and gizmo are build-verified + manual.
- Reuse: gizmo follows the `BaseViewport` `ImGuizmo` pattern (op/mode, `SetRect`/`Manipulate`/
  `IsUsing`, decompose/recompose); the transform helper mirrors `collectDisplayMasks`; the Inspector
  branch and the interface mirror `ITransferFunctionNode`/`TFCurveEditor`.
- Flagged for the implementer (verify against source, adjust minimally, report): the exact
  `Layer::root()` + `LayerNodeData::setAsTransform` accessor the bridge calls; `ANARITypeFor<mat4>`
  is irrelevant now (transform is no longer an `Any`); ANARI `cylinder` params on VisRTX
  (`vertex.position` pairs + `radius`); the fovy/near/far used to build `GraphViewport`'s gizmo
  projection (π/3, not read from the opaque camera); gating `GraphViewport` camera input on
  `ImGuizmo::IsUsing()/IsOver()` against its `InvisibleButton`.
