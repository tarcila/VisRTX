# tsdFlow Phase 4g — Viewport & Graph UX Design

**Status:** approved (brainstorm), pending implementation plan
**Date:** 2026-06-23
**Depends on:** Phase 4d (GraphEditor/Inspector), Phase 4f (multi-viewport + `viewportMask`)
**Sketches:** `docs/superpowers/sketches/2026-06-23-viewport-ux/` (P1/P2/P3 + README)

## Goal

Three independent UX refinements to tsdFlow: (1) a discoverable **viewport rail** to
reveal/show/hide the viewport windows, (2) a **leaner** per-display viewport-mask
control in the Inspector (toggle-chips instead of an 8-checkbox row), and (3)
**graph auto-layout** — layered placement of programmatically-created nodes plus a
one-click "Clean Up Layout" button — while preserving click-placement for
mouse-added nodes.

## Decisions (from brainstorm)

| # | Decision |
|---|----------|
| Q1 | Unify the viewport controls; chosen direction = **P3 hybrid**: a slim **viewport rail** owns reveal/show-hide of windows; per-display masking stays in the Inspector. |
| Q2 | Mask chips live **in the Inspector** (selected display only), restyled from the 8-checkbox row to compact highlight **toggle-chips** — NOT on the canvas node (nodes keep title+pins). |
| Q3 | Auto-layout uses a **layered DAG** placement (sources left, displays right, by topological depth). |
| Q4 | Auto-place applies **only to programmatically/un-positioned nodes** (e.g. the startup demo). **Mouse-added nodes keep click-position placement** (unchanged). |
| Q5 | A one-click **"Clean Up Layout"** button re-lays-out the whole graph on demand. |
| Q6 | The layout-position computation is a UI-free, unit-tested helper; the GUI applies it. |

The three features are decoupled (different files, any order). No engine/bridge
changes; one small framework addition (a new generic `Window`).

## Architecture

```
(1) Viewport rail            (2) Lean mask chips           (3) Graph auto-layout
─────────────────            ───────────────────           ─────────────────────
new ViewportRail Window      Inspector.cpp edit            computeLayeredLayout(Graph&)  [tested, tsd_graph_nodes]
(tsd_ui_imgui): vertical      viewportMask branch:               │ applied by
strip of N toggle cells       8 checkboxes → compact        GraphEditor:
bound to each viewport        highlight toggle-chips          · un-positioned nodes → auto-place (grid)
window's visiblePtr()         (selected display only)         · mouse-added nodes → keep click pos
        │                                                     · "Clean Up Layout" button → re-layout all
   app wires the pool
```

## Component 1: `ViewportRail` window (`tsd_ui_imgui`)

A new generic window that reveals/toggles viewport windows.

```cpp
// tsd/src/tsd/ui/imgui/windows/ViewportRail.hpp
struct ViewportRail : public Window
{
  ViewportRail(Application *app,
      std::vector<Window *> viewports,   // borrowed; owned by the app's WindowArray
      const char *name = "Viewports");
  void buildUI() override;
 private:
  std::vector<Window *> m_viewports;
};
```

- `buildUI` renders a slim vertical column of small square toggle cells, one per
  entry in `m_viewports`. A cell is highlighted when that window's `*visiblePtr()`
  is true; clicking it flips visibility. Cell label = its 1-based index.
- Implementation uses only the existing `Window` API (`visiblePtr()` /
  `show()` / `hide()` / `name()`); no bridge or `Application` change. The
  framework's View menu (`Application::uiMainMenuBar_View`) remains as a secondary
  path; the rail is the always-visible, discoverable one.
- **App wiring (explicit).** In `setupWindows`, the 8 `GraphViewport` are created as
  locals and `emplace_back`'d into the `WindowArray`; gather those raw `Window*`
  into a `std::vector<Window*>` *as they are created*, then construct
  `ViewportRail(this, thatVector, "Viewports")` and **also `emplace_back` the rail
  into the same `WindowArray`** (otherwise it is never rendered — `Application`
  iterates the array). The rail borrows the same raw pointers; ownership stays with
  the `WindowArray`'s `unique_ptr`s.
- **Docking.** `tsdFlow`'s `getDefaultLayout()` must gain a `[Window][Viewports]`
  entry docked into a thin strip node (e.g. a narrow left-edge split), or the rail
  floats at the default size on first launch. Because it takes `Window*`, the rail
  is reusable for any window set.
- **Out of scope:** dimming cells whose viewport has no display masked into it
  (would couple the rail to the bridge) — the rail is visibility-only.

**Testing:** GUI — build-verified + manual (toggling a cell shows/hides the
matching viewport).

## Component 2: lean Inspector mask chips (`Inspector.cpp`)

A focused restyle of the Phase 4f `viewportMask` branch — same data path, leaner
widget.

- Replace the row of `kMaxViewports` `ImGui::Checkbox` calls with a compact row of
  highlight **toggle-chips**: a small fixed-size `ImGui::Selectable(label, isActive,
  ImGuiSelectableFlags_None, ImVec2(chipW, chipH))` per viewport (the chips are not
  in a popup, so no `DontClosePopups`/`NoAutoClosePopups` flag is needed; or use a
  `Button` wrapped in an active-color `PushStyleColor`), laid out with `SameLine`,
  4 per row.
- Identical logic: `mask = p.value.get<int>()`; clicking chip *i* toggles bit *i*;
  on change `params.set(name, mask)` + `*m_graphDirty = true`. Still keyed on the
  param name `viewportMask`, still `kMaxViewports` chips, unique `PushID(i)` per
  chip. No behavior change — only the rendering is leaner.

**Testing:** GUI — build-verified + manual (chips fill/clear; editing re-renders).

## Component 3: graph auto-layout + Clean-Up button

### Tested helper (UI-free, `tsd_graph_nodes`)

```cpp
// tsd/src/tsd/graph_nodes/GraphLayout.hpp
struct NodePlacement { tsd::graph::NodeId node; int col; int row; };

// Layered DAG layout: col = longest-path depth from a source (a node with no
// incoming connection); else max(col(producers)) + 1. Within a column, rows are
// assigned 0,1,2,... in g.nodeIds() ascending order. Pure topology — no pixels.
// The engine guarantees acyclicity, so the memoized depth recursion terminates.
std::vector<NodePlacement> computeLayeredLayout(const tsd::graph::Graph &g);
```

- Producers of a node come from `g.connections()` (`c.toNode == n` → `c.fromNode`
  is a producer). `col(n)` via memoized DFS over producers. Takes **`const Graph&`**
  — `nodeIds()`/`connections()`/`node()` are all const-qualified and the helper
  reads no parameters (unlike `collectDisplayMasks`, which needs non-const for
  `Node::parameters()`).
- For the demo graph: `GenerateNoiseVolume`=col 0; `ScalarRange`,`BoundingBox`=col 1;
  `TransferFunction`,`DisplaySurface`=col 2; `DisplayVolume`=col 3 (fan-out depth:
  fed by `TransferFunction` (col 2) and `GenerateNoiseVolume` (col 0), so
  `max(col(producers)) + 1 = max(2,0) + 1 = 3`).

### `GraphEditor` application (pixel spacing lives here)

All grid positioning happens in **one place** — a sweep run *inside* the editor
scope, immediately after `ImNodes::BeginNodeEditor()` and **before** the `drawNode`
loop — so a position takes effect the same frame the node is submitted (imnodes'
`BeginNode` reads the node's grid Origin) and every positioned node is guaranteed a
valid submission index that frame (no `EndNodeEditor` assert).

- Add members `std::set<tsd::graph::NodeId> m_positioned;` and `bool m_relayoutAll{false};` (and `#include <set>` in `GraphEditor.hpp`).
- The context-menu add (`contextMenu`, which already calls
  `ImNodes::SetNodeScreenSpacePos(nodeImId(id), clickPos)`) inserts that id into
  `m_positioned` — mouse-added nodes are thus "positioned" and never auto-moved.
- The **sweep** (new private `applyAutoLayout()`, called right after
  `BeginNodeEditor()`):
  - if `m_relayoutAll`: target = **all** `g.nodeIds()`; clear the flag;
  - else: target = `g.nodeIds()` not in `m_positioned` (programmatic/startup-demo
    nodes); if empty, return immediately (no layout work most frames);
  - run `computeLayeredLayout(*m_graph)`, and for each target id call
    `ImNodes::SetNodeGridSpacePos(nodeImId(id), ImVec2(col*COL_W, row*ROW_H))`
    (use the same `nodeImId(id)` int cast the draw/link code uses — **not** the raw
    `uint64_t`); insert all targets into `m_positioned`. Fixed spacing `COL_W =
    360.f`, `ROW_H = 170.f` (no dependence on measured node sizes).
- A **"Clean Up Layout"** button at the top of the Graph Editor window (regular
  ImGui, emitted *before* `ImNodes::BeginNodeEditor()`) simply sets
  `m_relayoutAll = true`; the next sweep re-lays-out all nodes. This is the only
  path that moves hand-placed nodes.
- `handleDeletion` erases removed ids from `m_positioned` (so the set doesn't
  accumulate dead ids; harmless if missed since `SetNodeGridSpacePos` on a dead id
  no-ops, but kept tidy).

**Testing:** `computeLayeredLayout` unit test (headless, device-free); the
`GraphEditor` application + button are build-verified + manual.

## Testing summary

**Headless Catch2 (`tsd/tests`):**
- `test_nodes_GraphLayout.cpp` — `computeLayeredLayout` on the demo graph:
  - every node placed exactly once;
  - columns match topological depth: `GenerateNoiseVolume` col 0;
    `ScalarRange`/`BoundingBox` col 1; `TransferFunction`/`DisplaySurface` col 2;
    `DisplayVolume` col 3;
  - no two nodes share the same `(col,row)`;
  - within a column, rows are `0,1,2,…` assigned in `nodeIds()` ascending order
    (row determinism — the only thing pinning rows);
  - a producer's col is strictly less than each consumer's col (layering invariant).
  (Device-free — pure graph topology.)

**Build-verified + manual (GUI):** `ViewportRail` (toggle cells show/hide
viewports), lean Inspector chips (fill/clear, re-render), the auto-layout on the
demo (tidy at startup, not stacked), the "Clean Up Layout" button (reshapes after
manual dragging), and that mouse-added nodes stay at the click location until
Clean Up is pressed.

## Scope

**In scope (4g):** `ViewportRail` window + app wiring + dock placement; Inspector
mask chip restyle; `computeLayeredLayout` helper + its unit test; `GraphEditor`
auto-place of un-positioned nodes + "Clean Up Layout" button + `m_positioned`
tracking.

**Out of scope (deferred):** rail content-awareness (dimming empty viewports);
on-node chips; viewport rename; spline/force-directed layouts (layered only);
animated layout transitions; persistence of node positions (Phase 5); collapsing
multiple selected nodes / grouping.

## File plan (indicative)

| File | New/Mod | Responsibility |
|------|---------|----------------|
| `tsd/src/tsd/ui/imgui/windows/ViewportRail.hpp` | New | rail window decl |
| `tsd/src/tsd/ui/imgui/windows/ViewportRail.cpp` | New | rail toggle-cell rendering |
| `tsd/src/tsd/ui/imgui/CMakeLists.txt` | Mod | add `ViewportRail.cpp` |
| `tsd/src/tsd/ui/imgui/windows/Inspector.cpp` | Mod | `viewportMask` branch → toggle-chips |
| `tsd/src/tsd/graph_nodes/GraphLayout.hpp` | New | `NodePlacement` + `computeLayeredLayout` decl |
| `tsd/src/tsd/graph_nodes/GraphLayout.cpp` | New | layered-layout impl |
| `tsd/src/tsd/graph_nodes/CMakeLists.txt` | Mod | add `GraphLayout.cpp` |
| `tsd/src/tsd/ui/imgui/windows/GraphEditor.hpp/.cpp` | Mod | `m_positioned`, auto-place un-positioned, "Clean Up Layout" button |
| `tsd/apps/interactive/tsdFlow/tsdFlow.cpp` | Mod | gather the 8 `GraphViewport*`, construct `ViewportRail` (also `emplace_back` it into the `WindowArray`); add a `[Window][Viewports]` strip to `getDefaultLayout()` |
| `tsd/tests/test_nodes_GraphLayout.cpp` | New | `computeLayeredLayout` unit test |
| `tsd/tests/CMakeLists.txt` | Mod | register the test |

## Self-review notes

- Each brainstorm decision maps to a component: P3 rail (Component 1), Inspector
  lean chips (Component 2), layered auto-layout + programmatic-only auto-place +
  Clean-Up button + tested helper (Component 3, Q3–Q6).
- Tested seam: the only silent-failure-prone logic is the layout-position
  computation, which is unit-tested headlessly; the rest is GUI (build-verified +
  manual) — consistent with prior phases.
- `GraphEditor` depends on `tsd_graph_nodes` already (Phase 4d), so including
  `GraphLayout.hpp` adds no new link edge; `ViewportRail` is pure `tsd_ui_imgui`.
- Flagged for the implementer (verify against source, adjust minimally, report):
  the exact `Window` ctor/`visiblePtr()` usage for the rail; whether the
  "Clean Up Layout" button should sit in its own row or share the existing menu
  bar; the imnodes grid-vs-screen space call (`SetNodeGridSpacePos` is panning-
  independent and the right choice for layout).
