# tsdFlow Phase 4f — Multi-Viewport & Display Masking Design

**Status:** approved (brainstorm), pending meta-review → implementation plan
**Date:** 2026-06-23
**Depends on:** Phase 3 (render bridge + viewport masks), Phase 4c (app shell), Phase 4d (interactive editing)
**Relation to roadmap:** slots ahead of 4b (CUDA residency), 4e (undo/redo), Phase 5 (Lua + persistence).

## Goal

Replace the two hardcoded purpose-named viewports ("Volume" / "Surface") with a
pool of generic viewports and make each display object carry an editable
**viewport bitmask** that selects which viewport(s) it renders into. This
realizes the foundational design ("display into one or more viewports, selected
per display object by a viewport mask; N viewports each render the union of
display nodes masked to it") at the interactive-UI level — the bridge already
implements the mechanism.

## Decisions (from brainstorm)

| # | Decision |
|---|----------|
| Q1 | Fixed pool of `kMaxViewports = 8` generic viewports, shown/hidden via the existing View menu. No `Application` framework change, no runtime change to the bridge's viewport count. |
| Q2 | The viewport mask is **graph-owned state**: a `viewportMask` parameter on the display node (single source of truth, read by the bridge, serialized for free in Phase 5). |
| Q2b | Stored as a plain `ParameterList` value (`int`/`ANARI_INT32`), not a typed interface — no new interface, flows through existing dirty/eval/persistence paths; the Inspector special-cases its *rendering* as checkboxes. |
| Q3 | Viewports have fixed names "Viewport 1".."Viewport 8" (no rename). 8 max, 1 visible at startup. |
| Q4 | Minimal UI: per-display masking via a "Viewports" checkbox row in the **Inspector**; show/hide of viewport windows via the **existing View menu**. No dedicated Viewports panel. |
| Q5 | Extract a UI-free `collectDisplayMasks(Graph&)` helper (tested); the app's `syncDisplays()` is a thin consumer. Headless unit + VisRTX render tests; GUI build-verified + manual. |

## Architecture

```
Display node (DisplayVolume/DisplaySurface)
  └─ ParameterList gains "viewportMask" (ANARI_INT32, default 0b01)   ← graph-owned source of truth
        ▲ edited by                         │ read by
        │                                    ▼
   Inspector (kMaxViewports checkboxes,  collectDisplayMasks(Graph&) → [{NodeId, mask}]   [tested, tsd_graph_nodes]
   special-cased on param name)              │ called by
        │ markDirty + graphDirty             ▼
        └──────────────────────────►  app syncDisplays() → bridge.setDisplay(id, mask, mask!=0) / removeDisplay
                                             │
                                       GraphRenderBridge (built once, numViewports = kMaxViewports = 8)
                                             │ world(i)
                              Viewport pool: 8 GraphViewport windows ("Viewport 1".."Viewport 8")
                                             1 visible at startup; rest toggled via View menu
```

- **No engine/bridge code changes.** The bridge already supports N viewports +
  per-display masks (`setDisplay(NodeId, uint64_t viewportMask, bool enabled)`,
  `world(int)`, `removeDisplay`). This phase only adds a node param, a helper, app
  wiring, and Inspector UI.
- **No `Application` framework change.** "Add/remove a viewport" = show/hide one
  of a fixed pool via the framework's existing per-window visibility (`hide()` /
  `show()` / the View-menu checkboxes).
- **Mask is graph state** — one source of truth (the display node's param),
  edited by the Inspector, consumed by the bridge through the helper, serialized
  by Phase 5's graph persistence with no special handling.

## Component: mask param + `collectDisplayMasks` (tested core)

### The node param

`DisplayVolume` and `DisplaySurface` seed a `viewportMask` parameter at
construction so it is present for both the Inspector and the helper:

```cpp
// in each display node's (new) constructor:
DisplayVolume()  { params.set(tsd::core::Token("viewportMask"), kDefaultViewportMask); }
DisplaySurface() { params.set(tsd::core::Token("viewportMask"), kDefaultViewportMask); }
```

- Stored as **`int` / `ANARI_INT32`** — matches the existing `getOr<int>(...)`
  precedent in the catalog (`Any` handles it; the Inspector already recognizes
  `ANARI_INT32`). Used purely as an 8-bit mask; the bridge's `setDisplay` widens
  it to `uint64_t`.
- **Not consumed by `evaluate()`** — the node never reads the mask, so the
  renderable's *content* is unchanged by a toggle. **Routing** is driven entirely
  by the app: a toggle sets the param + `m_graphDirty`; `syncDisplays()` re-reads
  the mask and calls `bridge.setDisplay(node, newMask, …)`; `bridge.update()` →
  `layersForViewport()` re-derives each viewport's layer set from the stored
  `d.mask & bit`. No `markDirty` is needed for routing (and the existing Inspector
  convention sets only `*m_graphDirty`, not `markDirty`).
- **Cost note (precise):** because the mask lives in the `ParameterList`, a toggle
  changes `ParameterList::hash()`, so the next `bridge.update()` pull recomputes
  the display node (the Evaluator recomputes on `paramHash` mismatch regardless of
  the dirty flag) and bumps its `outputVersion`, which makes the bridge's
  `rebuildLayer` (`out->version != d.lastVersion`) **clear and rebuild that
  display's ANARI layer objects**. This is correct but a full per-toggle layer
  rebuild, not just a re-route. It is cheap at interactive rates (one display, on
  demand). Avoiding it would require storing the mask as node state *outside* the
  `ParameterList` (a typed interface like `ITransferFunctionNode`) so it doesn't
  feed `hash()` — explicitly **deferred** (Q2b chose the simpler param for generic
  inspection + free persistence).

### Shared constant + helper

New header/impl `tsd/src/tsd/graph_nodes/DisplayMask.hpp` / `.cpp` (UI-free, in
`tsd_graph_nodes`):

```cpp
#pragma once
#include "tsd/graph/Graph.hpp"
#include <cstdint>
#include <vector>

namespace tsd::graph_nodes {

constexpr int kMaxViewports = 8;
constexpr int kDefaultViewportMask = 0b01;   // bit 0 → "Viewport 1"

struct DisplayMask { tsd::graph::NodeId node; uint64_t mask; };

// Every display node (DisplayVolume/DisplaySurface) and its viewport mask, read
// from the node's "viewportMask" param (kDefaultViewportMask if absent).
std::vector<DisplayMask> collectDisplayMasks(tsd::graph::Graph &g);

} // namespace tsd::graph_nodes
```

Implementation: iterate `g.nodeIds()`; for each, fetch `gn = g.node(id)` and skip
if `!gn || !gn->impl`. **Bind `typeInfo()` to a local** — it returns a temporary
(`NodeTypeInfo` by value; `Graph.hpp` warns never to hold a reference into it):
`const auto info = gn->impl->typeInfo();`. If `info.name` is `Token("DisplayVolume")`
or `Token("DisplaySurface")`, read
`gn->impl->parameters().getOr<int>(Token("viewportMask"), kDefaultViewportMask)`
and push `{id, uint64_t(mask)}`. Takes `Graph&` non-const because both
`Graph::node()` (mutating overload) and `Node::parameters()` are non-const; the
function is logically read-only.

`kMaxViewports` is the single source for the bridge's `numViewports`, the
window-pool size, and the Inspector's checkbox count — all three include this
header.

### Unit tests (`test_nodes_DisplayMask.cpp`, headless)

- demo graph → `collectDisplayMasks` returns exactly the 2 display nodes, each
  mask `0b01` (default).
- set one display's `viewportMask` param to `0b11` → helper reports `0b11` for
  it, `0b01` for the other.
- non-display nodes (source / ScalarRange / TransferFunction) are excluded.

## Component: viewport pool, app `syncDisplays`, Inspector checkboxes

### Viewport pool (`tsdFlow`)

Bridge built once with `numViewports = kMaxViewports`. Construct 8 generic
viewports, show only the first:

```cpp
for (int i = 0; i < tsd::graph_nodes::kMaxViewports; ++i) {
  char nm[16]; std::snprintf(nm, sizeof nm, "Viewport %d", i + 1);
  auto *vp = new ui::GraphViewport(this, m_bridge.get(), i, m_device, nm);
  if (i > 0) vp->hide();              // Window::hide() sets m_visible=false
  windows.emplace_back(vp);
}
```

**Show/hide UI is the framework's existing View menu** — `Application` already
renders a main menu bar with a "View" menu (`Application::uiMainMenuBar_View`,
`Application.cpp:694`) that auto-lists every window with a `visiblePtr()`
checkbox. The 8 viewports appear there automatically; **no menu code is added**
(tsdFlow's own `uiFrameStart` "Regenerate" bar coexists — ImGui merges multiple
`BeginMainMenuBar` calls in a frame).

**`getDefaultLayout()` must be rewritten (in scope).** The current INI hardcodes
`[Window][Volume]`/`[Window][Surface]` in an X-split central node with fixed dock
ids. Replace the `[Docking][Data]` block and add **8 `[Window][Viewport N]`
stanzas that all share one central dock node id** (a tab group where the 4d split
was); GraphEditor + Inspector stay in the left column, Log along the bottom. The
`[Window][...]` names must match the `GraphViewport` `name` args exactly
("Viewport 1"…"Viewport 8") or docking silently fails. A viewport that is hidden
at startup but **present in the INI** tabs into the center when later shown; one
absent from the INI would float. Window visibility (`m_visible`) is independent of
the docking INI — but note `Window::saveSettings/loadSettings` persist `visible`,
so "1 visible at startup" holds on first run / default layout; a previously-saved
layout can restore other viewports visible.

### `syncDisplays()` rewrite

Becomes a thin consumer of the helper (replaces the hardcoded `0b01` logic and
manual prune from Phase 4d):

```cpp
void syncDisplays() {
  const auto masks = tsd::graph_nodes::collectDisplayMasks(m_graph);
  std::set<tsd::graph::NodeId> current;
  for (const auto &dm : masks) {
    m_bridge->setDisplay(dm.node, dm.mask, /*enabled=*/dm.mask != 0);  // mask 0 = nowhere
    current.insert(dm.node);
  }
  for (auto id : m_knownDisplays)
    if (!current.count(id)) m_bridge->removeDisplay(id);
  m_knownDisplays = std::move(current);
}
```

Startup: build demo → `syncDisplays()` (both displays default `0b01` → both in
Viewport 1, the union) → `bridge.update()`. The Phase 4d explicit
`setDisplay(volume, 0b01)` / `setDisplay(surface, 0b10)` calls and the old
`0b01`-default prune loop are removed.

### Inspector checkboxes

In `drawParameters`, special-case the param named `viewportMask` (include
`DisplayMask.hpp` for `kMaxViewports`): render a "Viewports" row of
`kMaxViewports` checkboxes (bit *i* ↔ "Viewport *i*+1") instead of the generic
int field; on toggle, set/clear the bit, `params.set(name, mask)` and
`*m_graphDirty = true` — **no `markDirty`** (matches the existing generic-param
convention; routing is handled by `syncDisplays`+`bridge.update`). Each checkbox
gets a unique `PushID` so the numeric labels don't collide. Other params fall
through to the existing generic dispatch.

This is the one place the generic Inspector knows a specific param name — a
documented, contained coupling (the alternative, a separate mask widget/panel,
was ruled out in Q4).

## Data flow

```
user toggles a viewport checkbox (Inspector, display node selected)
   → params.set("viewportMask", newMask) + m_graphDirty=true        (no markDirty)
   → app frame hook: if m_graphDirty → syncDisplays() (collectDisplayMasks → setDisplay/removeDisplay) → bridge.update()
       · setDisplay updates the bridge's stored d.mask; update() re-derives layersForViewport from d.mask&bit (the re-route)
       · the param-hash change makes update()'s pull recompute the display node → layer rebuild (the cost note above)
   → each visible GraphViewport renders bridge.world(i); the display now appears in exactly the masked viewports
```

Same dirty-coalesced re-render loop as Phase 4d; `syncDisplays()` is just
mask-aware now.

## Testing

The bridge's mask→viewport *rendering* is already proven by the existing
`test_bridge_Mask` (a masked display lands in the right viewports' layer sets;
disabled → empty). 4f's genuinely new logic is `collectDisplayMasks` (param→mask)
and its wiring into the bridge — so the new tests target **that** seam, not a
redundant pixel re-render.

**Headless Catch2 (`tsd/tests`):**
- `test_nodes_DisplayMask.cpp` — the three `collectDisplayMasks` unit cases above
  (no device needed; pure graph/param logic).
- `test_nodes_MultiViewport.cpp` — proves the param→helper→bridge **routing**
  end-to-end via the `syncDisplays` path, following the existing
  `test_bridge_Mask` precedent (assert **layer membership**, not pixels): build the
  demo, set a display's `viewportMask` param to `0b11`, run
  `collectDisplayMasks` → `bridge.setDisplay(...)` for each → `bridge.update()`,
  then assert `bridge.layersForViewport(0)` and `layersForViewport(1)` both
  include that display's layer; set it to `0b01` and assert it is in viewport 0's
  set but **not** viewport 1's; set `0` and assert it is in neither. The bridge
  ctor needs a device, so match `test_bridge_Mask`'s setup and register with
  `set_tests_properties(... PROPERTIES TIMEOUT 300)`. (No camera/renderer/frame or
  pixel readback — that heavier path is already covered by `test_bridge_RenderVolume`.)

**Build-verified + manual (GUI):** Inspector checkbox row, the 8-viewport pool,
View-menu show/hide, docking layout. Manual checklist: select a display → toggle
its viewport checkboxes → it appears/disappears in the matching viewport; show
"Viewport 2" from the View menu; mask one display into two viewports → it renders
in both; mask to none → it vanishes.

## Scope

**In scope (4f):** `viewportMask` param on `DisplayVolume`/`DisplaySurface`
(seeded `0b01`); `collectDisplayMasks` + `kMaxViewports`/`kDefaultViewportMask`
(`DisplayMask.hpp`); app changes (bridge `numViewports = 8`, 8-viewport pool with
1 visible, `syncDisplays` via the helper, updated default layout); Inspector
`viewportMask` checkbox special-case; the two headless tests. No engine/bridge
code changes.

**Out of scope (deferred):** runtime change of the viewport *count* / true
dynamic bridge resize + `Application` window add-remove (fixed pool of 8);
viewport rename (fixed "Viewport N"); a dedicated Viewports management panel;
per-viewport renderer settings or camera sync; mask persistence (the mask is now
graph state, so Phase 5's graph serialization picks it up — not implemented
here); the routing-vs-data eval optimization (toggling a mask re-evaluates the
display node — harmless).

## File plan (indicative)

| File | New/Mod | Responsibility |
|------|---------|----------------|
| `tsd/src/tsd/graph_nodes/DisplayMask.hpp` | New | `kMaxViewports`, `kDefaultViewportMask`, `DisplayMask`, `collectDisplayMasks` decl |
| `tsd/src/tsd/graph_nodes/DisplayMask.cpp` | New | `collectDisplayMasks` impl |
| `tsd/src/tsd/graph_nodes/DisplayVolume.cpp` | Mod | seed `viewportMask` default in a ctor |
| `tsd/src/tsd/graph_nodes/DisplaySurface.cpp` | Mod | seed `viewportMask` default in a ctor |
| `tsd/src/tsd/graph_nodes/CMakeLists.txt` | Mod | add `DisplayMask.cpp` |
| `tsd/src/tsd/ui/imgui/windows/Inspector.cpp` | Mod | special-case `viewportMask` → checkbox row |
| `tsd/apps/interactive/tsdFlow/tsdFlow.cpp` | Mod | bridge `numViewports=8`, 8-viewport pool, `syncDisplays` via helper, default layout |
| `tsd/tests/test_nodes_DisplayMask.cpp` | New | helper unit tests |
| `tsd/tests/test_nodes_MultiViewport.cpp` | New | VisRTX multi-viewport render test |
| `tsd/tests/CMakeLists.txt` | Mod | register both tests (+ TIMEOUT 300 on the render one) |

## Self-review notes

- Every brainstorm decision (Q1–Q5) maps to a component: fixed pool + visibility
  (app pool + View menu), graph-owned mask (node param + `DisplayMask.hpp`),
  fixed names (string literals), Inspector-only UI (checkbox special-case),
  tested helper + render test.
- The tested seam is the UI-free `collectDisplayMasks` + the render test of the
  mask→viewport routing; the GUI (checkboxes, pool, layout) is build-verified +
  manually checked — consistent with prior phases.
- Contained coupling flagged: the Inspector special-cases one param name
  (`viewportMask`); the `int`-as-mask choice (vs unsigned) is deliberate for
  `Any`/Inspector compatibility.
- No engine/bridge/framework code changes — the capability already exists; this
  phase is node-param + helper + app + Inspector only.
