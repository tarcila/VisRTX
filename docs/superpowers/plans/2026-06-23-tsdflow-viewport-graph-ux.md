# tsdFlow Phase 4g — Viewport & Graph UX Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Three decoupled tsdFlow UX refinements — a `ViewportRail` to reveal/show-hide viewport windows, leaner Inspector viewport-mask toggle-chips, and graph auto-layout (layered placement of programmatic nodes + a "Clean Up Layout" button) preserving click-placement for mouse-added nodes.

**Architecture:** A UI-free `computeLayeredLayout(Graph&)` helper (tested) in `tsd_graph_nodes`; `GraphEditor` applies it in a single in-editor-scope sweep. A new generic `ViewportRail : Window` in `tsd_ui_imgui` toggling other windows' visibility. A focused restyle of the Inspector's existing `viewportMask` branch. No engine/bridge changes.

**Tech Stack:** C++17, `tsd_graph` / `tsd_graph_nodes`, `tsd_ui_imgui` (Window/GraphEditor/Inspector/GraphViewport), vendored imnodes v0.5, ImGui, Catch2, jj.

## Global Constraints

- Version control is **jj**, not git. Commit ONLY a task's files with explicit paths: `jj commit <paths> -m "..."`. **NEVER** a bare `jj commit` — an unrelated `.envrc` in the working copy must stay uncommitted. **Raw `git` is sandboxed and will fail** — never call git.
- Build tree `_out/_cmake` (Ninja Multi-Config, RelWithDebInfo). Do NOT create a `build/` dir.
  - Build tests: `cmake --build _out/_cmake --config RelWithDebInfo --target tsdTests`
  - Build UI lib: `cmake --build _out/_cmake --config RelWithDebInfo --target tsd_ui_imgui`
  - Build app: `cmake --build _out/_cmake --config RelWithDebInfo --target tsdFlow`
  - Run a test: `ctest --test-dir _out/_cmake/tsd/tests -C RelWithDebInfo -R '<name>' --output-on-failure`
- `clang-format -i` ONLY `.cpp`/`.hpp` files — **NEVER** clang-format `CMakeLists.txt` (edit by hand).
- File header on every new file: `// Copyright 2026 NVIDIA Corporation` then `// SPDX-License-Identifier: Apache-2.0`. Headers use `#pragma once`.
- Namespaces: catalog `tsd::graph_nodes`; UI `tsd::ui::imgui`; engine `tsd::graph`.
- No engine/bridge changes. Tasks are independent except Task 2 depends on Task 1.

## Verified API reference (confirmed against source — cite exactly)

```cpp
// tsd/graph/Graph.hpp
using NodeId = uint64_t;
std::vector<NodeId> Graph::nodeIds() const;             // ascending
const std::vector<Connection>& Graph::connections() const;
struct Connection { ConnectionId id; NodeId fromNode; tsd::core::Token fromPort; NodeId toNode; tsd::core::Token toPort; };
GraphNode *Graph::node(NodeId);  const GraphNode *Graph::node(NodeId) const;
struct GraphNode { NodeId id; std::unique_ptr<Node> impl; /* ... */ };
// Node::typeInfo() returns NodeTypeInfo BY VALUE (bind to a local).

// tsd/ui/imgui/windows/Window.h
struct Window {
  Window(Application*, const char*);
  void show(); void hide(); bool *visiblePtr(); const char *name();
  virtual void buildUI() = 0;
 protected: Application *m_app; std::string m_name; bool m_visible;
};

// tsd/ui/imgui/windows/GraphEditor.hpp/.cpp  (Phase 4d/4f)
//   buildUI(): ImNodes::BeginNodeEditor(); for nodeIds drawNode; links; contextMenu(); MiniMap(); EndNodeEditor(); then handleCreation/handleDeletion/selection.
//   contextMenu(): on add → ImNodes::SetNodeScreenSpacePos(nodeImId(id), clickPos); *m_graphDirty=true.
//   handleDeletion(): selection+Delete → m_model->removeNode(id); *m_graphDirty=true.
//   file-local: int nodeImId(NodeId id) { return int(id); }
// tsd/ui/imgui/windows/Inspector.cpp  (Phase 4f) — viewportMask branch is the FIRST in the per-param if/else-if chain.

// imnodes (v0.5) — namespace ImNodes
void SetNodeGridSpacePos(int node_id, const ImVec2& grid_pos);   // panning-independent — use for layout
void SetNodeScreenSpacePos(int node_id, const ImVec2&);          // used by mouse-add
void BeginNodeEditor(); void EndNodeEditor();

// tsd/graph_nodes/DemoGraph.hpp
struct DemoDisplays { tsd::graph::NodeId source; tsd::graph::NodeId volumeDisplay; tsd::graph::NodeId surfaceDisplay; };
DemoDisplays buildVolumeSurfaceDemo(tsd::graph::Graph&, tsd::graph::NodeRegistry&);
// demo add order (→ ascending ids): GenerateNoiseVolume, ScalarRange, TransferFunction, DisplayVolume, BoundingBox, DisplaySurface
// tsd/graph_nodes/DisplayMask.hpp:  constexpr int kMaxViewports = 8;

// tsdFlow.cpp setupWindows (Phase 4f): builds 8 GraphViewport "Viewport 1".."Viewport 8" (i>0 → vp->hide()); getDefaultLayout() docks them to central node 0x00000003.
```

## File structure

| File | New/Mod | Responsibility |
|------|---------|----------------|
| `tsd/src/tsd/graph_nodes/GraphLayout.hpp` | New | `NodePlacement` + `computeLayeredLayout` decl |
| `tsd/src/tsd/graph_nodes/GraphLayout.cpp` | New | layered-layout impl |
| `tsd/src/tsd/graph_nodes/CMakeLists.txt` | Mod | add `GraphLayout.cpp` |
| `tsd/tests/test_nodes_GraphLayout.cpp` | New | `computeLayeredLayout` unit test |
| `tsd/tests/CMakeLists.txt` | Mod | register the test |
| `tsd/src/tsd/ui/imgui/windows/GraphEditor.hpp/.cpp` | Mod | `m_positioned`/`m_relayoutAll`, `applyAutoLayout`, "Clean Up Layout" button |
| `tsd/src/tsd/ui/imgui/windows/Inspector.cpp` | Mod | `viewportMask` checkboxes → toggle-chips |
| `tsd/src/tsd/ui/imgui/windows/ViewportRail.hpp/.cpp` | New | viewport reveal/show-hide rail |
| `tsd/src/tsd/ui/imgui/CMakeLists.txt` | Mod | add `ViewportRail.cpp` |
| `tsd/apps/interactive/tsdFlow/tsdFlow.cpp` | Mod | gather viewports → `ViewportRail`; dock it in `getDefaultLayout()` |

---

## Task 1: `computeLayeredLayout` helper + unit test

**Files:**
- Create: `tsd/src/tsd/graph_nodes/GraphLayout.hpp`, `tsd/src/tsd/graph_nodes/GraphLayout.cpp`
- Modify: `tsd/src/tsd/graph_nodes/CMakeLists.txt`
- Test: `tsd/tests/test_nodes_GraphLayout.cpp`, `tsd/tests/CMakeLists.txt`

**Interfaces:**
- Produces: `struct tsd::graph_nodes::NodePlacement { tsd::graph::NodeId node; int col; int row; };` and `std::vector<NodePlacement> tsd::graph_nodes::computeLayeredLayout(const tsd::graph::Graph &g);` — layered DAG: `col` = longest-path depth from a source; rows `0,1,2,…` per column in `nodeIds()` ascending order.

- [ ] **Step 1: Write the failing test** — create `tsd/tests/test_nodes_GraphLayout.cpp`:

```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/graph/Graph.hpp"
#include "tsd/graph/NodeRegistry.hpp"
#include "tsd/graph_nodes/BuiltinNodes.hpp"
#include "tsd/graph_nodes/DemoGraph.hpp"
#include "tsd/graph_nodes/GraphLayout.hpp"
// std
#include <set>
#include <utility>

using tsd::core::Token;
using namespace tsd::graph;
using tsd::graph_nodes::computeLayeredLayout;
using tsd::graph_nodes::NodePlacement;

namespace {
// col of the (single) node whose typeInfo name matches `typeName`.
int colOfType(Graph &g, const std::vector<NodePlacement> &p, const char *typeName)
{
  for (const auto &np : p) {
    auto *gn = g.node(np.node);
    if (gn && gn->impl && gn->impl->typeInfo().name == Token(typeName))
      return np.col;
  }
  return -1;
}
} // namespace

SCENARIO("computeLayeredLayout lays the demo graph out by topological depth", "[graph-layout]")
{
  NodeRegistry reg;
  tsd::graph_nodes::registerBuiltinNodes(reg);
  Graph g;
  tsd::graph_nodes::buildVolumeSurfaceDemo(g, reg);

  auto placements = computeLayeredLayout(g);

  THEN("every node is placed exactly once") {
    REQUIRE(placements.size() == g.nodeIds().size());
    std::set<NodeId> ids;
    for (const auto &p : placements) ids.insert(p.node);
    REQUIRE(ids.size() == placements.size());
  }

  THEN("columns match topological depth") {
    REQUIRE(colOfType(g, placements, "GenerateNoiseVolume") == 0);
    REQUIRE(colOfType(g, placements, "ScalarRange") == 1);
    REQUIRE(colOfType(g, placements, "BoundingBox") == 1);
    REQUIRE(colOfType(g, placements, "TransferFunction") == 2);
    REQUIRE(colOfType(g, placements, "DisplaySurface") == 2);
    REQUIRE(colOfType(g, placements, "DisplayVolume") == 3);
  }

  THEN("no two nodes share the same (col,row)") {
    std::set<std::pair<int, int>> cells;
    for (const auto &p : placements) cells.insert({p.col, p.row});
    REQUIRE(cells.size() == placements.size());
  }

  THEN("every producer's column is strictly less than its consumer's") {
    auto colOf = [&](NodeId id) {
      for (const auto &p : placements) if (p.node == id) return p.col;
      return -1;
    };
    for (const auto &c : g.connections())
      REQUIRE(colOf(c.fromNode) < colOf(c.toNode));
  }

  THEN("rows within a column are 0..k-1 (determinism via nodeIds order)") {
    // column 1 has exactly ScalarRange and BoundingBox → rows {0,1}.
    std::set<int> col1rows;
    for (const auto &p : placements) if (p.col == 1) col1rows.insert(p.row);
    REQUIRE(col1rows == std::set<int>({0, 1}));
  }
}
```

- [ ] **Step 2: Register the test** — edit `tsd/tests/CMakeLists.txt` by hand: add `test_nodes_GraphLayout.cpp` to the executable source list (near the other `test_nodes_*`), and:

```cmake
add_test(NAME tsd::nodes::GraphLayout COMMAND ${PROJECT_NAME} "[graph-layout]")
```

- [ ] **Step 3: Build, confirm FAIL**

Run: `cmake --build _out/_cmake --config RelWithDebInfo --target tsdTests 2>&1 | tail -20`
Expected: `tsd/graph_nodes/GraphLayout.hpp: No such file or directory`.

- [ ] **Step 4: Create `tsd/src/tsd/graph_nodes/GraphLayout.hpp`:**

```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/graph/Graph.hpp"
// std
#include <vector>

namespace tsd::graph_nodes {

struct NodePlacement
{
  tsd::graph::NodeId node{0};
  int col{0};
  int row{0};
};

// Layered DAG layout: col = longest-path depth from a source (no incoming
// connection), else max(col(producers)) + 1. Rows are 0,1,2,... per column in
// g.nodeIds() ascending order. Pure topology, no pixels. The engine guarantees
// acyclicity, so the memoized depth recursion terminates.
std::vector<NodePlacement> computeLayeredLayout(const tsd::graph::Graph &g);

} // namespace tsd::graph_nodes
```

- [ ] **Step 5: Create `tsd/src/tsd/graph_nodes/GraphLayout.cpp`:**

```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/graph_nodes/GraphLayout.hpp"
// std
#include <algorithm>
#include <functional>
#include <map>

namespace tsd::graph_nodes {

using tsd::graph::NodeId;

std::vector<NodePlacement> computeLayeredLayout(const tsd::graph::Graph &g)
{
  // Build producer adjacency: toNode <- [fromNode...].
  std::map<NodeId, std::vector<NodeId>> producers;
  for (const NodeId id : g.nodeIds())
    producers[id]; // ensure an entry for every node (incl. sources)
  for (const auto &c : g.connections())
    producers[c.toNode].push_back(c.fromNode);

  // Memoized longest-path depth (terminates: graph is acyclic by construction).
  std::map<NodeId, int> depth;
  std::function<int(NodeId)> col = [&](NodeId n) -> int {
    auto it = depth.find(n);
    if (it != depth.end())
      return it->second;
    int d = 0;
    for (const NodeId p : producers[n])
      d = std::max(d, col(p) + 1);
    depth[n] = d;
    return d;
  };

  // Assign rows per column in nodeIds() ascending order.
  std::map<int, int> nextRow;
  std::vector<NodePlacement> out;
  out.reserve(g.nodeIds().size());
  for (const NodeId id : g.nodeIds()) {
    const int c = col(id);
    out.push_back({id, c, nextRow[c]++});
  }
  return out;
}

} // namespace tsd::graph_nodes
```

- [ ] **Step 6: Add to CMake** — edit `tsd/src/tsd/graph_nodes/CMakeLists.txt` by hand, adding `GraphLayout.cpp` to `project_sources(PRIVATE ...)`.

- [ ] **Step 7: Build + run**

Run: `cmake --build _out/_cmake --config RelWithDebInfo --target tsdTests && ctest --test-dir _out/_cmake/tsd/tests -C RelWithDebInfo -R 'tsd::nodes::GraphLayout' --output-on-failure`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
clang-format -i tsd/src/tsd/graph_nodes/GraphLayout.hpp tsd/src/tsd/graph_nodes/GraphLayout.cpp tsd/tests/test_nodes_GraphLayout.cpp
jj commit tsd/src/tsd/graph_nodes/GraphLayout.hpp tsd/src/tsd/graph_nodes/GraphLayout.cpp tsd/src/tsd/graph_nodes/CMakeLists.txt tsd/tests/test_nodes_GraphLayout.cpp tsd/tests/CMakeLists.txt -m "feat(graph_nodes): computeLayeredLayout — layered DAG node placement"
```

---

## Task 2: GraphEditor auto-layout + "Clean Up Layout" button

**Files:**
- Modify: `tsd/src/tsd/ui/imgui/windows/GraphEditor.hpp`, `tsd/src/tsd/ui/imgui/windows/GraphEditor.cpp`

**Interfaces:**
- Consumes: `computeLayeredLayout` + `NodePlacement` (Task 1); existing `nodeImId`, `drawNode`, `contextMenu`, `handleDeletion`, `m_graph`.
- Produces: nothing new for later tasks — un-positioned/programmatic nodes auto-lay-out on first appearance; a "Clean Up Layout" button re-lays-out all.

No automated test (GUI). Deliverable: `tsd_ui_imgui` compiles + links; verified via the app (Task 4 manual checklist).

- [ ] **Step 1: Edit `GraphEditor.hpp`** — add the include, the new members, and the helper decl. Add `#include <set>` to the std includes (it has `<map>`/`<vector>`); add `#include "tsd/graph_nodes/GraphLayout.hpp"` to the tsd includes. Add the method decl `void applyAutoLayout();` next to the other private methods, and these members after `m_linkId`:

```cpp
  std::set<tsd::graph::NodeId> m_positioned; // nodes already given a position
  bool m_relayoutAll{false};                 // "Clean Up Layout" request
```

- [ ] **Step 2: Add `applyAutoLayout()` and spacing constants to `GraphEditor.cpp`.** Near the top of the file (in the anonymous namespace with `nodeImId`/`kConversionColor`), add:

```cpp
constexpr float kColW = 360.f;
constexpr float kRowH = 170.f;
```

Then add the method (anywhere in the `tsd::ui::imgui` namespace, e.g. before `buildUI`):

```cpp
void GraphEditor::applyAutoLayout()
{
  std::vector<NodeId> targets;
  if (m_relayoutAll) {
    targets = m_graph->nodeIds();
    m_relayoutAll = false;
  } else {
    for (const NodeId id : m_graph->nodeIds())
      if (!m_positioned.count(id))
        targets.push_back(id);
    if (targets.empty())
      return; // nothing new — skip the layout work this frame
  }

  const auto placements = tsd::graph_nodes::computeLayeredLayout(*m_graph);
  std::map<NodeId, const tsd::graph_nodes::NodePlacement *> byId;
  for (const auto &p : placements)
    byId[p.node] = &p;

  for (const NodeId id : targets) {
    auto it = byId.find(id);
    if (it == byId.end())
      continue;
    ImNodes::SetNodeGridSpacePos(nodeImId(id),
        ImVec2(it->second->col * kColW, it->second->row * kRowH));
    m_positioned.insert(id);
  }
}
```

- [ ] **Step 3: Wire the button + sweep into `buildUI`.** At the very top of `GraphEditor::buildUI()`, BEFORE `ImNodes::BeginNodeEditor();`, add the button; immediately AFTER `ImNodes::BeginNodeEditor();` and before the `drawNode` loop, call the sweep:

```cpp
void GraphEditor::buildUI()
{
  if (ImGui::Button("Clean Up Layout"))
    m_relayoutAll = true;

  ImNodes::BeginNodeEditor();

  applyAutoLayout(); // positions un-positioned (programmatic) nodes, or all on request

  for (const NodeId id : m_graph->nodeIds())
    drawNode(id);
  // ... rest of buildUI unchanged ...
```

(`#include "imgui.h"` is already present for `ImGui::Button`.)

- [ ] **Step 4: Record mouse-added nodes as positioned.** In `GraphEditor::contextMenu()`, immediately after the existing `ImNodes::SetNodeScreenSpacePos(nodeImId(id), clickPos);` line, add:

```cpp
          m_positioned.insert(id);
```

(So mouse-placed nodes are never moved by the auto-layout sweep.)

- [ ] **Step 5: Forget deleted nodes.** In `GraphEditor::handleDeletion()`, in the node-deletion loop, immediately after `m_model->removeNode(id);` add:

```cpp
      m_positioned.erase(id);
```

- [ ] **Step 6: Build `tsd_ui_imgui`**

Run: `cmake --build _out/_cmake --config RelWithDebInfo --target tsd_ui_imgui 2>&1 | tail -20`
Expected: compiles + links, warning-free.

- [ ] **Step 7: Commit**

```bash
clang-format -i tsd/src/tsd/ui/imgui/windows/GraphEditor.hpp tsd/src/tsd/ui/imgui/windows/GraphEditor.cpp
jj commit tsd/src/tsd/ui/imgui/windows/GraphEditor.hpp tsd/src/tsd/ui/imgui/windows/GraphEditor.cpp -m "feat(ui): GraphEditor auto-layout for programmatic nodes + Clean Up Layout button"
```

---

## Task 3: Lean Inspector mask chips

**Files:**
- Modify: `tsd/src/tsd/ui/imgui/windows/Inspector.cpp`

**Interfaces:**
- Consumes: existing `viewportMask` branch, `tsd::graph_nodes::kMaxViewports`. No new produced interface.

No automated test (GUI). Deliverable: `tsd_ui_imgui` compiles + links.

- [ ] **Step 1: Replace the checkbox with a toggle-chip.** In `tsd/src/tsd/ui/imgui/windows/Inspector.cpp`, inside the `if (name == tsd::core::Token("viewportMask"))` branch, replace the `ImGui::Checkbox(...)` block with an `ImGui::Selectable` chip. The current inner body is:

```cpp
        bool on = (mask >> i) & 1;
        char lbl[8];
        std::snprintf(lbl, sizeof(lbl), "%d", i + 1);
        if (ImGui::Checkbox(lbl, &on)) {
          if (on)
            mask |= (1 << i);
          else
            mask &= ~(1 << i);
          changed = true;
        }
```

Replace it with:

```cpp
        const bool on = (mask >> i) & 1;
        char lbl[8];
        std::snprintf(lbl, sizeof(lbl), "%d", i + 1);
        if (ImGui::Selectable(
                lbl, on, ImGuiSelectableFlags_None, ImVec2(24.f, 0.f))) {
          mask ^= (1 << i); // Selectable returns true on click → flip the bit
          changed = true;
        }
```

Leave the surrounding loop (`ImGui::PushID(i)`, the `if (i % 4 != 0) ImGui::SameLine();`, `ImGui::PopID()`), the `ImGui::TextUnformatted("Viewports")` label, and the post-loop `if (changed) { params.set(name, mask); *m_graphDirty = true; }` exactly as they are. The chips render as small highlighted squares (filled when the bit is set), 4 per row.

- [ ] **Step 2: Build `tsd_ui_imgui`**

Run: `cmake --build _out/_cmake --config RelWithDebInfo --target tsd_ui_imgui 2>&1 | tail -20`
Expected: compiles + links, warning-free.

- [ ] **Step 3: Commit**

```bash
clang-format -i tsd/src/tsd/ui/imgui/windows/Inspector.cpp
jj commit tsd/src/tsd/ui/imgui/windows/Inspector.cpp -m "feat(ui): Inspector viewport mask as compact toggle-chips"
```

---

## Task 4: `ViewportRail` window + app wiring + dock layout

**Files:**
- Create: `tsd/src/tsd/ui/imgui/windows/ViewportRail.hpp`, `tsd/src/tsd/ui/imgui/windows/ViewportRail.cpp`
- Modify: `tsd/src/tsd/ui/imgui/CMakeLists.txt`, `tsd/apps/interactive/tsdFlow/tsdFlow.cpp`

**Interfaces:**
- Consumes: `Window` base (`visiblePtr()`/`name()`), the app's 8 `GraphViewport*`.
- Produces: `tsd::ui::imgui::ViewportRail(Application*, std::vector<Window*> viewports, const char *name = "Viewports")`.

No automated test (GUI). Deliverable: `tsdFlow` builds + the full suite stays green; manual checklist recorded.

- [ ] **Step 1: Create `tsd/src/tsd/ui/imgui/windows/ViewportRail.hpp`:**

```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/ui/imgui/windows/Window.h"
// std
#include <vector>

namespace tsd::ui::imgui {

// A slim strip of toggle cells, one per supplied window: a cell is highlighted
// when its window is visible; clicking it flips visibility. Borrows the window
// pointers (owned by the app's WindowArray).
struct ViewportRail : public Window
{
  ViewportRail(Application *app,
      std::vector<Window *> viewports,
      const char *name = "Viewports");
  void buildUI() override;

 private:
  std::vector<Window *> m_viewports;
};

} // namespace tsd::ui::imgui
```

- [ ] **Step 2: Create `tsd/src/tsd/ui/imgui/windows/ViewportRail.cpp`:**

```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/ui/imgui/windows/ViewportRail.hpp"
// imgui
#include "imgui.h"
// std
#include <cstdio>

namespace tsd::ui::imgui {

ViewportRail::ViewportRail(
    Application *app, std::vector<Window *> viewports, const char *name)
    : Window(app, name), m_viewports(std::move(viewports))
{}

void ViewportRail::buildUI()
{
  for (size_t i = 0; i < m_viewports.size(); ++i) {
    ImGui::PushID(int(i));
    bool *vis = m_viewports[i]->visiblePtr();
    char lbl[8];
    std::snprintf(lbl, sizeof(lbl), "%zu", i + 1);
    // Stacked vertically (no SameLine) → a slim vertical rail.
    if (ImGui::Selectable(lbl, *vis, ImGuiSelectableFlags_None, ImVec2(28.f, 28.f)))
      *vis = !*vis;
    ImGui::PopID();
  }
}

} // namespace tsd::ui::imgui
```

- [ ] **Step 3: Add to CMake** — edit `tsd/src/tsd/ui/imgui/CMakeLists.txt` by hand, adding `windows/ViewportRail.cpp` to the sources (after `windows/GraphEditor.cpp`).

- [ ] **Step 4: Wire the rail into `tsdFlow.cpp`.** Add the include near the other window includes:

```cpp
#include "tsd/ui/imgui/windows/ViewportRail.hpp"
```

In `setupWindows()`, change the viewport-pool loop to collect the pointers, then construct the rail and add it. The current loop is:

```cpp
    for (int i = 0; i < tsd::graph_nodes::kMaxViewports; ++i) {
      char nm[16];
      std::snprintf(nm, sizeof(nm), "Viewport %d", i + 1);
      auto *vp = new ui::GraphViewport(this, m_bridge.get(), i, m_device, nm);
      if (i > 0)
        vp->hide();
      windows.emplace_back(vp);
    }
    windows.emplace_back(new ui::Log(this));
```

Replace it with:

```cpp
    std::vector<ui::Window *> viewportPtrs;
    for (int i = 0; i < tsd::graph_nodes::kMaxViewports; ++i) {
      char nm[16];
      std::snprintf(nm, sizeof(nm), "Viewport %d", i + 1);
      auto *vp = new ui::GraphViewport(this, m_bridge.get(), i, m_device, nm);
      if (i > 0)
        vp->hide();
      viewportPtrs.push_back(vp);
      windows.emplace_back(vp);
    }
    windows.emplace_back(new ui::ViewportRail(this, viewportPtrs, "Viewports"));
    windows.emplace_back(new ui::Log(this));
```

(`<vector>` is already included in `tsdFlow.cpp`.)

- [ ] **Step 5: Dock the rail in `getDefaultLayout()`.** Add a `[Window][Viewports]` entry and a thin left-edge dock node. Replace the current `getDefaultLayout()` return string's `[Window][Graph Editor]` entry region and the `[Docking][Data]` block so the rail gets its own narrow node. Concretely: add this `[Window]` stanza (anywhere among the others):

```
[Window][Viewports]
Pos=0,26
Size=44,790
Collapsed=0
DockId=0x00000009,0
```

and replace the `[Docking][Data]` block with this tree (adds node `0x00000009` for the rail and a wrapper `0x0000000A` for the rest of the top row):

```
[Docking][Data]
DockSpace        ID=0x80F5B4C5 Window=0x079D3A04 Pos=0,26 Size=1920,1054 Split=Y
  DockNode       ID=0x00000001 Parent=0x80F5B4C5 SizeRef=1920,790 Split=X
    DockNode     ID=0x00000009 Parent=0x00000001 SizeRef=44,790
    DockNode     ID=0x0000000A Parent=0x00000001 SizeRef=1872,790 Split=X
      DockNode   ID=0x00000007 Parent=0x0000000A SizeRef=420,790 Split=Y
        DockNode ID=0x00000005 Parent=0x00000007 SizeRef=420,395
        DockNode ID=0x00000006 Parent=0x00000007 SizeRef=420,395
      DockNode   ID=0x00000003 Parent=0x0000000A SizeRef=1408,790 CentralNode=1
  DockNode       ID=0x00000002 Parent=0x80F5B4C5 SizeRef=1920,262
```

(Keep the existing `[Window][Graph Editor]`/`[Inspector]`/`[Viewport 1..8]`/`[Log]` stanzas unchanged — their DockIds `0x5/0x6/0x3/0x2` still exist in this tree. Docking INIs are finicky: if the rail renders mis-docked, it still functions; verify in the manual test and nudge SizeRefs if needed.)

- [ ] **Step 6: Build the app**

Run: `cmake --build _out/_cmake --config RelWithDebInfo --target tsdFlow 2>&1 | tail -20`
Expected: compiles + links.

- [ ] **Step 7: Full suite gate**

Run:
```bash
cmake --build _out/_cmake --config RelWithDebInfo --target tsdTests --parallel
ctest --test-dir _out/_cmake/tsd/tests -C RelWithDebInfo --output-on-failure
```
Expected: all green — the prior suite plus the new `tsd::nodes::GraphLayout`. Report the summary line.

- [ ] **Step 8: Confirm `.envrc` uncommitted**

Run: `jj status`
Expected: working copy shows `A .envrc` and nothing from this task after the commit. NEVER commit `.envrc`.

- [ ] **Step 9: Commit**

```bash
clang-format -i tsd/src/tsd/ui/imgui/windows/ViewportRail.hpp tsd/src/tsd/ui/imgui/windows/ViewportRail.cpp tsd/apps/interactive/tsdFlow/tsdFlow.cpp
jj commit tsd/src/tsd/ui/imgui/windows/ViewportRail.hpp tsd/src/tsd/ui/imgui/windows/ViewportRail.cpp tsd/src/tsd/ui/imgui/CMakeLists.txt tsd/apps/interactive/tsdFlow/tsdFlow.cpp -m "feat(app): ViewportRail to reveal/show-hide viewports"
```

- [ ] **Step 10: Record the manual test checklist** (GUI not CI-tested) in the task report:
  - `tsdFlow` launches; the demo graph comes up **tidy** in the Graph Editor (nodes in left→right columns, not stacked at the origin).
  - A "Viewports" rail strip shows numbered cells; cell 1 highlighted (Viewport 1 visible). Click cell 2 → Viewport 2 appears (tabs into the center); click again → hides.
  - Select a display node → Inspector "Viewports" shows compact toggle-chips (filled = member); clicking a chip re-routes the display and re-renders.
  - Right-click the canvas → add a node: it appears **at the cursor** (not auto-moved).
  - Drag nodes around, then click **"Clean Up Layout"** → the whole graph snaps back to the tidy layered arrangement.

---

## Phase 4g completion checklist

- [ ] `computeLayeredLayout` + `tsd::nodes::GraphLayout` test pass (Task 1)
- [ ] GraphEditor auto-places programmatic nodes + "Clean Up Layout" button; mouse-added nodes keep click pos (Task 2)
- [ ] Inspector viewport mask renders as compact toggle-chips (Task 3)
- [ ] `ViewportRail` reveals/show-hides viewports; docked in the app (Task 4)
- [ ] full suite green; `.envrc` uncommitted; manual checklist recorded

## Out of scope (per spec)

Rail content-awareness (dimming empty viewports); on-node chips; viewport rename;
spline/force-directed layouts; animated transitions; node-position persistence
(Phase 5); node grouping.

## Self-review notes

- **Spec coverage:** `ViewportRail` + app wiring + dock (Task 4 = Component 1);
  Inspector lean chips (Task 3 = Component 2); `computeLayeredLayout` + its test
  (Task 1) + GraphEditor application with the single in-scope sweep, programmatic-
  only auto-place, Clean-Up button, `m_positioned` (Task 2 = Component 3, Q3–Q6).
- **Type consistency:** `computeLayeredLayout(const Graph&) → vector<NodePlacement{NodeId,int col,int row}>`,
  `kColW`/`kRowH`, `nodeImId(id)` cast, `m_positioned`/`m_relayoutAll`, and the
  `ViewportRail(Application*, vector<Window*>, const char*)` ctor are used
  identically across tasks and match the verified API reference.
- **Tested seam:** only the layout-position computation is silent-failure-prone and
  device-free → unit-tested (Task 1); the three GUI pieces are build-verified +
  manually checked (Task 4 checklist) — consistent with prior phases.
- **Sweep placement (from meta-review):** the auto-place sweep runs inside the
  editor scope, right after `BeginNodeEditor()` and before the `drawNode` loop, so
  positions apply same-frame and every swept node is submitted (no `EndNodeEditor`
  assert); the Clean-Up button (before `BeginNodeEditor`) only sets a flag.
- **Flagged for the implementer (adjust minimally, report):** the docking-INI tree
  is finicky — verify the rail docks as a thin left strip and nudge SizeRefs if
  ImGui mis-renders (functional even if mis-docked); confirm `ImGui::Selectable`'s
  4-arg overload + `ImVec2` sizing render as compact chips/cells.
```
