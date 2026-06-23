# tsdFlow Phase 4f — Multi-Viewport & Display Masking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace tsdFlow's two hardcoded "Volume"/"Surface" viewports with a pool of 8 generic viewports, and make each display object carry a user-editable `viewportMask` (graph-owned) that selects which viewport(s) it renders into.

**Architecture:** The mask becomes an `ANARI_INT32` param on the `DisplayVolume`/`DisplaySurface` nodes (graph-owned source of truth). A UI-free `collectDisplayMasks(Graph&)` helper (in `tsd_graph_nodes`) reads those masks; the app's `syncDisplays()` pushes them to the existing `GraphRenderBridge` (which already supports N viewports + per-display masks — no engine/bridge change). The Inspector special-cases the `viewportMask` param as a viewport-checkbox row; the app builds a fixed pool of 8 `GraphViewport` windows (1 visible), shown/hidden via the framework's existing View menu.

**Tech Stack:** C++17, `tsd_graph` / `tsd_graph_nodes`, `tsd_rendering` `GraphRenderBridge`, `tsd_ui_imgui` (Application/Window/Inspector/GraphViewport), ANARI + VisRTX, ImGui, Catch2, jj.

## Global Constraints

- Version control is **jj**, not git. Commit ONLY a task's files with explicit paths: `jj commit <paths> -m "..."`. **NEVER** a bare `jj commit` — an unrelated `.envrc` in the working copy must stay uncommitted. **Raw `git` is sandboxed and will fail** — never call git.
- Build tree is `_out/_cmake` (Ninja Multi-Config, RelWithDebInfo). Do NOT create a new `build/` dir.
  - Build tests: `cmake --build _out/_cmake --config RelWithDebInfo --target tsdTests`
  - Build the app: `cmake --build _out/_cmake --config RelWithDebInfo --target tsdFlow`
  - Build the UI lib: `cmake --build _out/_cmake --config RelWithDebInfo --target tsd_ui_imgui`
  - Run a test: `ctest --test-dir _out/_cmake/tsd/tests -C RelWithDebInfo -R '<name>' --output-on-failure`
- `clang-format -i` ONLY `.cpp`/`.hpp` files — **NEVER** clang-format `CMakeLists.txt` (edit it by hand).
- File header on every new file: `// Copyright 2026 NVIDIA Corporation` then `// SPDX-License-Identifier: Apache-2.0`. Headers use `#pragma once`.
- Namespaces: catalog `tsd::graph_nodes`; UI `tsd::ui::imgui`; engine `tsd::graph`; core `tsd::core`.
- VisRTX is the render device. Any test that constructs a `GraphRenderBridge` (needs a device) gets `set_tests_properties(<name> PROPERTIES TIMEOUT 300)`.
- **No engine/bridge/framework code changes.** The bridge already supports N viewports + per-display masks; this phase is node-param + helper + app + Inspector only.

## Verified API reference (confirmed against source — cite exactly)

```cpp
// tsd/graph/Graph.hpp
using NodeId = uint64_t;
std::vector<NodeId> Graph::nodeIds() const;
GraphNode *Graph::node(NodeId);                 // non-const overload exists
struct GraphNode { NodeId id; std::unique_ptr<Node> impl; /* ... */ };
void Graph::markDirty(NodeId);
// tsd/graph/Node.hpp
class Node { virtual NodeTypeInfo typeInfo() const; virtual ParameterList &parameters(); /*...*/ };
// NOTE: typeInfo() returns NodeTypeInfo BY VALUE — bind to a local; never hold a ref into it.
// tsd/graph/Parameter.hpp
struct Parameter { tsd::core::Token name; tsd::core::Any value; };
struct ParameterList {
  template<class T> void set(tsd::core::Token, T);          // set(Token, int) → ANARI_INT32
  template<class T> T getOr(tsd::core::Token, const T&) const;
  const std::vector<Parameter>& items() const;
  uint64_t hash() const;                                    // includes every param
};
// tsd/core/Any.hpp:  anari::DataType Any::type();  template<class T> T Any::get();   Any(int) → ANARI_INT32

// tsd/graph_nodes/DemoGraph.hpp
struct DemoDisplays { tsd::graph::NodeId source; tsd::graph::NodeId volumeDisplay; tsd::graph::NodeId surfaceDisplay; };
DemoDisplays buildVolumeSurfaceDemo(tsd::graph::Graph&, tsd::graph::NodeRegistry&);
// tsd/graph_nodes/BuiltinNodes.hpp:  void registerBuiltinNodes(tsd::graph::NodeRegistry&);
// display node typeInfo names (anonymous-namespace structs, public `ParameterList params;` member, no user ctor today):
//   DisplayVolume.cpp:  i.name = Token("DisplayVolume")   (category "sink")
//   DisplaySurface.cpp: i.name = Token("DisplaySurface")

// tsd/rendering/bridge/GraphRenderBridge.hpp
GraphRenderBridge(tsd::graph::Graph&, tsd::graph::Evaluator&, tsd::core::Token lib, anari::Device, int numViewports); // numViewports validated 1..64
void setDisplay(tsd::graph::NodeId, uint64_t viewportMask, bool enabled);
void removeDisplay(tsd::graph::NodeId);
void update();
anari::World world(int viewport) const;
std::vector<const tsd::scene::Layer*> layersForViewport(int i) const;  // membership; gated on enabled && realized && (mask&bit)

// tsd/ui/imgui/windows/Window.h
struct Window { Window(Application*, const char*); void hide(); void show(); bool *visiblePtr(); /* m_visible gates renderUI */ };
// Application base renders a main menu bar incl. uiMainMenuBar_View() — a "View" menu auto-listing every window's visiblePtr() checkbox.
// tsd/ui/imgui/windows/GraphViewport.hpp
GraphViewport(Application*, tsd::rendering::GraphRenderBridge*, int viewportIndex, anari::Device, const char *name);
```

## File structure

| File | New/Mod | Responsibility |
|------|---------|----------------|
| `tsd/src/tsd/graph_nodes/DisplayMask.hpp` | New | `kMaxViewports`, `kDefaultViewportMask`, `DisplayMask`, `collectDisplayMasks` decl |
| `tsd/src/tsd/graph_nodes/DisplayMask.cpp` | New | `collectDisplayMasks` impl |
| `tsd/src/tsd/graph_nodes/DisplayVolume.cpp` | Mod | seed `viewportMask` default in a ctor |
| `tsd/src/tsd/graph_nodes/DisplaySurface.cpp` | Mod | seed `viewportMask` default in a ctor |
| `tsd/src/tsd/graph_nodes/CMakeLists.txt` | Mod | add `DisplayMask.cpp` |
| `tsd/tests/test_nodes_DisplayMask.cpp` | New | `collectDisplayMasks` unit tests (device-free) |
| `tsd/tests/test_nodes_MultiViewport.cpp` | New | param→helper→bridge routing membership test (VisRTX) |
| `tsd/tests/CMakeLists.txt` | Mod | register both tests (+ TIMEOUT 300 on the routing one) |
| `tsd/src/tsd/ui/imgui/windows/Inspector.cpp` | Mod | special-case `viewportMask` → checkbox row |
| `tsd/apps/interactive/tsdFlow/tsdFlow.cpp` | Mod | bridge `numViewports=8`, 8-viewport pool, `syncDisplays` via helper, default layout |

---

## Task 1: `viewportMask` param + `collectDisplayMasks` helper (+ both tests)

**Files:**
- Create: `tsd/src/tsd/graph_nodes/DisplayMask.hpp`, `tsd/src/tsd/graph_nodes/DisplayMask.cpp`
- Modify: `tsd/src/tsd/graph_nodes/DisplayVolume.cpp`, `tsd/src/tsd/graph_nodes/DisplaySurface.cpp`, `tsd/src/tsd/graph_nodes/CMakeLists.txt`
- Test: `tsd/tests/test_nodes_DisplayMask.cpp`, `tsd/tests/test_nodes_MultiViewport.cpp`, `tsd/tests/CMakeLists.txt`

**Interfaces:**
- Produces:
  - `tsd::graph_nodes::kMaxViewports` (`constexpr int` = 8), `kDefaultViewportMask` (`constexpr int` = `0b01`).
  - `struct tsd::graph_nodes::DisplayMask { tsd::graph::NodeId node; uint64_t mask; };`
  - `std::vector<DisplayMask> tsd::graph_nodes::collectDisplayMasks(tsd::graph::Graph &g);` — every `DisplayVolume`/`DisplaySurface` node and its `viewportMask` (default `kDefaultViewportMask` if absent).
  - `DisplayVolume`/`DisplaySurface` nodes now seed a `viewportMask` (`ANARI_INT32`) param at construction.

- [ ] **Step 1: Write the failing unit test** — create `tsd/tests/test_nodes_DisplayMask.cpp`:

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
#include "tsd/graph_nodes/DisplayMask.hpp"
// std
#include <algorithm>

using tsd::core::Token;
using namespace tsd::graph;
using tsd::graph_nodes::collectDisplayMasks;
using tsd::graph_nodes::kDefaultViewportMask;

namespace {
const tsd::graph_nodes::DisplayMask *find(
    const std::vector<tsd::graph_nodes::DisplayMask> &v, NodeId id)
{
  for (const auto &dm : v)
    if (dm.node == id)
      return &dm;
  return nullptr;
}
} // namespace

SCENARIO("collectDisplayMasks reports display nodes and their masks", "[display-mask]")
{
  NodeRegistry reg;
  tsd::graph_nodes::registerBuiltinNodes(reg);
  Graph g;
  auto d = tsd::graph_nodes::buildVolumeSurfaceDemo(g, reg);

  WHEN("masks are read from a fresh demo graph") {
    auto masks = collectDisplayMasks(g);
    THEN("exactly the two display nodes appear, each at the default mask") {
      REQUIRE(masks.size() == 2);
      REQUIRE(find(masks, d.volumeDisplay) != nullptr);
      REQUIRE(find(masks, d.surfaceDisplay) != nullptr);
      REQUIRE(find(masks, d.volumeDisplay)->mask == uint64_t(kDefaultViewportMask));
      REQUIRE(find(masks, d.surfaceDisplay)->mask == uint64_t(kDefaultViewportMask));
    }
    THEN("non-display nodes are excluded") {
      REQUIRE(find(masks, d.source) == nullptr); // GenerateNoiseVolume
    }
  }

  WHEN("one display's viewportMask param is changed to 0b11") {
    g.node(d.volumeDisplay)->impl->parameters().set(Token("viewportMask"), 0b11);
    auto masks = collectDisplayMasks(g);
    THEN("the helper reports the new mask for it, default for the other") {
      REQUIRE(find(masks, d.volumeDisplay)->mask == uint64_t(0b11));
      REQUIRE(find(masks, d.surfaceDisplay)->mask == uint64_t(kDefaultViewportMask));
    }
  }
}
```

- [ ] **Step 2: Register the unit test in CMake** — edit `tsd/tests/CMakeLists.txt` by hand: add `test_nodes_DisplayMask.cpp` to the executable source list (near the other `test_nodes_*`), and add this line near the other `add_test(NAME tsd::nodes::*)` registrations:

```cmake
add_test(NAME tsd::nodes::DisplayMask COMMAND ${PROJECT_NAME} "[display-mask]")
```

- [ ] **Step 3: Build, confirm FAIL**

Run: `cmake --build _out/_cmake --config RelWithDebInfo --target tsdTests 2>&1 | tail -20`
Expected: `tsd/graph_nodes/DisplayMask.hpp: No such file or directory`.

- [ ] **Step 4: Create `tsd/src/tsd/graph_nodes/DisplayMask.hpp`:**

```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/graph/Graph.hpp"
// std
#include <cstdint>
#include <vector>

namespace tsd::graph_nodes {

constexpr int kMaxViewports = 8;
constexpr int kDefaultViewportMask = 0b01;   // bit 0 → "Viewport 1"

struct DisplayMask
{
  tsd::graph::NodeId node{0};
  uint64_t mask{0};
};

// Every display node (DisplayVolume/DisplaySurface) and its viewport mask, read
// from the node's "viewportMask" param (kDefaultViewportMask if absent).
// Takes Graph& non-const: Graph::node() and Node::parameters() are non-const,
// though this is logically read-only.
std::vector<DisplayMask> collectDisplayMasks(tsd::graph::Graph &g);

} // namespace tsd::graph_nodes
```

- [ ] **Step 5: Create `tsd/src/tsd/graph_nodes/DisplayMask.cpp`:**

```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/graph_nodes/DisplayMask.hpp"
#include "tsd/graph/NodeRegistry.hpp" // for Node/typeInfo via Graph.hpp chain

namespace tsd::graph_nodes {

using tsd::core::Token;
using tsd::graph::NodeId;

std::vector<DisplayMask> collectDisplayMasks(tsd::graph::Graph &g)
{
  std::vector<DisplayMask> out;
  for (const NodeId id : g.nodeIds()) {
    auto *gn = g.node(id);
    if (!gn || !gn->impl)
      continue;
    const auto info = gn->impl->typeInfo(); // bind temporary to a local
    if (info.name != Token("DisplayVolume") && info.name != Token("DisplaySurface"))
      continue;
    const int mask =
        gn->impl->parameters().getOr<int>(Token("viewportMask"), kDefaultViewportMask);
    out.push_back({id, uint64_t(mask)});
  }
  return out;
}

} // namespace tsd::graph_nodes
```

- [ ] **Step 6: Seed the param in both display nodes.** In `tsd/src/tsd/graph_nodes/DisplayVolume.cpp`, find the `struct DisplayVolume : Node {` with its `ParameterList params;` member and add a constructor that seeds the default mask. Also add the include for the constant. Concretely, add near the top includes:

```cpp
#include "tsd/graph_nodes/DisplayMask.hpp"
```

and inside the struct, immediately after the `ParameterList params;` line, add:

```cpp
  DisplayVolume() { params.set(tsd::core::Token("viewportMask"), kDefaultViewportMask); }
```

Do the identical change in `tsd/src/tsd/graph_nodes/DisplaySurface.cpp` (struct `DisplaySurface`, ctor `DisplaySurface() { params.set(tsd::core::Token("viewportMask"), kDefaultViewportMask); }`, same include). `kDefaultViewportMask` is in namespace `tsd::graph_nodes`, the same namespace these structs live in, so it resolves unqualified.

- [ ] **Step 7: Add `DisplayMask.cpp` to CMake** — edit `tsd/src/tsd/graph_nodes/CMakeLists.txt` by hand, adding `DisplayMask.cpp` to `project_sources(PRIVATE ...)` (after `DemoGraph.cpp`).

- [ ] **Step 8: Build + run the unit test**

Run: `cmake --build _out/_cmake --config RelWithDebInfo --target tsdTests && ctest --test-dir _out/_cmake/tsd/tests -C RelWithDebInfo -R 'tsd::nodes::DisplayMask' --output-on-failure`
Expected: PASS (2 scenarios). If a display node fails to seed the param (size mismatch or default-only masks), confirm the ctor compiled in and `reg.create` runs it.

- [ ] **Step 9: Write the routing membership test** — create `tsd/tests/test_nodes_MultiViewport.cpp` (follows the `test_bridge_Mask` precedent: build a bridge on VisRTX, set masks, `update()`, assert `layersForViewport(i)` membership — no camera/renderer/pixels):

```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/graph/Evaluator.hpp"
#include "tsd/graph/NodeRegistry.hpp"
#include "tsd/graph_nodes/BuiltinNodes.hpp"
#include "tsd/graph_nodes/DemoGraph.hpp"
#include "tsd/graph_nodes/DisplayMask.hpp"
#include "tsd/rendering/bridge/GraphRenderBridge.hpp"
// anari
#include <anari/anari_cpp.hpp>
// std
#include <memory>

using tsd::core::Token;
using namespace tsd::graph;
using tsd::rendering::GraphRenderBridge;

namespace {
anari::Device makeVisRTX()
{
  auto lib = anari::loadLibrary("visrtx", nullptr, nullptr);
  return lib ? anari::newDevice(lib, "default") : nullptr;
}

// Mirror the app's syncDisplays() routing: read masks from the graph, push to bridge.
void sync(GraphRenderBridge &bridge, Graph &g)
{
  for (const auto &dm : tsd::graph_nodes::collectDisplayMasks(g))
    bridge.setDisplay(dm.node, dm.mask, /*enabled=*/dm.mask != 0);
  bridge.update();
}
} // namespace

SCENARIO("display viewportMask routes its layer into the masked viewports", "[multi-viewport]")
{
  anari::Device dev = makeVisRTX();
  REQUIRE(dev != nullptr);

  NodeRegistry reg;
  tsd::graph_nodes::registerBuiltinNodes(reg);
  Graph g;
  auto d = tsd::graph_nodes::buildVolumeSurfaceDemo(g, reg);
  Evaluator e(g);

  GraphRenderBridge bridge(g, e, Token("visrtx"), dev, /*numViewports=*/3);

  WHEN("the volume display is masked into viewports 0 and 1 (surface stays 0b01)") {
    g.node(d.volumeDisplay)->impl->parameters().set(Token("viewportMask"), 0b11);
    sync(bridge, g);
    THEN("vp0 has both displays, vp1 has the volume, vp2 is empty") {
      REQUIRE(bridge.layersForViewport(0).size() == 2); // volume + surface
      REQUIRE(bridge.layersForViewport(1).size() == 1); // volume only
      REQUIRE(bridge.layersForViewport(2).empty());
    }
  }

  WHEN("the volume display is masked to no viewports") {
    g.node(d.volumeDisplay)->impl->parameters().set(Token("viewportMask"), 0);
    sync(bridge, g);
    THEN("vp0 has only the surface; the volume appears nowhere") {
      REQUIRE(bridge.layersForViewport(0).size() == 1); // surface only
      REQUIRE(bridge.layersForViewport(1).empty());
      REQUIRE(bridge.layersForViewport(2).empty());
    }
  }

  anari::release(dev, dev);
}
```

- [ ] **Step 10: Register the routing test (with TIMEOUT)** — edit `tsd/tests/CMakeLists.txt` by hand: add `test_nodes_MultiViewport.cpp` to the source list, then:

```cmake
add_test(NAME tsd::nodes::MultiViewport COMMAND ${PROJECT_NAME} "[multi-viewport]")
set_tests_properties(tsd::nodes::MultiViewport PROPERTIES TIMEOUT 300)
```

- [ ] **Step 11: Build + run the routing test**

Run: `cmake --build _out/_cmake --config RelWithDebInfo --target tsdTests && ctest --test-dir _out/_cmake/tsd/tests -C RelWithDebInfo -R 'tsd::nodes::MultiViewport' --output-on-failure`
Expected: PASS (first run does an OptiX warmup — may take tens of seconds). If a viewport size is off, recall `layersForViewport` counts `enabled && realized && (mask & bit)` displays; both demo displays must realize (the full demo graph is wired so they evaluate).

- [ ] **Step 12: Commit**

```bash
clang-format -i tsd/src/tsd/graph_nodes/DisplayMask.hpp tsd/src/tsd/graph_nodes/DisplayMask.cpp tsd/src/tsd/graph_nodes/DisplayVolume.cpp tsd/src/tsd/graph_nodes/DisplaySurface.cpp tsd/tests/test_nodes_DisplayMask.cpp tsd/tests/test_nodes_MultiViewport.cpp
jj commit tsd/src/tsd/graph_nodes/DisplayMask.hpp tsd/src/tsd/graph_nodes/DisplayMask.cpp tsd/src/tsd/graph_nodes/DisplayVolume.cpp tsd/src/tsd/graph_nodes/DisplaySurface.cpp tsd/src/tsd/graph_nodes/CMakeLists.txt tsd/tests/test_nodes_DisplayMask.cpp tsd/tests/test_nodes_MultiViewport.cpp tsd/tests/CMakeLists.txt -m "feat(graph_nodes): viewportMask on display nodes + collectDisplayMasks helper"
```

---

## Task 2: Inspector `viewportMask` checkbox row

**Files:**
- Modify: `tsd/src/tsd/ui/imgui/windows/Inspector.cpp`

**Interfaces:**
- Consumes: `tsd::graph_nodes::kMaxViewports` (Task 1); `ParameterList::items()`/`set`, `Any::get<int>()`.
- Produces: nothing new — when the selected node has a `viewportMask` param, the Inspector renders a "Viewports" checkbox row instead of the generic int field.

No automated test (GUI). Deliverable: `tsd_ui_imgui` compiles + links; behavior verified via the app in Task 3.

- [ ] **Step 1: Add the include.** In `tsd/src/tsd/ui/imgui/windows/Inspector.cpp`, add near the other tsd includes:

```cpp
#include "tsd/graph_nodes/DisplayMask.hpp"
```

- [ ] **Step 2: Special-case `viewportMask` as the first branch** in `drawParameters`'s per-param `if/else-if` chain. The chain currently begins with `if (t == ANARI_BOOL) {`. Insert this branch BEFORE it (so it takes precedence; the existing trailing `ImGui::PopID();` still runs for it):

```cpp
    if (name == tsd::core::Token("viewportMask")) {
      int mask = p.value.get<int>();
      ImGui::TextUnformatted("Viewports");
      bool changed = false;
      for (int i = 0; i < tsd::graph_nodes::kMaxViewports; ++i) {
        ImGui::PushID(i);
        if (i % 4 != 0)
          ImGui::SameLine();
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
        ImGui::PopID();
      }
      if (changed) {
        params.set(name, mask);
        *m_graphDirty = true;
      }
    } else if (t == ANARI_BOOL) {
```

(That is: change the existing `if (t == ANARI_BOOL) {` line into the `} else if (t == ANARI_BOOL) {` tail of the new branch. Everything else in the loop — the outer `ImGui::PushID(name.c_str())`, the other `else if` branches, the final `ImGui::PopID()` — is unchanged. `<cstdio>` for `snprintf` is already included for the existing `ANARI_STRING` branch.)

- [ ] **Step 3: Build `tsd_ui_imgui`**

Run: `cmake --build _out/_cmake --config RelWithDebInfo --target tsd_ui_imgui 2>&1 | tail -20`
Expected: compiles + links, warning-free. (If `<cstdio>` turns out not to be transitively included, add `#include <cstdio>`.)

- [ ] **Step 4: Commit**

```bash
clang-format -i tsd/src/tsd/ui/imgui/windows/Inspector.cpp
jj commit tsd/src/tsd/ui/imgui/windows/Inspector.cpp -m "feat(ui): Inspector renders viewportMask as a viewport checkbox row"
```

---

## Task 3: tsdFlow app — 8-viewport pool, mask-driven `syncDisplays`, layout

**Files:**
- Modify: `tsd/apps/interactive/tsdFlow/tsdFlow.cpp`

**Interfaces:**
- Consumes: `tsd::graph_nodes::collectDisplayMasks` + `kMaxViewports` (Task 1); `Inspector` checkbox UI (Task 2); existing `GraphViewport`, `GraphRenderBridge`, `Window::hide()`.

No automated test (GUI). Deliverable: `tsdFlow` builds + the full suite stays green; manual checklist recorded.

- [ ] **Step 1: Add the helper include.** In `tsd/apps/interactive/tsdFlow/tsdFlow.cpp`, add to the tsd includes:

```cpp
#include "tsd/graph_nodes/DisplayMask.hpp"
```

- [ ] **Step 2: Build the bridge with `kMaxViewports` and drop the hardcoded masks.** Replace the bridge construction + explicit `setDisplay`/seed block. The current code is:

```cpp
    m_bridge = std::make_unique<tsd::rendering::GraphRenderBridge>(
        m_graph, *m_eval, Token(lib.c_str()), m_device, /*numViewports=*/2);
    m_bridge->setDisplay(m_displays.volumeDisplay, 0b01, true);
    m_bridge->setDisplay(m_displays.surfaceDisplay, 0b10, true);
    m_bridge->update();

    // Seed known displays so syncDisplays() doesn't re-register them.
    m_knownDisplays.insert(m_displays.volumeDisplay);
    m_knownDisplays.insert(m_displays.surfaceDisplay);
```

Replace it with (masks now come from the nodes via the helper; `syncDisplays()` does the registration):

```cpp
    m_bridge = std::make_unique<tsd::rendering::GraphRenderBridge>(
        m_graph, *m_eval, Token(lib.c_str()), m_device,
        /*numViewports=*/tsd::graph_nodes::kMaxViewports);
    syncDisplays();      // reads each display node's viewportMask → setDisplay
    m_bridge->update();
```

(Both demo displays default to `0b01` via their ctors, so both land in Viewport 1 — the union. `m_knownDisplays` is now populated by `syncDisplays()` itself; see Step 4.)

- [ ] **Step 3: Replace the two named viewports with the 8-viewport pool.** The current window-construction block has:

```cpp
    windows.emplace_back(
        new ui::GraphViewport(this, m_bridge.get(), 0, m_device, "Volume"));
    windows.emplace_back(
        new ui::GraphViewport(this, m_bridge.get(), 1, m_device, "Surface"));
```

Replace those two lines with:

```cpp
    for (int i = 0; i < tsd::graph_nodes::kMaxViewports; ++i) {
      char nm[16];
      std::snprintf(nm, sizeof(nm), "Viewport %d", i + 1);
      auto *vp = new ui::GraphViewport(this, m_bridge.get(), i, m_device, nm);
      if (i > 0)
        vp->hide();
      windows.emplace_back(vp);
    }
```

(Keep the `GraphEditor`, `Inspector`, and `Log` window constructions as they are. Ensure `<cstdio>` is included for `snprintf` — it already is via the existing `Regenerate`/layout code; if not, add `#include <cstdio>`.)

- [ ] **Step 4: Rewrite `syncDisplays()`** to route node masks to the bridge. Replace the current `syncDisplays()` body with:

```cpp
  void syncDisplays()
  {
    const auto masks = tsd::graph_nodes::collectDisplayMasks(m_graph);
    std::set<tsd::graph::NodeId> current;
    for (const auto &dm : masks) {
      m_bridge->setDisplay(dm.node, dm.mask, /*enabled=*/dm.mask != 0);
      current.insert(dm.node);
    }
    for (auto id : m_knownDisplays)
      if (!current.count(id))
        m_bridge->removeDisplay(id);
    m_knownDisplays = std::move(current);
  }
```

(`<set>` and `<algorithm>` are already included from the Phase 4d version. The old name-matching loop and `0b01` default are gone — masks now come from `collectDisplayMasks`.)

- [ ] **Step 5: Update `getDefaultLayout()`** so the central node is a tab group of all 8 viewports (replacing the Volume|Surface split), with GraphEditor+Inspector left and Log bottom. Replace the entire `getDefaultLayout()` return string with:

```cpp
    return R"layout(
[Window][MainDockSpace]
Pos=0,26
Size=1920,1054
Collapsed=0

[Window][Graph Editor]
Pos=0,26
Size=420,528
Collapsed=0
DockId=0x00000005,0

[Window][Inspector]
Pos=0,556
Size=420,526
Collapsed=0
DockId=0x00000006,0

[Window][Viewport 1]
Pos=422,26
Size=1498,790
Collapsed=0
DockId=0x00000003,0

[Window][Viewport 2]
Pos=422,26
Size=1498,790
Collapsed=0
DockId=0x00000003,1

[Window][Viewport 3]
Pos=422,26
Size=1498,790
Collapsed=0
DockId=0x00000003,2

[Window][Viewport 4]
Pos=422,26
Size=1498,790
Collapsed=0
DockId=0x00000003,3

[Window][Viewport 5]
Pos=422,26
Size=1498,790
Collapsed=0
DockId=0x00000003,4

[Window][Viewport 6]
Pos=422,26
Size=1498,790
Collapsed=0
DockId=0x00000003,5

[Window][Viewport 7]
Pos=422,26
Size=1498,790
Collapsed=0
DockId=0x00000003,6

[Window][Viewport 8]
Pos=422,26
Size=1498,790
Collapsed=0
DockId=0x00000003,7

[Window][Log]
Pos=0,818
Size=1920,262
Collapsed=0
DockId=0x00000002,0

[Docking][Data]
DockSpace      ID=0x80F5B4C5 Window=0x079D3A04 Pos=0,26 Size=1920,1054 Split=Y
  DockNode     ID=0x00000001 Parent=0x80F5B4C5 SizeRef=1920,790 Split=X
    DockNode   ID=0x00000007 Parent=0x00000001 SizeRef=420,790 Split=Y
      DockNode ID=0x00000005 Parent=0x00000007 SizeRef=420,395
      DockNode ID=0x00000006 Parent=0x00000007 SizeRef=420,395
    DockNode   ID=0x00000003 Parent=0x00000001 SizeRef=1498,790 CentralNode=1
  DockNode     ID=0x00000002 Parent=0x80F5B4C5 SizeRef=1920,262
)layout";
```

(The dockspace id `0x80F5B4C5` / window id `0x079D3A04` are the framework's fixed `MainDockSpace` hashes — keep them. All 8 viewports share central `DockId=0x00000003` as tabs `,0`..`,7`; the `[Window][Viewport N]` names must match the pool's `GraphViewport` names exactly. Only "Viewport 1" is visible at startup — the rest are hidden via `vp->hide()` and shown from the View menu, where they tab into this central node.)

- [ ] **Step 6: Build the app**

Run: `cmake --build _out/_cmake --config RelWithDebInfo --target tsdFlow 2>&1 | tail -20`
Expected: compiles + links. Fix any mismatch against the real member names in `tsdFlow.cpp` (e.g. confirm `m_knownDisplays` is a `std::set<tsd::graph::NodeId>` member, `m_displays` is `DemoDisplays`); report changes.

- [ ] **Step 7: Full suite gate**

Run:
```bash
cmake --build _out/_cmake --config RelWithDebInfo --target tsdTests --parallel
ctest --test-dir _out/_cmake/tsd/tests -C RelWithDebInfo --output-on-failure
```
Expected: all green — the prior suite plus the 2 new Phase 4f tests (`tsd::nodes::DisplayMask`, `tsd::nodes::MultiViewport`). Report the summary line.

- [ ] **Step 8: Confirm `.envrc` uncommitted**

Run: `jj status`
Expected: working copy shows `A .envrc` and nothing from this task after the commit below. NEVER commit `.envrc`.

- [ ] **Step 9: Commit**

```bash
clang-format -i tsd/apps/interactive/tsdFlow/tsdFlow.cpp
jj commit tsd/apps/interactive/tsdFlow/tsdFlow.cpp -m "feat(app): tsdFlow 8-viewport pool + mask-driven display routing"
```

- [ ] **Step 10: Record the manual test checklist** (GUI not CI-tested) in the task report:
  - `tsdFlow` launches; "Viewport 1" shows both the volume and the bounding-box surface (the union); Graph Editor + Inspector dock left, Log bottom.
  - View menu lists "Viewport 1".."Viewport 8"; enabling "Viewport 2" tabs a second viewport into the center.
  - Select the `DisplayVolume` node → Inspector shows a "Viewports" row of 8 checkboxes (bit 1 checked). Check box 2 → the volume now also renders in Viewport 2.
  - Uncheck all boxes for a display → it vanishes from every viewport; re-check one → it returns.
  - The `DisplaySurface` node masks independently of the volume.

---

## Phase 4f completion checklist

- [ ] `viewportMask` param seeded on both display nodes; `collectDisplayMasks` + `kMaxViewports`/`kDefaultViewportMask` (Task 1)
- [ ] `tsd::nodes::DisplayMask` unit test + `tsd::nodes::MultiViewport` routing test pass (Task 1)
- [ ] Inspector renders `viewportMask` as a viewport checkbox row (Task 2)
- [ ] tsdFlow: bridge `numViewports=8`, 8-viewport pool (1 visible), mask-driven `syncDisplays`, tab-group layout (Task 3)
- [ ] full suite green; `.envrc` uncommitted; manual checklist recorded

## Out of scope (per spec)

Runtime change of the viewport *count* / true dynamic bridge resize + framework window add-remove (fixed pool of 8); viewport rename (fixed "Viewport N"); a dedicated Viewports management panel; per-viewport renderer settings or camera sync; mask persistence (the mask is now graph state, so Phase 5's serialization picks it up — not implemented here); the routing-vs-data optimization (a mask toggle re-evaluates + rebuilds the display's layer — correct, not free, acceptable at interactive rates).

## Self-review notes

- **Spec coverage:** the `viewportMask` param + `collectDisplayMasks` + constants (Task 1), the two tests at the right seam — device-free unit + VisRTX layer-membership routing (Task 1), the Inspector checkbox special-case (Task 2), and the 8-viewport pool + mask-driven `syncDisplays` + tab-group layout (Task 3) cover every spec component. The View menu is the framework's existing one (no new code), per the spec.
- **Type consistency:** `collectDisplayMasks(Graph&) → vector<DisplayMask{NodeId,uint64_t}>`, `kMaxViewports`/`kDefaultViewportMask` (`int`), the `viewportMask` param name, and `GraphViewport`/`setDisplay`/`layersForViewport` signatures are used identically across tasks and match the verified API reference.
- **Tested seam:** the silent-failure-prone logic (param→mask read, mask→viewport routing into the bridge) is headlessly tested (Task 1); the GUI (checkboxes, pool, layout) is build-verified + manually checked (Task 3) — consistent with prior phases.
- **Cost note carried from the spec:** toggling a mask changes the display node's param-hash, so the next `bridge.update()` recomputes that node and rebuilds its layer — correct, not free; the routing itself is driven by `setDisplay` + `layersForViewport`, not the re-eval. Deferred optimization, documented.
- **Flagged for the implementer (adjust minimally, report):** confirm `<cstdio>` is included where `snprintf` is used (Inspector + app); confirm `tsdFlow.cpp`'s `m_knownDisplays`/`m_displays` member types; the docking-INI dock ids are the framework's fixed `MainDockSpace` hashes.
