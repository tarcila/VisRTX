# tsdFlow Phase 4h — Spatial Display & Manipulation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `BoundingBox` a non-occluding wireframe, and make displayed spatial data transformable via a render-time instance transform editable in the Inspector (numeric TRS) and a viewport `ImGuizmo` overlay.

**Architecture:** The bounds change is localized to `BoundingBox` (emit `cylinder` edges). The transform is typed node state behind a new `ITransformableNode` interface (NOT a `ParameterList` param — keeps it off the eval hash so editing never re-evaluates/rebuilds); a UI-free `collectDisplayTransforms` helper feeds `GraphRenderBridge::setDisplayTransform`, which applies the matrix to each display's **layer-root** transform every `update()` (no object rebuild; picked up by the existing repopulate). Inspector + a ported `ImGuizmo` overlay in `GraphViewport` edit it.

**Tech Stack:** C++17, `tsd_graph`/`tsd_graph_nodes`, `tsd_rendering` `GraphRenderBridge` + `tsd::scene` Layer, `tsd_ui_imgui` (Inspector/GraphViewport/BaseViewport), vendored `ImGuizmo`, ANARI + VisRTX, Catch2, jj.

## Global Constraints

- Version control is **jj**, not git. Commit ONLY a task's files with explicit paths: `jj commit <paths> -m "..."`. **NEVER** a bare `jj commit` — an unrelated `.envrc` must stay uncommitted. **Raw `git` is sandboxed and will fail** — never call git.
- Build tree `_out/_cmake` (Ninja Multi-Config, RelWithDebInfo). No new `build/` dir.
  - Tests: `cmake --build _out/_cmake --config RelWithDebInfo --target tsdTests`
  - UI lib: `cmake --build _out/_cmake --config RelWithDebInfo --target tsd_ui_imgui`
  - App: `cmake --build _out/_cmake --config RelWithDebInfo --target tsdFlow`
  - Run a test: `ctest --test-dir _out/_cmake/tsd/tests -C RelWithDebInfo -R '<name>' --output-on-failure`
- `clang-format -i` ONLY `.cpp`/`.hpp` — **NEVER** clang-format `CMakeLists.txt` (edit by hand).
- New-file header: `// Copyright 2026 NVIDIA Corporation` then `// SPDX-License-Identifier: Apache-2.0`; `#pragma once` in headers.
- Namespaces: catalog `tsd::graph_nodes`; UI `tsd::ui::imgui`. VisRTX render device; any test that constructs a `GraphRenderBridge` (needs a device) gets `set_tests_properties(<name> PROPERTIES TIMEOUT 300)`.
- The transform is **typed state behind `ITransformableNode`, NOT a `ParameterList` param** (off the eval hash → no re-eval/rebuild on edit). The Inspector transform section is **additive** to `drawParameters` and must **not** call `markDirty`.

## Verified API reference (confirmed against source)

```cpp
// BoundingBox.cpp (current output): emits SurfaceData (NOT Renderable)
//   s->geomSubtype = Token("triangle"); s->prim.arrays.push_back({Token("vertex.position"), pos /*36*/});
//   s->appearance.scalars color; out.type = PortType{portSurface()}; payload = s
// Descriptors.hpp:  struct SurfaceData { tsd::core::Token geomSubtype; RenderableParams prim; RenderableParams appearance; };
// DisplaySurface.cpp: r->primSubtype = s->geomSubtype; r->prim = s->prim;  (maps SurfaceData → Renderable)
// DisplayVolume/DisplaySurface: `struct X : Node { ParameterList params; X(){ params.set(Token("viewportMask"), kDefaultViewportMask); } ... }`  (anon namespace)

// tsd/core/TSDMath.hpp:  using mat4 = float4x4;  extern const mat4 IDENTITY_MAT4;  (tsd::core::math; tsd::math is an alias)
// tsd/graph/Graph.hpp:  std::vector<NodeId> nodeIds() const;  GraphNode *node(NodeId);  GraphNode{ NodeId id; std::unique_ptr<Node> impl; };
// tsd/graph_nodes/DisplayMask.hpp:  std::vector<DisplayMask> collectDisplayMasks(Graph&);  constexpr int kMaxViewports;

// GraphRenderBridge.hpp (private): struct Display { uint64_t mask; bool enabled; tsd::scene::Layer *layer; uint64_t lastVersion; bool realized; };
//   public: setDisplay(NodeId,uint64_t,bool); removeDisplay(NodeId); std::vector<const Layer*> layersForViewport(int) const; update(); world(int);
//   buildSurface: createObject<Geometry>(r.primSubtype); applyParams(prim); ... insertChildObjectNode(layer->root(), surf);
//   update(): for each enabled display rebuildLayer(); then for each viewport m_indices[i]->setIncludedLayers(layersForViewport(i));
// tsd/scene/Layer.hpp:  LayerNodeRef root() const;   // root is a transform node, survives clearLayerObjects
// tsd/scene/LayerNodeData.hpp: void setAsTransform(const tsd::math::mat4&); tsd::math::mat4 getTransform() const;  (LayerNodeRef::operator-> yields LayerNodeData*)
// BaseViewport.cpp uses: (*nodeRef)->setAsTransform(m); ImGuizmo::{BeginFrame,SetOrthographic,SetDrawlist,SetRect,Manipulate,IsUsing,IsOver,DecomposeMatrixToComponents,RecomposeMatrixFromComponents,OPERATION,MODE}

// tsd/ui/imgui/windows/Inspector.cpp buildUI(): if (dynamic_cast<ITransferFunctionNode*>(impl)) {TF; markDirty} else {drawParameters}
// tsd/ui/imgui/windows/GraphViewport: ctor (Application*, GraphRenderBridge*, int viewportIndex, anari::Device, const char* name);
//   members m_bridge,m_viewportIndex,m_device,m_camera(anari, opaque),m_renderer,m_manip(Manipulator),m_size(int2); buildUI draws ImGui::Image at GetCursorScreenPos()/m_size, then InvisibleButton("##viewport") + handleNavigation().
//   Manipulator: float3 eye()/at()/up()/dir(); float distance();
// tsdFlow.cpp: viewport pool `new ui::GraphViewport(this, m_bridge.get(), i, m_device, nm)` (line ~208); syncDisplays(){ for collectDisplayMasks → setDisplay }; app owns m_graph,m_selected,m_graphDirty.
```

## File structure / tasks

1. Wireframe `BoundingBox` (+ test). 2. `ITransformableNode` + display nodes + `collectDisplayTransforms` (+ test). 3. Bridge `setDisplayTransform` + layer-root application (+ test). 4. Inspector TRS section. 5. `GraphViewport` gizmo + app wiring (+ suite gate + manual). Order 1→2→3→4→5 (4 needs 2; 5 needs 2/3/4).

---

## Task 1: wireframe `BoundingBox`

**Files:** Modify `tsd/src/tsd/graph_nodes/BoundingBox.cpp`; Test `tsd/tests/test_nodes_Surface.cpp` (extend).

**Interfaces:** Produces: `BoundingBox` `SurfaceData` with `geomSubtype == Token("cylinder")`, `prim.arrays` `vertex.position` (24 float3 = 12 edges × 2), `prim.scalars` `radius` (>0). Downstream `DisplaySurface` → `Renderable` `primSubtype == "cylinder"` (unchanged mapping).

- [ ] **Step 1: Update the existing failing test** — in `tsd/tests/test_nodes_Surface.cpp`, the `[nodes-surface]` BoundingBox scenario currently asserts `geomSubtype == Token("triangle")` and 36 vertices. Change those assertions to:

```cpp
    THEN("it is a cylinder wireframe with 24 vertex positions and a radius")
    {
      REQUIRE(s != nullptr);
      REQUIRE(s->geomSubtype == Token("cylinder"));
      // 12 box edges x 2 endpoints
      bool foundPos = false, foundRadius = false;
      for (const auto &a : s->prim.arrays)
        if (a.first == Token("vertex.position")) {
          REQUIRE(a.second.size() == 24);
          foundPos = true;
        }
      for (const auto &sc : s->prim.scalars)
        if (sc.first == Token("radius")) {
          REQUIRE(sc.second.get<float>() > 0.f);
          foundRadius = true;
        }
      REQUIRE(foundPos);
      REQUIRE(foundRadius);
    }
```
And update the downstream `DisplaySurface` assertion in the same scenario from `primSubtype == Token("triangle")` to `primSubtype == Token("cylinder")`. (Keep the rest of the scenario; `AnyArray::size()` and `Any::get<float>()` are the accessors already used elsewhere in the suite.)

- [ ] **Step 2: Build + confirm FAIL**

Run: `cmake --build _out/_cmake --config RelWithDebInfo --target tsdTests && ctest --test-dir _out/_cmake/tsd/tests -C RelWithDebInfo -R '\[nodes-surface\]' --output-on-failure 2>&1 | tail -15`
Expected: FAIL (still emits triangle/36).

- [ ] **Step 3: Rewrite the geometry in `BoundingBox::evaluate`.** Replace the 36-index triangle table + the `pos(...,36)` build + `geomSubtype = Token("triangle")` block (everything from the `const int tri[36] = {...}` through the `s->prim.arrays.push_back({Token("vertex.position"), pos});` line) with the 12-edge cylinder build. The 8 corners `c[8]` and `lo`/`hi` are computed above and stay. Use:

```cpp
    // 12 box edges as cylinder segments (consecutive vertex.position pairs).
    static const int edge[12][2] = {{0, 1}, {1, 2}, {2, 3}, {3, 0}, // bottom
        {4, 5}, {5, 6}, {6, 7}, {7, 4}, // top
        {0, 4}, {1, 5}, {2, 6}, {3, 7}}; // verticals
    tsd::core::AnyArray pos(ANARI_FLOAT32_VEC3, 24);
    for (int e = 0; e < 12; ++e) {
      pos.get<float3>(size_t(2 * e)) = c[edge[e][0]];
      pos.get<float3>(size_t(2 * e + 1)) = c[edge[e][1]];
    }
    const float3 d = hi - lo;
    const float radius = std::max(0.004f * std::sqrt(dot(d, d)), 1e-4f);

    auto s = std::make_shared<SurfaceData>();
    s->geomSubtype = Token("cylinder");
    s->prim.arrays.push_back({Token("vertex.position"), pos});
    s->prim.scalars.push_back({Token("radius"), tsd::core::Any(radius)});
    s->appearance.scalars.push_back({Token("color"),
        tsd::core::Any(params.getOr<float3>(Token("color"), float3(0.8f)))});
```
(Keep the trailing `Value out; out.type = PortType{portSurface()}; out.residency = hostResidency(); out.payload = s; ctx.setOutput(...)` unchanged. Add `#include <cmath>` if not present — it is, for nothing yet; `std::sqrt`/`std::max` need `<cmath>`/`<algorithm>`, add both. `dot` is `tsd::core::math::dot` (linalg), available via the existing math include.)

- [ ] **Step 4: Build + run the test → PASS**

Run: `cmake --build _out/_cmake --config RelWithDebInfo --target tsdTests && ctest --test-dir _out/_cmake/tsd/tests -C RelWithDebInfo -R '\[nodes-surface\]' --output-on-failure 2>&1 | tail -8`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
clang-format -i tsd/src/tsd/graph_nodes/BoundingBox.cpp tsd/tests/test_nodes_Surface.cpp
jj commit tsd/src/tsd/graph_nodes/BoundingBox.cpp tsd/tests/test_nodes_Surface.cpp -m "feat(graph_nodes): BoundingBox emits a cylinder wireframe (non-occluding)"
```

---

## Task 2: `ITransformableNode` + display nodes + `collectDisplayTransforms`

**Files:** Create `tsd/src/tsd/graph_nodes/TransformableNode.hpp`, `tsd/src/tsd/graph_nodes/DisplayTransform.hpp`, `tsd/src/tsd/graph_nodes/DisplayTransform.cpp`; Modify `DisplayVolume.cpp`, `DisplaySurface.cpp`, `tsd/src/tsd/graph_nodes/CMakeLists.txt`; Test `tsd/tests/test_nodes_DisplayTransform.cpp`, `tsd/tests/CMakeLists.txt`.

**Interfaces:**
- Produces: `struct tsd::graph_nodes::ITransformableNode { virtual ~ITransformableNode()=default; virtual tsd::core::math::mat4 &transform()=0; };`
- Produces: `struct DisplayTransform { tsd::graph::NodeId node; tsd::core::math::mat4 xfm; };` and `std::vector<DisplayTransform> collectDisplayTransforms(tsd::graph::Graph &g);`
- `DisplayVolume`/`DisplaySurface` now implement `ITransformableNode` (identity default).

- [ ] **Step 1: Write the failing test** — create `tsd/tests/test_nodes_DisplayTransform.cpp`:

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
#include "tsd/graph_nodes/DisplayTransform.hpp"
#include "tsd/graph_nodes/TransformableNode.hpp"

using tsd::core::Token;
using namespace tsd::graph;
using tsd::graph_nodes::collectDisplayTransforms;
using mat4 = tsd::core::math::mat4;

namespace {
const tsd::graph_nodes::DisplayTransform *find(
    const std::vector<tsd::graph_nodes::DisplayTransform> &v, NodeId id)
{
  for (const auto &dt : v)
    if (dt.node == id)
      return &dt;
  return nullptr;
}
} // namespace

SCENARIO("collectDisplayTransforms reports display transforms", "[display-transform]")
{
  NodeRegistry reg;
  tsd::graph_nodes::registerBuiltinNodes(reg);
  Graph g;
  auto d = tsd::graph_nodes::buildVolumeSurfaceDemo(g, reg);

  WHEN("transforms are default") {
    auto xs = collectDisplayTransforms(g);
    THEN("both display nodes appear at identity") {
      REQUIRE(xs.size() == 2);
      REQUIRE(find(xs, d.volumeDisplay) != nullptr);
      REQUIRE(find(xs, d.surfaceDisplay) != nullptr);
      REQUIRE(find(xs, d.volumeDisplay)->xfm == tsd::core::math::IDENTITY_MAT4);
    }
    THEN("non-display nodes are excluded") {
      REQUIRE(find(xs, d.source) == nullptr);
    }
  }

  WHEN("one display's transform is set via ITransformableNode") {
    mat4 m = tsd::core::math::IDENTITY_MAT4;
    m[3].x = 5.f; // translate +5 in x (column-major: column 3 = translation)
    auto *itf = dynamic_cast<tsd::graph_nodes::ITransformableNode *>(
        g.node(d.volumeDisplay)->impl.get());
    REQUIRE(itf != nullptr);
    itf->transform() = m;
    auto xs = collectDisplayTransforms(g);
    THEN("the helper reports it; the other stays identity") {
      REQUIRE(find(xs, d.volumeDisplay)->xfm == m);
      REQUIRE(find(xs, d.surfaceDisplay)->xfm == tsd::core::math::IDENTITY_MAT4);
    }
  }
}
```
(`mat4 operator==` is linalg's element-wise compare — available. If exact `==` is flaky, compare `m[3].x`; but identity vs a single edited element is exact.)

- [ ] **Step 2: Register the test** — edit `tsd/tests/CMakeLists.txt` by hand: add `test_nodes_DisplayTransform.cpp` to the source list and:
```cmake
add_test(NAME tsd::nodes::DisplayTransform COMMAND ${PROJECT_NAME} "[display-transform]")
```

- [ ] **Step 3: Build + confirm FAIL** — `cmake --build _out/_cmake --config RelWithDebInfo --target tsdTests 2>&1 | tail -15` → `TransformableNode.hpp`/`DisplayTransform.hpp` not found.

- [ ] **Step 4: Create `tsd/src/tsd/graph_nodes/TransformableNode.hpp`:**

```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/TSDMath.hpp"

namespace tsd::graph_nodes {

// Implemented by nodes carrying a render-time instance transform. Kept OUT of
// the node's ParameterList (so it never enters ParameterList::hash() and a
// transform edit never triggers re-evaluation / layer rebuild). UI reaches it
// via dynamic_cast, like ITransferFunctionNode.
struct ITransformableNode
{
  virtual ~ITransformableNode() = default;
  virtual tsd::core::math::mat4 &transform() = 0;
};

} // namespace tsd::graph_nodes
```

- [ ] **Step 5: Make the display nodes implement it.** In `tsd/src/tsd/graph_nodes/DisplayVolume.cpp`: add `#include "tsd/graph_nodes/TransformableNode.hpp"` near the includes; change `struct DisplayVolume : Node` to `struct DisplayVolume : Node, ITransformableNode`; add a member and accessor:

```cpp
  tsd::core::math::mat4 m_transform{tsd::core::math::IDENTITY_MAT4};
  tsd::core::math::mat4 &transform() override { return m_transform; }
```
(place the member next to `ParameterList params;`, the accessor among the other overrides). Do the **identical** change in `tsd/src/tsd/graph_nodes/DisplaySurface.cpp` (`struct DisplaySurface : Node, ITransformableNode`, same member + accessor + include). `evaluate()` is unchanged — it must NOT read `m_transform`.

- [ ] **Step 6: Create `tsd/src/tsd/graph_nodes/DisplayTransform.hpp`:**

```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/TSDMath.hpp"
#include "tsd/graph/Graph.hpp"
// std
#include <vector>

namespace tsd::graph_nodes {

struct DisplayTransform
{
  tsd::graph::NodeId node{0};
  tsd::core::math::mat4 xfm{tsd::core::math::IDENTITY_MAT4};
};

// Every node implementing ITransformableNode and its transform. Graph& non-const
// (node()->impl is non-const for the dynamic_cast). Logically read-only.
std::vector<DisplayTransform> collectDisplayTransforms(tsd::graph::Graph &g);

} // namespace tsd::graph_nodes
```

- [ ] **Step 7: Create `tsd/src/tsd/graph_nodes/DisplayTransform.cpp`:**

```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/graph_nodes/DisplayTransform.hpp"
#include "tsd/graph_nodes/TransformableNode.hpp"

namespace tsd::graph_nodes {

using tsd::graph::NodeId;

std::vector<DisplayTransform> collectDisplayTransforms(tsd::graph::Graph &g)
{
  std::vector<DisplayTransform> out;
  for (const NodeId id : g.nodeIds()) {
    auto *gn = g.node(id);
    if (!gn || !gn->impl)
      continue;
    if (auto *itf = dynamic_cast<ITransformableNode *>(gn->impl.get()))
      out.push_back({id, itf->transform()});
  }
  return out;
}

} // namespace tsd::graph_nodes
```

- [ ] **Step 8: Add to CMake** — edit `tsd/src/tsd/graph_nodes/CMakeLists.txt` by hand: add `DisplayTransform.cpp` to `project_sources(PRIVATE ...)`.

- [ ] **Step 9: Build + run → PASS**

Run: `cmake --build _out/_cmake --config RelWithDebInfo --target tsdTests && ctest --test-dir _out/_cmake/tsd/tests -C RelWithDebInfo -R 'tsd::nodes::DisplayTransform' --output-on-failure`
Expected: PASS.

- [ ] **Step 10: Commit**

```bash
clang-format -i tsd/src/tsd/graph_nodes/TransformableNode.hpp tsd/src/tsd/graph_nodes/DisplayTransform.hpp tsd/src/tsd/graph_nodes/DisplayTransform.cpp tsd/src/tsd/graph_nodes/DisplayVolume.cpp tsd/src/tsd/graph_nodes/DisplaySurface.cpp tsd/tests/test_nodes_DisplayTransform.cpp
jj commit tsd/src/tsd/graph_nodes/TransformableNode.hpp tsd/src/tsd/graph_nodes/DisplayTransform.hpp tsd/src/tsd/graph_nodes/DisplayTransform.cpp tsd/src/tsd/graph_nodes/DisplayVolume.cpp tsd/src/tsd/graph_nodes/DisplaySurface.cpp tsd/src/tsd/graph_nodes/CMakeLists.txt tsd/tests/test_nodes_DisplayTransform.cpp tsd/tests/CMakeLists.txt -m "feat(graph_nodes): ITransformableNode on display nodes + collectDisplayTransforms"
```

---

## Task 3: bridge `setDisplayTransform` + layer-root application

**Files:** Modify `tsd/src/tsd/rendering/bridge/GraphRenderBridge.hpp`, `GraphRenderBridge.cpp`; Test `tsd/tests/test_bridge_Transform.cpp`, `tsd/tests/CMakeLists.txt`.

**Interfaces:** Produces: `void GraphRenderBridge::setDisplayTransform(tsd::graph::NodeId, const tsd::math::mat4 &);` — stores the matrix per display; `update()` applies it to that display's layer-root transform.

- [ ] **Step 1: Write the failing test** — create `tsd/tests/test_bridge_Transform.cpp` (mirrors `test_bridge_Mask`'s setup; asserts the layer-root transform, no pixel render):

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
#include "tsd/rendering/bridge/GraphRenderBridge.hpp"
#include "tsd/scene/Layer.hpp"
// anari
#include <anari/anari_cpp.hpp>

using tsd::core::Token;
using namespace tsd::graph;
using tsd::rendering::GraphRenderBridge;
using mat4 = tsd::math::mat4;

namespace {
anari::Device makeVisRTX()
{
  auto lib = anari::loadLibrary("visrtx", nullptr, nullptr);
  return lib ? anari::newDevice(lib, "default") : nullptr;
}
} // namespace

SCENARIO("setDisplayTransform applies to the display's layer-root", "[bridge-transform]")
{
  anari::Device dev = makeVisRTX();
  REQUIRE(dev != nullptr);

  NodeRegistry reg;
  tsd::graph_nodes::registerBuiltinNodes(reg);
  Graph g;
  auto d = tsd::graph_nodes::buildVolumeSurfaceDemo(g, reg);
  Evaluator e(g);

  GraphRenderBridge bridge(g, e, Token("visrtx"), dev, /*numViewports=*/1);
  bridge.setDisplay(d.volumeDisplay, 0b01, true);   // only the volume, in viewport 0
  bridge.setDisplay(d.surfaceDisplay, 0b00, false); // exclude the surface

  mat4 m = tsd::math::IDENTITY_MAT4;
  m[3].x = 5.f; // translate +5 x
  bridge.setDisplayTransform(d.volumeDisplay, m);
  bridge.update();

  WHEN("inspecting the volume display's layer root") {
    auto layers = bridge.layersForViewport(0);
    THEN("its root transform is the matrix we set") {
      REQUIRE(layers.size() == 1);
      const auto root = layers[0]->root();
      REQUIRE(root->getTransform()[3].x == Approx(5.f));
    }
  }

  anari::release(dev, dev);
}
```
(If `layersForViewport` returns `const Layer*` and `root()`/`getTransform()` are const-accessible, this compiles as-is; if `getTransform()` isn't const, fetch the layer differently or assert via a non-const path — adjust minimally and report.)

- [ ] **Step 2: Register the test (TIMEOUT 300)** — edit `tsd/tests/CMakeLists.txt` by hand: add `test_bridge_Transform.cpp` to the source list, then:
```cmake
add_test(NAME tsd::rendering::BridgeTransform COMMAND ${PROJECT_NAME} "[bridge-transform]")
set_tests_properties(tsd::rendering::BridgeTransform PROPERTIES TIMEOUT 300)
```

- [ ] **Step 3: Build + confirm FAIL** — `cmake --build _out/_cmake --config RelWithDebInfo --target tsdTests 2>&1 | tail -15` → `setDisplayTransform` not a member.

- [ ] **Step 4: Add the field + method to `GraphRenderBridge.hpp`.** In the `Display` struct add:
```cpp
    tsd::math::mat4 transform{tsd::math::IDENTITY_MAT4};
```
In the `public:` section after `removeDisplay`:
```cpp
  void setDisplayTransform(tsd::graph::NodeId node, const tsd::math::mat4 &xfm);
```
(Ensure `tsd/core/TSDMath.hpp` is included — it is, transitively via the graph/scene headers; add it if the build complains.)

- [ ] **Step 5: Implement in `GraphRenderBridge.cpp`.** Add the setter:
```cpp
void GraphRenderBridge::setDisplayTransform(
    tsd::graph::NodeId node, const tsd::math::mat4 &xfm)
{
  auto it = m_displays.find(node);
  if (it != m_displays.end())
    it->second.transform = xfm;
}
```
In `update()`, apply the transform to the layer root for each enabled display **after** `rebuildLayer`. Change the loop body:
```cpp
  for (auto &kv : m_displays) {
    Display &d = kv.second;
    if (!d.enabled) {
      d.realized = false;
      continue;
    }
    rebuildLayer(kv.first, d);
    if (d.layer)
      d.layer->root()->setAsTransform(d.transform); // root survives rebuild; no object rebuild
  }
```
(`d.layer->root()` returns a `LayerNodeRef`; `->setAsTransform(const mat4&)` mutates the root transform node. The subsequent `setIncludedLayers` repopulate re-reads it. Confirm `root()`/`setAsTransform` exact spelling against `Layer.hpp`/`LayerNodeData.hpp`.)

- [ ] **Step 6: Build + run → PASS**

Run: `cmake --build _out/_cmake --config RelWithDebInfo --target tsdTests && ctest --test-dir _out/_cmake/tsd/tests -C RelWithDebInfo -R 'tsd::rendering::BridgeTransform' --output-on-failure`
Expected: PASS (first run may OptiX-warmup).

- [ ] **Step 7: Commit**

```bash
clang-format -i tsd/src/tsd/rendering/bridge/GraphRenderBridge.hpp tsd/src/tsd/rendering/bridge/GraphRenderBridge.cpp tsd/tests/test_bridge_Transform.cpp
jj commit tsd/src/tsd/rendering/bridge/GraphRenderBridge.hpp tsd/src/tsd/rendering/bridge/GraphRenderBridge.cpp tsd/tests/test_bridge_Transform.cpp tsd/tests/CMakeLists.txt -m "feat(rendering): GraphRenderBridge::setDisplayTransform applies to the display layer-root"
```

---

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

## Task 5: `GraphViewport` gizmo + app wiring (+ suite gate)

**Files:** Modify `tsd/src/tsd/ui/imgui/windows/GraphViewport.hpp`, `GraphViewport.cpp`, `tsd/apps/interactive/tsdFlow/tsdFlow.cpp`.

**Interfaces:** Consumes `ITransformableNode` + `collectDisplayTransforms` (Task 2), `setDisplayTransform` (Task 3). `GraphViewport` ctor gains `tsd::graph::Graph*`, `tsd::graph::NodeId* selected`, `bool* graphDirty`.

No automated test (GUI). Deliverable: `tsdFlow` builds; full suite green; manual checklist.

- [ ] **Step 1: Extend `GraphViewport.hpp`.** Add includes `#include "tsd/graph/Graph.hpp"` and `#include "ImGuizmo.h"`. Change the ctor signature to:
```cpp
  GraphViewport(Application *app,
      tsd::rendering::GraphRenderBridge *bridge,
      int viewportIndex,
      anari::Device device,
      tsd::graph::Graph *graph,
      tsd::graph::NodeId *selected,
      bool *graphDirty,
      const char *name = "Viewport");
```
Add members:
```cpp
  tsd::graph::Graph *m_graph{nullptr};
  tsd::graph::NodeId *m_selected{nullptr};
  bool *m_graphDirty{nullptr};
  ImGuizmo::OPERATION m_gizmoOp{ImGuizmo::TRANSLATE};
  ImGuizmo::MODE m_gizmoMode{ImGuizmo::WORLD};
```

- [ ] **Step 2: Store the new ctor args** in `GraphViewport.cpp`'s ctor init list (`m_graph(graph), m_selected(selected), m_graphDirty(graphDirty)`), matching the new signature.

- [ ] **Step 3: Add a `drawGizmo()` helper + call it.** In `GraphViewport::buildUI`, after the `ImGui::Image(...)` blit (where `pos = ImGui::GetCursorScreenPos()` before the image and `imgSize` are known) and **before** `handleNavigation()`, call `drawGizmo(pos, imgSize)`, and gate navigation:
```cpp
  const bool gizmoActive = drawGizmo(pos, imgSize);
  if (!gizmoActive)
    handleNavigation();
```
Implement `drawGizmo` (returns true if the gizmo is using/hovered, so navigation is suppressed):

```cpp
bool GraphViewport::drawGizmo(const ImVec2 &imgPos, const ImVec2 &imgSize)
{
  if (!m_selected || *m_selected == tsd::graph::INVALID_NODE || !m_graph)
    return false;
  auto *gn = m_graph->node(*m_selected);
  if (!gn || !gn->impl)
    return false;
  auto *itf = dynamic_cast<tsd::graph_nodes::ITransformableNode *>(gn->impl.get());
  if (!itf)
    return false;
  // Only show the gizmo if this display is masked into this viewport.
  const int mask = gn->impl->parameters().getOr<int>(Token("viewportMask"), 0);
  if (!((mask >> m_viewportIndex) & 1))
    return false;

  using tsd::math::float3;
  using tsd::math::mat4;
  const float3 eye = m_manip.eye(), at = m_manip.at(), up = m_manip.up();
  const mat4 view = linalg::lookat_matrix(eye, at, up);
  const float aspect = float(m_size.x) / float(m_size.y);
  constexpr float kFovy = 1.04719755f; // π/3 — the ANARI/VisRTX perspective default
  const float focusDist = linalg::length(at - eye);
  const mat4 proj = linalg::perspective_matrix(
      kFovy, aspect, std::max(0.01f * focusDist, 1e-3f), 100.f * focusDist + 10.f);

  ImGuizmo::BeginFrame();
  ImGuizmo::SetOrthographic(false);
  ImGuizmo::SetDrawlist();
  ImGuizmo::SetRect(imgPos.x, imgPos.y, imgSize.x, imgSize.y);

  mat4 m = itf->transform();
  if (ImGuizmo::Manipulate(
          &view[0].x, &proj[0].x, m_gizmoOp, m_gizmoMode, &m[0].x)) {
    itf->transform() = m; // root has no parent → manipulated matrix IS the transform
    *m_graphDirty = true;
  }
  return ImGuizmo::IsUsing() || ImGuizmo::IsOver();
}
```
Declare `bool drawGizmo(const ImVec2 &, const ImVec2 &);` in the header's private section. (Add `#include <algorithm>` for `std::max` and confirm `linalg::lookat_matrix`/`perspective_matrix`/`length` are the right names against the linalg the project uses — `BaseViewport.cpp` is the reference; adjust to match and report. Optional: key toggles for `m_gizmoOp`/`m_gizmoMode` à la `BaseViewport` — include if quick, else default TRANSLATE/WORLD is fine for v1.)

- [ ] **Step 4: Wire the app.** In `tsd/apps/interactive/tsdFlow/tsdFlow.cpp`, update the viewport-pool construction to pass the new args:
```cpp
      auto *vp = new ui::GraphViewport(
          this, m_bridge.get(), i, m_device, &m_graph, &m_selected, &m_graphDirty, nm);
```
And in `syncDisplays()`, after the existing mask loop, add the transform sync:
```cpp
    for (const auto &dt : tsd::graph_nodes::collectDisplayTransforms(m_graph))
      m_bridge->setDisplayTransform(dt.node, dt.xfm);
```
(Add `#include "tsd/graph_nodes/DisplayTransform.hpp"` to tsdFlow.cpp. The mask loop runs first so each display's layer exists before its transform is set; combined with the bridge's `if (d.layer)` guard this is safe.)

- [ ] **Step 5: Build the app** — `cmake --build _out/_cmake --config RelWithDebInfo --target tsdFlow 2>&1 | tail -15` → compiles + links. Fix linalg name mismatches per Step 3's note; report.

- [ ] **Step 6: Full suite gate**

```bash
cmake --build _out/_cmake --config RelWithDebInfo --target tsdTests --parallel
ctest --test-dir _out/_cmake/tsd/tests -C RelWithDebInfo --output-on-failure
```
Expected: all green — prior suite + the 3 new/changed Phase 4h tests (`[nodes-surface]` updated, `tsd::nodes::DisplayTransform`, `tsd::rendering::BridgeTransform`). Report the summary line.

- [ ] **Step 7: Confirm `.envrc` uncommitted** — `jj status` shows `.envrc` untracked, nothing else from this task after the commit. NEVER commit `.envrc`.

- [ ] **Step 8: Commit**

```bash
clang-format -i tsd/src/tsd/ui/imgui/windows/GraphViewport.hpp tsd/src/tsd/ui/imgui/windows/GraphViewport.cpp tsd/apps/interactive/tsdFlow/tsdFlow.cpp
jj commit tsd/src/tsd/ui/imgui/windows/GraphViewport.hpp tsd/src/tsd/ui/imgui/windows/GraphViewport.cpp tsd/apps/interactive/tsdFlow/tsdFlow.cpp -m "feat(app): GraphViewport ImGuizmo overlay for the selected display's transform"
```

- [ ] **Step 9: Record the manual test checklist** (GUI not CI-tested) in the report:
  - `tsdFlow` launches; the bounding box is a **wireframe** and the volume is visible through it.
  - Select `DisplayVolume`/`DisplaySurface` → the Inspector shows **Translate/Rotate/Scale** fields (+ "Transform" header, "Reset"); editing them moves/rotates/scales the rendered object live.
  - The `ImGuizmo` overlay appears on the selected display **only in viewports it's masked into**; dragging it moves the object **smoothly** (no stutter/rebuild), and the Inspector fields track it.
  - "Reset" restores identity; camera orbit/pan/zoom still work when not interacting with the gizmo.

---

## Phase 4h completion checklist

- [ ] `BoundingBox` wireframe (cylinder edges) + `[nodes-surface]` test updated (Task 1)
- [ ] `ITransformableNode` + display nodes + `collectDisplayTransforms` + test (Task 2)
- [ ] `GraphRenderBridge::setDisplayTransform` + layer-root application + test (Task 3)
- [ ] Inspector additive transform TRS section, no `markDirty` (Task 4)
- [ ] `GraphViewport` gizmo + app wiring + transform sync (Task 5)
- [ ] full suite green; `.envrc` uncommitted; manual checklist recorded

## Out of scope (per spec)

Data-resampling `Transform` node; transform on non-display nodes; gizmo snapping/extra modes; per-viewport independent transforms; transform persistence (Phase 5 must serialize the typed `transform()` state explicitly, like TF control points).

## Self-review notes

- **Spec coverage:** Component 1 → Task 1; Component 2 data model + helper → Task 2; bridge application → Task 3; Inspector → Task 4; gizmo + app wiring + flow → Task 5. The Q5 meta-review decision (transform = `ITransformableNode` typed state, off the hash) is realized in Task 2 and relied on by 3/4/5.
- **Type consistency:** `ITransformableNode::transform()→mat4&`, `DisplayTransform{NodeId,mat4 xfm}`, `collectDisplayTransforms(Graph&)`, `setDisplayTransform(NodeId,const mat4&)`, `Display.transform`, and the new `GraphViewport` ctor args are used identically across tasks and match the verified API reference.
- **Tested seam:** the device-free silent-failure-prone logic (`collectDisplayTransforms`, the `BoundingBox` geometry) is unit-tested; the bridge layer-root application gets a VisRTX membership/transform test (Task 3); Inspector + gizmo are build-verified + manual (Task 5) — consistent with prior phases.
- **Flagged for the implementer (adjust minimally, report):** exact `Layer::root()`/`LayerNodeData::setAsTransform`/`getTransform` spelling + const-ness (Task 3 test); `&m[0].x` 16-float column-major layout for ImGuizmo; linalg `lookat_matrix`/`perspective_matrix`/`length` names (match `BaseViewport.cpp`); whether `getTransform()` is readable from a `const Layer*` in the Task 3 test (fall back to build+manual if not, and report); `<cmath>`/`<algorithm>` includes.
