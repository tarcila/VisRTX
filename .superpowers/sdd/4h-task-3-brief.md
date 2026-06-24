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

