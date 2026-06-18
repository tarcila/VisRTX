# tsdFlow Phase 3 — Render Bridge & Viewport Masks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans, task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** A `GraphRenderBridge` that turns `tsd_graph` `renderable` outputs into rendered ANARI worlds — one per viewport — selected by a per-display viewport mask, reusing TSD `RenderIndex` for TSD→ANARI translation. Headless; validated with VisRTX.

**Architecture:** Bridge owns a derived, single-writer `tsd::Scene` (render-scene) with one `Layer` per display node. Each display node's `renderable` descriptor is translated to `tsd::Surface`/`tsd::Volume` in its layer. Per viewport `i`: a `RenderIndexAllLayers` whose `setIncludedLayers(...)` = the layers of displays whose mask includes `i`. `RenderIndex` builds the ANARI world.

**Tech Stack:** C++17, `tsd_graph` (Phase 1/2), `tsd_scene`/`tsd_rendering` (Scene, Layer, Surface, Volume, RenderIndexAllLayers), ANARI + VisRTX, Catch2, CTest, jj.

**Spec:** `docs/superpowers/specs/2026-06-17-tsdflow-phase3-render-bridge-design.md`.

---

## Conventions (every task)

- **jj**, not git. Commit ONLY task files with explicit paths: `jj commit <paths> -m "..."`. NEVER bare `jj commit` (an unrelated `.envrc` must stay uncommitted).
- Build tree `_out/_cmake` (Ninja Multi-Config, RelWithDebInfo). No new `build/` dir.
  - Build: `cmake --build _out/_cmake --config RelWithDebInfo --target tsdTests`
  - Test: `ctest --test-dir _out/_cmake/tsd/tests -C RelWithDebInfo -R '<name>' --output-on-failure`
- File header `// Copyright 2026 NVIDIA Corporation` / `// SPDX-License-Identifier: Apache-2.0`; `#pragma once`. `clang-format -i` before commit. Namespace `tsd::graph` (engine types) / `tsd::rendering` (bridge).
- Register a test: source into `project_add_executable(...)` in `tsd/tests/CMakeLists.txt` + an `add_test(...)` line.
- **VisRTX is the reference device** (`anari::loadLibrary("visrtx")`, `anari::newDevice(lib,"default")`), confirmed working in this GPU sandbox. First render triggers OptiX module compilation (seconds) — render tests get a generous timeout (set `TIMEOUT 300` on those `add_test`s via `set_tests_properties`).

## File structure

| File | Change |
|------|--------|
| `tsd/src/tsd/graph/Renderable.hpp` | NEW (core): `RenderableParams`, `Renderable` descriptor |
| `tsd/src/tsd/rendering/bridge/GraphRenderBridge.hpp` | NEW: bridge interface |
| `tsd/src/tsd/rendering/bridge/GraphRenderBridge.cpp` | NEW: bridge impl |
| `tsd/src/tsd/rendering/CMakeLists.txt` | Modify: add bridge source; link `tsd_graph` |
| `tsd/tests/test_graph_Renderable.cpp` | NEW: descriptor round-trips through a node |
| `tsd/tests/test_bridge_Mask.cpp` | NEW: `layersForViewport` membership (visrtx device, no render) |
| `tsd/tests/test_bridge_RenderSurface.cpp` | NEW: visrtx objectId smoke, masked surface |
| `tsd/tests/test_bridge_RenderVolume.cpp` | NEW: visrtx volume smoke + multi-viewport |
| `tsd/tests/CMakeLists.txt` | Modify: register tests; link `tsd_rendering` (already linked) |

`tsd/tests` already links `tsd_rendering` (and thus ANARI). The bridge lives in `tsd_rendering`, which will newly link `tsd_graph`.

---

## Task 1: `Renderable` descriptor (core, no ANARI)

**Files:**
- Create: `tsd/src/tsd/graph/Renderable.hpp`
- Test: `tsd/tests/test_graph_Renderable.cpp`

- [ ] **Step 1: Write the failing test** `tsd/tests/test_graph_Renderable.cpp`:
```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/graph/Evaluator.hpp"
#include "tsd/graph/Renderable.hpp"
// std
#include <memory>

using tsd::core::Token;
using tsd::graph::EvalContext;
using tsd::graph::Evaluator;
using tsd::graph::Graph;
using tsd::graph::Node;
using tsd::graph::NodeTypeInfo;
using tsd::graph::ParameterList;
using tsd::graph::PortType;
using tsd::graph::Renderable;
using tsd::graph::Value;
using tsd::graph::hostResidency;

namespace {

// Emits a "renderable" Value describing a sphere surface.
struct EmitSphere : Node
{
  ParameterList params;
  NodeTypeInfo typeInfo() const override
  {
    NodeTypeInfo i;
    i.name = Token("EmitSphere");
    i.outputs.push_back(
        {Token("out"), PortType{Token("renderable")}, true, {}});
    return i;
  }
  ParameterList &parameters() override { return params; }
  void evaluate(EvalContext &ctx) override
  {
    auto r = std::make_shared<Renderable>();
    r->kind = Renderable::Kind::Surface;
    r->primSubtype = Token("sphere");
    r->prim.scalars.push_back({Token("radius"), tsd::core::Any(0.5f)});
    Value v;
    v.type = PortType{Token("renderable")};
    v.residency = hostResidency();
    v.payload = r;
    ctx.setOutput(Token("out"), v);
  }
};

} // namespace

SCENARIO("tsd::graph::Renderable travels as a Value payload", "[graph-renderable]")
{
  Graph g;
  auto n = g.addNode(std::make_unique<EmitSphere>());
  Evaluator e(g);

  WHEN("pulling the emitter")
  {
    REQUIRE(e.pull(n));
    const Value *out = e.output(n, Token("out"), hostResidency());
    THEN("the renderable describes a sphere surface")
    {
      REQUIRE(out != nullptr);
      auto r = std::static_pointer_cast<Renderable>(out->payload);
      REQUIRE(r->kind == Renderable::Kind::Surface);
      REQUIRE(r->primSubtype == Token("sphere"));
      REQUIRE(r->prim.scalars.size() == 1);
      REQUIRE(r->prim.scalars[0].first == Token("radius"));
      REQUIRE(r->prim.scalars[0].second.get<float>() == 0.5f);
    }
  }
}
```
Register `add_test(NAME tsd::graph::Renderable COMMAND ${PROJECT_NAME} "[graph-renderable]")`.

- [ ] **Step 2: Build, confirm FAIL** (`tsd/graph/Renderable.hpp` not found).

- [ ] **Step 3: Create `tsd/src/tsd/graph/Renderable.hpp`:**
```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/Any.hpp"
#include "tsd/core/AnyArray.hpp"
#include "tsd/core/Token.hpp"
// std
#include <utility>
#include <vector>

namespace tsd::graph {

// Named scalar and array parameters, mapped 1:1 onto tsd Object::setParameter /
// setParameterObject by the render bridge. Backend-agnostic (host data only).
struct RenderableParams
{
  std::vector<std::pair<tsd::core::Token, tsd::core::Any>> scalars;
  std::vector<std::pair<tsd::core::Token, tsd::core::AnyArray>> arrays;
};

// A backend-agnostic description of one renderable thing. The bridge turns it
// into a tsd::Surface (geometry+material) or tsd::Volume (spatial field + TF).
struct Renderable
{
  enum class Kind
  {
    Surface,
    Volume
  };
  Kind kind{Kind::Surface};
  // Surface: geometry subtype (e.g. "sphere"); Volume: spatial-field subtype
  // (e.g. "structuredRegular").
  tsd::core::Token primSubtype;
  RenderableParams prim;       // geometry params, or spatial-field params
  RenderableParams appearance; // material params, or volume/TF params
};

} // namespace tsd::graph
```

- [ ] **Step 4: Build + run** `ctest ... -R 'tsd::graph::Renderable' --output-on-failure` → PASS.

- [ ] **Step 5: Commit**
```bash
clang-format -i tsd/src/tsd/graph/Renderable.hpp tsd/tests/test_graph_Renderable.cpp
jj commit tsd/src/tsd/graph/Renderable.hpp tsd/tests/test_graph_Renderable.cpp tsd/tests/CMakeLists.txt -m "feat(graph): add Renderable descriptor (surface/volume) carried as a Value"
```

---

## Task 2: `GraphRenderBridge` (render-scene, layers, mask, translation, update) + CMake

**Files:**
- Create: `tsd/src/tsd/rendering/bridge/GraphRenderBridge.hpp`, `GraphRenderBridge.cpp`
- Modify: `tsd/src/tsd/rendering/CMakeLists.txt` (add source; link `tsd_graph`)
- Test: `tsd/tests/test_bridge_Mask.cpp`

This is the core task. The bridge owns a render-scene, a per-display Layer + display
registration (mask+enabled), one `RenderIndexAllLayers` per viewport, and an
`update()` that (re)builds each display's layer from its `renderable` and refreshes
each viewport's included layers + ANARI world.

- [ ] **Step 1: Write the failing test** `tsd/tests/test_bridge_Mask.cpp` (mask→layers membership; constructs a real visrtx device but does NOT render):
```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/graph/Evaluator.hpp"
#include "tsd/graph/Renderable.hpp"
#include "tsd/rendering/bridge/GraphRenderBridge.hpp"
// anari
#include <anari/anari_cpp.hpp>
// std
#include <memory>

using tsd::core::Token;
using tsd::graph::EvalContext;
using tsd::graph::Evaluator;
using tsd::graph::Graph;
using tsd::graph::Node;
using tsd::graph::NodeTypeInfo;
using tsd::graph::ParameterList;
using tsd::graph::PortType;
using tsd::graph::Renderable;
using tsd::graph::Value;
using tsd::graph::hostResidency;
using tsd::rendering::GraphRenderBridge;

namespace {

struct EmitSphere : Node
{
  ParameterList params;
  NodeTypeInfo typeInfo() const override
  {
    NodeTypeInfo i;
    i.name = Token("EmitSphere");
    i.outputs.push_back(
        {Token("out"), PortType{Token("renderable")}, true, {}});
    return i;
  }
  ParameterList &parameters() override { return params; }
  void evaluate(EvalContext &ctx) override
  {
    auto r = std::make_shared<Renderable>();
    r->kind = Renderable::Kind::Surface;
    r->primSubtype = Token("sphere");
    r->prim.scalars.push_back({Token("radius"), tsd::core::Any(0.5f)});
    Value v;
    v.type = PortType{Token("renderable")};
    v.residency = hostResidency();
    v.payload = r;
    ctx.setOutput(Token("out"), v);
  }
};

anari::Device makeVisRTX()
{
  auto lib = anari::loadLibrary("visrtx", nullptr, nullptr);
  if (!lib)
    return nullptr;
  return anari::newDevice(lib, "default");
}

} // namespace

SCENARIO("tsd::rendering::GraphRenderBridge maps viewport masks to layers",
    "[bridge-mask]")
{
  anari::Device dev = makeVisRTX();
  REQUIRE(dev != nullptr); // VisRTX must be available in this environment

  Graph g;
  auto a = g.addNode(std::make_unique<EmitSphere>());
  auto b = g.addNode(std::make_unique<EmitSphere>());
  Evaluator e(g);

  GraphRenderBridge bridge(g, e, Token("visrtx"), dev, /*numViewports=*/3);

  // a -> viewports {0,1}; b -> viewport {2}
  bridge.setDisplay(a, /*mask=*/0b011, /*enabled=*/true);
  bridge.setDisplay(b, /*mask=*/0b100, /*enabled=*/true);
  bridge.update();

  WHEN("inspecting per-viewport layer membership")
  {
    THEN("each viewport includes exactly the masked displays' layers")
    {
      REQUIRE(bridge.layersForViewport(0).size() == 1); // a
      REQUIRE(bridge.layersForViewport(1).size() == 1); // a
      REQUIRE(bridge.layersForViewport(2).size() == 1); // b
    }
  }

  WHEN("disabling b and re-updating")
  {
    bridge.setDisplay(b, 0b100, false);
    bridge.update();
    THEN("viewport 2 has no layers; 0 and 1 still have a")
    {
      REQUIRE(bridge.layersForViewport(2).empty());
      REQUIRE(bridge.layersForViewport(0).size() == 1);
    }
  }

  WHEN("removing a")
  {
    bridge.removeDisplay(a);
    bridge.update();
    THEN("viewports 0 and 1 are empty")
    {
      REQUIRE(bridge.layersForViewport(0).empty());
      REQUIRE(bridge.layersForViewport(1).empty());
    }
  }

  anari::release(dev, dev);
}
```
Register `add_test(NAME tsd::rendering::BridgeMask COMMAND ${PROJECT_NAME} "[bridge-mask]")`.

- [ ] **Step 2: Build, confirm FAIL** (`GraphRenderBridge.hpp` not found).

- [ ] **Step 3: Create `tsd/src/tsd/rendering/bridge/GraphRenderBridge.hpp`:**
```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/graph/Evaluator.hpp"
#include "tsd/graph/Graph.hpp"
#include "tsd/graph/Renderable.hpp"
#include "tsd/rendering/index/RenderIndexAllLayers.hpp"
#include "tsd/scene/Scene.hpp"
// anari
#include <anari/anari_cpp.hpp>
// std
#include <cstdint>
#include <map>
#include <memory>
#include <vector>

namespace tsd::rendering {

// Renders evaluated graph `renderable` outputs into one ANARI world per viewport,
// selected by a per-display viewport mask. Owns a derived, single-writer render
// Scene (one Layer per display) and one RenderIndexAllLayers per viewport.
class GraphRenderBridge
{
 public:
  GraphRenderBridge(tsd::graph::Graph &graph,
      tsd::graph::Evaluator &eval,
      tsd::core::Token deviceName,
      anari::Device device,
      int numViewports);
  ~GraphRenderBridge();

  GraphRenderBridge(const GraphRenderBridge &) = delete;
  GraphRenderBridge &operator=(const GraphRenderBridge &) = delete;

  // Register/replace a display node's viewport mask + enabled flag.
  void setDisplay(tsd::graph::NodeId node, uint64_t viewportMask, bool enabled);
  void removeDisplay(tsd::graph::NodeId node);

  // The render-scene layers included by viewport i (display enabled && masked).
  std::vector<const tsd::scene::Layer *> layersForViewport(int i) const;

  // Pull enabled displays, (re)build their layers, refresh each viewport index.
  void update();

  anari::World world(int viewport) const;
  int numViewports() const { return int(m_indices.size()); }

 private:
  struct Display
  {
    uint64_t mask{0};
    bool enabled{true};
    tsd::scene::Layer *layer{nullptr}; // this display's layer in m_renderScene
    uint64_t lastVersion{0};           // producer outputVersion last realized
    bool realized{false};
  };

  void rebuildLayer(tsd::graph::NodeId node, Display &d);
  void buildSurface(tsd::scene::Layer *layer, const tsd::graph::Renderable &r);
  void buildVolume(tsd::scene::Layer *layer, const tsd::graph::Renderable &r);
  void applyParams(
      tsd::scene::Object &obj, const tsd::graph::RenderableParams &p);

  tsd::graph::Graph &m_graph;
  tsd::graph::Evaluator &m_eval;
  tsd::core::Token m_deviceName;
  anari::Device m_device;

  tsd::scene::Scene m_renderScene;
  std::map<tsd::graph::NodeId, Display> m_displays;
  std::vector<std::unique_ptr<RenderIndexAllLayers>> m_indices; // one per viewport
};

} // namespace tsd::rendering
```

> **Note for the implementer:** confirm the exact namespace of `Scene`/`Layer`
> (`tsd::scene::` per the headers) and `Object` (for `applyParams`). If
> `tsd::core::Object` is wrong, use the actual base type that has
> `setParameter`/`setParameterObject` (check `tsd/src/tsd/scene/Object.hpp`).
> `setParameterObject` takes an object/array ref; `setParameter` takes a scalar
> Any-convertible value.

- [ ] **Step 4: Create `tsd/src/tsd/rendering/bridge/GraphRenderBridge.cpp`:**
```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/rendering/bridge/GraphRenderBridge.hpp"
// tsd
#include "tsd/core/Logging.hpp"
#include "tsd/scene/objects/Geometry.hpp"
#include "tsd/scene/objects/Material.hpp"
#include "tsd/scene/objects/SpatialField.hpp"
#include "tsd/scene/objects/Surface.hpp"
#include "tsd/scene/objects/Volume.hpp"
// anari
#include <anari/anari_cpp/ext/linalg.h>
// std
#include <stdexcept>
#include <string>

namespace tsd::rendering {

using namespace tsd::scene;
using tsd::core::Token;
using tsd::graph::NodeId;
using tsd::graph::Renderable;
using tsd::graph::RenderableParams;
using float3 = anari::math::float3;

GraphRenderBridge::GraphRenderBridge(tsd::graph::Graph &graph,
    tsd::graph::Evaluator &eval,
    Token deviceName,
    anari::Device device,
    int numViewports)
    : m_graph(graph), m_eval(eval), m_deviceName(deviceName), m_device(device)
{
  if (!m_device)
    throw std::runtime_error("GraphRenderBridge: null ANARI device");
  if (numViewports < 1 || numViewports > 64)
    throw std::runtime_error("GraphRenderBridge: numViewports must be 1..64");
  for (int i = 0; i < numViewports; ++i) {
    m_indices.push_back(std::make_unique<RenderIndexAllLayers>(
        m_renderScene, m_deviceName, m_device));
  }
}

GraphRenderBridge::~GraphRenderBridge() = default;

void GraphRenderBridge::setDisplay(NodeId node, uint64_t mask, bool enabled)
{
  auto &d = m_displays[node];
  d.mask = mask;
  d.enabled = enabled;
  if (!d.layer) {
    const std::string name = "display_" + std::to_string(node);
    d.layer = m_renderScene.addLayer(Token(name.c_str()));
  }
}

void GraphRenderBridge::removeDisplay(NodeId node)
{
  auto it = m_displays.find(node);
  if (it == m_displays.end())
    return;
  // Clear the layer's content (leave the empty layer; simplest + safe).
  if (it->second.layer)
    m_renderScene.removeLayer(it->second.layer->name());
  m_displays.erase(it);
}

std::vector<const Layer *> GraphRenderBridge::layersForViewport(int i) const
{
  std::vector<const Layer *> out;
  if (i < 0 || i >= int(m_indices.size())) // i < numViewports (<= 64, see ctor)
    return out;
  const uint64_t bit = uint64_t(1) << i;
  for (const auto &kv : m_displays) {
    const Display &d = kv.second;
    if (d.enabled && d.realized && d.layer && (d.mask & bit))
      out.push_back(d.layer);
  }
  return out;
}

void GraphRenderBridge::update()
{
  // 1) (Re)build each enabled display's layer from its renderable.
  for (auto &kv : m_displays) {
    Display &d = kv.second;
    if (!d.enabled) {
      d.realized = false;
      continue;
    }
    rebuildLayer(kv.first, d);
  }

  // 2) Refresh each viewport's included layers + ANARI world.
  for (int i = 0; i < int(m_indices.size()); ++i) {
    m_indices[i]->setIncludedLayers(layersForViewport(i));
    m_indices[i]->populate();
  }
}

void GraphRenderBridge::rebuildLayer(NodeId node, Display &d)
{
  if (!m_eval.pull(node)) {
    d.realized = false;
    return;
  }
  const auto *out =
      m_eval.output(node, Token("out"), tsd::graph::hostResidency());
  if (!out || !out->payload || out->type.name != Token("renderable")) {
    d.realized = false;
    return;
  }

  // Skip rebuild if the producer's output is unchanged and already realized.
  // (Phase 2 `version` is the producer's monotonic outputVersion, bumped on each
  // recompute and stable across pulls — safe to compare.)
  if (d.realized && out->version == d.lastVersion)
    return;

  // Clear the layer's content IN PLACE — keeps the Layer* identity stable, so
  // pointers held in any viewport's included-layers list never dangle, and the
  // bridge (which constructs RenderIndexAllLayers directly, NOT via
  // updateDelegate().emplace) does not observe the scene live, so there is no
  // mid-rebuild delegate callback into freed state.
  d.layer->clear();

  auto r = std::static_pointer_cast<Renderable>(out->payload);
  if (r->kind == Renderable::Kind::Surface)
    buildSurface(d.layer, *r);
  else
    buildVolume(d.layer, *r);

  d.lastVersion = out->version;
  d.realized = true;
}

void GraphRenderBridge::applyParams(
    tsd::scene::Object &obj, const RenderableParams &p)
{
  for (const auto &s : p.scalars)
    obj.setParameter(s.first, s.second);
  for (const auto &a : p.arrays) {
    auto arr = m_renderScene.createArray(a.second.elementType(), a.second.size());
    arr->setData(a.second.data()); // Array::setData(const void*, byteOffset=0)
    obj.setParameterObject(a.first, *arr);
  }
}

void GraphRenderBridge::buildSurface(Layer *layer, const Renderable &r)
{
  auto geom = m_renderScene.createObject<Geometry>(r.primSubtype);
  applyParams(*geom, r.prim);

  auto mat = m_renderScene.createObject<Material>(tokens::material::matte);
  applyParams(*mat, r.appearance);

  auto surf = m_renderScene.createSurface("renderable", geom, mat);
  m_renderScene.insertChildObjectNode(layer->root(), surf);
}

void GraphRenderBridge::buildVolume(Layer *layer, const Renderable &r)
{
  auto field = m_renderScene.createObject<SpatialField>(r.primSubtype);

  // structuredRegular needs its `data` as a 3D array; the generic applyParams
  // path only makes 1D arrays. Read a `dims` scalar (float3) from prim and build
  // `data` as createArray(type, nx, ny, nz); apply the rest generically.
  if (r.primSubtype == Token("structuredRegular")) {
    float3 dims(0.f);
    for (const auto &s : r.prim.scalars)
      if (s.first == Token("dims"))
        dims = s.second.get<float3>();
    for (const auto &a : r.prim.arrays) {
      if (a.first == Token("data")) {
        auto arr = m_renderScene.createArray(a.second.elementType(),
            size_t(dims.x), size_t(dims.y), size_t(dims.z));
        arr->setData(a.second.data());
        field->setParameterObject(Token("data"), *arr);
      }
    }
    for (const auto &s : r.prim.scalars)
      if (s.first != Token("dims"))
        field->setParameter(s.first, s.second);
  } else {
    applyParams(*field, r.prim);
  }

  auto vol = m_renderScene.createObject<Volume>(tokens::volume::transferFunction1D);
  vol->setParameterObject("value", *field);
  applyParams(*vol, r.appearance); // e.g. "color" as a float4 array

  m_renderScene.insertChildObjectNode(layer->root(), vol);
}

anari::World GraphRenderBridge::world(int viewport) const
{
  return m_indices.at(viewport)->world();
}

} // namespace tsd::rendering
```

> **API points — RESOLVED against the real headers (already reflected in the code above):**
> 1. `Layer::clear()` exists (`scene/Layer.hpp:49`) — used to reset a display's layer in place, keeping `Layer*` identity stable. `Scene::removeLayer(Token)` (`Scene.hpp:182`) is used only in `removeDisplay`.
> 2. `Array::setData(const void *data, size_t byteOffset=0)` (`scene/objects/Array.hpp:78`) — call `arr->setData(a.second.data())` (single arg) to copy the whole buffer. `AnyArray::data()/size()/elementType()` confirmed.
> 3. `setParameter`/`setParameterObject` live on `tsd::scene::Object` (`scene/Object.hpp:241,125`) — the base of Geometry/Material/SpatialField/Volume. `applyParams` takes `tsd::scene::Object&`.
> 4. The bridge constructs `RenderIndexAllLayers` directly (NOT via `updateDelegate().emplace`), so it does NOT observe the scene live. `setIncludedLayers(...)` calls `signalActiveLayersChanged()`→`updateWorld()`; the explicit `populate()` in `update()` re-snapshots and is safe to call every time. Multiple indices over one Scene are independent (each owns its world/cache). Confirmed in `RenderIndex.cpp`/`RenderIndexAllLayers.cpp`.
> 5. `addLayer(Token)`→distinct `Layer*`; `layer->root()` is the insert parent (per `generate_randomSpheres`/`generate_noiseVolume`). `transferFunction1D` `color` is a **float4 array** via `setParameterObject`, `value`=field (per `Volume.cpp`/`generate_noiseVolume.cpp`).

- [ ] **Step 5: Edit `tsd/src/tsd/rendering/CMakeLists.txt`:** add `bridge/GraphRenderBridge.cpp` to `project_sources(PRIVATE ...)`, and add `tsd_graph` to `project_link_libraries(PUBLIC ...)` (alongside `tsd_scene tsd_algorithms`).

- [ ] **Step 6: Build + run** `ctest ... -R 'tsd::rendering::BridgeMask' --output-on-failure` → PASS (membership correct across enable/disable/remove). Set a generous timeout:
  in `tsd/tests/CMakeLists.txt` after the `add_test`, add
  `set_tests_properties(tsd::rendering::BridgeMask PROPERTIES TIMEOUT 300)`.

- [ ] **Step 7: Commit**
```bash
clang-format -i tsd/src/tsd/rendering/bridge/GraphRenderBridge.hpp tsd/src/tsd/rendering/bridge/GraphRenderBridge.cpp tsd/tests/test_bridge_Mask.cpp
jj commit tsd/src/tsd/rendering/bridge/GraphRenderBridge.hpp tsd/src/tsd/rendering/bridge/GraphRenderBridge.cpp tsd/src/tsd/rendering/CMakeLists.txt tsd/tests/test_bridge_Mask.cpp tsd/tests/CMakeLists.txt -m "feat(rendering): GraphRenderBridge (render-scene, per-viewport masks via RenderIndex layers)"
```

---

## Task 3: VisRTX surface render smoke (objectId AOV)

**Files:**
- Test: `tsd/tests/test_bridge_RenderSurface.cpp`

Render two viewports; a sphere display masked to viewport 0 only. Assert viewport 0
produces objectId hits and viewport 1 is empty.

- [ ] **Step 1: Write the test.** A small helper renders one viewport's world to an
objectId framebuffer and returns the count of non-zero pixels.

`tsd/tests/test_bridge_RenderSurface.cpp`:
```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/graph/Evaluator.hpp"
#include "tsd/graph/Renderable.hpp"
#include "tsd/rendering/bridge/GraphRenderBridge.hpp"
// anari
#include <anari/anari_cpp.hpp>
#include <anari/anari_cpp/ext/linalg.h>
// std
#include <memory>
#include <vector>

using tsd::core::Token;
using namespace tsd::graph;
using tsd::rendering::GraphRenderBridge;
using uint2 = anari::math::uint2;
using float3 = anari::math::float3;

namespace {

struct EmitSphere : Node
{
  ParameterList params;
  NodeTypeInfo typeInfo() const override
  {
    NodeTypeInfo i;
    i.name = Token("EmitSphere");
    i.outputs.push_back({Token("out"), PortType{Token("renderable")}, true, {}});
    return i;
  }
  ParameterList &parameters() override { return params; }
  void evaluate(EvalContext &ctx) override
  {
    auto r = std::make_shared<Renderable>();
    r->kind = Renderable::Kind::Surface;
    r->primSubtype = Token("sphere");
    // single sphere at the origin via a one-element vertex.position array
    tsd::core::AnyArray pos(ANARI_FLOAT32_VEC3, 1);
    pos.get<float3>(0) = float3(0.f, 0.f, 0.f);
    r->prim.arrays.push_back({Token("vertex.position"), pos});
    r->prim.scalars.push_back({Token("radius"), tsd::core::Any(0.5f)});
    r->appearance.scalars.push_back(
        {Token("color"), tsd::core::Any(float3(0.8f, 0.2f, 0.2f))});
    Value v;
    v.type = PortType{Token("renderable")};
    v.residency = hostResidency();
    v.payload = r;
    ctx.setOutput(Token("out"), v);
  }
};

anari::Device makeVisRTX()
{
  auto lib = anari::loadLibrary("visrtx", nullptr, nullptr);
  return lib ? anari::newDevice(lib, "default") : nullptr;
}

// Render `world` with a camera framing the origin; return # non-zero objectId px.
size_t countObjectIdHits(anari::Device d, anari::World world)
{
  auto cam = anari::newObject<anari::Camera>(d, "perspective");
  anari::setParameter(d, cam, "aspect", 1.f);
  anari::setParameter(d, cam, "position", float3(0.f, 0.f, 3.f));
  anari::setParameter(d, cam, "direction", float3(0.f, 0.f, -1.f));
  anari::setParameter(d, cam, "up", float3(0.f, 1.f, 0.f));
  anari::commitParameters(d, cam);

  auto rnd = anari::newObject<anari::Renderer>(d, "default");
  anari::commitParameters(d, rnd);

  auto frame = anari::newObject<anari::Frame>(d);
  uint2 sz{64, 64};
  anari::setParameter(d, frame, "size", sz);
  anari::setParameter(d, frame, "channel.color", ANARI_UFIXED8_RGBA_SRGB);
  anari::setParameter(d, frame, "channel.objectId", ANARI_UINT32);
  anari::setParameter(d, frame, "world", world);
  anari::setParameter(d, frame, "camera", cam);
  anari::setParameter(d, frame, "renderer", rnd);
  anari::commitParameters(d, frame);

  anari::render(d, frame);
  anari::wait(d, frame);

  size_t hits = 0;
  auto fb = anari::map<uint32_t>(d, frame, "channel.objectId");
  if (fb.data) {
    for (uint32_t i = 0; i < fb.width * fb.height; ++i)
      if (fb.data[i] != 0u)
        ++hits;
  }
  anari::unmap(d, frame, "channel.objectId");

  anari::release(d, frame);
  anari::release(d, rnd);
  anari::release(d, cam);
  return hits;
}

} // namespace

SCENARIO("GraphRenderBridge renders a masked surface into its viewport only",
    "[bridge-render-surface]")
{
  anari::Device dev = makeVisRTX();
  REQUIRE(dev != nullptr);

  Graph g;
  auto sphere = g.addNode(std::make_unique<EmitSphere>());
  Evaluator e(g);

  GraphRenderBridge bridge(g, e, Token("visrtx"), dev, /*numViewports=*/2);
  bridge.setDisplay(sphere, /*mask=*/0b01, /*enabled=*/true); // viewport 0 only
  bridge.update();

  WHEN("rendering both viewports")
  {
    size_t hits0 = countObjectIdHits(dev, bridge.world(0));
    size_t hits1 = countObjectIdHits(dev, bridge.world(1));
    THEN("viewport 0 shows the sphere; viewport 1 is empty")
    {
      REQUIRE(hits0 > 0);
      REQUIRE(hits1 == 0);
    }
  }

  WHEN("the mask is swapped to viewport 1")
  {
    // Control: proves viewport 1 is not structurally empty — the same sphere
    // masked to vp1 now renders there and not in vp0 (the hit counts swap).
    bridge.setDisplay(sphere, /*mask=*/0b10, /*enabled=*/true);
    bridge.update();
    THEN("the hit counts swap")
    {
      REQUIRE(countObjectIdHits(dev, bridge.world(0)) == 0);
      REQUIRE(countObjectIdHits(dev, bridge.world(1)) > 0);
    }
  }

  anari::release(dev, dev);
}
```
Register `add_test(NAME tsd::rendering::BridgeRenderSurface COMMAND ${PROJECT_NAME} "[bridge-render-surface]")` and `set_tests_properties(tsd::rendering::BridgeRenderSurface PROPERTIES TIMEOUT 300)`.

- [ ] **Step 2: Build + run** `ctest ... -R 'tsd::rendering::BridgeRenderSurface' --output-on-failure` → PASS.
  - If `hits0 == 0`: the sphere isn't in frame or objectId isn't populated — check the camera frames the origin (sphere radius 0.5 at origin, camera at z=3 looking -z) and that the device populates `channel.objectId` (it does for surfaces). If objectId is unavailable, fall back to asserting a non-background `channel.color` (some pixel != the clear color). Report whichever channel was used.
  - If `vertex.position` array handling fails, verify the `AnyArray` element-type/`get<float3>` usage and the bridge's array translation (Task 2 verification point 2).

- [ ] **Step 3: Commit**
```bash
clang-format -i tsd/tests/test_bridge_RenderSurface.cpp
jj commit tsd/tests/test_bridge_RenderSurface.cpp tsd/tests/CMakeLists.txt -m "test(rendering): visrtx objectId smoke — masked surface renders in its viewport only"
```

---

## Task 4: VisRTX volume render smoke + multi-viewport sharing

**Files:**
- Test: `tsd/tests/test_bridge_RenderVolume.cpp`

- [ ] **Step 1: Write the test.** One display emits a small `structuredRegular`
volume; a second emits a sphere. Mask the volume to viewports {0,1} and the sphere
to {1}. Assert viewport 0 renders something (volume only), viewport 1 renders more
(volume + sphere). Use a hit/non-background count helper (reuse the color channel for
volumes, since objectId may be surface-only).

`tsd/tests/test_bridge_RenderVolume.cpp`:
```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/graph/Evaluator.hpp"
#include "tsd/graph/Renderable.hpp"
#include "tsd/rendering/bridge/GraphRenderBridge.hpp"
// anari
#include <anari/anari_cpp.hpp>
#include <anari/anari_cpp/ext/linalg.h>
// std
#include <memory>
#include <vector>

using tsd::core::Token;
using namespace tsd::graph;
using tsd::rendering::GraphRenderBridge;
using uint2 = anari::math::uint2;
using float2 = anari::math::float2;
using float3 = anari::math::float3;
using float4 = anari::math::float4;

namespace {

// Minimal sphere emitter (also defined in the surface test; kept local here).
struct EmitSphere : Node
{
  ParameterList params;
  NodeTypeInfo typeInfo() const override
  {
    NodeTypeInfo i;
    i.name = Token("EmitSphere");
    i.outputs.push_back({Token("out"), PortType{Token("renderable")}, true, {}});
    return i;
  }
  ParameterList &parameters() override { return params; }
  void evaluate(EvalContext &ctx) override
  {
    auto r = std::make_shared<Renderable>();
    r->kind = Renderable::Kind::Surface;
    r->primSubtype = Token("sphere");
    tsd::core::AnyArray pos(ANARI_FLOAT32_VEC3, 1);
    pos.get<float3>(0) = float3(0.f, 0.f, 0.f);
    r->prim.arrays.push_back({Token("vertex.position"), pos});
    r->prim.scalars.push_back({Token("radius"), tsd::core::Any(0.5f)});
    r->appearance.scalars.push_back(
        {Token("color"), tsd::core::Any(float3(0.2f, 0.8f, 0.2f))});
    Value v;
    v.type = PortType{Token("renderable")};
    v.residency = hostResidency();
    v.payload = r;
    ctx.setOutput(Token("out"), v);
  }
};

struct EmitVolume : Node
{
  ParameterList params;
  NodeTypeInfo typeInfo() const override
  {
    NodeTypeInfo i;
    i.name = Token("EmitVolume");
    i.outputs.push_back({Token("out"), PortType{Token("renderable")}, true, {}});
    return i;
  }
  ParameterList &parameters() override { return params; }
  void evaluate(EvalContext &ctx) override
  {
    auto r = std::make_shared<Renderable>();
    r->kind = Renderable::Kind::Volume;
    r->primSubtype = Token("structuredRegular");
    // 8^3 scalar field, ramp values 0..1
    const int N = 8;
    tsd::core::AnyArray data(ANARI_FLOAT32, size_t(N) * N * N);
    for (size_t i = 0; i < data.size(); ++i)
      data.get<float>(i) = float(i) / float(data.size());
    r->prim.arrays.push_back({Token("data"), data});
    r->prim.scalars.push_back({Token("dims"), tsd::core::Any(float3(N, N, N))});
    r->prim.scalars.push_back({Token("origin"), tsd::core::Any(float3(-1.f))});
    r->prim.scalars.push_back({Token("spacing"), tsd::core::Any(float3(2.f / N))});
    // transferFunction1D "color" is a float4 (RGBA) colormap array; alpha ramps
    // up so the volume is visibly non-background.
    tsd::core::AnyArray color(ANARI_FLOAT32_VEC4, 256);
    for (size_t i = 0; i < color.size(); ++i) {
      float t = float(i) / float(color.size());
      color.get<float4>(i) = float4(1.f - t, 0.f, t, t);
    }
    r->appearance.arrays.push_back({Token("color"), color});
    r->appearance.scalars.push_back(
        {Token("valueRange"), tsd::core::Any(float2(0.f, 1.f))});
    Value v;
    v.type = PortType{Token("renderable")};
    v.residency = hostResidency();
    v.payload = r;
    ctx.setOutput(Token("out"), v);
  }
};

anari::Device makeVisRTX()
{
  auto lib = anari::loadLibrary("visrtx", nullptr, nullptr);
  return lib ? anari::newDevice(lib, "default") : nullptr;
}

// Count non-background pixels in channel.color (alpha>0 or rgb!=0).
size_t countColorHits(anari::Device d, anari::World world)
{
  auto cam = anari::newObject<anari::Camera>(d, "perspective");
  anari::setParameter(d, cam, "aspect", 1.f);
  anari::setParameter(d, cam, "position", float3(0.f, 0.f, 3.f));
  anari::setParameter(d, cam, "direction", float3(0.f, 0.f, -1.f));
  anari::setParameter(d, cam, "up", float3(0.f, 1.f, 0.f));
  anari::commitParameters(d, cam);
  auto rnd = anari::newObject<anari::Renderer>(d, "default");
  anari::commitParameters(d, rnd);
  auto frame = anari::newObject<anari::Frame>(d);
  uint2 sz{64, 64};
  anari::setParameter(d, frame, "size", sz);
  anari::setParameter(d, frame, "channel.color", ANARI_UFIXED8_RGBA_SRGB);
  anari::setParameter(d, frame, "world", world);
  anari::setParameter(d, frame, "camera", cam);
  anari::setParameter(d, frame, "renderer", rnd);
  anari::commitParameters(d, frame);
  anari::render(d, frame);
  anari::wait(d, frame);
  size_t hits = 0;
  auto fb = anari::map<uint32_t>(d, frame, "channel.color");
  if (fb.data) {
    for (uint32_t i = 0; i < fb.width * fb.height; ++i)
      if ((fb.data[i] & 0x00ffffffu) != 0u) // any non-black RGB
        ++hits;
  }
  anari::unmap(d, frame, "channel.color");
  anari::release(d, frame);
  anari::release(d, rnd);
  anari::release(d, cam);
  return hits;
}

} // namespace

SCENARIO("GraphRenderBridge renders volumes and differentiates viewports by mask",
    "[bridge-render-volume]")
{
  anari::Device dev = makeVisRTX();
  REQUIRE(dev != nullptr);

  Graph g;
  auto vol = g.addNode(std::make_unique<EmitVolume>());
  auto sphere = g.addNode(std::make_unique<EmitSphere>());
  Evaluator e(g);

  GraphRenderBridge bridge(g, e, Token("visrtx"), dev, /*numViewports=*/2);
  bridge.setDisplay(vol, /*mask=*/0b11, /*enabled=*/true);    // both viewports
  bridge.setDisplay(sphere, /*mask=*/0b10, /*enabled=*/true); // viewport 1 only
  bridge.update();

  WHEN("rendering both viewports")
  {
    const size_t hits0 = countColorHits(dev, bridge.world(0)); // volume only
    const size_t hits1 = countColorHits(dev, bridge.world(1)); // volume + sphere
    THEN("both show the shared volume, and viewport 1 also shows the sphere")
    {
      REQUIRE(hits0 > 0);          // shared volume present in viewport 0
      REQUIRE(hits1 > 0);          // viewport 1 populated
      REQUIRE(hits1 > hits0);      // the masked-only sphere adds coverage in vp1
    }
  }

  anari::release(dev, dev);
}
```
Register `add_test(NAME tsd::rendering::BridgeRenderVolume COMMAND ${PROJECT_NAME} "[bridge-render-volume]")` + `set_tests_properties(... TIMEOUT 300)`.

- [ ] **Step 2: Build + run.** Expected: PASS. Volume gotchas to resolve if it fails:
  - **3D data array:** `structuredRegular` expects `data` as a 3D array
    (`ANARI_ARRAY3D`). The descriptor carries a flat `AnyArray` of N*N*N scalars;
    the bridge's `applyParams` array path creates a 1D `createArray(type, count)`.
    For a volume's `data`, the field needs 3D dims. **Implementer:** extend the
    bridge so a volume `data` array is created as 3D — either add explicit dims to
    the descriptor (e.g. a `dims` scalar param the bridge reads to call
    `createArray(type, nx, ny, nz)`) or special-case `data` in `buildVolume`. Pick
    the minimal correct approach and note it. (This is the one place the generic
    param mapping needs help.)
  - If the volume doesn't render, verify `valueRange`/`color`/`value` param names
    against `tsd/src/tsd/scene/objects/Volume.cpp` (color may need to be a float4
    array or a sampler — mirror `generate_noiseVolume.cpp` exactly).

- [ ] **Step 3: Commit**
```bash
clang-format -i tsd/tests/test_bridge_RenderVolume.cpp
jj commit tsd/tests/test_bridge_RenderVolume.cpp tsd/tests/CMakeLists.txt -m "test(rendering): visrtx volume smoke + multi-viewport display sharing"
```

---

## Task 5: Full-suite gate

- [ ] **Step 1: Build + run the entire graph + bridge suite**
```bash
cmake --build _out/_cmake --config RelWithDebInfo --target tsdTests
ctest --test-dir _out/_cmake/tsd/tests -C RelWithDebInfo -R 'tsd::graph|tsd::rendering::Bridge' --output-on-failure
```
Expected: all green — 20 Phase 1/2 graph tests + Renderable + BridgeMask + BridgeRenderSurface + BridgeRenderVolume (= 24). Report the summary line and confirm `.envrc` uncommitted (`jj status`).

- [ ] **Step 2:** Verification gate (no new commit unless a fix was needed).

---

## Phase 3 completion checklist

- [ ] `Renderable` descriptor (surface + volume) carried as a `Value`; `tsd_graph` stays core-only
- [ ] `GraphRenderBridge` in `tsd_rendering` (newly links `tsd_graph`); owns a derived render-scene
- [ ] One `Layer` per display; `setDisplay(node, mask, enabled)` / `removeDisplay`
- [ ] viewport mask → `RenderIndexAllLayers::setIncludedLayers` per viewport; `layersForViewport` verified
- [ ] `update()` pulls displays, rebuilds layers (surface + volume translation), refreshes worlds; version-skip when unchanged
- [ ] VisRTX objectId smoke: masked surface renders in its viewport only
- [ ] VisRTX volume smoke + multi-viewport display sharing
- [ ] full suite green (24)

## Out of scope (per spec)

Interactive ImGui viewport / camera manipulator / picking / AOV UI; the real node
catalog + CUDA residency at the device boundary; async bridge updates; Lua;
persistence. (Phases 4–5.)

## Self-review notes

- The bridge does **no hand-written ANARI object code** — it builds `tsd::Surface`/
  `tsd::Volume` and lets `RenderIndex` translate. Mask = `setIncludedLayers`.
- Biggest implementer risks, all flagged inline: (1) `Scene::removeLayer`/layer-clear
  API exact shape; (2) `Array::setData` raw-bytes vs typed; (3) `RenderIndex`
  `populate()` re-snapshot semantics on repeated `update()`; (4) `structuredRegular`
  needs a **3D** `data` array (the one spot the generic param mapping needs a
  special case); (5) volume TF param names — mirror `generate_noiseVolume.cpp`.
- Render assertions use the **objectId AOV** (surfaces) / non-background color
  (volumes) — robust vs lighting. Render tests carry `TIMEOUT 300` for OptiX warmup.
- All tests require VisRTX; each asserts `dev != nullptr` up front (guaranteed in
  this GPU sandbox).
