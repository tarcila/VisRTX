# tsdFlow Phase 4c — Interactive App Shell & Viewports Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans, task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** A `tsdFlow` ImGui/SDL3 app that builds the Phase 4a demo graph (via a shared builder), drives it through the Phase 3 `GraphRenderBridge`, and shows it in two camera-controlled `GraphViewport` windows (volume + bounding-box surface, one per viewport mask).

**Architecture:** A shared core-only `buildVolumeSurfaceDemo` (in `tsd_graph_nodes`) — used by both the app and a headless Catch2 smoke test. A standalone `GraphViewport : tsd::ui::imgui::Window` (in `tsd_ui_imgui`) that owns its own ANARI camera/renderer/manipulator/`ImagePipeline` and renders `bridge.world(index)` via `AnariSceneRenderPass::setWorld`. A `tsdFlow` app wiring graph → bridge → two viewports.

**Tech Stack:** C++17, `tsd_ui_imgui` (Application/Window/ImagePipeline/AnariSceneRenderPass/CopyToSDLTexturePass/Manipulator), `tsd_graph` + `tsd_graph_nodes` (catalog), `tsd_rendering` `GraphRenderBridge` (Phase 3), ANARI + VisRTX, ImGui/SDL3, Catch2, jj.

## Global Constraints

- Version control is **jj**, not git. Commit ONLY a task's files with explicit paths: `jj commit <paths> -m "..."`. NEVER a bare `jj commit` (an unrelated `.envrc` must stay uncommitted).
- Build tree `_out/_cmake` (Ninja Multi-Config, RelWithDebInfo). No new `build/` dir.
  - Build tests: `cmake --build _out/_cmake --config RelWithDebInfo --target tsdTests`
  - Build the app: `cmake --build _out/_cmake --config RelWithDebInfo --target tsdFlow`
  - Run a test: `ctest --test-dir _out/_cmake/tsd/tests -C RelWithDebInfo -R '<name>' --output-on-failure`
- `clang-format -i` ONLY `.cpp`/`.hpp` — NEVER clang-format `CMakeLists.txt` (it mangles CMake; edit by hand).
- File header: `// Copyright 2026 NVIDIA Corporation` / `// SPDX-License-Identifier: Apache-2.0`; `#pragma once` for headers. Namespaces: catalog `tsd::graph_nodes`; UI window `tsd::ui::imgui`.
- VisRTX is the render device (GPU sandbox; first render = OptiX warmup; render tests get `set_tests_properties(<name> PROPERTIES TIMEOUT 300)`).
- The app is under the `TSD_BUILD_INTERACTIVE_APPS` CMake guard.
- 4c is host-residency only; no engine/bridge changes.

## File structure

| File | Responsibility |
|------|----------------|
| `tsd/src/tsd/graph_nodes/DemoGraph.hpp/.cpp` | `buildVolumeSurfaceDemo(Graph&, NodeRegistry&) -> DemoDisplays` (shared builder) |
| `tsd/tests/test_tsdflow_Smoke.cpp` | headless Catch2 smoke: builder → bridge → VisRTX render assertions |
| `tsd/src/tsd/ui/imgui/windows/GraphViewport.hpp/.cpp` | standalone viewport rendering `bridge.world(i)` |
| `tsd/src/tsd/ui/imgui/CMakeLists.txt` | add `GraphViewport.cpp` |
| `tsd/apps/interactive/tsdFlow/tsdFlow.cpp` | the app (Application subclass + main) |
| `tsd/apps/interactive/tsdFlow/CMakeLists.txt` | `tsdFlow` executable |
| `tsd/apps/interactive/CMakeLists.txt` | `add_subdirectory(tsdFlow)` |
| `tsd/src/tsd/graph_nodes/CMakeLists.txt`, `tsd/tests/CMakeLists.txt` | wiring |

---

## Task 1: shared `buildVolumeSurfaceDemo` builder + headless smoke test

**Files:**
- Create: `tsd/src/tsd/graph_nodes/DemoGraph.hpp`, `tsd/src/tsd/graph_nodes/DemoGraph.cpp`
- Modify: `tsd/src/tsd/graph_nodes/CMakeLists.txt`
- Test: `tsd/tests/test_tsdflow_Smoke.cpp`

**Interfaces:**
- Produces: `tsd::graph_nodes::DemoDisplays { tsd::graph::NodeId volumeDisplay, surfaceDisplay; }` and `DemoDisplays buildVolumeSurfaceDemo(tsd::graph::Graph &g, tsd::graph::NodeRegistry &reg)`. Builds the 4a demo, returns the `DisplayVolume`/`DisplaySurface` node ids.

- [ ] **Step 1: Write the failing test** `tsd/tests/test_tsdflow_Smoke.cpp`:
```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/graph/Evaluator.hpp"
#include "tsd/graph_nodes/BuiltinNodes.hpp"
#include "tsd/graph_nodes/DemoGraph.hpp"
#include "tsd/rendering/bridge/GraphRenderBridge.hpp"
// anari
#include <anari/anari_cpp.hpp>
#include <anari/anari_cpp/ext/linalg.h>
// std
#include <memory>

using tsd::core::Token;
using namespace tsd::graph;
using tsd::rendering::GraphRenderBridge;
using uint2 = anari::math::uint2;
using float3 = anari::math::float3;

namespace {

anari::Device makeVisRTX()
{
  auto lib = anari::loadLibrary("visrtx", nullptr, nullptr);
  return lib ? anari::newDevice(lib, "default") : nullptr;
}

struct Counts { size_t color{0}; size_t objectId{0}; };

Counts renderCounts(anari::Device d, anari::World world)
{
  auto cam = anari::newObject<anari::Camera>(d, "perspective");
  anari::setParameter(d, cam, "aspect", 1.f);
  anari::setParameter(d, cam, "position", float3(0.f, 0.f, 3.f));
  anari::setParameter(d, cam, "direction", float3(0.f, 0.f, -1.f));
  anari::setParameter(d, cam, "up", float3(0.f, 1.f, 0.f));
  anari::commitParameters(d, cam);
  auto rnd = anari::newObject<anari::Renderer>(d, "default");
  anari::setParameter(d, rnd, "ambientRadiance", 1.f);
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
  Counts c;
  auto col = anari::map<uint32_t>(d, frame, "channel.color");
  if (col.data)
    for (uint32_t i = 0; i < col.width * col.height; ++i)
      if ((col.data[i] & 0x00ffffffu) != 0u)
        ++c.color;
  anari::unmap(d, frame, "channel.color");
  auto oid = anari::map<uint32_t>(d, frame, "channel.objectId");
  if (oid.data)
    for (uint32_t i = 0; i < oid.width * oid.height; ++i)
      if (oid.data[i] != ~0u)
        ++c.objectId;
  anari::unmap(d, frame, "channel.objectId");
  anari::release(d, frame);
  anari::release(d, rnd);
  anari::release(d, cam);
  return c;
}

} // namespace

SCENARIO("the shared demo graph renders via the bridge", "[tsdflow-smoke]")
{
  anari::Device dev = makeVisRTX();
  REQUIRE(dev != nullptr);

  NodeRegistry reg;
  tsd::graph_nodes::registerBuiltinNodes(reg);
  Graph g;
  auto d = tsd::graph_nodes::buildVolumeSurfaceDemo(g, reg);
  Evaluator e(g);

  GraphRenderBridge bridge(g, e, Token("visrtx"), dev, /*numViewports=*/2);
  bridge.setDisplay(d.volumeDisplay, 0b01, true);
  bridge.setDisplay(d.surfaceDisplay, 0b10, true);
  bridge.update();

  WHEN("rendering both viewports")
  {
    Counts c0 = renderCounts(dev, bridge.world(0));
    Counts c1 = renderCounts(dev, bridge.world(1));
    THEN("vp0 has the volume (color) and vp1 the box surface (objectId)")
    {
      REQUIRE(c0.color > 0);
      REQUIRE(c1.objectId > 0);
    }
  }

  anari::release(dev, dev);
}
```
Register `add_test(NAME tsd::tsdflow::Smoke COMMAND ${PROJECT_NAME} "[tsdflow-smoke]")` + `set_tests_properties(tsd::tsdflow::Smoke PROPERTIES TIMEOUT 300)` (edit `tsd/tests/CMakeLists.txt` by hand; add the source to the executable list too).

- [ ] **Step 2: Build, confirm FAIL** (`tsd/graph_nodes/DemoGraph.hpp` not found).

- [ ] **Step 3: Create `tsd/src/tsd/graph_nodes/DemoGraph.hpp`:**
```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/graph/Graph.hpp"
#include "tsd/graph/NodeRegistry.hpp"

namespace tsd::graph_nodes {

struct DemoDisplays
{
  tsd::graph::NodeId source{0};         // GenerateNoiseVolume (for Regenerate)
  tsd::graph::NodeId volumeDisplay{0};
  tsd::graph::NodeId surfaceDisplay{0};
};

// Builds the 4a demo graph into `g` using node types from `reg` (the caller has
// already registerBuiltinNodes'd it). Returns the source + display node ids.
DemoDisplays buildVolumeSurfaceDemo(
    tsd::graph::Graph &g, tsd::graph::NodeRegistry &reg);

} // namespace tsd::graph_nodes
```

- [ ] **Step 4: Create `tsd/src/tsd/graph_nodes/DemoGraph.cpp`:**
```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/graph_nodes/DemoGraph.hpp"
// std
#include <cassert>
#include <memory>

namespace tsd::graph_nodes {

using tsd::core::Token;
using tsd::graph::NodeId;

DemoDisplays buildVolumeSurfaceDemo(
    tsd::graph::Graph &g, tsd::graph::NodeRegistry &reg)
{
  auto add = [&](const char *t) -> NodeId {
    return g.addNode(reg.create(Token(t)));
  };
  // Graph::connect returns LinkResult; a mistyped port must fail loudly, not
  // silently drop the edge (which would surface as an opaque render miss).
  auto link = [&](NodeId f, const char *fp, NodeId t, const char *tp) {
    const bool ok = g.connect(f, Token(fp), t, Token(tp)).ok;
    assert(ok && "buildVolumeSurfaceDemo: connect failed");
    (void)ok;
  };

  const NodeId src = add("GenerateNoiseVolume");
  const NodeId sr = add("ScalarRange");
  const NodeId tf = add("TransferFunction");
  const NodeId dv = add("DisplayVolume");
  const NodeId bb = add("BoundingBox");
  const NodeId ds = add("DisplaySurface");

  link(src, "out", sr, "in");
  link(sr, "out", tf, "in");
  link(src, "out", dv, "field"); // fan-out
  link(tf, "out", dv, "tf");      // multi-input
  link(src, "out", bb, "in");
  link(bb, "out", ds, "in");

  return DemoDisplays{src, dv, ds};
}

} // namespace tsd::graph_nodes
```
Add `DemoGraph.cpp` to `project_sources(PRIVATE ...)` in `tsd/src/tsd/graph_nodes/CMakeLists.txt` (edit by hand).

- [ ] **Step 5: Build + run** `ctest ... -R 'tsd::tsdflow::Smoke' --output-on-failure` → PASS (vp0 color>0, vp1 objectId>0). This reuses the proven Phase 3/4a render path; if a count is 0, see the 4a render-test troubleshooting (volume colormap alpha; objectId `~0u` sentinel).

- [ ] **Step 6: Commit.**
```bash
clang-format -i tsd/src/tsd/graph_nodes/DemoGraph.hpp tsd/src/tsd/graph_nodes/DemoGraph.cpp tsd/tests/test_tsdflow_Smoke.cpp
jj commit tsd/src/tsd/graph_nodes/DemoGraph.hpp tsd/src/tsd/graph_nodes/DemoGraph.cpp tsd/src/tsd/graph_nodes/CMakeLists.txt tsd/tests/test_tsdflow_Smoke.cpp tsd/tests/CMakeLists.txt -m "feat(graph_nodes): shared buildVolumeSurfaceDemo + headless render smoke"
```

---

## Task 2: `GraphViewport` standalone window (`tsd_ui_imgui`)

**Files:**
- Create: `tsd/src/tsd/ui/imgui/windows/GraphViewport.hpp`, `.cpp`
- Modify: `tsd/src/tsd/ui/imgui/CMakeLists.txt`

**Interfaces:**
- Consumes: `tsd::rendering::GraphRenderBridge::world(int)`, `Window`, `AnariSceneRenderPass`, `CopyToSDLTexturePass`, `ImagePipeline`, `Manipulator`, `updateCameraParametersPerspective`.
- Produces: `tsd::ui::imgui::GraphViewport(Application*, tsd::rendering::GraphRenderBridge*, int viewportIndex, anari::Device, const char *name)`.

No automated test (GUI). Deliverable: it compiles + links into `tsd_ui_imgui`; behavior is manually verified via the app in Task 3/4.

- [ ] **Step 1: Create `tsd/src/tsd/ui/imgui/windows/GraphViewport.hpp`:**
```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/ui/imgui/windows/Window.h"
#include "tsd/rendering/bridge/GraphRenderBridge.hpp"
#include "tsd/rendering/pipeline/ImagePipeline.h"
#include "tsd/rendering/pipeline/passes/AnariSceneRenderPass.h"
#include "tsd/rendering/pipeline/passes/CopyToSDLTexturePass.h"
#include "tsd/rendering/view/Manipulator.hpp"
// anari
#include <anari/anari_cpp.hpp>

namespace tsd::ui::imgui {

// Standalone viewport that renders one of a GraphRenderBridge's per-viewport
// ANARI worlds. Owns its own ANARI camera/renderer + manipulator + pipeline; it
// does NOT use BaseViewport (which is bound to TSD scene cameras/renderers).
struct GraphViewport : public Window
{
  GraphViewport(Application *app,
      tsd::rendering::GraphRenderBridge *bridge,
      int viewportIndex,
      anari::Device device,
      const char *name = "Viewport");
  ~GraphViewport() override;

  void buildUI() override;

 private:
  tsd::rendering::GraphRenderBridge *m_bridge{nullptr};
  int m_viewportIndex{0};
  anari::Device m_device{nullptr};
  anari::Camera m_camera{nullptr};
  anari::Renderer m_renderer{nullptr};
  tsd::rendering::Manipulator m_manip;
  tsd::rendering::UpdateToken m_manipToken{0};
  tsd::rendering::ImagePipeline m_pipeline;
  tsd::rendering::AnariSceneRenderPass *m_anariPass{nullptr};
  tsd::rendering::CopyToSDLTexturePass *m_outputPass{nullptr};
  tsd::math::int2 m_size{0, 0};
};

} // namespace tsd::ui::imgui
```

- [ ] **Step 2: Create `tsd/src/tsd/ui/imgui/windows/GraphViewport.cpp`:**
```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/ui/imgui/windows/GraphViewport.hpp"
#include "tsd/ui/imgui/Application.h"
#include "tsd/rendering/view/ManipulatorToAnari.hpp"
// imgui
#include "imgui.h"
// anari
#include <anari/anari_cpp/ext/linalg.h>

namespace tsd::ui::imgui {

using float3 = anari::math::float3;

GraphViewport::GraphViewport(Application *app,
    tsd::rendering::GraphRenderBridge *bridge,
    int viewportIndex,
    anari::Device device,
    const char *name)
    : Window(app, name),
      m_bridge(bridge),
      m_viewportIndex(viewportIndex),
      m_device(device)
{
  // Camera + renderer on the bridge's device.
  m_camera = anari::newObject<anari::Camera>(m_device, "perspective");
  anari::commitParameters(m_device, m_camera);
  m_renderer = anari::newObject<anari::Renderer>(m_device, "default");
  anari::setParameter(m_device, m_renderer, "ambientRadiance", 1.f);
  anari::commitParameters(m_device, m_renderer);

  // Frame the demo's content (field spans ~[-1,1]).
  m_manip.setConfig(float3(0.f, 0.f, 0.f), 3.f);

  // Pipeline: render the (external) world, then copy to an SDL texture.
  m_anariPass = m_pipeline.emplace_back<tsd::rendering::AnariSceneRenderPass>(m_device);
  m_anariPass->setCamera(m_camera);
  m_anariPass->setRenderer(m_renderer);
  m_anariPass->setRunAsync(false);
  m_outputPass =
      m_pipeline.emplace_back<tsd::rendering::CopyToSDLTexturePass>(
          m_app->sdlRenderer());
}

GraphViewport::~GraphViewport()
{
  if (m_camera)
    anari::release(m_device, m_camera);
  if (m_renderer)
    anari::release(m_device, m_renderer);
  // m_pipeline owns its passes; m_device is owned by the app/bridge.
}

void GraphViewport::buildUI()
{
  const ImVec2 avail = ImGui::GetContentRegionAvail();
  const tsd::math::int2 size{int(avail.x), int(avail.y)};
  if (size.x > 0 && size.y > 0 && (size.x != m_size.x || size.y != m_size.y)) {
    m_size = size;
    m_pipeline.setDimensions(uint32_t(size.x), uint32_t(size.y));
    anari::setParameter(
        m_device, m_camera, "aspect", float(size.x) / float(size.y));
    anari::commitParameters(m_device, m_camera);
  }
  if (m_size.x <= 0 || m_size.y <= 0)
    return;

  // Drive the camera from the current world + manipulator.
  if (m_manip.hasChanged(m_manipToken)) {
    tsd::rendering::updateCameraParametersPerspective(
        m_device, m_camera, m_manip);
    anari::commitParameters(m_device, m_camera);
  }
  m_anariPass->setWorld(m_bridge->world(m_viewportIndex));
  m_pipeline.render();

  // Blit + handle mouse navigation over the image.
  ImGui::Image((ImTextureID)m_outputPass->getTexture(),
      ImVec2(float(m_size.x), float(m_size.y)),
      ImVec2(0, 1),
      ImVec2(1, 0));

  if (ImGui::IsItemHovered()) {
    ImGuiIO &io = ImGui::GetIO();
    const float rotScale = 0.4f;
    if (ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
      auto del = ImGui::GetMouseDragDelta(ImGuiMouseButton_Left);
      m_manip.rotate(anari::math::float2(del.x * rotScale, del.y * rotScale));
      ImGui::ResetMouseDragDelta(ImGuiMouseButton_Left);
    } else if (ImGui::IsMouseDragging(ImGuiMouseButton_Right)) {
      auto del = ImGui::GetMouseDragDelta(ImGuiMouseButton_Right);
      m_manip.pan(anari::math::float2(del.x, del.y));
      ImGui::ResetMouseDragDelta(ImGuiMouseButton_Right);
    }
    if (io.MouseWheel != 0.f)
      m_manip.zoom(io.MouseWheel);
  }
}

} // namespace tsd::ui::imgui
```

- [ ] **Step 3: Add to CMake.** In `tsd/src/tsd/ui/imgui/CMakeLists.txt`, add `windows/GraphViewport.cpp` to the sources (edit by hand). Confirm `tsd_ui_imgui` already links `tsd_rendering` (it does — Viewport uses it); the bridge header is in `tsd_rendering`.

- [ ] **Step 4: Build** `cmake --build _out/_cmake --config RelWithDebInfo --target tsd_ui_imgui` → compiles + links. (No automated test; verified via the app.)
  - If `CopyToSDLTexturePass`/`AnariSceneRenderPass`/`Manipulator`/`ImagePipeline` include paths differ, fix per the real headers (paths: `tsd/rendering/pipeline/passes/…`, `tsd/rendering/view/Manipulator.hpp`, `tsd/rendering/view/ManipulatorToAnari.hpp`).
  - If `Window`'s ctor or `m_app->sdlRenderer()` differ, adjust per `Window.h`/`Application.h`.

- [ ] **Step 5: Commit.**
```bash
clang-format -i tsd/src/tsd/ui/imgui/windows/GraphViewport.hpp tsd/src/tsd/ui/imgui/windows/GraphViewport.cpp
jj commit tsd/src/tsd/ui/imgui/windows/GraphViewport.hpp tsd/src/tsd/ui/imgui/windows/GraphViewport.cpp tsd/src/tsd/ui/imgui/CMakeLists.txt -m "feat(ui): GraphViewport — standalone window rendering a GraphRenderBridge world"
```

---

## Task 3: `tsdFlow` app

**Files:**
- Create: `tsd/apps/interactive/tsdFlow/tsdFlow.cpp`, `tsd/apps/interactive/tsdFlow/CMakeLists.txt`
- Modify: `tsd/apps/interactive/CMakeLists.txt`

**Interfaces:**
- Consumes: `buildVolumeSurfaceDemo`, `registerBuiltinNodes`, `GraphRenderBridge`, `GraphViewport`, `Application`, `Log`.

No automated test (GUI). Deliverable: `tsdFlow` builds; manual launch shows two viewports.

- [ ] **Step 1: Create `tsd/apps/interactive/tsdFlow/tsdFlow.cpp`:**
```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// tsd_ui_imgui
#include "tsd/ui/imgui/Application.h"
#include "tsd/ui/imgui/windows/GraphViewport.hpp"
#include "tsd/ui/imgui/windows/Log.h"
// tsd
#include "tsd/core/Logging.hpp"
#include "tsd/graph/Evaluator.hpp"
#include "tsd/graph/Graph.hpp"
#include "tsd/graph/NodeRegistry.hpp"
#include "tsd/graph_nodes/BuiltinNodes.hpp"
#include "tsd/graph_nodes/DemoGraph.hpp"
#include "tsd/rendering/bridge/GraphRenderBridge.hpp"
// imgui
#include "imgui.h"
// std
#include <cstdlib>
#include <memory>

namespace tsd_flow {

namespace ui = tsd::ui::imgui;
using tsd::core::Token;

class Application : public ui::Application
{
 public:
  Application(int argc, const char *argv[]) : ui::Application(argc, argv) {}
  ~Application() override = default;

  const char *getDefaultLayout() const override
  {
    return R"layout(
[Window][MainDockSpace]
Pos=0,0
Size=1920,1080
Collapsed=0

[Docking][Data]
DockSpace   ID=0x782A6D04 Window=0x39AB4DB7 Pos=0,0 Size=1920,1080 Split=X
  DockNode  ID=0x00000001 Parent=0x782A6D04 SizeRef=960,1080 CentralNode=1 Selected=Volume
  DockNode  ID=0x00000002 Parent=0x782A6D04 SizeRef=958,1080 Selected=Surface
)layout";
  }

  ui::WindowArray setupWindows() override
  {
    auto windows = ui::Application::setupWindows();
    auto *ctx = appContext();

    // 1) Registry + demo graph.
    tsd::graph_nodes::registerBuiltinNodes(m_registry);
    m_displays = tsd::graph_nodes::buildVolumeSurfaceDemo(m_graph, m_registry);
    m_eval = std::make_unique<tsd::graph::Evaluator>(m_graph);

    // 2) Device (default visrtx; CLI/TSD_ANARI_LIBRARIES picks the first listed).
    const std::string lib = ctx->anari.libraryList().empty()
        ? std::string("visrtx")
        : ctx->anari.libraryList().front();
    m_device = ctx->anari.loadDevice(lib);
    if (!m_device) {
      tsd::core::logError("[tsdFlow] failed to load ANARI device '%s'; exiting",
          lib.c_str());
      std::exit(1);
    }

    // 3) Bridge: 2 viewports; volume->vp0, surface->vp1.
    m_bridge = std::make_unique<tsd::rendering::GraphRenderBridge>(
        m_graph, *m_eval, Token(lib.c_str()), m_device, /*numViewports=*/2);
    m_bridge->setDisplay(m_displays.volumeDisplay, 0b01, true);
    m_bridge->setDisplay(m_displays.surfaceDisplay, 0b10, true);
    m_bridge->update();

    // 4) Windows: two viewports + a log.
    windows.emplace_back(
        new ui::GraphViewport(this, m_bridge.get(), 0, m_device, "Volume"));
    windows.emplace_back(
        new ui::GraphViewport(this, m_bridge.get(), 1, m_device, "Surface"));
    windows.emplace_back(new ui::Log(this));

    setWindowArray(windows);
    return windows;
  }

  void uiFrameStart() override
  {
    // Minimal "Regenerate" control: bump the noise seed and re-evaluate/render.
    if (ImGui::BeginMainMenuBar()) {
      if (ImGui::MenuItem("Regenerate"))
        regenerate();
      ImGui::EndMainMenuBar();
    }
  }

 private:
  void regenerate()
  {
    auto *n = m_graph.node(m_displays.source);
    if (!n)
      return;
    n->impl->parameters().set(Token("seed"), ++m_seed);
    m_graph.markDirty(m_displays.source);
    m_bridge->update(); // synchronous: pulls the dirty chain before returning
  }

  // Member declaration order is load-bearing — DO NOT REORDER. Reverse-order
  // destruction yields m_bridge -> m_device -> m_eval -> m_graph, which is
  // required because m_bridge holds Graph&, Evaluator&, and the device, and
  // m_eval holds Graph&. The device handle is intentionally NOT hand-released
  // (releasing it in this dtor's body would run BEFORE m_bridge destructs ->
  // use-after-free); process/manager teardown reclaims it.
  tsd::graph::Graph m_graph;
  tsd::graph::NodeRegistry m_registry;
  std::unique_ptr<tsd::graph::Evaluator> m_eval;
  tsd::graph_nodes::DemoDisplays m_displays;
  anari::Device m_device{nullptr};
  std::unique_ptr<tsd::rendering::GraphRenderBridge> m_bridge;
  int m_seed{0};
};

} // namespace tsd_flow

int main(int argc, const char *argv[])
{
  {
    tsd_flow::Application app(argc, argv);
    app.run(1920, 1080, "tsdFlow");
  }
  return 0;
}
```

> **Implementer notes:**
> 1. `getDefaultLayout()` is pure-virtual on `Application` (verified). The layout INI above is a starting point — verify the two viewports dock side-by-side at launch; if not, adjust the dock IDs (cosmetic; the app still works with the default-docked fallback).
> 2. The source NodeId comes from `m_displays.source` (the builder returns it — Task 1). Do NOT hardcode an id; there is no `m_srcNodeId()` helper.
> 3. Do NOT override `applicationStateMetadata()` — it is NOT pure-virtual (it has a default impl); only `getDefaultLayout()` must be overridden. (Signatures `Application(int, const char**)`, `setWindowArray(const WindowArray&)`, `appContext()->anari.loadDevice(std::string)`, `libraryList()`, `Log(Application*, bool=true)` are all verified against the headers.)
> 4. Member declaration order is fixed and load-bearing (see the comment in the code) — do not reorder. The device handle is intentionally not hand-released.

- [ ] **Step 2: Create `tsd/apps/interactive/tsdFlow/CMakeLists.txt`:**
```cmake
project(tsdFlow)
project_add_executable(tsdFlow.cpp)
project_link_libraries(PUBLIC tsd_ui_imgui tsd_graph_nodes)
```

- [ ] **Step 3: Wire** `tsd/apps/interactive/CMakeLists.txt`: add `add_subdirectory(tsdFlow)` (edit by hand; keep alongside the other interactive apps, inside whatever guard they use — the parent already gates interactive apps via `TSD_BUILD_INTERACTIVE_APPS`).

- [ ] **Step 4: Build** `cmake --build _out/_cmake --config RelWithDebInfo --target tsdFlow` → links. Fix any header/signature mismatches per the notes (report changes).

- [ ] **Step 5: Commit.**
```bash
clang-format -i tsd/apps/interactive/tsdFlow/tsdFlow.cpp
jj commit tsd/apps/interactive/tsdFlow/tsdFlow.cpp tsd/apps/interactive/tsdFlow/CMakeLists.txt tsd/apps/interactive/CMakeLists.txt -m "feat(app): tsdFlow interactive app — demo graph in two bridge viewports"
```

---

## Task 4: build + suite gate + manual test checklist

- [ ] **Step 1: Build everything + run the suite.**
```bash
cmake --build _out/_cmake --config RelWithDebInfo --target tsdTests --parallel
cmake --build _out/_cmake --config RelWithDebInfo --target tsdFlow --parallel
ctest --test-dir _out/_cmake/tsd/tests -C RelWithDebInfo --output-on-failure
```
Expected: all green (prior 53 + `tsd::tsdflow::Smoke` = 54). Report the summary line; confirm `.envrc` uncommitted (`jj status`).

- [ ] **Step 2: Record the manual test checklist** (in the task report; GUI not CI-tested):
  - `tsdFlow` launches; two viewport windows ("Volume", "Surface") dock side-by-side.
  - "Volume" shows a colored volume blob; "Surface" shows a green bounding box.
  - Left-drag orbits, right-drag pans, wheel zooms — independently per viewport.
  - "Regenerate" changes the volume (re-render via `bridge.update()`).
  - The Log window shows startup messages; resize + clean close work.

- [ ] **Step 3:** Verification gate (no new commit unless a fix was needed).

---

## Phase 4c completion checklist

- [ ] Shared `buildVolumeSurfaceDemo` builder (core-only) used by app + smoke test
- [ ] Headless `tsd::tsdflow::Smoke` test passes (VisRTX; volume color + surface objectId)
- [ ] `GraphViewport` standalone window renders `bridge.world(i)` with mouse camera control
- [ ] `tsdFlow` app builds under `TSD_BUILD_INTERACTIVE_APPS`; two viewports + Log + Regenerate
- [ ] full suite green; `.envrc` uncommitted
- [ ] manual test checklist recorded

## Out of scope (per spec)

Node editor / inspector / TF editor (4d); in-app device switching (deferred); CUDA
(4b); undo/redo, persistence (4e/Phase 5); picking, AOV viz, multi-device compositing.

## Self-review notes

- The only automated test is the headless smoke (Task 1), which drives the SAME
  shared builder + bridge the app uses — so app graph-wiring is CI-guarded; the
  GUI (Tasks 2–3) is build-verified + manually checked (Task 4 checklist).
- `GraphViewport` is a standalone `Window` (not `BaseViewport`) — it owns its ANARI
  camera/renderer/manipulator/pipeline and renders the bridge's external world via
  `AnariSceneRenderPass::setWorld`, avoiding BaseViewport's scene-camera coupling.
- Engine/bridge unchanged. The app links `tsd_ui_imgui` + `tsd_graph_nodes`; the
  bridge comes transitively via `tsd_rendering` (which `tsd_ui_imgui` links).
- Risk areas flagged inline: the docking-layout string (cosmetic), the source-node
  id for Regenerate (prefer returning it from the builder), and a handful of
  Application/Window/device signatures to confirm against headers.