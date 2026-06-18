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

// Must match the Size= in getDefaultLayout()'s docking INI.
constexpr int kDefaultWindowWidth = 1920;
constexpr int kDefaultWindowHeight = 1080;

class Application : public ui::Application
{
 public:
  Application(int argc, const char *argv[]) : ui::Application(argc, argv) {}
  ~Application() override = default;

  const char *getDefaultLayout() const override
  {
    // The dockspace/window IDs are fixed: ImGui::GetID("MainDockSpaceID")
    // inside the "MainDockSpace" window (created by the Application base) always
    // hashes to 0x80F5B4C5 / 0x079D3A04. Sizes are seed ratios — ImGui rescales
    // them to the live window — so Log's 262/1054 height yields ~1/4.
    return R"layout(
[Window][MainDockSpace]
Pos=0,26
Size=1920,1054
Collapsed=0

[Window][Surface]
Pos=0,26
Size=959,790
Collapsed=0
DockId=0x00000003,0

[Window][Volume]
Pos=961,26
Size=959,790
Collapsed=0
DockId=0x00000004,0

[Window][Log]
Pos=0,818
Size=1920,262
Collapsed=0
DockId=0x00000002,0

[Docking][Data]
DockSpace    ID=0x80F5B4C5 Window=0x079D3A04 Pos=0,26 Size=1920,1054 Split=Y
  DockNode   ID=0x00000001 Parent=0x80F5B4C5 SizeRef=1920,790 Split=X
    DockNode ID=0x00000003 Parent=0x00000001 SizeRef=959,790
    DockNode ID=0x00000004 Parent=0x00000001 SizeRef=959,790
  DockNode   ID=0x00000002 Parent=0x80F5B4C5 SizeRef=1920,262
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

    // 2) Device (default visrtx; CLI/TSD_ANARI_LIBRARIES picks the first
    // listed).
    const std::string lib = ctx->anari.libraryList().empty()
        ? std::string("visrtx")
        : ctx->anari.libraryList().front();
    m_device = ctx->anari.loadDevice(lib);
    if (!m_device) {
      tsd::core::logError(
          "[tsdFlow] failed to load ANARI device '%s'; exiting", lib.c_str());
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
    m_bridge->update();
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
    app.run(tsd_flow::kDefaultWindowWidth,
        tsd_flow::kDefaultWindowHeight,
        "tsdFlow");
  }
  return 0;
}
