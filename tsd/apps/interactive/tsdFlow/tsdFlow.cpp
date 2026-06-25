// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// tsd_ui_imgui
#include "tsd/ui/imgui/Application.h"
#include "tsd/ui/imgui/windows/GraphEditor.hpp"
#include "tsd/ui/imgui/windows/GraphViewport.hpp"
#include "tsd/ui/imgui/windows/LayerDebug.hpp"
#include "tsd/ui/imgui/windows/Inspector.hpp"
#include "tsd/ui/imgui/windows/Log.h"
#include "tsd/ui/imgui/windows/ViewportRail.hpp"
// tsd
#include "tsd/core/Logging.hpp"
#include "tsd/graph/Evaluator.hpp"
#include "tsd/graph/Graph.hpp"
#include "tsd/graph/NodeRegistry.hpp"
#include "tsd/graph_nodes/BuiltinNodes.hpp"
#include "tsd/graph_nodes/DemoGraph.hpp"
#include "tsd/graph_nodes/DisplayMask.hpp"
#include "tsd/graph_nodes/DisplayTransform.hpp"
#include "tsd/graph_nodes/GraphEditModel.hpp"
#include "tsd/rendering/bridge/GraphRenderBridge.hpp"
// imgui
#include "imgui.h"
#include "imnodes.h"
// std
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <set>
#include <vector>

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

  void teardown() override
  {
    ImNodes::DestroyContext();
    ui::Application::teardown();
  }

  const char *getDefaultLayout() const override
  {
    // The dockspace/window IDs are fixed: ImGui::GetID("MainDockSpaceID")
    // inside the "MainDockSpace" window (created by the Application base)
    // always hashes to 0x80F5B4C5 / 0x079D3A04. Sizes are seed ratios — ImGui
    // rescales them to the live window — so Log's 262/1054 height yields ~1/4.
    // Layout: left column (420px) = Graph Editor (top) + Inspector (bottom);
    // center = 8-viewport tab group; bottom strip = Log.
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

[Window][Viewports]
Pos=0,26
Size=44,790
Collapsed=0
DockId=0x00000009,0

[Window][Log]
Pos=0,818
Size=1920,262
Collapsed=0
DockId=0x00000002,0

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
)layout";
  }

  ui::WindowArray setupWindows() override
  {
    ImNodes::CreateContext();

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

    // 3) Bridge: 8 viewports; masks come from each display node's viewportMask.
    m_bridge = std::make_unique<tsd::rendering::GraphRenderBridge>(m_graph,
        *m_eval,
        Token(lib.c_str()),
        m_device,
        /*numViewports=*/tsd::graph_nodes::kMaxViewports);
    syncDisplays(); // reads each display node's viewportMask → setDisplay
    m_bridge->update();

    // 4) Graph edit model.
    // No ConversionRegistry yet: no builtin node types declare convertible
    // cross-type ports, so every cross-type link is rejected as incompatible
    // and the amber "conversion" link feedback (implemented + unit-tested in
    // GraphEditModel) stays dormant until conversions are registered. Passing a
    // registry here changes nothing until then.
    m_model = std::make_unique<tsd::graph_nodes::GraphEditModel>(
        m_graph, m_registry, /*conversions=*/nullptr);

    // 5) Windows: graph editor, inspector, 8-viewport pool (1 visible), log.
    windows.emplace_back(new ui::GraphEditor(this,
        &m_graph,
        m_model.get(),
        &m_selected,
        &m_graphDirty,
        "Graph Editor"));
    windows.emplace_back(new ui::Inspector(
        this, &m_graph, &m_selected, &m_graphDirty, "Inspector"));
    static_assert(tsd::graph_nodes::kMaxViewports <= 99,
        "Viewport name buffer (nm[16]) and getDefaultLayout() INI assume <= 99 viewports");
    std::vector<ui::Window *> viewportPtrs;
    for (int i = 0; i < tsd::graph_nodes::kMaxViewports; ++i) {
      char nm[16];
      std::snprintf(nm, sizeof(nm), "Viewport %d", i + 1);
      auto *vp = new ui::GraphViewport(this,
          m_bridge.get(),
          i,
          m_device,
          &m_graph,
          &m_selected,
          &m_graphDirty,
          nm);
      if (i > 0)
        vp->hide();
      viewportPtrs.push_back(vp);
      windows.emplace_back(vp);
    }
    windows.emplace_back(new ui::ViewportRail(this, viewportPtrs, "Viewports"));
    auto *layerDebug = new ui::LayerDebug(this, m_bridge.get(), "Layer Debug");
    layerDebug->hide(); // debug tool: off by default; toggle via the View menu
    windows.emplace_back(layerDebug);
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
      uiMainMenuBar_View(); // "View" menu: per-window visibility toggles
      ImGui::EndMainMenuBar();
    }

    if (m_graphDirty) {
      syncDisplays();
      m_bridge->update();
      m_graphDirty = false;
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
    for (const auto &dt : tsd::graph_nodes::collectDisplayTransforms(m_graph))
      m_bridge->setDisplayTransform(dt.node, dt.xfm);
  }

  // Member declaration order is load-bearing — DO NOT REORDER. Reverse-order
  // destruction yields m_model -> m_bridge -> m_device -> m_eval -> m_graph,
  // which is required because m_bridge holds Graph&, Evaluator&, and the
  // device, and m_eval holds Graph&. The device handle is intentionally NOT
  // hand-released (releasing it in this dtor's body would run BEFORE m_bridge
  // destructs -> use-after-free); process/manager teardown reclaims it.
  tsd::graph::Graph m_graph;
  tsd::graph::NodeRegistry m_registry;
  std::unique_ptr<tsd::graph::Evaluator> m_eval;
  tsd::graph_nodes::DemoDisplays m_displays;
  anari::Device m_device{nullptr};
  std::unique_ptr<tsd::rendering::GraphRenderBridge> m_bridge;
  std::unique_ptr<tsd::graph_nodes::GraphEditModel> m_model;
  tsd::graph::NodeId m_selected{0}; // 0 == INVALID_NODE
  bool m_graphDirty{false};
  std::set<tsd::graph::NodeId> m_knownDisplays;
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
