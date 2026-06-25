// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/ui/imgui/windows/LayerDebug.hpp"
#include "tsd/scene/Object.hpp"
#include "tsd/ui/imgui/tsd_ui_imgui.h"
// imgui
#include "imgui.h"

namespace tsd::ui::imgui {

LayerDebug::LayerDebug(
    Application *app, tsd::rendering::GraphRenderBridge *bridge, const char *name)
    : Window(app, name), m_bridge(bridge)
{
  if (m_bridge) {
    m_tree = std::make_unique<LayerTree>(
        app, "Render Layers", &m_bridge->renderScene(), /*readOnly=*/true);
  }
}

void LayerDebug::buildUI()
{
  if (!m_bridge || !m_tree) {
    ImGui::TextDisabled("No bridge");
    return;
  }

  m_tree->buildUI(); // the real LayerTree, read-only, over the render scene

  ImGui::Separator();
  if (auto *o = m_tree->readOnlySelectedObject()) {
    ImGui::BeginDisabled(true);
    tsd::ui::buildUI_object(*o, m_bridge->renderScene(), /*useTable=*/true);
    ImGui::EndDisabled();
  } else {
    ImGui::TextDisabled("No object selected");
  }
}

} // namespace tsd::ui::imgui
