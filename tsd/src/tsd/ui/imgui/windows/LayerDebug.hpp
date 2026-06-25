// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/rendering/bridge/GraphRenderBridge.hpp"
#include "tsd/ui/imgui/windows/LayerTree.h"
#include "tsd/ui/imgui/windows/Window.h"
// std
#include <memory>

namespace tsd::ui::imgui {

// Hidden-by-default debug panel: the real LayerTree, read-only, over the
// bridge's render scene, plus a greyed param pane for the selected object.
struct LayerDebug : public Window
{
  LayerDebug(Application *app,
      tsd::rendering::GraphRenderBridge *bridge,
      const char *name = "Layer Debug");
  void buildUI() override;

 private:
  tsd::rendering::GraphRenderBridge *m_bridge{nullptr};
  std::unique_ptr<LayerTree> m_tree;
};

} // namespace tsd::ui::imgui
