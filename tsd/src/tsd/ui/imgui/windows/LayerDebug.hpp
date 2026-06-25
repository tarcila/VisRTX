// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/rendering/bridge/GraphRenderBridge.hpp"
#include "tsd/ui/imgui/windows/Window.h"
// std
#include <cstddef>

namespace tsd::ui::imgui {

// Read-only, hidden-by-default debug view of the GraphRenderBridge's realized
// per-viewport layers, with a greyed param pane for the selected object.
struct LayerDebug : public Window
{
  LayerDebug(Application *app,
      tsd::rendering::GraphRenderBridge *bridge,
      const char *name = "Layer Debug");
  void buildUI() override;

 private:
  tsd::rendering::GraphRenderBridge *m_bridge{nullptr};
  // Selection key, re-resolved each frame (never a persisted pointer).
  int m_selViewport{-1};
  int m_selLayer{-1};
  std::size_t m_selNodeIndex{~std::size_t(0)};
};

} // namespace tsd::ui::imgui
