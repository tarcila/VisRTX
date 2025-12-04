// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Window.h"
// tsd
#include "tsd/core/FlatMap.hpp"

namespace tsd::ui::imgui {

struct ImportFileDialog;

struct LayerTree : public Window
{
  LayerTree(Application *app, const char *name = "Layers");
  void buildUI() override;

  void setEnableAddRemoveLayers(bool enable);

 private:
  void buildUI_layerHeader();
  void buildUI_tree();
  void buildUI_activateObjectSceneMenu();
  void buildUI_objectSceneMenu();
  void buildUI_newLayerSceneMenu();
  void buildUI_setActiveLayersSceneMenus();

  // Data //

  bool m_enableAddRemove{true};
  size_t m_hoveredNode{TSD_INVALID_INDEX};
  size_t m_menuNode{TSD_INVALID_INDEX};
  bool m_activeLayerMenuTriggered{false};
  bool m_editingNodeName{false};
  bool m_menuVisible{false};
  std::vector<int> m_needToTreePop;
  int m_layerIdx{0};
};

} // namespace tsd::ui::imgui
