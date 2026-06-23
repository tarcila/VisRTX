// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/graph/Graph.hpp"
#include "tsd/graph_nodes/TransferFunctionNode.hpp"
#include "tsd/ui/imgui/windows/TFCurveEditor.hpp"
#include "tsd/ui/imgui/windows/Window.h"
// std
#include <memory>

namespace tsd::ui::imgui {

struct Inspector : public Window
{
  Inspector(Application *app,
      tsd::graph::Graph *graph,
      tsd::graph::NodeId *selected,
      bool *graphDirty,
      const char *name = "Inspector");

  void buildUI() override;

 private:
  void drawParameters(tsd::graph::NodeId);

  tsd::graph::Graph *m_graph{nullptr};
  tsd::graph::NodeId *m_selected{nullptr};
  bool *m_graphDirty{nullptr};
  std::unique_ptr<TFCurveEditor> m_tfEditor;
};

} // namespace tsd::ui::imgui
