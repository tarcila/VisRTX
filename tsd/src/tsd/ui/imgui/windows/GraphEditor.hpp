// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/ui/imgui/windows/Window.h"
// tsd
#include "tsd/graph/Graph.hpp"
#include "tsd/graph_nodes/GraphEditModel.hpp"
// std
#include <map>
#include <vector>

namespace tsd::ui::imgui {

// imnodes canvas over a Graph + GraphEditModel. Owns the int<->id maps imnodes
// needs (imnodes addresses everything by int; tsd uses NodeId/ConnectionId +
// (NodeId, port Token, direction) for pins).
struct GraphEditor : public Window
{
  GraphEditor(Application *app,
      tsd::graph::Graph *graph,
      tsd::graph_nodes::GraphEditModel *model,
      tsd::graph::NodeId *selected,
      bool *graphDirty,
      const char *name = "Graph Editor");

  void buildUI() override;

 private:
  struct PinKey
  {
    tsd::graph::NodeId node{0};
    tsd::core::Token port;
    bool isInput{false};
  };

  int pinId(tsd::graph::NodeId, tsd::core::Token port, bool isInput);
  void drawNode(tsd::graph::NodeId);
  void handleCreation();
  void handleDeletion();
  void contextMenu();

  tsd::graph::Graph *m_graph{nullptr};
  tsd::graph_nodes::GraphEditModel *m_model{nullptr};
  tsd::graph::NodeId *m_selected{nullptr};
  bool *m_graphDirty{nullptr};

  std::vector<PinKey> m_pins; // index+1 == imnodes pin id
  std::map<int, tsd::graph::ConnectionId>
      m_linkId; // imnodes link id -> ConnectionId
  bool m_placedInitial{false};
};

} // namespace tsd::ui::imgui
