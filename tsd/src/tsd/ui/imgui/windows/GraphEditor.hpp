// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/ui/imgui/windows/Window.h"
// tsd
#include "tsd/graph/Graph.hpp"
#include "tsd/graph_nodes/GraphEditModel.hpp"
// std
#include <map>
#include <set>
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
  bool isCollapsed(tsd::graph::NodeId) const;
  // Map a (possibly proxy) pin to a concrete port; false (logged) if the
  // collapsed node's direction has zero or multiple ports.
  bool resolvePort(const PinKey &pin, tsd::core::Token &outPort) const;
  void applyPendingPlacements();
  void applyAutoLayout();
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
  std::set<tsd::graph::NodeId> m_positioned; // nodes already given a position
  std::set<tsd::graph::NodeId> m_expanded;   // collapsed unless present here
  bool m_relayoutAll{false}; // "Clean Up Layout" request
  // Mouse-added nodes: placed next frame (inside the editor, before drawNode)
  // so the node is submitted that frame — calling SetNode*Pos on an
  // un-submitted node trips an imnodes EndNodeEditor assert.
  std::map<tsd::graph::NodeId, ImVec2> m_pendingScreenPos;
};

} // namespace tsd::ui::imgui
