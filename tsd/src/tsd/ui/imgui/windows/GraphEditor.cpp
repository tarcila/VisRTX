// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/ui/imgui/windows/GraphEditor.hpp"
#include "tsd/core/Logging.hpp"
#include "tsd/ui/imgui/Application.h"
// imnodes
#include <imnodes.h>
// imgui
#include "imgui.h"

namespace tsd::ui::imgui {

using namespace tsd::graph;
using tsd::core::Token;

namespace {
int nodeImId(NodeId id)
{
  return int(id);
} // NodeId small in practice
constexpr unsigned int kConversionColor = IM_COL32(204, 148, 81, 255); // amber
constexpr float kColW = 360.f;
constexpr float kRowH = 170.f;
} // namespace

GraphEditor::GraphEditor(Application *app,
    Graph *graph,
    tsd::graph_nodes::GraphEditModel *model,
    NodeId *selected,
    bool *graphDirty,
    const char *name)
    : Window(app, name),
      m_graph(graph),
      m_model(model),
      m_selected(selected),
      m_graphDirty(graphDirty)
{}

int GraphEditor::pinId(NodeId node, Token port, bool isInput)
{
  for (size_t i = 0; i < m_pins.size(); ++i) {
    const auto &p = m_pins[i];
    if (p.node == node && p.port == port && p.isInput == isInput)
      return int(i) + 1; // 0 reserved
  }
  m_pins.push_back({node, port, isInput});
  return int(m_pins.size()); // size after push == index+1
}

void GraphEditor::drawNode(NodeId id)
{
  const auto *gn = m_graph->node(id);
  if (!gn || !gn->impl)
    return;
  const auto info = gn->impl->typeInfo();

  ImNodes::BeginNode(nodeImId(id));

  ImNodes::BeginNodeTitleBar();
  ImGui::TextUnformatted(info.name.c_str());
  ImNodes::EndNodeTitleBar();

  for (const auto &in : info.inputs) {
    ImNodes::BeginInputAttribute(
        pinId(id, in.name, true), ImNodesPinShape_CircleFilled);
    ImGui::TextUnformatted(in.name.c_str());
    ImNodes::EndInputAttribute();
  }
  for (const auto &out : info.outputs) {
    ImNodes::BeginOutputAttribute(
        pinId(id, out.name, false), ImNodesPinShape_TriangleFilled);
    ImGui::TextUnformatted(out.name.c_str());
    ImNodes::EndOutputAttribute();
  }

  ImNodes::EndNode();
}

void GraphEditor::handleCreation()
{
  int startAttr = 0, endAttr = 0;
  if (!ImNodes::IsLinkCreated(&startAttr, &endAttr))
    return;
  // Attr ids are pin ids (index+1). Output pin is the "from"; figure direction.
  auto resolve = [&](int attr) -> PinKey * {
    const int idx = attr - 1;
    return (idx >= 0 && idx < int(m_pins.size())) ? &m_pins[size_t(idx)]
                                                  : nullptr;
  };
  PinKey *a = resolve(startAttr), *b = resolve(endAttr);
  if (!a || !b)
    return;
  PinKey *outPin = a->isInput ? b : a;
  PinKey *inPin = a->isInput ? a : b;
  if (outPin->isInput || !inPin->isInput)
    return; // not an out->in pairing

  auto chk =
      m_model->canConnect(outPin->node, outPin->port, inPin->node, inPin->port);
  if (!chk.ok()) {
    tsd::core::logWarning(
        "[GraphEditor] link rejected: %s", chk.detail.c_str());
    return;
  }
  m_model->connect(outPin->node, outPin->port, inPin->node, inPin->port);
  *m_graphDirty = true;
}

void GraphEditor::handleDeletion()
{
  if (!ImGui::IsKeyPressed(ImGuiKey_Delete))
    return;

  const int nLinks = ImNodes::NumSelectedLinks();
  if (nLinks > 0) {
    std::vector<int> sel(static_cast<size_t>(nLinks));
    ImNodes::GetSelectedLinks(sel.data());
    for (int lid : sel) {
      auto it = m_linkId.find(lid);
      if (it != m_linkId.end()) {
        m_model->disconnect(it->second);
        *m_graphDirty = true;
      }
    }
  }
  const int nNodes = ImNodes::NumSelectedNodes();
  if (nNodes > 0) {
    std::vector<int> sel(static_cast<size_t>(nNodes));
    ImNodes::GetSelectedNodes(sel.data());
    for (int nid : sel) {
      const NodeId id = NodeId(nid);
      if (*m_selected == id)
        *m_selected = INVALID_NODE;
      m_model->removeNode(id);
      m_positioned.erase(id);
      *m_graphDirty = true;
    }
  }
}

void GraphEditor::contextMenu()
{
  if (ImNodes::IsEditorHovered()
      && ImGui::IsMouseClicked(ImGuiMouseButton_Right))
    ImGui::OpenPopup("addNode");
  if (ImGui::BeginPopup("addNode")) {
    const ImVec2 clickPos = ImGui::GetMousePosOnOpeningCurrentPopup();
    for (const auto &type : m_model->nodeCatalog()) {
      if (ImGui::MenuItem(type.c_str())) {
        const NodeId id = m_model->addNode(type);
        if (id != INVALID_NODE) {
          // Defer placement to next frame: this node is added after the
          // drawNode loop, so it is not submitted to imnodes this frame. Calling
          // SetNodeScreenSpacePos now flags it InUse without a submission index,
          // which asserts in EndNodeEditor.
          m_pendingScreenPos[id] = clickPos;
          *m_graphDirty = true;
        }
      }
    }
    ImGui::EndPopup();
  }
}

void GraphEditor::applyPendingPlacements()
{
  // Place mouse-added nodes from a previous frame. Runs inside the editor scope
  // before the drawNode loop, so each placed node is submitted this frame and
  // SetNodeScreenSpacePos never trips the EndNodeEditor submission assert.
  if (m_pendingScreenPos.empty())
    return;
  for (const auto &kv : m_pendingScreenPos) {
    if (!m_graph->node(kv.first))
      continue; // added then deleted before it was ever placed
    ImNodes::SetNodeScreenSpacePos(nodeImId(kv.first), kv.second);
    m_positioned.insert(kv.first);
  }
  m_pendingScreenPos.clear();
}

void GraphEditor::applyAutoLayout()
{
  std::vector<NodeId> targets;
  if (m_relayoutAll) {
    targets = m_graph->nodeIds();
    m_relayoutAll = false;
  } else {
    for (const NodeId id : m_graph->nodeIds())
      if (!m_positioned.count(id))
        targets.push_back(id);
    if (targets.empty())
      return; // nothing new — skip the layout work this frame
  }

  const auto placements = tsd::graph_nodes::computeLayeredLayout(*m_graph);
  std::map<NodeId, const tsd::graph_nodes::NodePlacement *> byId;
  for (const auto &p : placements)
    byId[p.node] = &p;

  for (const NodeId id : targets) {
    auto it = byId.find(id);
    if (it == byId.end())
      continue;
    ImNodes::SetNodeGridSpacePos(
        nodeImId(id), ImVec2(it->second->col * kColW, it->second->row * kRowH));
    m_positioned.insert(id);
  }
}

void GraphEditor::buildUI()
{
  if (ImGui::Button("Clean Up Layout"))
    m_relayoutAll = true;

  ImNodes::BeginNodeEditor();

  applyPendingPlacements();
  applyAutoLayout(); // positions un-positioned (programmatic) nodes, or all on
                     // request

  for (const NodeId id : m_graph->nodeIds())
    drawNode(id);

  // Links: assign a stable imnodes link id per ConnectionId and color
  // conversions.
  m_linkId.clear();
  int linkCounter = 1;
  for (const auto &c : m_graph->connections()) {
    const int lid = linkCounter++;
    m_linkId[lid] = c.id;
    const bool conv =
        m_model->classify(c) == tsd::graph_nodes::LinkKind::Conversion;
    if (conv)
      ImNodes::PushColorStyle(ImNodesCol_Link, kConversionColor);
    ImNodes::Link(lid,
        pinId(c.fromNode, c.fromPort, false),
        pinId(c.toNode, c.toPort, true));
    if (conv)
      ImNodes::PopColorStyle();
  }

  contextMenu();
  ImNodes::MiniMap();
  ImNodes::EndNodeEditor();

  // After EndNodeEditor: creation, deletion, selection.
  handleCreation();
  handleDeletion();

  if (ImNodes::NumSelectedNodes() == 1) {
    int sel = 0;
    ImNodes::GetSelectedNodes(&sel);
    const NodeId id = NodeId(sel);
    *m_selected = m_graph->node(id) ? id : INVALID_NODE;
  } else {
    *m_selected = INVALID_NODE;
  }
}

} // namespace tsd::ui::imgui
