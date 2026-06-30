// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/ui/imgui/windows/GraphEditor.hpp"
#include "tsd/core/Logging.hpp"
#include "tsd/graph_nodes/GraphLayout.hpp"
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
// Display label for a category token: capitalize the first letter of each word
// ("source" -> "Source", "spatial field" -> "Spatial Field").
std::string titleCase(const char *s)
{
  std::string out(s ? s : "");
  bool atWordStart = true;
  for (char &c : out) {
    if (atWordStart && c >= 'a' && c <= 'z')
      c = char(c - 'a' + 'A');
    atWordStart = (c == ' ');
  }
  return out;
}
constexpr unsigned int kConversionColor = IM_COL32(204, 148, 81, 255); // amber
// Auto-layout grid pitch — tight, sized for compact (collapsed) nodes.
constexpr float kColW = 190.f;
constexpr float kRowH = 70.f;
// Shrink the whole editor a notch to fit more graph on screen (static, not a
// live zoom — imnodes v0.5 has no canvas scaling).
constexpr float kEditorFontScale = 0.85f;
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

bool GraphEditor::isCollapsed(NodeId id) const
{
  return m_expanded.find(id) == m_expanded.end();
}

void GraphEditor::drawNode(NodeId id)
{
  const auto *gn = m_graph->node(id);
  if (!gn || !gn->impl)
    return;
  const auto info = gn->impl->typeInfo();

  ImNodes::BeginNode(nodeImId(id));

  if (isCollapsed(id)) {
    const bool hasIn = !info.inputs.empty();
    const bool hasOut = !info.outputs.empty();
    // Each proxy attribute needs a real item: an empty ImGui group adopts
    // g.LastItemData.Rect.Max (the previously drawn node's right edge) as its
    // bounding box, which stretches the node across the canvas. A text-height
    // stub gives the group a small, well-defined rect and centers the pin.
    const ImVec2 pinStub(1.f, ImGui::GetTextLineHeight());
    if (hasIn) {
      ImNodes::BeginInputAttribute(
          pinId(id, Token("##in"), true), ImNodesPinShape_CircleFilled);
      ImGui::Dummy(pinStub);
      ImNodes::EndInputAttribute();
      ImGui::SameLine();
    }
    ImGui::TextUnformatted(info.name.c_str());
    if (hasOut) {
      ImGui::SameLine();
      ImNodes::BeginOutputAttribute(
          pinId(id, Token("##out"), false), ImNodesPinShape_TriangleFilled);
      ImGui::Dummy(pinStub);
      ImNodes::EndOutputAttribute();
    }
  } else {
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
  }

  ImNodes::EndNode();
}

bool GraphEditor::resolvePort(const PinKey &pin, Token &outPort) const
{
  const Token proxy = pin.isInput ? Token("##in") : Token("##out");
  if (pin.port != proxy) { // already a concrete port
    outPort = pin.port;
    return true;
  }
  const auto *gn = m_graph->node(pin.node);
  if (!gn || !gn->impl)
    return false;
  const auto info = gn->impl->typeInfo(); // by value — copy the Token out
  const auto &ports = pin.isInput ? info.inputs : info.outputs;
  if (ports.size() != 1) {
    tsd::core::logWarning("[GraphEditor] expand node to choose a port");
    return false;
  }
  outPort = ports.front().name;
  return true;
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

  Token outPort, inPort;
  if (!resolvePort(*outPin, outPort) || !resolvePort(*inPin, inPort))
    return; // ambiguous proxy on a multi-port collapsed node — expand first

  auto chk = m_model->canConnect(outPin->node, outPort, inPin->node, inPort);
  if (!chk.ok()) {
    tsd::core::logWarning(
        "[GraphEditor] link rejected: %s", chk.detail.c_str());
    return;
  }
  m_model->connect(outPin->node, outPort, inPin->node, inPort);
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
    for (int nid : sel)
      deleteNode(NodeId(nid));
  }
}

void GraphEditor::deleteNode(NodeId id)
{
  if (*m_selected == id)
    *m_selected = INVALID_NODE;
  m_model->removeNode(
      id); // removes the node + its connections, dirties consumers
  m_positioned.erase(id);
  m_expanded.erase(id);
  m_pendingScreenPos.erase(id);
  *m_graphDirty = true;
}

void GraphEditor::contextMenu()
{
  // Right-click a node → per-node menu; right-click elsewhere in the editor →
  // add menu. Runs after EndNodeEditor, where IsNodeHovered is valid;
  // IsEditorHovered is NOT valid here (it needs the live editor scope), so gate
  // on the window/child hover instead.
  if (ImGui::IsMouseClicked(ImGuiMouseButton_Right)
      && ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows)) {
    int hovered = -1;
    if (ImNodes::IsNodeHovered(&hovered)) {
      m_menuNode = NodeId(hovered);
      ImGui::OpenPopup("nodeMenu");
    } else {
      ImGui::OpenPopup("addNode");
    }
  }

  if (ImGui::BeginPopup("nodeMenu")) {
    if (ImGui::MenuItem("Delete Node") && m_menuNode != INVALID_NODE)
      deleteNode(m_menuNode);
    ImGui::EndPopup();
  }

  if (ImGui::BeginPopup("addNode")) {
    const ImVec2 clickPos = ImGui::GetMousePosOnOpeningCurrentPopup();
    auto addType = [&](const Token &type) {
      const NodeId id = m_model->addNode(type);
      if (id != INVALID_NODE) {
        // Defer placement to next frame: this node is added after the
        // drawNode loop, so it is not submitted to imnodes this frame.
        // Calling SetNodeScreenSpacePos now flags it InUse without a
        // submission index, which asserts in EndNodeEditor.
        m_pendingScreenPos[id] = clickPos;
        *m_graphDirty = true;
      }
    };
    for (const auto &cat : m_model->nodeCatalogByCategory()) {
      if (ImGui::BeginMenu(titleCase(cat.first.c_str()).c_str())) {
        for (const auto &type : cat.second)
          if (ImGui::MenuItem(type.c_str()))
            addType(type);
        ImGui::EndMenu();
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
    if (it != byId.end())
      ImNodes::SetNodeGridSpacePos(nodeImId(id),
          ImVec2(it->second->col * kColW, it->second->row * kRowH));
    m_positioned.insert(id); // always — never retry this id
  }
}

void GraphEditor::buildUI()
{
  ImGui::SetWindowFontScale(kEditorFontScale);
  ImNodes::GetStyle().NodePadding = ImVec2(6.f, 3.f); // tighter node interiors

  if (ImGui::Button("Clean Up Layout"))
    m_relayoutAll = true;

  ImNodes::BeginNodeEditor();

  applyPendingPlacements();
  applyAutoLayout();

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
    const int fromPin = isCollapsed(c.fromNode)
        ? pinId(c.fromNode, Token("##out"), false)
        : pinId(c.fromNode, c.fromPort, false);
    const int toPin = isCollapsed(c.toNode)
        ? pinId(c.toNode, Token("##in"), true)
        : pinId(c.toNode, c.toPort, true);
    ImNodes::Link(lid, fromPin, toPin);
    if (conv)
      ImNodes::PopColorStyle();
  }

  ImNodes::MiniMap();
  ImNodes::EndNodeEditor();

  // Context menus query ImNodes::IsNodeHovered/IsEditorHovered, which are only
  // valid AFTER EndNodeEditor (same reason the double-click handler below runs
  // here).
  contextMenu();

  // Double-click a node to toggle compact/expanded. IsNodeHovered returns the
  // topmost node under the cursor, so this targets the right node when stacked.
  int hoveredNode = 0;
  if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)
      && ImNodes::IsNodeHovered(&hoveredNode)) {
    const NodeId id = NodeId(hoveredNode);
    if (m_expanded.count(id))
      m_expanded.erase(id);
    else
      m_expanded.insert(id);
  }

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
