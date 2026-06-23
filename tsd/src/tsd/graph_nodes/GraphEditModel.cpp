// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/graph_nodes/GraphEditModel.hpp"

namespace tsd::graph_nodes {

using namespace tsd::graph;
using tsd::core::Token;

GraphEditModel::GraphEditModel(
    Graph &graph, NodeRegistry &registry, const ConversionRegistry *conversions)
    : m_graph(graph), m_registry(registry), m_conversions(conversions)
{
  m_catalog = m_registry.types();
}

NodeId GraphEditModel::addNode(Token type)
{
  auto node = m_registry.create(type);
  if (!node)
    return INVALID_NODE;
  const NodeId id = m_graph.addNode(std::move(node));
  m_graph.markDirty(id);
  return id;
}

void GraphEditModel::removeNode(NodeId id)
{
  m_graph.removeNode(id); // already dirties downstream consumers
}

LinkResult GraphEditModel::connect(
    NodeId from, Token fromPort, NodeId to, Token toPort)
{
  const LinkResult r = m_graph.connect(from, fromPort, to, toPort);
  if (r.ok)
    m_graph.markDirty(to);
  return r;
}

void GraphEditModel::disconnect(ConnectionId id)
{
  // Capture the consumer before removing so we can dirty it.
  NodeId consumer = INVALID_NODE;
  for (const auto &c : m_graph.connections())
    if (c.id == id) {
      consumer = c.toNode;
      break;
    }
  m_graph.disconnect(id);
  if (consumer != INVALID_NODE)
    m_graph.markDirty(consumer);
}

bool GraphEditModel::outputType(NodeId id, Token port, PortType &out) const
{
  const auto *gn = m_graph.node(id);
  if (!gn || !gn->impl)
    return false;
  for (const auto &p : gn->impl->typeInfo().outputs)
    if (p.name == port) {
      out = p.type;
      return true;
    }
  return false;
}

bool GraphEditModel::inputType(NodeId id, Token port, PortType &out) const
{
  const auto *gn = m_graph.node(id);
  if (!gn || !gn->impl)
    return false;
  for (const auto &p : gn->impl->typeInfo().inputs)
    if (p.name == port) {
      out = p.type;
      return true;
    }
  return false;
}

ConnectCheck GraphEditModel::canConnect(
    NodeId from, Token fromPort, NodeId to, Token toPort) const
{
  const LinkResult r = m_graph.canConnect(from, fromPort, to, toPort);
  if (!r.ok) {
    const bool cycle = r.reason.find("cycle") != std::string::npos;
    return {cycle ? LinkKind::Cycle : LinkKind::Incompatible, r.reason};
  }

  PortType o, i;
  if (outputType(from, fromPort, o) && inputType(to, toPort, i) && o != i) {
    std::string detail =
        std::string(o.name.c_str()) + "->" + std::string(i.name.c_str());
    return {LinkKind::Conversion, std::move(detail)};
  }
  return {LinkKind::Direct, ""};
}

LinkKind GraphEditModel::classify(const Connection &c) const
{
  PortType o, i;
  if (outputType(c.fromNode, c.fromPort, o) && inputType(c.toNode, c.toPort, i)
      && o != i)
    return LinkKind::Conversion;
  return LinkKind::Direct; // committed links are never Incompatible/Cycle
}

const std::vector<Token> &GraphEditModel::nodeCatalog() const
{
  return m_catalog;
}

} // namespace tsd::graph_nodes
