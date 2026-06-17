// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/graph/Graph.hpp"
// std
#include <functional>

namespace tsd::graph {

Graph::Graph(const ConversionRegistry *conversions) : m_conversions(conversions)
{}

void Graph::setConversionRegistry(const ConversionRegistry *r)
{
  m_conversions = r;
}

NodeId Graph::addNode(std::unique_ptr<Node> node)
{
  NodeId id = m_nextNodeId++;
  GraphNode gn;
  gn.id = id;
  gn.impl = std::move(node);
  gn.state = EvalState::Dirty;
  m_nodes.emplace(id, std::move(gn));
  return id;
}

GraphNode *Graph::node(NodeId id)
{
  auto it = m_nodes.find(id);
  return it == m_nodes.end() ? nullptr : &it->second;
}

const GraphNode *Graph::node(NodeId id) const
{
  auto it = m_nodes.find(id);
  return it == m_nodes.end() ? nullptr : &it->second;
}

const std::vector<Connection> &Graph::connections() const
{
  return m_connections;
}

bool Graph::findOutputSpec(
    const GraphNode &n, tsd::core::Token port, PortSpec &out) const
{
  auto info = n.impl->typeInfo();
  for (const auto &p : info.outputs) {
    if (p.name == port) {
      out = p;
      return true;
    }
  }
  return false;
}

bool Graph::findInputSpec(
    const GraphNode &n, tsd::core::Token port, PortSpec &out) const
{
  auto info = n.impl->typeInfo();
  for (const auto &p : info.inputs) {
    if (p.name == port) {
      out = p;
      return true;
    }
  }
  return false;
}

const Connection *Graph::inputConnection(
    NodeId to, tsd::core::Token toPort) const
{
  for (const auto &c : m_connections)
    if (c.toNode == to && c.toPort == toPort)
      return &c;
  return nullptr;
}

bool Graph::wouldCreateCycle(NodeId from, NodeId to) const
{
  // A new edge from->to creates a cycle iff `from` is reachable starting at
  // `to` along existing edges. Self-edge is the trivial case.
  if (from == to)
    return true;
  std::function<bool(NodeId)> reaches = [&](NodeId start) {
    for (const auto &c : m_connections) {
      if (c.fromNode == start) {
        if (c.toNode == from)
          return true;
        if (reaches(c.toNode))
          return true;
      }
    }
    return false;
  };
  return reaches(to);
}

LinkResult Graph::connect(
    NodeId from, tsd::core::Token fromPort, NodeId to, tsd::core::Token toPort)
{
  auto *fromN = node(from);
  auto *toN = node(to);
  if (!fromN || !toN)
    return {false, INVALID_CONNECTION, "unknown node"};

  PortSpec outSpec, inSpec;
  if (!findOutputSpec(*fromN, fromPort, outSpec))
    return {false, INVALID_CONNECTION, "no such output port"};
  if (!findInputSpec(*toN, toPort, inSpec))
    return {false, INVALID_CONNECTION, "no such input port"};

  if (wouldCreateCycle(from, to))
    return {false, INVALID_CONNECTION, "connection would create a cycle"};

  // Type compatibility: exact match, or a registered conversion exists.
  if (outSpec.type != inSpec.type) {
    const bool convertible = m_conversions
        && m_conversions->find(outSpec.type, inSpec.type) != nullptr;
    if (!convertible) {
      return {false,
          INVALID_CONNECTION,
          "incompatible port types and no registered conversion"};
    }
  }

  ConnectionId id = m_nextConnId++;
  m_connections.push_back(Connection{id, from, fromPort, to, toPort});

  // New incoming data invalidates the consumer's cached output.
  toN->state = EvalState::Dirty;
  toN->cache.clear();
  return {true, id, ""};
}

void Graph::disconnect(ConnectionId id)
{
  for (auto it = m_connections.begin(); it != m_connections.end(); ++it) {
    if (it->id == id) {
      if (auto *toN = node(it->toNode)) {
        toN->state = EvalState::Dirty;
        toN->cache.clear();
      }
      m_connections.erase(it);
      return;
    }
  }
}

void Graph::markDirty(NodeId id)
{
  auto *n = node(id);
  if (!n)
    return;
  if (n->state != EvalState::Dirty) {
    n->state = EvalState::Dirty;
    n->cache.clear();
  }
  for (const auto &c : m_connections) {
    if (c.fromNode == id)
      markDirty(c.toNode);
  }
}

void Graph::revalidateRequiredInputs(NodeId id)
{
  auto *n = node(id);
  if (!n)
    return;
  auto info = n->impl->typeInfo();
  for (const auto &port : info.inputs) {
    if (port.required && inputConnection(id, port.name) == nullptr) {
      n->state = EvalState::Error;
      n->error = "missing required input: " + port.name.str();
      return;
    }
  }
}

void Graph::removeNode(NodeId id)
{
  std::vector<NodeId> affectedConsumers;
  for (auto it = m_connections.begin(); it != m_connections.end();) {
    if (it->fromNode == id || it->toNode == id) {
      if (it->fromNode == id)
        affectedConsumers.push_back(it->toNode);
      it = m_connections.erase(it);
    } else {
      ++it;
    }
  }
  m_nodes.erase(id);
  for (NodeId c : affectedConsumers) {
    markDirty(c);
    revalidateRequiredInputs(c);
  }
}

} // namespace tsd::graph
