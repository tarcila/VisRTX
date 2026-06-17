// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/graph/ConversionRegistry.hpp"
#include "tsd/graph/Node.hpp"
#include "tsd/graph/Value.hpp"
// std
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace tsd::graph {

using NodeId = uint64_t;
using ConnectionId = uint64_t;

constexpr NodeId INVALID_NODE = 0;
constexpr ConnectionId INVALID_CONNECTION = 0;

enum class EvalState
{
  Clean,
  Dirty,
  Computing,
  Error
};

struct Connection
{
  ConnectionId id{INVALID_CONNECTION};
  NodeId fromNode{INVALID_NODE};
  tsd::core::Token fromPort;
  NodeId toNode{INVALID_NODE};
  tsd::core::Token toPort;
};

struct LinkResult
{
  bool ok{false};
  ConnectionId id{INVALID_CONNECTION};
  std::string reason;
};

// One output's cached results, keyed by full Residency (backend + deviceId) so
// a copy on device 0 never satisfies a device-1 consumer. Fan-out to consumers
// on different residencies each gets a correctly-resident copy.
using OutputCache = std::map<Residency, Value, ResidencyLess>;

struct GraphNode
{
  NodeId id{INVALID_NODE};
  std::unique_ptr<Node> impl;
  EvalState state{EvalState::Dirty};
  std::string error;
  uint64_t outputVersion{0}; // bumped on each recompute that changes outputs
  bool hasEvaluated{false};
  uint64_t lastParamHash{0};
  // input port -> producer outputVersion consumed at last evaluate
  std::map<tsd::core::Token, uint64_t, TokenLess> consumedInputVersions;
  // outputName -> (residency -> Value)
  std::map<tsd::core::Token, OutputCache, TokenLess> cache;
};

// Owns nodes and connections. Validates connections at link time (type compat
// via exact match or a registered conversion; cycle rejection). Residency
// mismatch is never a link error — it is resolved during evaluation.
class Graph
{
 public:
  explicit Graph(const ConversionRegistry *conversions = nullptr);

  NodeId addNode(std::unique_ptr<Node> node);
  void removeNode(NodeId id);

  LinkResult connect(NodeId from,
      tsd::core::Token fromPort,
      NodeId to,
      tsd::core::Token toPort);
  void disconnect(ConnectionId id);

  GraphNode *node(NodeId id);
  const GraphNode *node(NodeId id) const;
  const std::vector<Connection> &connections() const;

  // Connection feeding a given (node,input). Null if unconnected.
  const Connection *inputConnection(NodeId to, tsd::core::Token toPort) const;

  void setConversionRegistry(const ConversionRegistry *r);

  // Mark a node and all transitive downstream consumers Dirty.
  void markDirty(NodeId id);

 private:
  // After topology changes, set any node missing a required input to Error.
  void revalidateRequiredInputs(NodeId id);
  bool wouldCreateCycle(NodeId from, NodeId to) const;
  // Search by value (typeInfo() returns a temporary, so never return a pointer
  // into it). Returns true and fills `out` if the port exists.
  bool findOutputSpec(
      const GraphNode &n, tsd::core::Token port, PortSpec &out) const;
  bool findInputSpec(
      const GraphNode &n, tsd::core::Token port, PortSpec &out) const;

  std::map<NodeId, GraphNode> m_nodes;
  std::vector<Connection> m_connections;
  NodeId m_nextNodeId{1};
  ConnectionId m_nextConnId{1};
  const ConversionRegistry *m_conversions{nullptr};
};

} // namespace tsd::graph
