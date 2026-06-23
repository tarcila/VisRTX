// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/ColorMapUtil.hpp"
#include "tsd/core/TSDMath.hpp"
#include "tsd/graph/ConversionRegistry.hpp"
#include "tsd/graph/Graph.hpp"
#include "tsd/graph/NodeRegistry.hpp"
// std
#include <string>
#include <vector>

namespace tsd::graph_nodes {

enum class LinkKind
{
  Direct,
  Conversion,
  Incompatible,
  Cycle
};

struct ConnectCheck
{
  LinkKind kind{LinkKind::Incompatible};
  std::string detail; // "from->to" for Conversion, else the reject reason
  bool ok() const
  {
    return kind == LinkKind::Direct || kind == LinkKind::Conversion;
  }
};

// UI-free editor logic over a Graph + NodeRegistry (+ optional
// ConversionRegistry). Every mutating op marks the graph dirty so the bridge
// re-renders on update().
class GraphEditModel
{
 public:
  GraphEditModel(tsd::graph::Graph &graph,
      tsd::graph::NodeRegistry &registry,
      const tsd::graph::ConversionRegistry *conversions);

  // Mutating ops.
  tsd::graph::NodeId addNode(tsd::core::Token type);
  void removeNode(tsd::graph::NodeId id);
  tsd::graph::LinkResult connect(tsd::graph::NodeId from,
      tsd::core::Token fromPort,
      tsd::graph::NodeId to,
      tsd::core::Token toPort);
  void disconnect(tsd::graph::ConnectionId id);

  // Non-mutating queries.
  ConnectCheck canConnect(tsd::graph::NodeId from,
      tsd::core::Token fromPort,
      tsd::graph::NodeId to,
      tsd::core::Token toPort) const;
  LinkKind classify(const tsd::graph::Connection &c) const;

  const std::vector<tsd::core::Token> &nodeCatalog() const;

  // Pure TF sampling (implemented in Task 3): control points -> RGBA colormap.
  // ColorPoint is {position, R, G, B}; OpacityPoint is {position, opacity}.
  static std::vector<tsd::core::math::float4> sampleColormap(
      const std::vector<tsd::core::ColorPoint> &colorPoints,
      const std::vector<tsd::core::OpacityPoint> &opacityPoints,
      int samples);

 private:
  // Resolve a node's declared PortType for a named output/input port.
  bool outputType(
      tsd::graph::NodeId, tsd::core::Token, tsd::graph::PortType &) const;
  bool inputType(
      tsd::graph::NodeId, tsd::core::Token, tsd::graph::PortType &) const;

  tsd::graph::Graph &m_graph;
  tsd::graph::NodeRegistry &m_registry;
  const tsd::graph::ConversionRegistry *m_conversions{nullptr};
  std::vector<tsd::core::Token> m_catalog; // cached NodeRegistry::types()
};

} // namespace tsd::graph_nodes
