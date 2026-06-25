// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/graph/Graph.hpp"
#include "tsd/graph/NodeRegistry.hpp"
#include "tsd/graph_nodes/BuiltinNodes.hpp"
#include "tsd/graph_nodes/DemoGraph.hpp"
#include "tsd/graph_nodes/GraphLayout.hpp"
// std
#include <set>
#include <utility>

using tsd::core::Token;
using namespace tsd::graph;
using tsd::graph_nodes::computeLayeredLayout;
using tsd::graph_nodes::NodePlacement;

namespace {
// col of the (single) node whose typeInfo name matches `typeName`.
int colOfType(
    Graph &g, const std::vector<NodePlacement> &p, const char *typeName)
{
  for (const auto &np : p) {
    auto *gn = g.node(np.node);
    if (gn && gn->impl && gn->impl->typeInfo().name == Token(typeName))
      return np.col;
  }
  return -1;
}
} // namespace

SCENARIO("computeLayeredLayout lays the demo graph out by topological depth",
    "[graph-layout]")
{
  NodeRegistry reg;
  tsd::graph_nodes::registerBuiltinNodes(reg);
  Graph g;
  tsd::graph_nodes::buildVolumeSurfaceDemo(g, reg);

  auto placements = computeLayeredLayout(g);

  THEN("every node is placed exactly once")
  {
    REQUIRE(placements.size() == g.nodeIds().size());
    std::set<NodeId> ids;
    for (const auto &p : placements)
      ids.insert(p.node);
    REQUIRE(ids.size() == placements.size());
  }

  THEN("columns match topological depth")
  {
    REQUIRE(colOfType(g, placements, "GenerateNoiseVolume") == 0);
    REQUIRE(colOfType(g, placements, "ScalarRange") == 1);
    REQUIRE(colOfType(g, placements, "BoundingBox") == 1);
    REQUIRE(colOfType(g, placements, "TransferFunction") == 2);
    REQUIRE(colOfType(g, placements, "DisplaySurface") == 2);
    REQUIRE(colOfType(g, placements, "DisplayVolume") == 3);
  }

  THEN("no two nodes share the same (col,row)")
  {
    std::set<std::pair<int, int>> cells;
    for (const auto &p : placements)
      cells.insert({p.col, p.row});
    REQUIRE(cells.size() == placements.size());
  }

  THEN("every producer's column is strictly less than its consumer's")
  {
    auto colOf = [&](NodeId id) {
      for (const auto &p : placements)
        if (p.node == id)
          return p.col;
      return -1;
    };
    for (const auto &c : g.connections())
      REQUIRE(colOf(c.fromNode) < colOf(c.toNode));
  }

  THEN("rows within a column are 0..k-1 (determinism via nodeIds order)")
  {
    // Under TSD_GRAPH_NODES_HAVE_VISKORES the isosurface branch adds
    // IsosurfaceExtract (col 1) and its DisplaySurface (col 2).
    // Column 1: ScalarRange, BoundingBox, [IsosurfaceExtract] → rows {0,1[,2]}.
    std::set<int> col1rows;
    for (const auto &p : placements)
      if (p.col == 1)
        col1rows.insert(p.row);
#ifdef TSD_GRAPH_NODES_HAVE_VISKORES
    REQUIRE(col1rows == std::set<int>({0, 1, 2}));
#else
    REQUIRE(col1rows == std::set<int>({0, 1}));
#endif
  }
}
