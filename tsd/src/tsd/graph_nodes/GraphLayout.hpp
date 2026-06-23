// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/graph/Graph.hpp"
// std
#include <vector>

namespace tsd::graph_nodes {

struct NodePlacement
{
  tsd::graph::NodeId node{0};
  int col{0};
  int row{0};
};

// Layered DAG layout: col = longest-path depth from a source (no incoming
// connection), else max(col(producers)) + 1. Rows are 0,1,2,... per column in
// g.nodeIds() ascending order. Pure topology, no pixels. The engine guarantees
// acyclicity, so the memoized depth recursion terminates.
std::vector<NodePlacement> computeLayeredLayout(const tsd::graph::Graph &g);

} // namespace tsd::graph_nodes
