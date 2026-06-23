// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/graph_nodes/GraphLayout.hpp"
// std
#include <algorithm>
#include <functional>
#include <map>

namespace tsd::graph_nodes {

using tsd::graph::NodeId;

std::vector<NodePlacement> computeLayeredLayout(const tsd::graph::Graph &g)
{
  // Build producer adjacency: toNode <- [fromNode...].
  std::map<NodeId, std::vector<NodeId>> producers;
  for (const NodeId id : g.nodeIds())
    producers[id]; // ensure an entry for every node (incl. sources)
  for (const auto &c : g.connections())
    producers[c.toNode].push_back(c.fromNode);

  // Memoized longest-path depth (terminates: graph is acyclic by construction).
  std::map<NodeId, int> depth;
  std::function<int(NodeId)> col = [&](NodeId n) -> int {
    auto it = depth.find(n);
    if (it != depth.end())
      return it->second;
    int d = 0;
    for (const NodeId p : producers[n])
      d = std::max(d, col(p) + 1);
    depth[n] = d;
    return d;
  };

  // Assign rows per column in nodeIds() ascending order.
  std::map<int, int> nextRow;
  std::vector<NodePlacement> out;
  out.reserve(g.nodeIds().size());
  for (const NodeId id : g.nodeIds()) {
    const int c = col(id);
    out.push_back({id, c, nextRow[c]++});
  }
  return out;
}

} // namespace tsd::graph_nodes
