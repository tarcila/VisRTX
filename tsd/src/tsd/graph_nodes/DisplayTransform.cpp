// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/graph_nodes/DisplayTransform.hpp"
#include "tsd/graph_nodes/TransformableNode.hpp"

namespace tsd::graph_nodes {

using tsd::graph::NodeId;

std::vector<DisplayTransform> collectDisplayTransforms(tsd::graph::Graph &g)
{
  std::vector<DisplayTransform> out;
  for (const NodeId id : g.nodeIds()) {
    auto *gn = g.node(id);
    if (!gn || !gn->impl)
      continue;
    if (auto *itf = dynamic_cast<ITransformableNode *>(gn->impl.get()))
      out.push_back({id, itf->transform()});
  }
  return out;
}

} // namespace tsd::graph_nodes
