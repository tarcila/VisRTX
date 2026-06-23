// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/graph_nodes/DisplayMask.hpp"

namespace tsd::graph_nodes {

using tsd::core::Token;
using tsd::graph::NodeId;

std::vector<DisplayMask> collectDisplayMasks(tsd::graph::Graph &g)
{
  std::vector<DisplayMask> out;
  for (const NodeId id : g.nodeIds()) {
    auto *gn = g.node(id);
    if (!gn || !gn->impl)
      continue;
    const auto info = gn->impl->typeInfo(); // bind temporary to a local
    if (info.name != Token("DisplayVolume")
        && info.name != Token("DisplaySurface"))
      continue;
    const int mask = gn->impl->parameters().getOr<int>(
        Token("viewportMask"), kDefaultViewportMask);
    out.push_back({id, uint64_t(mask)});
  }
  return out;
}

} // namespace tsd::graph_nodes
