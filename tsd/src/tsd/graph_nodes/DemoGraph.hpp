// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/graph/Graph.hpp"
#include "tsd/graph/NodeRegistry.hpp"

namespace tsd::graph_nodes {

struct DemoDisplays
{
  tsd::graph::NodeId source{0}; // GenerateNoiseVolume (for Regenerate)
  tsd::graph::NodeId volumeDisplay{0};
  tsd::graph::NodeId surfaceDisplay{0};
};

// Builds the 4a demo graph into `g` using node types from `reg` (the caller has
// already registerBuiltinNodes'd it). Returns the source + display node ids.
DemoDisplays buildVolumeSurfaceDemo(
    tsd::graph::Graph &g, tsd::graph::NodeRegistry &reg);

} // namespace tsd::graph_nodes
