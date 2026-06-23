// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/graph/Graph.hpp"
// std
#include <cstdint>
#include <vector>

namespace tsd::graph_nodes {

constexpr int kMaxViewports = 8;
constexpr int kDefaultViewportMask = 0b01; // bit 0 → "Viewport 1"

struct DisplayMask
{
  tsd::graph::NodeId node{0};
  uint64_t mask{0};
};

// Every display node (DisplayVolume/DisplaySurface) and its viewport mask, read
// from the node's "viewportMask" param (kDefaultViewportMask if absent).
// Takes Graph& non-const: Graph::node() and Node::parameters() are non-const,
// though this is logically read-only.
std::vector<DisplayMask> collectDisplayMasks(tsd::graph::Graph &g);

} // namespace tsd::graph_nodes
