// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/TSDMath.hpp"
#include "tsd/graph/Graph.hpp"
// std
#include <vector>

namespace tsd::graph_nodes {

struct DisplayTransform
{
  tsd::graph::NodeId node{0};
  tsd::core::math::mat4 xfm{tsd::core::math::IDENTITY_MAT4};
};

// Every node implementing ITransformableNode and its transform. Graph&
// non-const (node()->impl is non-const for the dynamic_cast). Logically
// read-only.
std::vector<DisplayTransform> collectDisplayTransforms(tsd::graph::Graph &g);

} // namespace tsd::graph_nodes
