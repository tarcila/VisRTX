// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/TSDMath.hpp"

namespace tsd::graph_nodes {

// Implemented by nodes carrying a render-time instance transform. Kept OUT of
// the node's ParameterList (so it never enters ParameterList::hash() and a
// transform edit never triggers re-evaluation / layer rebuild). UI reaches it
// via dynamic_cast, like ITransferFunctionNode.
struct ITransformableNode
{
  virtual ~ITransformableNode() = default;
  virtual tsd::core::math::mat4 &transform() = 0;
};

} // namespace tsd::graph_nodes
