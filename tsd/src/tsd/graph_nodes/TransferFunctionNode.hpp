// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/ColorMapUtil.hpp"

namespace tsd::graph_nodes {

// Implemented by the TransferFunction node so UI can edit its control points
// directly (these don't fit in the Any-based ParameterList).
struct ITransferFunctionNode
{
  virtual ~ITransferFunctionNode() = default;
  virtual tsd::core::TransferFunction &
  tfState() = 0; // colorPoints/opacityPoints/range
  virtual int &samples() = 0;
};

} // namespace tsd::graph_nodes
