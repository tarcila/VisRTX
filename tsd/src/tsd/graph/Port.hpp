// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/graph/PortType.hpp"
// std
#include <vector>

namespace tsd::graph {

// A node port. `acceptedBackends` lists residency backends this input can
// consume directly; empty means "any" (host-preferred). For outputs the field
// is unused.
struct PortSpec
{
  tsd::core::Token name;
  PortType type;
  bool required{true};
  std::vector<tsd::core::Token> acceptedBackends;
};

// Static description of a node type. UI is generated from this plus the node's
// ParameterList.
struct NodeTypeInfo
{
  tsd::core::Token name;
  tsd::core::Token category;
  std::vector<PortSpec> inputs;
  std::vector<PortSpec> outputs;
  bool isCacheable{true};
};

} // namespace tsd::graph
