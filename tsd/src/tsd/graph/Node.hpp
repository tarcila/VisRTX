// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/graph/Parameter.hpp"
#include "tsd/graph/Port.hpp"

namespace tsd::graph {

// Forward-declared; defined alongside the Evaluator (later task) since it
// bridges a node to the evaluator's input cache and transfer machinery.
class EvalContext;

// Interface every node type implements. evaluate() pulls inputs and sets
// outputs through the EvalContext; it never performs transfers itself.
class Node
{
 public:
  virtual ~Node() = default;
  virtual NodeTypeInfo typeInfo() const = 0;
  virtual ParameterList &parameters() = 0;
  virtual void evaluate(EvalContext &ctx) = 0;
};

} // namespace tsd::graph
