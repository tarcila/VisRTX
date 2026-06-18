// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/graph/Evaluator.hpp"
#include "tsd/graph_nodes/BuiltinNodes.hpp"
#include "tsd/graph_nodes/Descriptors.hpp"
// std
#include <memory>

using tsd::core::Token;
using tsd::graph::EvalContext;
using tsd::graph::Evaluator;
using tsd::graph::Graph;
using tsd::graph::hostResidency;
using tsd::graph::Node;
using tsd::graph::NodeTypeInfo;
using tsd::graph::ParameterList;
using tsd::graph::PortType;

namespace {

struct AlwaysFails : Node
{
  ParameterList params;
  NodeTypeInfo typeInfo() const override
  {
    NodeTypeInfo i;
    i.name = Token("AlwaysFails");
    i.outputs.push_back({Token("out"), PortType{Token("field")}, true, {}});
    return i;
  }
  ParameterList &parameters() override
  {
    return params;
  }
  void evaluate(EvalContext &ctx) override
  {
    ctx.fail("boom");
  }
};

} // namespace

SCENARIO("EvalContext::fail marks the node Error", "[nodes-scaffold]")
{
  Graph g;
  auto n = g.addNode(std::make_unique<AlwaysFails>());
  Evaluator e(g);
  THEN("pull fails and the node carries the message")
  {
    REQUIRE_FALSE(e.pull(n));
    REQUIRE(g.node(n)->state == tsd::graph::EvalState::Error);
    REQUIRE(g.node(n)->error == "boom");
  }
}

SCENARIO("tsd_graph_nodes registry is callable", "[nodes-scaffold]")
{
  tsd::graph::NodeRegistry reg;
  tsd::graph_nodes::registerBuiltinNodes(reg);
  THEN("the registry exists (node set filled in later tasks)")
  {
    REQUIRE_FALSE(reg.isRegistered(Token("DoesNotExist")));
  }
}
