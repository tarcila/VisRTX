// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/graph/Graph.hpp"

using tsd::core::Token;
using tsd::graph::EvalContext;
using tsd::graph::EvalState;
using tsd::graph::Graph;
using tsd::graph::Node;
using tsd::graph::NodeTypeInfo;
using tsd::graph::ParameterList;
using tsd::graph::PortSpec;
using tsd::graph::PortType;

namespace {

struct PassThrough : Node
{
  ParameterList params;
  bool isSource;
  explicit PassThrough(bool source) : isSource(source) {}
  NodeTypeInfo typeInfo() const override
  {
    NodeTypeInfo i;
    i.name = Token("PT");
    if (!isSource)
      i.inputs.push_back({Token("in"), PortType{Token("field")}, true, {}});
    i.outputs.push_back({Token("out"), PortType{Token("field")}, true, {}});
    return i;
  }
  ParameterList &parameters() override
  {
    return params;
  }
  void evaluate(EvalContext &) override {}
};

} // namespace

SCENARIO("tsd::graph::Graph dirty propagation and deletion", "[graph-dirty]")
{
  GIVEN("a chain A -> B -> C, all Clean")
  {
    Graph g;
    auto a = g.addNode(std::make_unique<PassThrough>(true));
    auto b = g.addNode(std::make_unique<PassThrough>(false));
    auto c = g.addNode(std::make_unique<PassThrough>(false));
    g.connect(a, Token("out"), b, Token("in"));
    g.connect(b, Token("out"), c, Token("in"));
    g.node(a)->state = EvalState::Clean;
    g.node(b)->state = EvalState::Clean;
    g.node(c)->state = EvalState::Clean;

    WHEN("A is marked dirty")
    {
      g.markDirty(a);
      THEN("A, B, and C are all dirty")
      {
        REQUIRE(g.node(a)->state == EvalState::Dirty);
        REQUIRE(g.node(b)->state == EvalState::Dirty);
        REQUIRE(g.node(c)->state == EvalState::Dirty);
      }
    }

    WHEN("B is deleted")
    {
      g.node(a)->state = EvalState::Clean;
      g.node(c)->state = EvalState::Clean;
      g.removeNode(b);
      THEN("C lost a required input and is in Error")
      {
        REQUIRE(g.node(c)->state == EvalState::Error);
        REQUIRE_FALSE(g.node(c)->error.empty());
      }
      THEN("A is untouched")
      {
        REQUIRE(g.node(a)->state == EvalState::Clean);
      }
    }
  }
}

SCENARIO("tsd::graph::Graph markDirty visits a diamond subtree once",
    "[graph-dirty]")
{
  GIVEN("a diamond A->B, A->C, B->D, C->D, all Clean")
  {
    Graph g;
    auto a = g.addNode(std::make_unique<PassThrough>(true));
    auto b = g.addNode(std::make_unique<PassThrough>(false));
    auto c = g.addNode(std::make_unique<PassThrough>(false));
    auto d = g.addNode(std::make_unique<PassThrough>(false));
    g.connect(a, Token("out"), b, Token("in"));
    g.connect(a, Token("out"), c, Token("in"));
    g.connect(b, Token("out"), d, Token("in"));
    g.connect(c, Token("out"), d, Token("in"));
    for (auto id : {a, b, c, d})
      g.node(id)->state = EvalState::Clean;

    WHEN("A is marked dirty")
    {
      g.markDirty(a);
      THEN("all four nodes are dirty")
      {
        REQUIRE(g.node(a)->state == EvalState::Dirty);
        REQUIRE(g.node(b)->state == EvalState::Dirty);
        REQUIRE(g.node(c)->state == EvalState::Dirty);
        REQUIRE(g.node(d)->state == EvalState::Dirty);
      }
    }
  }
}
