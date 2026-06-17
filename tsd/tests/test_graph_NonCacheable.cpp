// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/graph/Evaluator.hpp"
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
using tsd::graph::PortSpec;
using tsd::graph::PortType;
using tsd::graph::Value;

namespace {

struct Counter : Node
{
  ParameterList params;
  int *count;
  bool cacheable;
  Counter(int *c, bool cache) : count(c), cacheable(cache) {}
  NodeTypeInfo typeInfo() const override
  {
    NodeTypeInfo i;
    i.name = Token("Counter");
    i.outputs.push_back({Token("out"), PortType{Token("scalar")}, true, {}});
    i.isCacheable = cacheable;
    return i;
  }
  ParameterList &parameters() override
  {
    return params;
  }
  void evaluate(EvalContext &ctx) override
  {
    (*count)++;
    Value v;
    v.type = PortType{Token("scalar")};
    v.residency = hostResidency();
    v.payload = std::make_shared<float>(1.0f);
    ctx.setOutput(Token("out"), v);
  }
};

} // namespace

SCENARIO("tsd::graph non-cacheable nodes always recompute", "[graph-noncache]")
{
  GIVEN("a cacheable node")
  {
    int n = 0;
    Graph g;
    auto id = g.addNode(std::make_unique<Counter>(&n, true));
    Evaluator e(g);
    e.pull(id);
    e.pull(id);
    THEN("it evaluates only once across two pulls")
    {
      REQUIRE(n == 1);
    }
  }

  GIVEN("a non-cacheable node")
  {
    int n = 0;
    Graph g;
    auto id = g.addNode(std::make_unique<Counter>(&n, false));
    Evaluator e(g);
    e.pull(id);
    e.pull(id);
    THEN("it evaluates on every pull")
    {
      REQUIRE(n == 2);
    }
  }
}
