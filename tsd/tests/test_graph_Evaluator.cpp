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

// Emits param "v" as a host scalar; counts evaluations.
struct ConstSource : Node
{
  ParameterList params;
  int *evalCount;
  explicit ConstSource(int *c) : evalCount(c) {}
  NodeTypeInfo typeInfo() const override
  {
    NodeTypeInfo i;
    i.name = Token("ConstSource");
    i.outputs.push_back({Token("out"), PortType{Token("scalar")}, true, {}});
    return i;
  }
  ParameterList &parameters() override
  {
    return params;
  }
  void evaluate(EvalContext &ctx) override;
};

// Reads "in", multiplies by 2, emits "out"; counts evaluations.
struct DoubleNode : Node
{
  ParameterList params;
  int *evalCount;
  explicit DoubleNode(int *c) : evalCount(c) {}
  NodeTypeInfo typeInfo() const override
  {
    NodeTypeInfo i;
    i.name = Token("Double");
    i.inputs.push_back({Token("in"), PortType{Token("scalar")}, true, {}});
    i.outputs.push_back({Token("out"), PortType{Token("scalar")}, true, {}});
    return i;
  }
  ParameterList &parameters() override
  {
    return params;
  }
  void evaluate(EvalContext &ctx) override;
};

} // namespace

// Out-of-line so EvalContext (from the top include) is fully defined here.
void ConstSource::evaluate(EvalContext &ctx)
{
  (*evalCount)++;
  auto out = std::make_shared<float>(params.getOr<float>(Token("v"), 0.0f));
  Value v;
  v.type = PortType{Token("scalar")};
  v.residency = hostResidency();
  v.payload = out;
  ctx.setOutput(Token("out"), v);
}

void DoubleNode::evaluate(EvalContext &ctx)
{
  (*evalCount)++;
  float in = *std::static_pointer_cast<float>(
      ctx.input(Token("in"), hostResidency()).payload);
  auto out = std::make_shared<float>(in * 2.0f);
  Value v;
  v.type = PortType{Token("scalar")};
  v.residency = hostResidency();
  v.payload = out;
  ctx.setOutput(Token("out"), v);
}

SCENARIO("tsd::graph::Evaluator lazy pull, caching, and version short-circuit",
    "[graph-eval]")
{
  int srcEvals = 0, dblEvals = 0;

  Graph g;
  auto src = g.addNode(std::make_unique<ConstSource>(&srcEvals));
  auto dbl = g.addNode(std::make_unique<DoubleNode>(&dblEvals));
  g.node(src)->impl->parameters().set(Token("v"), 5.0f);
  g.connect(src, Token("out"), dbl, Token("in"));

  Evaluator e(g);

  WHEN("pulling the sink once")
  {
    REQUIRE(e.pull(dbl));
    THEN("the result is 10 and each node evaluated once")
    {
      const Value *out = e.output(dbl, Token("out"), hostResidency());
      REQUIRE(out != nullptr);
      REQUIRE(*std::static_pointer_cast<float>(out->payload) == 10.0f);
      REQUIRE(srcEvals == 1);
      REQUIRE(dblEvals == 1);
    }
  }

  WHEN("pulling twice with no edits")
  {
    e.pull(dbl);
    e.pull(dbl);
    THEN("nothing re-evaluates the second time")
    {
      REQUIRE(srcEvals == 1);
      REQUIRE(dblEvals == 1);
    }
  }

  WHEN("a param edit dirties the source, then pulling again")
  {
    e.pull(dbl);
    g.node(src)->impl->parameters().set(Token("v"), 6.0f);
    g.markDirty(src);
    e.pull(dbl);
    THEN("both recompute and the result updates to 12")
    {
      REQUIRE(srcEvals == 2);
      REQUIRE(dblEvals == 2);
      const Value *out = e.output(dbl, Token("out"), hostResidency());
      REQUIRE(*std::static_pointer_cast<float>(out->payload) == 12.0f);
    }
  }

  WHEN("only the sink's own param changes, then pulling again")
  {
    e.pull(dbl);
    g.node(dbl)->impl->parameters().set(Token("scale"), 1.0f);
    g.markDirty(dbl);
    e.pull(dbl);
    THEN("the sink recomputes but the source is short-circuited")
    {
      REQUIRE(srcEvals == 1);
      REQUIRE(dblEvals == 2);
    }
  }
}
