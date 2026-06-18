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
using tsd::graph::EvalState;
using tsd::graph::Evaluator;
using tsd::graph::Graph;
using tsd::graph::hostResidency;
using tsd::graph::Node;
using tsd::graph::NodeTypeInfo;
using tsd::graph::ParameterList;
using tsd::graph::PortType;
using tsd::graph::PullHandle;
using tsd::graph::Value;

namespace {

struct ConstSource : Node
{
  ParameterList params;
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
  void evaluate(EvalContext &ctx) override
  {
    auto out = std::make_shared<float>(params.getOr<float>(Token("v"), 0.0f));
    Value v;
    v.type = PortType{Token("scalar")};
    v.residency = hostResidency();
    v.payload = out;
    ctx.setOutput(Token("out"), v);
  }
};

// Reads a required input that is intentionally left unconnected -> materialize
// fails -> EvalContext sets this node to Error.
struct NeedsInput : Node
{
  ParameterList params;
  NodeTypeInfo typeInfo() const override
  {
    NodeTypeInfo i;
    i.name = Token("NeedsInput");
    i.inputs.push_back({Token("in"), PortType{Token("scalar")}, true, {}});
    i.outputs.push_back({Token("out"), PortType{Token("scalar")}, true, {}});
    return i;
  }
  ParameterList &parameters() override
  {
    return params;
  }
  void evaluate(EvalContext &ctx) override
  {
    auto in = ctx.input(Token("in"), hostResidency()); // unconnected -> invalid
    if (!in.valid())
      return; // required input missing; leave node without output
    auto out = std::make_shared<float>(1.0f);
    Value v;
    v.type = PortType{Token("scalar")};
    v.residency = hostResidency();
    v.payload = out;
    ctx.setOutput(Token("out"), v);
  }
};

} // namespace

SCENARIO("tsd::graph::Evaluator isolates a node error in an async pull",
    "[graph-asyncerror]")
{
  Graph g;
  auto good = g.addNode(std::make_unique<ConstSource>());
  g.node(good)->impl->parameters().set(Token("v"), 7.0f);
  auto bad = g.addNode(std::make_unique<NeedsInput>());

  // Wire good->bad so the link is valid, then remove good. Graph calls
  // revalidateRequiredInputs which sets bad to EvalState::Error because its
  // required input "in" is now unconnected.
  g.connect(good, Token("out"), bad, Token("in"));
  g.removeNode(good);

  // Re-add the healthy source (different NodeId from `good`).
  auto healthy = g.addNode(std::make_unique<ConstSource>());
  g.node(healthy)->impl->parameters().set(Token("v"), 7.0f);

  Evaluator e(g);

  WHEN("pulling the node whose required input is unconnected")
  {
    PullHandle h = e.pullAsync(bad);
    e.waitIdle();
    THEN("the pull fails and the node is in Error")
    {
      REQUIRE(e.isReady(h));
      REQUIRE_FALSE(e.result(h));
      REQUIRE(g.node(bad)->state == EvalState::Error);
    }
  }

  WHEN("pulling an unrelated healthy node afterward")
  {
    e.pull(bad); // leaves `bad` in Error
    THEN("the healthy branch still resolves")
    {
      REQUIRE(e.pull(healthy));
      const Value *out = e.output(healthy, Token("out"), hostResidency());
      REQUIRE(*std::static_pointer_cast<float>(out->payload) == 7.0f);
    }
  }
}
