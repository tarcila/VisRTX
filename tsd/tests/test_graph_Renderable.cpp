// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/graph/Evaluator.hpp"
#include "tsd/graph/Renderable.hpp"
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
using tsd::graph::Renderable;
using tsd::graph::Value;

namespace {

struct EmitSphere : Node
{
  ParameterList params;
  NodeTypeInfo typeInfo() const override
  {
    NodeTypeInfo i;
    i.name = Token("EmitSphere");
    i.outputs.push_back(
        {Token("out"), PortType{Token("renderable")}, true, {}});
    return i;
  }
  ParameterList &parameters() override
  {
    return params;
  }
  void evaluate(EvalContext &ctx) override
  {
    auto r = std::make_shared<Renderable>();
    r->kind = Renderable::Kind::Surface;
    r->primSubtype = Token("sphere");
    r->prim.scalars.push_back({Token("radius"), tsd::core::Any(0.5f)});
    Value v;
    v.type = PortType{Token("renderable")};
    v.residency = hostResidency();
    v.payload = r;
    ctx.setOutput(Token("out"), v);
  }
};

} // namespace

SCENARIO(
    "tsd::graph::Renderable travels as a Value payload", "[graph-renderable]")
{
  Graph g;
  auto n = g.addNode(std::make_unique<EmitSphere>());
  Evaluator e(g);

  WHEN("pulling the emitter")
  {
    REQUIRE(e.pull(n));
    const Value *out = e.output(n, Token("out"), hostResidency());
    THEN("the renderable describes a sphere surface")
    {
      REQUIRE(out != nullptr);
      auto r = std::static_pointer_cast<Renderable>(out->payload);
      REQUIRE(r->kind == Renderable::Kind::Surface);
      REQUIRE(r->primSubtype == Token("sphere"));
      REQUIRE(r->prim.scalars.size() == 1);
      REQUIRE(r->prim.scalars[0].first == Token("radius"));
      REQUIRE(r->prim.scalars[0].second.get<float>() == 0.5f);
    }
  }
}
