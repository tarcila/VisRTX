// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/graph/Evaluator.hpp"
#include "tsd/graph_nodes/BuiltinNodes.hpp"
#include "tsd/graph_nodes/Descriptors.hpp"
// std
#include <cmath>
#include <memory>

using tsd::core::Token;
using tsd::graph::Evaluator;
using tsd::graph::Graph;
using tsd::graph::hostResidency;
using tsd::graph_nodes::Field;
using uint3 = tsd::core::math::uint3;

namespace {
tsd::graph::NodeId addBuiltin(Graph &g, const char *type)
{
  static tsd::graph::NodeRegistry reg = [] {
    tsd::graph::NodeRegistry r;
    tsd::graph_nodes::registerBuiltinNodes(r);
    return r;
  }();
  return g.addNode(reg.create(Token(type)));
}

// Pull `type`, require a [0,1]-valued field of the requested dims.
void checkField(const char *type)
{
  Graph g;
  auto n = addBuiltin(g, type);
  REQUIRE(g.node(n) != nullptr); // type is registered
  g.node(n)->impl->parameters().set(Token("dims"), uint3(16u, 16u, 16u));
  Evaluator e(g);

  REQUIRE(e.pull(n));
  auto out = e.output(n, Token("out"), hostResidency());
  REQUIRE(out != nullptr);
  auto f = std::static_pointer_cast<Field>(out->payload);
  REQUIRE(f->dims.x == 16u);
  REQUIRE(f->data.size() == 16u * 16u * 16u);
  REQUIRE(f->data.elementType() == ANARI_FLOAT32);

  bool finiteInRange = true;
  for (size_t i = 0; i < f->data.size(); ++i) {
    const float v = f->data.get<float>(i);
    finiteInRange = finiteInRange && std::isfinite(v) && v >= 0.f && v <= 1.f;
  }
  REQUIRE(finiteInRange);
}
} // namespace

SCENARIO("procedural volume nodes emit valid unit fields", "[nodes-procedural]")
{
  WHEN("GenerateGyroid is pulled")
  {
    THEN("it yields a [0,1] field")
    {
      checkField("GenerateGyroid");
    }
  }
  WHEN("GenerateTurbulence is pulled")
  {
    THEN("it yields a [0,1] field")
    {
      checkField("GenerateTurbulence");
    }
  }
  WHEN("GenerateMetaballs is pulled")
  {
    THEN("it yields a [0,1] field")
    {
      checkField("GenerateMetaballs");
    }
  }
}

SCENARIO("procedural volume nodes reject zero dims", "[nodes-procedural]")
{
  for (const char *type :
      {"GenerateGyroid", "GenerateTurbulence", "GenerateMetaballs"}) {
    Graph g;
    auto n = addBuiltin(g, type);
    g.node(n)->impl->parameters().set(Token("dims"), uint3(0u, 4u, 4u));
    Evaluator e(g);
    REQUIRE_FALSE(e.pull(n));
    REQUIRE(g.node(n)->state == tsd::graph::EvalState::Error);
  }
}
