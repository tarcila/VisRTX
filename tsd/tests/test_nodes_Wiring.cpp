// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include <memory>
#include "catch.hpp"
#include "tsd/graph/Evaluator.hpp"
#include "tsd/graph_nodes/BuiltinNodes.hpp"
#include "tsd/graph_nodes/Descriptors.hpp"

using tsd::core::Token;
using namespace tsd::graph;
using uint3 = tsd::core::math::uint3;

namespace {
NodeRegistry &builtins()
{
  static NodeRegistry reg = [] {
    NodeRegistry r;
    tsd::graph_nodes::registerBuiltinNodes(r);
    return r;
  }();
  return reg;
}
NodeId add(Graph &g, const char *t)
{
  return g.addNode(builtins().create(Token(t)));
}
} // namespace

SCENARIO("all six builtins register", "[nodes-wiring]")
{
  NodeRegistry reg;
  tsd::graph_nodes::registerBuiltinNodes(reg);
  for (auto *t : {"GenerateNoiseVolume",
           "ScalarRange",
           "TransferFunction",
           "DisplayVolume",
           "BoundingBox",
           "DisplaySurface"})
    REQUIRE(reg.isRegistered(Token(t)));
}

SCENARIO("the demo volume graph resolves end-to-end", "[nodes-wiring]")
{
  Graph g;
  auto src = add(g, "GenerateNoiseVolume");
  g.node(src)->impl->parameters().set(Token("dims"), uint3(16u, 16u, 16u));
  auto sr = add(g, "ScalarRange");
  auto tf = add(g, "TransferFunction");
  auto dv = add(g, "DisplayVolume");
  g.connect(src, Token("out"), sr, Token("in"));
  g.connect(sr, Token("out"), tf, Token("in"));
  g.connect(src, Token("out"), dv, Token("field")); // fan-out
  g.connect(tf, Token("out"), dv, Token("tf")); // multi-input
  Evaluator e(g);

  REQUIRE(e.pull(dv));
  auto r = std::static_pointer_cast<Renderable>(
      e.output(dv, Token("out"), hostResidency())->payload);
  REQUIRE(r->kind == Renderable::Kind::Volume);

  WHEN("the source seed changes")
  {
    g.node(src)->impl->parameters().set(Token("seed"), 3);
    g.markDirty(src);
    THEN("the graph re-pulls successfully")
    {
      REQUIRE(e.pull(dv));
    }
  }
}
