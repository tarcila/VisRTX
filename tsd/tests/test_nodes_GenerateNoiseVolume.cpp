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
} // namespace

SCENARIO("GenerateNoiseVolume emits a deterministic field", "[nodes-noise]")
{
  Graph g;
  auto n = addBuiltin(g, "GenerateNoiseVolume");
  g.node(n)->impl->parameters().set(Token("dims"), uint3(16u, 16u, 16u));
  g.node(n)->impl->parameters().set(Token("seed"), 7);
  Evaluator e(g);

  WHEN("pulled")
  {
    REQUIRE(e.pull(n));
    auto out = e.output(n, Token("out"), hostResidency());
    REQUIRE(out != nullptr);
    auto f = std::static_pointer_cast<Field>(out->payload);
    THEN("the field has the requested dims and matching data size")
    {
      REQUIRE(f->dims.x == 16u);
      REQUIRE(f->data.size() == 16u * 16u * 16u);
      REQUIRE(f->data.elementType() == ANARI_FLOAT32);
    }
  }

  WHEN("pulled again with the same seed (fresh graph)")
  {
    Graph g2;
    auto n2 = addBuiltin(g2, "GenerateNoiseVolume");
    g2.node(n2)->impl->parameters().set(Token("dims"), uint3(16u, 16u, 16u));
    g2.node(n2)->impl->parameters().set(Token("seed"), 7);
    Evaluator e2(g2);
    e.pull(n);
    e2.pull(n2);
    THEN("the data matches element-for-element (determinism)")
    {
      auto a = std::static_pointer_cast<Field>(
          e.output(n, Token("out"), hostResidency())->payload);
      auto b = std::static_pointer_cast<Field>(
          e2.output(n2, Token("out"), hostResidency())->payload);
      bool same = true;
      for (size_t i = 0; i < a->data.size(); ++i)
        same = same && (a->data.get<float>(i) == b->data.get<float>(i));
      REQUIRE(same);
    }
  }

  WHEN("given zero dims")
  {
    auto bad = addBuiltin(g, "GenerateNoiseVolume");
    g.node(bad)->impl->parameters().set(Token("dims"), uint3(0u, 4u, 4u));
    Evaluator e3(g);
    THEN("the pull fails with Error")
    {
      REQUIRE_FALSE(e3.pull(bad));
      REQUIRE(g.node(bad)->state == tsd::graph::EvalState::Error);
    }
  }
}
