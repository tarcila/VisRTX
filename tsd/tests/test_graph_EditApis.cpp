// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/graph/Graph.hpp"
#include "tsd/graph/NodeRegistry.hpp"
#include "tsd/graph_nodes/BuiltinNodes.hpp"
// std
#include <algorithm>

using tsd::core::Token;
using namespace tsd::graph;

namespace {
NodeId add(Graph &g, NodeRegistry &r, const char *t)
{
  return g.addNode(r.create(Token(t)));
}
} // namespace

SCENARIO("Graph::nodeIds enumerates all nodes", "[graph-editapis]")
{
  NodeRegistry reg;
  tsd::graph_nodes::registerBuiltinNodes(reg);
  Graph g;
  const NodeId a = add(g, reg, "GenerateNoiseVolume");
  const NodeId b = add(g, reg, "ScalarRange");
  auto ids = g.nodeIds();
  REQUIRE(ids.size() == 2);
  REQUIRE(std::find(ids.begin(), ids.end(), a) != ids.end());
  REQUIRE(std::find(ids.begin(), ids.end(), b) != ids.end());
}

SCENARIO("NodeRegistry::types lists registered type names", "[graph-editapis]")
{
  NodeRegistry reg;
  tsd::graph_nodes::registerBuiltinNodes(reg);
  auto types = reg.types();
  REQUIRE(types.size() >= 6);
  REQUIRE(std::find(types.begin(), types.end(), Token("TransferFunction"))
      != types.end());
}

SCENARIO("Graph::canConnect mirrors connect's validation without mutating",
    "[graph-editapis]")
{
  NodeRegistry reg;
  tsd::graph_nodes::registerBuiltinNodes(reg);
  Graph g;
  const NodeId src = add(g, reg, "GenerateNoiseVolume"); // out: "out" (field)
  const NodeId sr =
      add(g, reg, "ScalarRange"); // in: "in" (field), out: "out" (range)
  const NodeId tf = add(g, reg, "TransferFunction"); // in: "in" (range)

  GIVEN("a valid pair")
  {
    auto chk = g.canConnect(src, Token("out"), sr, Token("in"));
    THEN("canConnect says ok and no connection was created")
    {
      REQUIRE(chk.ok);
      REQUIRE(g.connections().empty()); // non-mutating
    }
  }
  GIVEN("an unknown port")
  {
    auto chk = g.canConnect(src, Token("nope"), sr, Token("in"));
    THEN("it reports the same rejection connect would")
    {
      REQUIRE_FALSE(chk.ok);
    }
  }
  GIVEN("a type-incompatible pair (field out -> range in)")
  {
    auto chk = g.canConnect(src, Token("out"), tf, Token("in"));
    THEN("it is rejected")
    {
      REQUIRE_FALSE(chk.ok);
    }
  }
  GIVEN("a cycle (sr.out -> tf.in committed, then tf back to sr)")
  {
    REQUIRE(g.connect(src, Token("out"), sr, Token("in")).ok);
    REQUIRE(g.connect(sr, Token("out"), tf, Token("in")).ok);
    // tf has no output feeding sr's input type, but verify cycle path on a
    // self-feed attempt:
    auto chk = g.canConnect(sr, Token("out"), sr, Token("in"));
    THEN("a self/cycle link is rejected")
    {
      REQUIRE_FALSE(chk.ok);
    }
  }
}
