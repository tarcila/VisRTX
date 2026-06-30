// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// std
#include <algorithm>
// catch
#include "catch.hpp"
// tsd
#include "tsd/graph_nodes/BuiltinNodes.hpp"
#include "tsd/graph_nodes/GraphEditModel.hpp"

using tsd::core::Token;
using namespace tsd::graph;
using tsd::graph_nodes::GraphEditModel;
using tsd::graph_nodes::LinkKind;

SCENARIO(
    "GraphEditModel adds, connects, classifies, and removes", "[edit-model]")
{
  NodeRegistry reg;
  tsd::graph_nodes::registerBuiltinNodes(reg);
  Graph g;
  GraphEditModel model(g, reg, nullptr);

  WHEN("the catalog is queried")
  {
    THEN("it lists builtin types")
    {
      const auto &cat = model.nodeCatalog();
      REQUIRE(std::find(cat.begin(), cat.end(), Token("TransferFunction"))
          != cat.end());
    }
  }

  WHEN("nodes are added and connected")
  {
    const NodeId src = model.addNode(Token("GenerateNoiseVolume"));
    const NodeId sr = model.addNode(Token("ScalarRange"));
    REQUIRE(src != INVALID_NODE);
    REQUIRE(g.nodeIds().size() == 2);

    auto chk = model.canConnect(src, Token("out"), sr, Token("in"));
    THEN("a valid link is Direct")
    {
      REQUIRE(chk.ok());
      REQUIRE(chk.kind == LinkKind::Direct);
    }

    auto res = model.connect(src, Token("out"), sr, Token("in"));
    REQUIRE(res.ok);
    THEN("the committed link classifies as Direct")
    {
      REQUIRE(g.connections().size() == 1);
      REQUIRE(model.classify(g.connections().front()) == LinkKind::Direct);
    }

    AND_WHEN("the connection is removed")
    {
      model.disconnect(res.id);
      THEN("the graph has no connections")
      {
        REQUIRE(g.connections().empty());
      }
    }
    AND_WHEN("a node is removed")
    {
      model.removeNode(sr);
      THEN("it is gone")
      {
        REQUIRE(g.nodeIds().size() == 1);
      }
    }
  }

  WHEN("an incompatible link is checked")
  {
    const NodeId src = model.addNode(Token("GenerateNoiseVolume")); // field out
    const NodeId tf = model.addNode(Token("TransferFunction")); // range in
    auto chk = model.canConnect(src, Token("out"), tf, Token("in"));
    THEN("it is Incompatible and not ok")
    {
      REQUIRE_FALSE(chk.ok());
      REQUIRE(chk.kind == LinkKind::Incompatible);
    }
  }

  WHEN("a cycle is checked")
  {
    const NodeId a = model.addNode(Token("GenerateNoiseVolume"));
    const NodeId b = model.addNode(Token("ScalarRange"));
    REQUIRE(model.connect(a, Token("out"), b, Token("in")).ok);
    auto chk = model.canConnect(b, Token("out"), b, Token("in"));
    THEN("it is rejected")
    {
      REQUIRE_FALSE(chk.ok());
      REQUIRE(chk.kind == LinkKind::Cycle);
    }
  }

  WHEN("downstream suggestions are requested for a field source")
  {
    const NodeId src = model.addNode(Token("GenerateNoiseVolume")); // field out
    const auto sugg = model.downstreamSuggestions(src);

    auto has = [&](const char *type) {
      return std::any_of(sugg.begin(), sugg.end(), [&](const auto &s) {
        return s.nodeType == Token(type) && s.fromPort == Token("out");
      });
    };

    THEN("field-consuming nodes are offered, wired from the 'out' port")
    {
      REQUIRE(has("ScalarRange"));
      REQUIRE(has("DisplayVolume"));
      // Incompatible sinks (range/surface inputs) are not offered.
      REQUIRE_FALSE(has("TransferFunction"));
      // The suggested DisplayVolume wires into its 'field' input.
      auto it = std::find_if(sugg.begin(), sugg.end(), [&](const auto &s) {
        return s.nodeType == Token("DisplayVolume");
      });
      REQUIRE(it != sugg.end());
      REQUIRE(it->toPort == Token("field"));
    }
  }
}
