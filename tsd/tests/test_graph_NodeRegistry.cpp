// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/graph/NodeRegistry.hpp"

using tsd::core::Token;
using tsd::graph::EvalContext;
using tsd::graph::Node;
using tsd::graph::NodeRegistry;
using tsd::graph::NodeTypeInfo;
using tsd::graph::ParameterList;

namespace {

struct DummyNode : Node
{
  ParameterList params;
  NodeTypeInfo typeInfo() const override
  {
    NodeTypeInfo info;
    info.name = Token("Dummy");
    return info;
  }
  ParameterList &parameters() override
  {
    return params;
  }
  void evaluate(EvalContext &) override {}
};

} // namespace

// Self-register DummyNode under a distinct name at static-init.
TSD_GRAPH_REGISTER_NODE("AutoDummy", DummyNode)

SCENARIO(
    "tsd::graph::NodeRegistry creates registered types", "[graph-noderegistry]")
{
  GIVEN("a registry with Dummy registered")
  {
    NodeRegistry reg;
    reg.registerType(
        Token("Dummy"), [] { return std::make_unique<DummyNode>(); });

    WHEN("creating a Dummy")
    {
      auto n = reg.create(Token("Dummy"));
      THEN("a node is returned with the right type name")
      {
        REQUIRE(n != nullptr);
        REQUIRE(n->typeInfo().name == Token("Dummy"));
      }
    }
    WHEN("creating an unknown type")
    {
      THEN("nullptr is returned")
      {
        REQUIRE(reg.create(Token("Nope")) == nullptr);
      }
    }
  }
}

SCENARIO("tsd::graph node types self-register at static init",
    "[graph-noderegistry]")
{
  GIVEN("the process-global registry")
  {
    THEN("a type registered via TSD_GRAPH_REGISTER_NODE is present")
    {
      REQUIRE(
          tsd::graph::GlobalNodeRegistry().isRegistered(Token("AutoDummy")));
      auto n = tsd::graph::GlobalNodeRegistry().create(Token("AutoDummy"));
      REQUIRE(n != nullptr);
    }
  }
}
