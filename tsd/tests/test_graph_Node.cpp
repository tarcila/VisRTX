// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/graph/Node.hpp"

using tsd::core::Token;
using tsd::graph::EvalContext;
using tsd::graph::Node;
using tsd::graph::NodeTypeInfo;
using tsd::graph::ParameterList;
using tsd::graph::PortSpec;
using tsd::graph::PortType;

namespace {

// A minimal concrete node used only to exercise the interface.
struct ConstantNode : Node
{
  ParameterList params;

  NodeTypeInfo typeInfo() const override
  {
    NodeTypeInfo info;
    info.name = Token("Constant");
    info.category = Token("source");
    info.outputs.push_back(
        PortSpec{Token("out"), PortType{Token("scalar")}, true, {}});
    info.isCacheable = true;
    return info;
  }

  ParameterList &parameters() override
  {
    return params;
  }

  void evaluate(EvalContext &) override {}
};

} // namespace

SCENARIO("tsd::graph::Node exposes type info and params", "[graph-node]")
{
  GIVEN("a ConstantNode")
  {
    ConstantNode n;
    THEN("its type info names one output and is cacheable")
    {
      auto info = n.typeInfo();
      REQUIRE(info.name == Token("Constant"));
      REQUIRE(info.outputs.size() == 1);
      REQUIRE(info.outputs[0].name == Token("out"));
      REQUIRE(info.outputs[0].required);
      REQUIRE(info.isCacheable);
    }
    THEN("its parameter list is reachable and mutable")
    {
      n.parameters().set(Token("v"), 2.0f);
      REQUIRE(n.parameters().get<float>(Token("v")) == 2.0f);
    }
  }
}
