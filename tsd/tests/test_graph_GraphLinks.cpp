// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/graph/Graph.hpp"

using tsd::core::Token;
using tsd::graph::EvalContext;
using tsd::graph::Graph;
using tsd::graph::Node;
using tsd::graph::NodeTypeInfo;
using tsd::graph::ParameterList;
using tsd::graph::PortSpec;
using tsd::graph::PortType;

namespace {

// Source: one "field" output. Sink: one "field" input.
struct SourceNode : Node
{
  ParameterList params;
  NodeTypeInfo typeInfo() const override
  {
    NodeTypeInfo i;
    i.name = Token("Source");
    i.outputs.push_back({Token("out"), PortType{Token("field")}, true, {}});
    return i;
  }
  ParameterList &parameters() override
  {
    return params;
  }
  void evaluate(EvalContext &) override {}
};

struct SinkNode : Node
{
  ParameterList params;
  NodeTypeInfo typeInfo() const override
  {
    NodeTypeInfo i;
    i.name = Token("Sink");
    i.inputs.push_back({Token("in"), PortType{Token("field")}, true, {}});
    return i;
  }
  ParameterList &parameters() override
  {
    return params;
  }
  void evaluate(EvalContext &) override {}
};

// A sink expecting an incompatible type with no registered conversion.
struct ColorSink : Node
{
  ParameterList params;
  NodeTypeInfo typeInfo() const override
  {
    NodeTypeInfo i;
    i.name = Token("ColorSink");
    i.inputs.push_back({Token("in"), PortType{Token("color")}, true, {}});
    return i;
  }
  ParameterList &parameters() override
  {
    return params;
  }
  void evaluate(EvalContext &) override {}
};

// Has BOTH a field input and a field output, so it can form real cycles.
struct PassThrough : Node
{
  ParameterList params;
  NodeTypeInfo typeInfo() const override
  {
    NodeTypeInfo i;
    i.name = Token("PassThrough");
    i.inputs.push_back({Token("in"), PortType{Token("field")}, true, {}});
    i.outputs.push_back({Token("out"), PortType{Token("field")}, true, {}});
    return i;
  }
  ParameterList &parameters() override
  {
    return params;
  }
  void evaluate(EvalContext &) override {}
};

} // namespace

SCENARIO("tsd::graph::Graph link validation", "[graph-links]")
{
  GIVEN("a graph with a source and a sink")
  {
    Graph g;
    auto src = g.addNode(std::make_unique<SourceNode>());
    auto sink = g.addNode(std::make_unique<SinkNode>());

    WHEN("connecting matching types")
    {
      auto r = g.connect(src, Token("out"), sink, Token("in"));
      THEN("the link succeeds with a stable id")
      {
        REQUIRE(r.ok);
        REQUIRE(r.id != tsd::graph::INVALID_CONNECTION);
        REQUIRE(g.connections().size() == 1);
      }
    }

    WHEN("connecting to a mismatched type with no conversion")
    {
      auto csink = g.addNode(std::make_unique<ColorSink>());
      auto r = g.connect(src, Token("out"), csink, Token("in"));
      THEN("the link is rejected with a reason")
      {
        REQUIRE_FALSE(r.ok);
        REQUIRE_FALSE(r.reason.empty());
        REQUIRE(g.connections().empty());
      }
    }

    WHEN("a connection would create a cycle between two passthrough nodes")
    {
      auto a = g.addNode(std::make_unique<PassThrough>());
      auto b = g.addNode(std::make_unique<PassThrough>());
      auto ab = g.connect(a, Token("out"), b, Token("in"));
      REQUIRE(ab.ok);
      auto ba = g.connect(b, Token("out"), a, Token("in"));
      THEN("it is rejected with the cycle reason")
      {
        REQUIRE_FALSE(ba.ok);
        REQUIRE(ba.reason == "connection would create a cycle");
        REQUIRE(g.connections().size() == 1);
      }
    }

    WHEN("a passthrough node is connected to itself")
    {
      auto p = g.addNode(std::make_unique<PassThrough>());
      auto r = g.connect(p, Token("out"), p, Token("in"));
      THEN("it is rejected with the cycle reason")
      {
        REQUIRE_FALSE(r.ok);
        REQUIRE(r.reason == "connection would create a cycle");
      }
    }
  }
}
