// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/graph/Evaluator.hpp"
// std
#include <memory>
#include <vector>

using tsd::core::Token;
using tsd::graph::ConversionRegistry;
using tsd::graph::EvalContext;
using tsd::graph::EvalReportEntry;
using tsd::graph::Evaluator;
using tsd::graph::Graph;
using tsd::graph::hostResidency;
using tsd::graph::Node;
using tsd::graph::NodeTypeInfo;
using tsd::graph::ParameterList;
using tsd::graph::PortSpec;
using tsd::graph::PortType;
using tsd::graph::Residency;
using tsd::graph::TransferRegistry;
using tsd::graph::Value;

namespace {

struct I32Source : Node
{
  ParameterList params;
  NodeTypeInfo typeInfo() const override
  {
    NodeTypeInfo i;
    i.name = Token("I32Source");
    i.outputs.push_back({Token("out"), PortType{Token("i32array")}, true, {}});
    return i;
  }
  ParameterList &parameters() override
  {
    return params;
  }
  void evaluate(EvalContext &ctx) override
  {
    auto buf = std::make_shared<std::vector<int>>(std::vector<int>{1, 2, 3});
    Value v;
    v.type = PortType{Token("i32array")};
    v.residency = hostResidency();
    v.payload = buf;
    ctx.setOutput(Token("out"), v);
  }
};

struct F32Sink : Node
{
  ParameterList params;
  float sum{0};
  NodeTypeInfo typeInfo() const override
  {
    NodeTypeInfo i;
    i.name = Token("F32Sink");
    i.inputs.push_back(
        {Token("in"), PortType{Token("f32array")}, true, {Token("host")}});
    return i;
  }
  ParameterList &parameters() override
  {
    return params;
  }
  void evaluate(EvalContext &ctx) override
  {
    auto v = ctx.input(Token("in"), hostResidency());
    auto b = std::static_pointer_cast<std::vector<float>>(v.payload);
    sum = 0;
    for (float x : *b)
      sum += x;
  }
};

// Requires an input on a backend for which no transfer is registered.
struct VulkanSink : Node
{
  ParameterList params;
  NodeTypeInfo typeInfo() const override
  {
    NodeTypeInfo i;
    i.name = Token("VulkanSink");
    i.inputs.push_back(
        {Token("in"), PortType{Token("i32array")}, true, {Token("vulkan")}});
    return i;
  }
  ParameterList &parameters() override
  {
    return params;
  }
  void evaluate(EvalContext &ctx) override
  {
    ctx.input(Token("in"), Residency{Token("vulkan"), 0});
  }
};

ConversionRegistry makeI32ToF32()
{
  ConversionRegistry reg;
  reg.registerConversion(
      PortType{Token("i32array")},
      PortType{Token("f32array")},
      [](const Value &src) {
        auto in = std::static_pointer_cast<std::vector<int>>(src.payload);
        auto out = std::make_shared<std::vector<float>>();
        for (int x : *in)
          out->push_back(static_cast<float>(x));
        Value v = src;
        v.type = PortType{Token("f32array")};
        v.payload = out;
        return v;
      },
      [](const Value &src) -> size_t {
        return std::static_pointer_cast<std::vector<int>>(src.payload)->size();
      });
  return reg;
}

} // namespace

SCENARIO("tsd::graph inserts and reports an implicit conversion",
    "[graph-implicitops]")
{
  ConversionRegistry conversions = makeI32ToF32();
  Graph g(&conversions);
  auto src = g.addNode(std::make_unique<I32Source>());
  auto sinkId = g.addNode(std::make_unique<F32Sink>());
  auto *sink = static_cast<F32Sink *>(g.node(sinkId)->impl.get());

  WHEN("linking i32 source to an f32 sink (conversion registered)")
  {
    auto r = g.connect(src, Token("out"), sinkId, Token("in"));
    REQUIRE(r.ok); // link allowed because a conversion exists

    Evaluator e(g, nullptr, &conversions);
    REQUIRE(e.pull(sinkId));
    THEN("the data is converted to float and summed")
    {
      REQUIRE(sink->sum == 6.0f);
    }
    THEN("the EvalReport records one i32array->f32array conversion")
    {
      const auto &rep = e.lastReport();
      REQUIRE(rep.entries.size() == 1);
      REQUIRE(rep.entries[0].kind == EvalReportEntry::Kind::Convert);
      REQUIRE(rep.entries[0].from == Token("i32array"));
      REQUIRE(rep.entries[0].to == Token("f32array"));
      REQUIRE(rep.entries[0].estCost == 3);
    }
  }
}

SCENARIO(
    "tsd::graph reports a missing transfer path as a failed op, not a crash",
    "[graph-implicitops]")
{
  TransferRegistry transfers; // empty: no host->vulkan transfer
  Graph g;
  auto src = g.addNode(std::make_unique<I32Source>());
  auto sinkId = g.addNode(std::make_unique<VulkanSink>());
  g.connect(src, Token("out"), sinkId, Token("in")); // same type, link ok

  Evaluator e(g, &transfers, nullptr);

  WHEN("pulling a sink whose backend has no registered transfer")
  {
    bool ok = e.pull(sinkId);
    THEN("the pull fails and the node is in Error")
    {
      REQUIRE_FALSE(ok);
      REQUIRE(g.node(sinkId)->state == tsd::graph::EvalState::Error);
    }
    THEN("the EvalReport records one failed host->vulkan op")
    {
      const auto &rep = e.lastReport();
      REQUIRE(rep.entries.size() == 1);
      REQUIRE(rep.entries[0].kind == EvalReportEntry::Kind::Failed);
      REQUIRE(rep.entries[0].from == Token("host"));
      REQUIRE(rep.entries[0].to == Token("vulkan"));
    }
  }
}
