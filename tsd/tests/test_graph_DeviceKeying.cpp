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
using tsd::graph::EvalContext;
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

struct HostArraySource : Node
{
  ParameterList params;
  NodeTypeInfo typeInfo() const override
  {
    NodeTypeInfo i;
    i.name = Token("HostArraySource");
    i.outputs.push_back({Token("out"), PortType{Token("array")}, true, {}});
    return i;
  }
  ParameterList &parameters() override
  {
    return params;
  }
  void evaluate(EvalContext &ctx) override
  {
    auto buf = std::make_shared<std::vector<float>>(std::vector<float>{1, 2});
    Value v;
    v.type = PortType{Token("array")};
    v.residency = hostResidency();
    v.payload = buf;
    ctx.setOutput(Token("out"), v);
  }
};

// Requests its input on the "test" backend at a specific device id.
struct TestDeviceSink : Node
{
  ParameterList params;
  int device{0};
  Residency got;
  explicit TestDeviceSink(int d) : device(d) {}
  NodeTypeInfo typeInfo() const override
  {
    NodeTypeInfo i;
    i.name = Token("TestDeviceSink");
    i.inputs.push_back(
        {Token("in"), PortType{Token("array")}, true, {Token("test")}});
    return i;
  }
  ParameterList &parameters() override
  {
    return params;
  }
  void evaluate(EvalContext &ctx) override
  {
    auto v = ctx.input(Token("in"), Residency{Token("test"), device});
    got = v.residency;
  }
};

} // namespace

SCENARIO("tsd::graph transfer cache is keyed on target deviceId",
    "[graph-devicekeying]")
{
  int transferCount = 0;
  TransferRegistry transfers;
  transfers.registerTransfer(
      PortType{Token("array")},
      Token("host"),
      Token("test"),
      [&transferCount](const Value &src, const Residency &target) {
        ++transferCount;
        auto in = std::static_pointer_cast<std::vector<float>>(src.payload);
        auto out = std::make_shared<std::vector<float>>(*in);
        Value v = src;
        v.payload = out;
        v.residency = target;
        return v;
      },
      [](const Value &src) -> size_t {
        return std::static_pointer_cast<std::vector<float>>(src.payload)->size()
            * sizeof(float);
      });

  Graph g;
  auto src = g.addNode(std::make_unique<HostArraySource>());
  auto s0Id = g.addNode(std::make_unique<TestDeviceSink>(0));
  auto s1Id = g.addNode(std::make_unique<TestDeviceSink>(1));
  auto *s0 = static_cast<TestDeviceSink *>(g.node(s0Id)->impl.get());
  auto *s1 = static_cast<TestDeviceSink *>(g.node(s1Id)->impl.get());
  g.connect(src, Token("out"), s0Id, Token("in"));
  g.connect(src, Token("out"), s1Id, Token("in"));

  Evaluator e(g, &transfers, nullptr);

  WHEN("two consumers request the same data on device 0 and device 1")
  {
    REQUIRE(e.pull(s0Id));
    REQUIRE(e.pull(s1Id));
    THEN("each gets its own device-resident copy (no cross-device collision)")
    {
      REQUIRE(s0->got == Residency{Token("test"), 0});
      REQUIRE(s1->got == Residency{Token("test"), 1});
    }
    THEN("two distinct transfers ran (device 1 did not reuse device 0's copy)")
    {
      REQUIRE(transferCount == 2);
    }
  }

  WHEN("the same device-0 consumer is recomputed against an unchanged producer")
  {
    REQUIRE(e.pull(s0Id)); // transferCount == 1
    g.markDirty(s0Id); // force the sink to re-evaluate
    REQUIRE(e.pull(s0Id));
    THEN("the transfer is served from cache, not re-run")
    {
      REQUIRE(transferCount == 1);
    }
  }
}
