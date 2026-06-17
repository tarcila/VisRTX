// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/graph/Evaluator.hpp"
#include "tsd/graph/TestBackend.hpp"
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

struct TestResidentSource : Node
{
  ParameterList params;
  NodeTypeInfo typeInfo() const override
  {
    NodeTypeInfo i;
    i.name = Token("TestSource");
    i.outputs.push_back({Token("out"), PortType{Token("array")}, true, {}});
    return i;
  }
  ParameterList &parameters() override
  {
    return params;
  }
  void evaluate(EvalContext &ctx) override
  {
    auto buf =
        std::make_shared<std::vector<float>>(std::vector<float>{1, 2, 3, 4});
    Value v;
    v.type = PortType{Token("array")};
    v.residency = Residency{Token("test"), 0};
    v.payload = buf;
    ctx.setOutput(Token("out"), v);
  }
};

struct HostOnlySink : Node
{
  ParameterList params;
  float sum{0};
  NodeTypeInfo typeInfo() const override
  {
    NodeTypeInfo i;
    i.name = Token("HostSink");
    i.inputs.push_back(
        {Token("in"), PortType{Token("array")}, true, {Token("host")}});
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

} // namespace

SCENARIO("tsd::graph implicit test->host transfer is inserted and reported",
    "[graph-testbackend]")
{
  TransferRegistry transfers;
  tsd::graph::registerTestBackendTransfers(transfers);

  Graph g;
  auto src = g.addNode(std::make_unique<TestResidentSource>());
  auto sinkId = g.addNode(std::make_unique<HostOnlySink>());
  auto *sinkNode = static_cast<HostOnlySink *>(g.node(sinkId)->impl.get());
  g.connect(src, Token("out"), sinkId, Token("in"));

  Evaluator e(g, &transfers, nullptr);

  WHEN("pulling the host-only sink")
  {
    REQUIRE(e.pull(sinkId));
    THEN("the data was transferred to host and summed")
    {
      REQUIRE(sinkNode->sum == 10.0f);
    }
    THEN("the EvalReport records one test->host transfer with nonzero cost")
    {
      const auto &r = e.lastReport();
      REQUIRE(r.entries.size() == 1);
      REQUIRE(r.entries[0].kind == tsd::graph::EvalReportEntry::Kind::Transfer);
      REQUIRE(r.entries[0].from == Token("test"));
      REQUIRE(r.entries[0].to == Token("host"));
      REQUIRE(r.entries[0].estCost == 4 * sizeof(float));
    }
  }
}
