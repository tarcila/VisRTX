// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/graph/Evaluator.hpp"
// std
#include <atomic>
#include <memory>

using tsd::core::Token;
using tsd::graph::EvalContext;
using tsd::graph::Evaluator;
using tsd::graph::Graph;
using tsd::graph::hostResidency;
using tsd::graph::Node;
using tsd::graph::NodeTypeInfo;
using tsd::graph::ParameterList;
using tsd::graph::PortType;
using tsd::graph::PullHandle;
using tsd::graph::Value;

namespace {

struct ConstSource : Node
{
  ParameterList params;
  NodeTypeInfo typeInfo() const override
  {
    NodeTypeInfo i;
    i.name = Token("ConstSource");
    i.outputs.push_back({Token("out"), PortType{Token("scalar")}, true, {}});
    return i;
  }
  ParameterList &parameters() override
  {
    return params;
  }
  void evaluate(EvalContext &ctx) override
  {
    auto out = std::make_shared<float>(params.getOr<float>(Token("v"), 0.0f));
    Value v;
    v.type = PortType{Token("scalar")};
    v.residency = hostResidency();
    v.payload = out;
    ctx.setOutput(Token("out"), v);
  }
};

} // namespace

SCENARIO("tsd::graph::Evaluator runs pullAsync on a worker", "[graph-async]")
{
  Graph g;
  auto src = g.addNode(std::make_unique<ConstSource>());
  g.node(src)->impl->parameters().set(Token("v"), 5.0f);

  Evaluator e(g);

  WHEN("pulling asynchronously and polling to completion")
  {
    std::atomic<int> cbCount{0};
    std::atomic<bool> cbOk{false};
    PullHandle h = e.pullAsync(src, [&](bool ok) {
      cbCount++;
      cbOk = ok;
    });

    // Bounded spin: fails fast (rather than hanging CI) if a regression ever
    // leaves the worker stuck without advancing m_doneEpoch.
    long spins = 0;
    const long kMaxSpins = 2000000000L;
    while (!e.isReady(h) && spins < kMaxSpins)
      ++spins;

    THEN("the pull is ready, succeeded, and produced the value")
    {
      REQUIRE(spins < kMaxSpins);
      REQUIRE(e.isReady(h));
      REQUIRE(e.result(h));
      const Value *out = e.output(src, Token("out"), hostResidency());
      REQUIRE(out != nullptr);
      REQUIRE(*std::static_pointer_cast<float>(out->payload) == 5.0f);
    }
    THEN("the completion callback fired exactly once with success")
    {
      e.waitIdle();
      REQUIRE(cbCount.load() == 1);
      REQUIRE(cbOk.load());
    }
  }

  WHEN("using the blocking pull() (Phase 1 compatibility)")
  {
    THEN("it returns true and the value is available immediately after")
    {
      REQUIRE(e.pull(src));
      const Value *out = e.output(src, Token("out"), hostResidency());
      REQUIRE(*std::static_pointer_cast<float>(out->payload) == 5.0f);
    }
  }
}
