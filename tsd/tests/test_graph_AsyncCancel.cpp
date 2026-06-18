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
using tsd::graph::EvalState;
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

// Spins until cancelled() or `release` is set; CAP prevents any hang.
struct SlowNode : Node
{
  ParameterList params;
  std::atomic<bool> *release;
  std::atomic<bool> finished{false};
  explicit SlowNode(std::atomic<bool> *r) : release(r) {}
  NodeTypeInfo typeInfo() const override
  {
    NodeTypeInfo i;
    i.name = Token("Slow");
    i.outputs.push_back({Token("out"), PortType{Token("scalar")}, true, {}});
    i.isCacheable = false; // always re-run so the second pull recomputes
    return i;
  }
  ParameterList &parameters() override
  {
    return params;
  }
  void evaluate(EvalContext &ctx) override
  {
    const long CAP = 2000000000L;
    for (long i = 0; i < CAP; ++i) {
      if (ctx.cancelled())
        return; // bail without setOutput
      if (release->load())
        break;
    }
    auto out = std::make_shared<float>(1.0f);
    Value v;
    v.type = PortType{Token("scalar")};
    v.residency = hostResidency();
    v.payload = out;
    ctx.setOutput(Token("out"), v);
    finished.store(true);
  }
};

} // namespace

SCENARIO("tsd::graph::Evaluator cancels a running pull cooperatively",
    "[graph-asynccancel]")
{
  std::atomic<bool> release{false};
  Graph g;
  auto id = g.addNode(std::make_unique<SlowNode>(&release));
  auto *slow = static_cast<SlowNode *>(g.node(id)->impl.get());

  Evaluator e(g);

  WHEN("a running pull is cancelled before release")
  {
    PullHandle h = e.pullAsync(id);
    e.cancel(); // cooperative: SlowNode sees cancelled() and bails
    e.waitIdle(); // worker observed cancel and returned
    THEN("the pull did not succeed and the node did not finish")
    {
      REQUIRE(e.isReady(h));
      REQUIRE_FALSE(e.result(h));
      REQUIRE_FALSE(slow->finished.load());
    }
    THEN("the worker recovers: a released blocking pull completes")
    {
      release.store(true);
      REQUIRE(e.pull(id));
      const Value *out = e.output(id, Token("out"), hostResidency());
      REQUIRE(out != nullptr);
      REQUIRE(slow->finished.load());
    }
  }
}
