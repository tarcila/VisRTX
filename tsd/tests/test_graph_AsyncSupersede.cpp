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

// Emits param "v"; spins until cancelled or released so a pull can be
// superseded.
struct SlowSource : Node
{
  ParameterList params;
  std::atomic<bool> *release;
  std::atomic<int> evals{0};
  explicit SlowSource(std::atomic<bool> *r) : release(r) {}
  NodeTypeInfo typeInfo() const override
  {
    NodeTypeInfo i;
    i.name = Token("SlowSource");
    i.outputs.push_back({Token("out"), PortType{Token("scalar")}, true, {}});
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
        return;
      if (release->load())
        break;
    }
    evals++;
    auto out = std::make_shared<float>(params.getOr<float>(Token("v"), 0.0f));
    Value v;
    v.type = PortType{Token("scalar")};
    v.residency = hostResidency();
    v.payload = out;
    ctx.setOutput(Token("out"), v);
  }
};

} // namespace

SCENARIO("tsd::graph::Evaluator supersedes an in-flight pull after an edit",
    "[graph-supersede]")
{
  std::atomic<bool> release{false};
  Graph g;
  auto id = g.addNode(std::make_unique<SlowSource>(&release));
  auto *src = static_cast<SlowSource *>(g.node(id)->impl.get());
  g.node(id)->impl->parameters().set(Token("v"), 1.0f);

  Evaluator e(g);

  WHEN("a first pull is in flight, then a param edit + second pull happen")
  {
    PullHandle h1 = e.pullAsync(id); // starts running, spins (not released)

    // Mutation protocol: cancel + waitIdle before touching the Graph.
    e.cancel();
    e.waitIdle();
    g.node(id)->impl->parameters().set(Token("v"), 2.0f);
    g.markDirty(id);

    release.store(true); // let the next run finish
    PullHandle h2 = e.pullAsync(id);
    e.waitIdle();

    THEN("the first pull was superseded and the second produced the new value")
    {
      REQUIRE_FALSE(e.result(h1));
      REQUIRE(e.result(h2));
      const Value *out = e.output(id, Token("out"), hostResidency());
      REQUIRE(*std::static_pointer_cast<float>(out->payload) == 2.0f);
    }
    THEN("the superseded run did not publish: exactly one successful eval")
    {
      REQUIRE(src->evals.load() == 1);
    }
  }
}
