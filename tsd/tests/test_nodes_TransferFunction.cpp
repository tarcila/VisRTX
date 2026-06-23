// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include <memory>
#include <string>
#include "catch.hpp"
#include "tsd/graph/Evaluator.hpp"
#include "tsd/graph_nodes/BuiltinNodes.hpp"
#include "tsd/graph_nodes/Descriptors.hpp"
#include "tsd/graph_nodes/GraphEditModel.hpp"
#include "tsd/graph_nodes/TransferFunctionNode.hpp"

using tsd::core::Token;
using namespace tsd::graph;
using tsd::graph_nodes::ITransferFunctionNode;
using tsd::graph_nodes::TransferFunctionData;
using float2 = tsd::core::math::float2;
using float4 = tsd::core::math::float4;

namespace {
struct EmitRange : Node
{
  ParameterList params;
  NodeTypeInfo typeInfo() const override
  {
    NodeTypeInfo i;
    i.name = Token("EmitRange");
    i.outputs.push_back({Token("out"), PortType{Token("range")}, true, {}});
    return i;
  }
  ParameterList &parameters() override
  {
    return params;
  }
  void evaluate(EvalContext &ctx) override
  {
    Value v;
    v.type = PortType{Token("range")};
    v.residency = hostResidency();
    v.payload = std::make_shared<float2>(0.f, 1.f);
    ctx.setOutput(Token("out"), v);
  }
};
NodeId addBuiltin(Graph &g, const char *t)
{
  static NodeRegistry reg = [] {
    NodeRegistry r;
    tsd::graph_nodes::registerBuiltinNodes(r);
    return r;
  }();
  return g.addNode(reg.create(Token(t)));
}
} // namespace

SCENARIO("TransferFunction builds a colormap from a range", "[nodes-tf]")
{
  Graph g;
  auto rng = g.addNode(std::make_unique<EmitRange>());
  auto tf = addBuiltin(g, "TransferFunction");
  // Drive state via ITransferFunctionNode (preset path removed).
  auto *itf = dynamic_cast<ITransferFunctionNode *>(g.node(tf)->impl.get());
  REQUIRE(itf != nullptr);
  itf->tfState().colorPoints = {{0.f, 0.f, 0.f, 0.f}, {1.f, 1.f, 1.f, 1.f}};
  itf->tfState().opacityPoints = {{0.f, 1.f}, {1.f, 1.f}};
  itf->samples() = 8;
  g.connect(rng, Token("out"), tf, Token("in"));
  Evaluator e(g);

  REQUIRE(e.pull(tf));
  auto d = std::static_pointer_cast<TransferFunctionData>(
      e.output(tf, Token("out"), hostResidency())->payload);
  THEN("colormap is float4 x8 with black-to-white ramp and valueRange {0,1}")
  {
    REQUIRE(d->colormap.elementType() == ANARI_FLOAT32_VEC4);
    REQUIRE(d->colormap.size() == 8);
    REQUIRE(d->valueRange.x == 0.f);
    REQUIRE(d->valueRange.y == 1.f);
    REQUIRE(d->colormap.get<float4>(0).x == Approx(0.f));
    REQUIRE(d->colormap.get<float4>(7).x == Approx(1.f));
  }
}

SCENARIO("TransferFunction node samples its editable control points",
    "[nodes-tf-state]")
{
  using namespace tsd::graph;
  using tsd::core::Token;
  NodeRegistry reg;
  tsd::graph_nodes::registerBuiltinNodes(reg);
  Graph g;
  const NodeId src = g.addNode(reg.create(Token("GenerateNoiseVolume")));
  const NodeId sr = g.addNode(reg.create(Token("ScalarRange")));
  const NodeId tf = g.addNode(reg.create(Token("TransferFunction")));
  REQUIRE(g.connect(src, Token("out"), sr, Token("in")).ok);
  REQUIRE(g.connect(sr, Token("out"), tf, Token("in")).ok);

  // The TF node exposes editable state.
  auto *node = g.node(tf)->impl.get();
  auto *itf = dynamic_cast<tsd::graph_nodes::ITransferFunctionNode *>(node);
  REQUIRE(itf != nullptr);

  // Set a known black->white ramp, opaque.
  itf->tfState().colorPoints = {{0.f, 0.f, 0.f, 0.f}, {1.f, 1.f, 1.f, 1.f}};
  itf->tfState().opacityPoints = {{0.f, 1.f}, {1.f, 1.f}};
  itf->samples() = 4;
  g.markDirty(tf);

  Evaluator e(g);
  REQUIRE(e.pull(tf));
  // The evaluated output has a 4-entry colormap; last entry is white & opaque.
  auto d = std::static_pointer_cast<TransferFunctionData>(
      e.output(tf, Token("out"), hostResidency())->payload);
  REQUIRE(d != nullptr);
  REQUIRE(d->colormap.size() == 4);
  REQUIRE(d->colormap.get<float4>(3).x == Approx(1.f));
  REQUIRE(d->colormap.get<float4>(3).y == Approx(1.f));
  REQUIRE(d->colormap.get<float4>(3).z == Approx(1.f));
  REQUIRE(d->colormap.get<float4>(3).w == Approx(1.f));
}
