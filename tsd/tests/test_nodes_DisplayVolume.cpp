// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include "catch.hpp"
#include "tsd/graph/Evaluator.hpp"
#include "tsd/graph_nodes/BuiltinNodes.hpp"
#include "tsd/graph_nodes/Descriptors.hpp"

using tsd::core::Token;
using namespace tsd::graph;
using tsd::graph_nodes::Field;
using tsd::graph_nodes::TransferFunctionData;
using uint3 = tsd::core::math::uint3;
using float2 = tsd::core::math::float2;
using float4 = tsd::core::math::float4;

namespace {
bool hasArray(const RenderableParams &p, Token n)
{
  for (auto &a : p.arrays)
    if (a.first == n)
      return true;
  return false;
}
bool hasScalar(const RenderableParams &p, Token n)
{
  for (auto &s : p.scalars)
    if (s.first == n)
      return true;
  return false;
}
struct KnownField : Node
{
  ParameterList params;
  NodeTypeInfo typeInfo() const override
  {
    NodeTypeInfo i;
    i.name = Token("KnownField");
    i.outputs.push_back({Token("out"), PortType{Token("field")}, true, {}});
    return i;
  }
  ParameterList &parameters() override
  {
    return params;
  }
  void evaluate(EvalContext &ctx) override
  {
    auto f = std::make_shared<Field>();
    f->dims = uint3(2u, 2u, 2u);
    f->data = tsd::core::AnyArray(ANARI_FLOAT32, 8);
    for (int k = 0; k < 8; ++k)
      f->data.get<float>(k) = float(k) / 7.f;
    Value v;
    v.type = PortType{Token("field")};
    v.residency = hostResidency();
    v.payload = f;
    ctx.setOutput(Token("out"), v);
  }
};
struct EmitTF : Node
{
  ParameterList params;
  NodeTypeInfo typeInfo() const override
  {
    NodeTypeInfo i;
    i.name = Token("EmitTF");
    i.outputs.push_back(
        {Token("out"), PortType{Token("transferFunction")}, true, {}});
    return i;
  }
  ParameterList &parameters() override
  {
    return params;
  }
  void evaluate(EvalContext &ctx) override
  {
    auto d = std::make_shared<TransferFunctionData>();
    d->valueRange = float2(0.f, 1.f);
    d->colormap = tsd::core::AnyArray(ANARI_FLOAT32_VEC4, 4);
    for (int k = 0; k < 4; ++k)
      d->colormap.get<float4>(k) = float4(float(k) / 3.f);
    Value v;
    v.type = PortType{Token("transferFunction")};
    v.residency = hostResidency();
    v.payload = d;
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

SCENARIO("DisplayVolume packs field+TF into a Renderable", "[nodes-dvol]")
{
  Graph g;
  auto fld = g.addNode(std::make_unique<KnownField>());
  auto tf = g.addNode(std::make_unique<EmitTF>());
  auto dv = addBuiltin(g, "DisplayVolume");
  g.connect(fld, Token("out"), dv, Token("field"));
  g.connect(tf, Token("out"), dv, Token("tf"));
  Evaluator e(g);

  REQUIRE(e.pull(dv));
  auto r = std::static_pointer_cast<Renderable>(
      e.output(dv, Token("out"), hostResidency())->payload);
  THEN("it is a structuredRegular volume renderable")
  {
    REQUIRE(r->kind == Renderable::Kind::Volume);
    REQUIRE(r->primSubtype == Token("structuredRegular"));
    REQUIRE(hasScalar(r->prim, Token("dims")));
    REQUIRE(hasArray(r->prim, Token("data")));
    REQUIRE(hasScalar(r->prim, Token("origin")));
    REQUIRE(hasArray(r->appearance, Token("color")));
  }
}
