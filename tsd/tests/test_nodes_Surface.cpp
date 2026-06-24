// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/graph/Evaluator.hpp"
#include "tsd/graph_nodes/BuiltinNodes.hpp"
#include "tsd/graph_nodes/Descriptors.hpp"
// std
#include <memory>

using tsd::core::Token;
using namespace tsd::graph;
using tsd::graph_nodes::Field;
using tsd::graph_nodes::SurfaceData;
using uint3 = tsd::core::math::uint3;
using float3 = tsd::core::math::float3;

namespace {

bool hasArray(const RenderableParams &p, Token n)
{
  for (auto &a : p.arrays)
    if (a.first == n)
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
    f->origin = float3(-1.f, -1.f, -1.f);
    f->spacing = float3(1.f, 1.f, 1.f);
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

SCENARIO(
    "BoundingBox -> DisplaySurface produces a cylinder wireframe renderable",
    "[nodes-surface]")
{
  Graph g;
  auto fld = g.addNode(std::make_unique<KnownField>());
  auto bb = addBuiltin(g, "BoundingBox");
  g.node(bb)->impl->parameters().set(Token("color"), float3(0.2f, 0.8f, 0.2f));
  auto ds = addBuiltin(g, "DisplaySurface");
  g.connect(fld, Token("out"), bb, Token("in"));
  g.connect(bb, Token("out"), ds, Token("in"));
  Evaluator e(g);

  REQUIRE(e.pull(ds));

  WHEN("inspecting the BoundingBox surface output")
  {
    auto s = std::static_pointer_cast<SurfaceData>(
        e.output(bb, Token("out"), hostResidency())->payload);
    THEN("it is a cylinder wireframe with 24 vertex positions and a radius")
    {
      REQUIRE(s != nullptr);
      REQUIRE(s->geomSubtype == Token("cylinder"));
      // 12 box edges x 2 endpoints
      bool foundPos = false, foundRadius = false;
      for (const auto &a : s->prim.arrays)
        if (a.first == Token("vertex.position")) {
          REQUIRE(a.second.size() == 24);
          foundPos = true;
        }
      for (const auto &sc : s->prim.scalars)
        if (sc.first == Token("radius")) {
          REQUIRE(sc.second.get<float>() > 0.f);
          foundRadius = true;
        }
      REQUIRE(foundPos);
      REQUIRE(foundRadius);
    }
  }

  WHEN("inspecting the DisplaySurface renderable output")
  {
    auto r = std::static_pointer_cast<Renderable>(
        e.output(ds, Token("out"), hostResidency())->payload);
    THEN("it is a Surface renderable carrying the geometry")
    {
      REQUIRE(r->kind == Renderable::Kind::Surface);
      REQUIRE(r->primSubtype == Token("cylinder"));
      REQUIRE(hasArray(r->prim, Token("vertex.position")));
    }
  }
}
