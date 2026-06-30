// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "catch.hpp"

#ifdef TSD_GRAPH_NODES_HAVE_VISKORES

#include <memory>
#include "tsd/graph/Evaluator.hpp"
#include "tsd/graph_nodes/BuiltinNodes.hpp"
#include "tsd/graph_nodes/Descriptors.hpp"

using tsd::core::Token;
using namespace tsd::graph;
using tsd::graph_nodes::Field;
using tsd::graph_nodes::SurfaceData;
using tsd::graph_nodes::TransferFunctionData;
using uint3 = tsd::core::math::uint3;
using float2 = tsd::core::math::float2;
using float3 = tsd::core::math::float3;
using float4 = tsd::core::math::float4;

namespace {

// Linear ramp v = x over [-1,1]^3, so a Z=0 slice spans the full value range.
struct RampField : Node
{
  ParameterList params;
  NodeTypeInfo typeInfo() const override
  {
    NodeTypeInfo i;
    i.name = Token("RampField");
    i.outputs.push_back({Token("out"), PortType{Token("field")}, true, {}});
    return i;
  }
  ParameterList &parameters() override
  {
    return params;
  }
  void evaluate(EvalContext &ctx) override
  {
    const uint32_t N = 16;
    auto f = std::make_shared<Field>();
    f->dims = uint3(N, N, N);
    f->origin = float3(-1.f, -1.f, -1.f);
    f->spacing = float3(2.f / (N - 1), 2.f / (N - 1), 2.f / (N - 1));
    f->data = tsd::core::AnyArray(ANARI_FLOAT32, size_t(N) * N * N);
    size_t idx = 0;
    for (uint32_t z = 0; z < N; ++z)
      for (uint32_t y = 0; y < N; ++y)
        for (uint32_t x = 0; x < N; ++x, ++idx)
          f->data.get<float>(idx) = f->origin.x + f->spacing.x * x; // = px
    Value v;
    v.type = PortType{Token("field")};
    v.residency = hostResidency();
    v.payload = f;
    ctx.setOutput(Token("out"), v);
  }
};

// Two-entry blue→red colormap over value range [-1,1].
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
    d->valueRange = float2(-1.f, 1.f);
    d->colormap = tsd::core::AnyArray(ANARI_FLOAT32_VEC4, 2);
    d->colormap.get<float4>(0) = float4(0.f, 0.f, 1.f, 1.f);
    d->colormap.get<float4>(1) = float4(1.f, 0.f, 0.f, 1.f);
    Value v;
    v.type = PortType{Token("transferFunction")};
    v.residency = hostResidency();
    v.payload = d;
    ctx.setOutput(Token("out"), v);
  }
};

const tsd::core::AnyArray *findArray(const RenderableParams &p, Token n)
{
  for (const auto &a : p.arrays)
    if (a.first == n)
      return &a.second;
  return nullptr;
}

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

SCENARIO("CrossSection slices a field into a triangle surface", "[nodes-slice]")
{
  Graph g;
  auto fld = g.addNode(std::make_unique<RampField>());
  auto cs = addBuiltin(g, "CrossSection");
  REQUIRE(g.node(cs) != nullptr); // registered when Viskores is present
  g.connect(fld, Token("out"), cs, Token("in"));

  GIVEN("a Z=0 plane through the volume, no TF")
  {
    Evaluator e(g);
    REQUIRE(e.pull(cs));
    auto s = std::static_pointer_cast<SurfaceData>(
        e.output(cs, Token("out"), hostResidency())->payload);
    THEN("it emits a non-empty triangle surface with normals and no colors")
    {
      REQUIRE(s != nullptr);
      REQUIRE(s->geomSubtype == Token("triangle"));
      const auto *pos = findArray(s->prim, Token("vertex.position"));
      const auto *idx = findArray(s->prim, Token("primitive.index"));
      const auto *nrm = findArray(s->prim, Token("vertex.normal"));
      REQUIRE(pos != nullptr);
      REQUIRE(idx != nullptr);
      REQUIRE(nrm != nullptr);
      REQUIRE(pos->size() > 0);
      REQUIRE(idx->size() > 0);
      REQUIRE(findArray(s->prim, Token("vertex.color")) == nullptr);
      const size_t nv = pos->size();
      for (size_t t = 0; t < idx->size(); ++t) {
        const uint3 tri = idx->get<uint3>(t);
        REQUIRE(tri.x < nv);
        REQUIRE(tri.y < nv);
        REQUIRE(tri.z < nv);
      }
    }
  }

  GIVEN("a transfer function wired into the tf input")
  {
    auto tf = g.addNode(std::make_unique<EmitTF>());
    g.connect(tf, Token("out"), cs, Token("tf"));
    Evaluator e(g);
    REQUIRE(e.pull(cs));
    auto s = std::static_pointer_cast<SurfaceData>(
        e.output(cs, Token("out"), hostResidency())->payload);
    THEN("the slice carries per-vertex colors in [0,1]")
    {
      const auto *pos = findArray(s->prim, Token("vertex.position"));
      const auto *col = findArray(s->prim, Token("vertex.color"));
      REQUIRE(pos != nullptr);
      REQUIRE(col != nullptr);
      REQUIRE(col->size() == pos->size());
      REQUIRE(col->elementType() == ANARI_FLOAT32_VEC4);
      for (size_t i = 0; i < col->size(); ++i) {
        const float4 c = col->get<float4>(i);
        REQUIRE(c.x >= 0.f);
        REQUIRE(c.x <= 1.f);
        REQUIRE(c.w >= 0.f);
        REQUIRE(c.w <= 1.f);
      }
    }
  }
}

#endif // TSD_GRAPH_NODES_HAVE_VISKORES
