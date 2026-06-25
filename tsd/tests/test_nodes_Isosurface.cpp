// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "catch.hpp"

#ifdef TSD_GRAPH_NODES_HAVE_VISKORES

#include <cmath>
#include <memory>
#include "tsd/graph/Evaluator.hpp"
#include "tsd/graph_nodes/BuiltinNodes.hpp"
#include "tsd/graph_nodes/Descriptors.hpp"

using tsd::core::Token;
using namespace tsd::graph;
using tsd::graph_nodes::Field;
using tsd::graph_nodes::SurfaceData;
using uint3 = tsd::core::math::uint3;
using float3 = tsd::core::math::float3;

namespace {

// Values are a sphere distance field v = R - |p|, sampled over [-1,1]^3.
// An isovalue in (0,R) yields a closed surface; outside the range yields none.
struct SphereField : Node
{
  ParameterList params;
  NodeTypeInfo typeInfo() const override
  {
    NodeTypeInfo i;
    i.name = Token("SphereField");
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
        for (uint32_t x = 0; x < N; ++x, ++idx) {
          const float px = f->origin.x + f->spacing.x * x;
          const float py = f->origin.y + f->spacing.y * y;
          const float pz = f->origin.z + f->spacing.z * z;
          const float r = std::sqrt(px * px + py * py + pz * pz);
          f->data.get<float>(idx) = 0.6f - r; // R = 0.6
        }
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

const tsd::core::AnyArray *findArray(const RenderableParams &p, Token n)
{
  for (const auto &a : p.arrays)
    if (a.first == n)
      return &a.second;
  return nullptr;
}

} // namespace

SCENARIO("IsosurfaceExtract contours a scalar field into a triangle mesh",
    "[nodes-isosurface]")
{
  Graph g;
  auto fld = g.addNode(std::make_unique<SphereField>());
  auto iso = addBuiltin(g, "IsosurfaceExtract");
  REQUIRE(g.node(iso) != nullptr); // registered when Viskores is present
  g.connect(fld, Token("out"), iso, Token("in"));

  GIVEN("an isovalue inside the field range")
  {
    g.node(iso)->impl->parameters().set(Token("isovalue"), 0.2f);
    g.node(iso)->impl->parameters().set(Token("computeNormals"), true);
    Evaluator e(g);
    REQUIRE(e.pull(iso));

    auto s = std::static_pointer_cast<SurfaceData>(
        e.output(iso, Token("out"), hostResidency())->payload);
    THEN("it emits a non-empty triangle surface, valid indices, and normals")
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
      REQUIRE(nrm->size() == pos->size());
      const size_t nv = pos->size();
      for (size_t t = 0; t < idx->size(); ++t) {
        const uint3 tri = idx->get<uint3>(t);
        REQUIRE(tri.x < nv);
        REQUIRE(tri.y < nv);
        REQUIRE(tri.z < nv);
      }
      for (size_t i = 0; i < nv; ++i) {
        const float3 p = pos->get<float3>(i);
        REQUIRE(p.x >= -1.001f);
        REQUIRE(p.x <= 1.001f);
        REQUIRE(p.y >= -1.001f);
        REQUIRE(p.y <= 1.001f);
        REQUIRE(p.z >= -1.001f);
        REQUIRE(p.z <= 1.001f);
      }
    }
  }

  GIVEN("an isovalue outside the field range")
  {
    g.node(iso)->impl->parameters().set(Token("isovalue"), 100.f);
    Evaluator e(g);
    REQUIRE(e.pull(iso));
    auto s = std::static_pointer_cast<SurfaceData>(
        e.output(iso, Token("out"), hostResidency())->payload);
    THEN("it emits an empty surface without crashing")
    {
      REQUIRE(s != nullptr);
      REQUIRE(s->geomSubtype == Token("triangle"));
      const auto *pos = findArray(s->prim, Token("vertex.position"));
      REQUIRE((pos == nullptr || pos->size() == 0));
    }
  }
}

#endif // TSD_GRAPH_NODES_HAVE_VISKORES
