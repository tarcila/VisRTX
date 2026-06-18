// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <memory>
#include "tsd/graph/Evaluator.hpp"
#include "tsd/graph_nodes/BuiltinNodes.hpp"
#include "tsd/graph_nodes/Descriptors.hpp"

namespace tsd::graph_nodes {
namespace {

using namespace tsd::graph;
using tsd::core::Token;
using float3 = tsd::core::math::float3;

struct BoundingBox : Node
{
  ParameterList params;
  NodeTypeInfo typeInfo() const override
  {
    NodeTypeInfo i;
    i.name = Token("BoundingBox");
    i.category = Token("processor");
    i.inputs.push_back({Token("in"), PortType{portField()}, true, {}});
    i.outputs.push_back({Token("out"), PortType{portSurface()}, true, {}});
    return i;
  }
  ParameterList &parameters() override
  {
    return params;
  }
  void evaluate(EvalContext &ctx) override
  {
    auto f = std::static_pointer_cast<Field>(
        ctx.input(Token("in"), hostResidency()).payload);
    if (!f) {
      ctx.fail("BoundingBox: missing field input");
      return;
    }
    const float3 lo = f->origin;
    const float3 hi = float3(lo.x + f->spacing.x * f->dims.x,
        lo.y + f->spacing.y * f->dims.y,
        lo.z + f->spacing.z * f->dims.z);
    const float3 c[8] = {{lo.x, lo.y, lo.z},
        {hi.x, lo.y, lo.z},
        {hi.x, hi.y, lo.z},
        {lo.x, hi.y, lo.z},
        {lo.x, lo.y, hi.z},
        {hi.x, lo.y, hi.z},
        {hi.x, hi.y, hi.z},
        {lo.x, hi.y, hi.z}};
    const int tri[36] = {0,
        1,
        2,
        0,
        2,
        3,
        4,
        6,
        5,
        4,
        7,
        6,
        0,
        4,
        5,
        0,
        5,
        1,
        1,
        5,
        6,
        1,
        6,
        2,
        2,
        6,
        7,
        2,
        7,
        3,
        3,
        7,
        4,
        3,
        4,
        0};
    tsd::core::AnyArray pos(ANARI_FLOAT32_VEC3, 36);
    for (int i = 0; i < 36; ++i)
      pos.get<float3>(size_t(i)) = c[tri[i]];

    auto s = std::make_shared<SurfaceData>();
    s->geomSubtype = Token("triangle");
    s->prim.arrays.push_back({Token("vertex.position"), pos});
    s->appearance.scalars.push_back({Token("color"),
        tsd::core::Any(params.getOr<float3>(Token("color"), float3(0.8f)))});

    Value out;
    out.type = PortType{portSurface()};
    out.residency = hostResidency();
    out.payload = s;
    ctx.setOutput(Token("out"), out);
  }
};

} // namespace

void registerBoundingBox(NodeRegistry &reg)
{
  reg.registerType(
      Token("BoundingBox"), [] { return std::make_unique<BoundingBox>(); });
}

} // namespace tsd::graph_nodes
