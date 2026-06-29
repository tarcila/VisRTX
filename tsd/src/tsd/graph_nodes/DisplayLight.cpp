// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include "tsd/graph/Evaluator.hpp"
#include "tsd/graph_nodes/BuiltinNodes.hpp"
#include "tsd/graph_nodes/Descriptors.hpp"
#include "tsd/graph_nodes/DisplayMask.hpp"

namespace tsd::graph_nodes {
namespace {

using namespace tsd::graph;
using tsd::core::Token;
using float2 = tsd::core::math::float2;
using float3 = tsd::core::math::float3;

struct DisplayLight : Node
{
  ParameterList params;
  DisplayLight()
  {
    params.set(Token("viewportMask"), kDefaultViewportMask);
    params.set(Token("color"), float3(1.f, 1.f, 1.f));
    params.set(Token("irradiance"), 1.f);
    params.set(Token("direction"), float2(0.f, 240.f)); // azimuth/elevation deg
  }
  NodeTypeInfo typeInfo() const override
  {
    NodeTypeInfo i;
    i.name = Token("DisplayLight");
    i.category = Token("sink");
    i.outputs.push_back({Token("out"), PortType{portRenderable()}, true, {}});
    return i;
  }
  ParameterList &parameters() override
  {
    return params;
  }
  void evaluate(EvalContext &ctx) override
  {
    auto r = std::make_shared<Renderable>();
    r->kind = Renderable::Kind::Light;
    r->primSubtype = Token("directional");
    r->appearance.scalars.push_back({Token("color"),
        tsd::core::Any(params.getOr<float3>(Token("color"), float3(1.f)))});
    r->appearance.scalars.push_back({Token("irradiance"),
        tsd::core::Any(params.getOr<float>(Token("irradiance"), 1.f))});
    r->appearance.scalars.push_back({Token("direction"),
        tsd::core::Any(
            params.getOr<float2>(Token("direction"), float2(0.f, 240.f)))});

    Value out;
    out.type = PortType{portRenderable()};
    out.residency = hostResidency();
    out.payload = r;
    ctx.setOutput(Token("out"), out);
  }
};

} // namespace

void registerDisplayLight(NodeRegistry &reg)
{
  reg.registerType(
      Token("DisplayLight"), [] { return std::make_unique<DisplayLight>(); });
}

} // namespace tsd::graph_nodes
