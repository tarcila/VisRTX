// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include "tsd/graph/Evaluator.hpp"
#include "tsd/graph_nodes/BuiltinNodes.hpp"
#include "tsd/graph_nodes/Descriptors.hpp"
#include "tsd/graph_nodes/DisplayMask.hpp"
#include "tsd/graph_nodes/TransformableNode.hpp"

namespace tsd::graph_nodes {
namespace {

using namespace tsd::graph;
using tsd::core::Token;
using float3 = tsd::core::math::float3;

struct DisplayVolume : Node, ITransformableNode
{
  ParameterList params;
  tsd::core::math::mat4 m_transform{tsd::core::math::IDENTITY_MAT4};
  tsd::core::math::mat4 &transform() override
  {
    return m_transform;
  }
  DisplayVolume()
  {
    params.set(tsd::core::Token("viewportMask"), kDefaultViewportMask);
  }
  NodeTypeInfo typeInfo() const override
  {
    NodeTypeInfo i;
    i.name = Token("DisplayVolume");
    i.category = Token("sink");
    i.inputs.push_back({Token("field"), PortType{portField()}, true, {}});
    i.inputs.push_back({Token("tf"), PortType{portTF()}, true, {}});
    i.outputs.push_back({Token("out"), PortType{portRenderable()}, true, {}});
    return i;
  }
  ParameterList &parameters() override
  {
    return params;
  }
  void evaluate(EvalContext &ctx) override
  {
    auto field = std::static_pointer_cast<Field>(
        ctx.input(Token("field"), hostResidency()).payload);
    auto tf = std::static_pointer_cast<TransferFunctionData>(
        ctx.input(Token("tf"), hostResidency()).payload);
    if (!field || !tf) {
      ctx.fail("DisplayVolume: missing field or transferFunction input");
      return;
    }
    if (field->data.size()
        != size_t(field->dims.x) * field->dims.y * field->dims.z) {
      ctx.fail("DisplayVolume: field data size does not match dims");
      return;
    }
    auto r = std::make_shared<Renderable>();
    r->kind = Renderable::Kind::Volume;
    r->primSubtype = Token("structuredRegular");
    r->prim.scalars.push_back({Token("dims"),
        tsd::core::Any(float3(float(field->dims.x),
            float(field->dims.y),
            float(field->dims.z)))});
    r->prim.scalars.push_back({Token("origin"), tsd::core::Any(field->origin)});
    r->prim.scalars.push_back(
        {Token("spacing"), tsd::core::Any(field->spacing)});
    r->prim.arrays.push_back({Token("data"), field->data});
    r->appearance.arrays.push_back({Token("color"), tf->colormap});
    r->appearance.scalars.push_back(
        {Token("valueRange"), tsd::core::Any(tf->valueRange)});

    Value out;
    out.type = PortType{portRenderable()};
    out.residency = hostResidency();
    out.payload = r;
    ctx.setOutput(Token("out"), out);
  }
};

} // namespace

void registerDisplayVolume(NodeRegistry &reg)
{
  reg.registerType(
      Token("DisplayVolume"), [] { return std::make_unique<DisplayVolume>(); });
}

} // namespace tsd::graph_nodes
