// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/graph/Evaluator.hpp"
#include "tsd/graph_nodes/BuiltinNodes.hpp"
#include "tsd/graph_nodes/Descriptors.hpp"
// std
#include <functional>
#include <memory>

namespace tsd::graph_nodes {
namespace {

using namespace tsd::graph;
using tsd::core::Token;
using float2 = tsd::core::math::float2;

struct ScalarRange : Node
{
  ParameterList params;
  NodeTypeInfo typeInfo() const override
  {
    NodeTypeInfo i;
    i.name = Token("ScalarRange");
    i.category = Token("processor");
    i.inputs.push_back({Token("in"), PortType{portField()}, true, {}});
    i.outputs.push_back({Token("out"), PortType{portRange()}, true, {}});
    return i;
  }
  ParameterList &parameters() override
  {
    return params;
  }
  void evaluate(EvalContext &ctx) override
  {
    auto in = ctx.input(Token("in"), hostResidency());
    auto f = std::static_pointer_cast<Field>(in.payload);
    if (!f) {
      ctx.fail("ScalarRange: missing field input");
      return;
    }
    const size_t expected = size_t(f->dims.x) * f->dims.y * f->dims.z;
    if (f->data.size() == 0 || f->data.size() != expected) {
      ctx.fail("ScalarRange: field data size does not match dims");
      return;
    }
    float lo = f->data.get<float>(0), hi = lo;
    for (size_t i = 1; i < f->data.size(); ++i) {
      const float v = f->data.get<float>(i);
      lo = v < lo ? v : lo;
      hi = v > hi ? v : hi;
    }
    auto r = std::make_shared<float2>(lo, hi);
    Value out;
    out.type = PortType{portRange()};
    out.residency = hostResidency();
    out.payload = r;
    out.contentTag = (uint64_t(std::hash<float>{}(lo)) << 32)
        ^ uint64_t(std::hash<float>{}(hi));
    ctx.setOutput(Token("out"), out);
  }
};

} // namespace

void registerScalarRange(NodeRegistry &reg)
{
  reg.registerType(
      Token("ScalarRange"), [] { return std::make_unique<ScalarRange>(); });
}

} // namespace tsd::graph_nodes
