// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <memory>
#include <string>
#include "tsd/core/Logging.hpp"
#include "tsd/graph/Evaluator.hpp"
#include "tsd/graph_nodes/BuiltinNodes.hpp"
#include "tsd/graph_nodes/Descriptors.hpp"

namespace tsd::graph_nodes {
namespace {

using namespace tsd::graph;
using tsd::core::Token;
using float2 = tsd::core::math::float2;
using float4 = tsd::core::math::float4;

std::string getPreset(const ParameterList &params)
{
  for (const auto &p : params.items()) {
    if (p.name == Token("preset") && p.value.is(ANARI_STRING))
      return p.value.getString();
  }
  return "coolToWarm";
}

struct TransferFunction : Node
{
  ParameterList params;
  NodeTypeInfo typeInfo() const override
  {
    NodeTypeInfo i;
    i.name = Token("TransferFunction");
    i.category = Token("processor");
    i.inputs.push_back({Token("in"), PortType{portRange()}, true, {}});
    i.outputs.push_back({Token("out"), PortType{portTF()}, true, {}});
    return i;
  }
  ParameterList &parameters() override
  {
    return params;
  }
  void evaluate(EvalContext &ctx) override
  {
    auto in = ctx.input(Token("in"), hostResidency());
    auto range = std::static_pointer_cast<float2>(in.payload);
    if (!range) {
      ctx.fail("TransferFunction: missing range input");
      return;
    }
    const int samples = params.getOr<int>(Token("samples"), 256);
    if (samples < 2) {
      ctx.fail("TransferFunction: samples must be >= 2");
      return;
    }
    const std::string preset = getPreset(params);
    const bool grayscale = (preset == "grayscale");
    bool coolToWarm = (preset == "coolToWarm");
    if (!grayscale && !coolToWarm) {
      tsd::core::logWarning(
          "[TransferFunction] unknown preset '%s', using grayscale",
          preset.c_str());
    }

    auto d = std::make_shared<TransferFunctionData>();
    d->valueRange = *range;
    d->colormap = tsd::core::AnyArray(ANARI_FLOAT32_VEC4, size_t(samples));
    for (int i = 0; i < samples; ++i) {
      const float t = float(i) / float(samples - 1);
      float4 c;
      if (coolToWarm)
        c = float4(t, 1.f - std::abs(0.5f - t) * 2.f, 1.f - t, t);
      else
        c = float4(t, t, t, t);
      d->colormap.get<float4>(size_t(i)) = c;
    }

    Value out;
    out.type = PortType{portTF()};
    out.residency = hostResidency();
    out.payload = d;
    ctx.setOutput(Token("out"), out);
  }
};

} // namespace

void registerTransferFunction(NodeRegistry &reg)
{
  reg.registerType(Token("TransferFunction"),
      [] { return std::make_unique<TransferFunction>(); });
}

} // namespace tsd::graph_nodes
