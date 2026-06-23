// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include "tsd/graph/Evaluator.hpp"
#include "tsd/graph_nodes/BuiltinNodes.hpp"
#include "tsd/graph_nodes/Descriptors.hpp"
#include "tsd/graph_nodes/GraphEditModel.hpp"
#include "tsd/graph_nodes/TransferFunctionNode.hpp"

namespace tsd::graph_nodes {

using namespace tsd::graph;
using tsd::core::Token;
using float2 = tsd::core::math::float2;
using float4 = tsd::core::math::float4;

// Named (not anonymous) so the inspector can dynamic_cast to
// ITransferFunctionNode.
struct TransferFunctionNode : Node, ITransferFunctionNode
{
  ParameterList params;
  tsd::core::TransferFunction state{tsd::core::makeDefaultTransferFunction()};
  int sampleCount{256};

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
  tsd::core::TransferFunction &tfState() override
  {
    return state;
  }
  int &samples() override
  {
    return sampleCount;
  }

  void evaluate(EvalContext &ctx) override
  {
    auto in = ctx.input(Token("in"), hostResidency());
    auto range = std::static_pointer_cast<float2>(in.payload);
    if (!range) {
      ctx.fail("TransferFunction: missing range input");
      return;
    }
    if (sampleCount < 2) {
      ctx.fail("TransferFunction: samples must be >= 2");
      return;
    }

    auto sampled = GraphEditModel::sampleColormap(
        state.colorPoints, state.opacityPoints, sampleCount);

    auto d = std::make_shared<TransferFunctionData>();
    d->valueRange = *range;
    d->colormap = tsd::core::AnyArray(ANARI_FLOAT32_VEC4, size_t(sampleCount));
    for (int i = 0; i < sampleCount; ++i)
      d->colormap.get<float4>(size_t(i)) = sampled[size_t(i)];

    Value out;
    out.type = PortType{portTF()};
    out.residency = hostResidency();
    out.payload = d;
    ctx.setOutput(Token("out"), out);
  }
};

void registerTransferFunction(NodeRegistry &reg)
{
  reg.registerType(Token("TransferFunction"),
      [] { return std::make_unique<TransferFunctionNode>(); });
}

} // namespace tsd::graph_nodes
