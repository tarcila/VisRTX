// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/graph/Evaluator.hpp"
#include "tsd/graph_nodes/BuiltinNodes.hpp"
#include "tsd/graph_nodes/ProceduralField.hpp"
// std
#include <cmath>
#include <memory>

namespace tsd::graph_nodes {
namespace {

using namespace tsd::graph;
using tsd::core::Token;
using uint3 = tsd::core::math::uint3;

// Gyroid (triply-periodic minimal surface): a shell field bright near the
// implicit surface G(p)=0, so it renders as an intricate lattice in both the
// transfer-function volume and the isosurface paths.
struct GenerateGyroid : Node
{
  ParameterList params;
  GenerateGyroid()
  {
    params.set(Token("dims"), uint3(64u, 64u, 64u));
    params.set(Token("frequency"), 3.f); // cycles across the unit cube
    params.set(Token("thickness"), 0.6f); // shell width around G=0
    params.set(Token("phase"), 0.f); // animatable offset
    params.set(Token("warp"), 0.f); // domain-warp amount
  }
  NodeTypeInfo typeInfo() const override
  {
    NodeTypeInfo i;
    i.name = Token("GenerateGyroid");
    i.category = Token("source");
    i.outputs.push_back({Token("out"), PortType{portField()}, true, {}});
    return i;
  }
  ParameterList &parameters() override
  {
    return params;
  }
  void evaluate(EvalContext &ctx) override
  {
    const uint3 dims = params.getOr<uint3>(Token("dims"), uint3(64u));
    const float freq = params.getOr<float>(Token("frequency"), 3.f);
    const float thickness = params.getOr<float>(Token("thickness"), 0.6f);
    const float phase = params.getOr<float>(Token("phase"), 0.f);
    const float warp = params.getOr<float>(Token("warp"), 0.f);
    if (dims.x == 0u || dims.y == 0u || dims.z == 0u) {
      ctx.fail("GenerateGyroid: dims must be > 0 on each axis");
      return;
    }

    auto f = std::make_shared<Field>(makeUnitField(dims));
    const float k = freq * kPi;
    const float band = thickness < 1e-3f ? 1e-3f : thickness;

    size_t idx = 0;
    for (uint32_t z = 0; z < dims.z; ++z) {
      const float pz = normCoord(z, dims.z);
      for (uint32_t y = 0; y < dims.y; ++y) {
        const float py = normCoord(y, dims.y);
        for (uint32_t x = 0; x < dims.x; ++x, ++idx) {
          const float px = normCoord(x, dims.x);
          // Cheap domain warp so the lattice ripples with `warp`/`phase`.
          const float wx = px + warp * std::sin(k * py + phase);
          const float wy = py + warp * std::sin(k * pz + phase);
          const float wz = pz + warp * std::sin(k * px + phase);
          const float g = std::sin(k * wx) * std::cos(k * wy)
              + std::sin(k * wy) * std::cos(k * wz)
              + std::sin(k * wz) * std::cos(k * wx);
          f->data.get<float>(idx) = clamp01(1.f - std::fabs(g) / band);
        }
      }
    }

    Value out;
    out.type = PortType{portField()};
    out.residency = hostResidency();
    out.payload = f;
    ctx.setOutput(Token("out"), out);
  }
};

} // namespace

void registerGenerateGyroid(NodeRegistry &reg)
{
  reg.registerType(Token("GenerateGyroid"),
      [] { return std::make_unique<GenerateGyroid>(); });
}

} // namespace tsd::graph_nodes
