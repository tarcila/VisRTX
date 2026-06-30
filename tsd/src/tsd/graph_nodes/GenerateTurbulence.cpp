// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/graph/Evaluator.hpp"
#include "tsd/graph_nodes/BuiltinNodes.hpp"
#include "tsd/graph_nodes/ProceduralField.hpp"
// std
#include <memory>

namespace tsd::graph_nodes {
namespace {

using namespace tsd::graph;
using tsd::core::Token;
using uint3 = tsd::core::math::uint3;

// Multi-octave fBm turbulence with domain warping: soft, cloudy/nebula-like
// volumes. Rich parameter set (octaves/lacunarity/gain/warp) for shaping.
struct GenerateTurbulence : Node
{
  ParameterList params;
  GenerateTurbulence()
  {
    params.set(Token("dims"), uint3(64u, 64u, 64u));
    params.set(Token("frequency"), 3.f);
    params.set(Token("octaves"), 5);
    params.set(Token("lacunarity"), 2.f);
    params.set(Token("gain"), 0.5f);
    params.set(Token("warp"), 0.4f);
    params.set(Token("seed"), 0);
  }
  NodeTypeInfo typeInfo() const override
  {
    NodeTypeInfo i;
    i.name = Token("GenerateTurbulence");
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
    const int octaves = params.getOr<int>(Token("octaves"), 5);
    const float lacunarity = params.getOr<float>(Token("lacunarity"), 2.f);
    const float gain = params.getOr<float>(Token("gain"), 0.5f);
    const float warp = params.getOr<float>(Token("warp"), 0.4f);
    const int seed = params.getOr<int>(Token("seed"), 0);
    if (dims.x == 0u || dims.y == 0u || dims.z == 0u) {
      ctx.fail("GenerateTurbulence: dims must be > 0 on each axis");
      return;
    }

    auto f = std::make_shared<Field>(makeUnitField(dims));

    size_t idx = 0;
    for (uint32_t z = 0; z < dims.z; ++z) {
      const float pz = normCoord(z, dims.z) * freq;
      for (uint32_t y = 0; y < dims.y; ++y) {
        const float py = normCoord(y, dims.y) * freq;
        for (uint32_t x = 0; x < dims.x; ++x, ++idx) {
          const float px = normCoord(x, dims.x) * freq;
          // Warp the sample point by a low-octave fBm offset for billows.
          const float ox =
              warp * fbm(px, py, pz, 3, lacunarity, gain, seed + 11);
          const float oy =
              warp * fbm(px, py, pz, 3, lacunarity, gain, seed + 23);
          const float oz =
              warp * fbm(px, py, pz, 3, lacunarity, gain, seed + 37);
          const float v =
              fbm(px + ox, py + oy, pz + oz, octaves, lacunarity, gain, seed);
          f->data.get<float>(idx) = clamp01(v);
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

void registerGenerateTurbulence(NodeRegistry &reg)
{
  reg.registerType(Token("GenerateTurbulence"),
      [] { return std::make_unique<GenerateTurbulence>(); });
}

} // namespace tsd::graph_nodes
