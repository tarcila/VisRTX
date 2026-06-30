// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/graph/Evaluator.hpp"
#include "tsd/graph_nodes/BuiltinNodes.hpp"
#include "tsd/graph_nodes/ProceduralField.hpp"
// std
#include <cmath>
#include <memory>
#include <vector>

namespace tsd::graph_nodes {
namespace {

using namespace tsd::graph;
using tsd::core::Token;
using uint3 = tsd::core::math::uint3;
using float3 = tsd::core::math::float3;

// Sum of Gaussian blobs at pseudo-random centers: smooth organic surfaces that
// merge. `phase` orbits the centers for an animated, liquid-metal look.
struct GenerateMetaballs : Node
{
  ParameterList params;
  GenerateMetaballs()
  {
    params.set(Token("dims"), uint3(64u, 64u, 64u));
    params.set(Token("count"), 8);
    params.set(Token("radius"), 0.4f);
    params.set(Token("falloff"), 1.f);
    params.set(Token("phase"), 0.f); // animatable center orbit
    params.set(Token("seed"), 0);
  }
  NodeTypeInfo typeInfo() const override
  {
    NodeTypeInfo i;
    i.name = Token("GenerateMetaballs");
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
    const int count = params.getOr<int>(Token("count"), 8);
    const float radius = params.getOr<float>(Token("radius"), 0.4f);
    const float falloff = params.getOr<float>(Token("falloff"), 1.f);
    const float phase = params.getOr<float>(Token("phase"), 0.f);
    const int seed = params.getOr<int>(Token("seed"), 0);
    if (dims.x == 0u || dims.y == 0u || dims.z == 0u) {
      ctx.fail("GenerateMetaballs: dims must be > 0 on each axis");
      return;
    }
    const int n = count < 1 ? 1 : (count > 64 ? 64 : count);
    const float r = radius < 1e-3f ? 1e-3f : radius;
    const float invR2 = (falloff < 0.f ? 0.f : falloff) / (r * r);

    // Pseudo-random centers in [-0.7, 0.7]^3, orbited by `phase`.
    std::vector<float3> centers;
    centers.reserve(size_t(n));
    for (int k = 0; k < n; ++k) {
      const float cx = hashLattice(k, 0, 0, seed) * 1.4f - 0.7f;
      const float cy = hashLattice(0, k, 0, seed) * 1.4f - 0.7f;
      const float cz = hashLattice(0, 0, k, seed) * 1.4f - 0.7f;
      centers.push_back(float3(cx + 0.2f * std::sin(phase + float(k)),
          cy + 0.2f * std::sin(phase * 1.3f + float(k)),
          cz + 0.2f * std::cos(phase + float(k))));
    }

    auto f = std::make_shared<Field>(makeUnitField(dims));
    size_t idx = 0;
    for (uint32_t z = 0; z < dims.z; ++z) {
      const float pz = normCoord(z, dims.z);
      for (uint32_t y = 0; y < dims.y; ++y) {
        const float py = normCoord(y, dims.y);
        for (uint32_t x = 0; x < dims.x; ++x, ++idx) {
          const float px = normCoord(x, dims.x);
          float sum = 0.f;
          for (int k = 0; k < n; ++k) {
            const float3 &c = centers[size_t(k)];
            const float dx = px - c.x, dy = py - c.y, dz = pz - c.z;
            sum += std::exp(-(dx * dx + dy * dy + dz * dz) * invR2);
          }
          f->data.get<float>(idx) = clamp01(sum);
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

void registerGenerateMetaballs(NodeRegistry &reg)
{
  reg.registerType(Token("GenerateMetaballs"),
      [] { return std::make_unique<GenerateMetaballs>(); });
}

} // namespace tsd::graph_nodes
