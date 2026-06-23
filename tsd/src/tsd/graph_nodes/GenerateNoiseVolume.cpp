// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/graph/Evaluator.hpp"
#include "tsd/graph_nodes/BuiltinNodes.hpp"
#include "tsd/graph_nodes/Descriptors.hpp"
// std
#include <cmath>
#include <memory>

namespace tsd::graph_nodes {
namespace {

using namespace tsd::graph;
using tsd::core::Token;
using uint3 = tsd::core::math::uint3;
using float3 = tsd::core::math::float3;

struct GenerateNoiseVolume : Node
{
  ParameterList params;
  // Seed editable scalars so they show in the inspector; editing either changes
  // the param hash and re-runs the pipeline downstream (handy for testing live
  // pipeline updates).
  GenerateNoiseVolume()
  {
    params.set(Token("seed"), 0);
    params.set(Token("frequency"), 6.f);
  }
  NodeTypeInfo typeInfo() const override
  {
    NodeTypeInfo i;
    i.name = Token("GenerateNoiseVolume");
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
    const uint3 dims = params.getOr<uint3>(Token("dims"), uint3(32u, 32u, 32u));
    const int seed = params.getOr<int>(Token("seed"), 0);
    const float freq = params.getOr<float>(Token("frequency"), 6.f);
    if (dims.x == 0u || dims.y == 0u || dims.z == 0u) {
      ctx.fail("GenerateNoiseVolume: dims must be > 0 on each axis");
      return;
    }
    auto f = std::make_shared<Field>();
    f->dims = dims;
    f->origin = float3(-1.f, -1.f, -1.f);
    f->spacing = float3(2.f / dims.x, 2.f / dims.y, 2.f / dims.z);
    f->data =
        tsd::core::AnyArray(ANARI_FLOAT32, size_t(dims.x) * dims.y * dims.z);

    const float sx = float(seed) * 0.137f;
    size_t idx = 0;
    for (uint32_t z = 0; z < dims.z; ++z)
      for (uint32_t y = 0; y < dims.y; ++y)
        for (uint32_t x = 0; x < dims.x; ++x, ++idx) {
          const float px =
              (float(x) / float(dims.x - 1 ? dims.x - 1 : 1)) * 2.f - 1.f;
          const float py =
              (float(y) / float(dims.y - 1 ? dims.y - 1 : 1)) * 2.f - 1.f;
          const float pz =
              (float(z) / float(dims.z - 1 ? dims.z - 1 : 1)) * 2.f - 1.f;
          const float r = std::sqrt(px * px + py * py + pz * pz);
          float v = 1.f - r;
          v += 0.1f * std::sin(px * freq + sx) * std::sin(py * freq + sx);
          f->data.get<float>(idx) = v < 0.f ? 0.f : (v > 1.f ? 1.f : v);
        }

    Value out;
    out.type = PortType{portField()};
    out.residency = hostResidency();
    out.payload = f;
    ctx.setOutput(Token("out"), out);
  }
};

} // namespace

void registerGenerateNoiseVolume(NodeRegistry &reg)
{
  reg.registerType(Token("GenerateNoiseVolume"),
      [] { return std::make_unique<GenerateNoiseVolume>(); });
}

} // namespace tsd::graph_nodes
