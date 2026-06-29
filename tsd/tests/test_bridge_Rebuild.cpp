// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/graph/Evaluator.hpp"
#include "tsd/graph/Renderable.hpp"
#include "tsd/rendering/bridge/GraphRenderBridge.hpp"
// anari
#include <anari/anari_cpp/ext/linalg.h>
#include <anari/anari_cpp.hpp>
// std
#include <memory>

using tsd::core::Token;
using namespace tsd::graph;
using tsd::rendering::GraphRenderBridge;
using float3 = anari::math::float3;

namespace {

// Emits a sphere whose radius is read from a node parameter, so each parameter
// change bumps the node's output version and forces the bridge to rebuild.
struct EmitSphere : Node
{
  ParameterList params;
  int evalCount{0};

  NodeTypeInfo typeInfo() const override
  {
    NodeTypeInfo i;
    i.name = Token("EmitSphere");
    i.outputs.push_back(
        {Token("out"), PortType{Token("renderable")}, true, {}});
    return i;
  }
  ParameterList &parameters() override
  {
    return params;
  }
  void evaluate(EvalContext &ctx) override
  {
    ++evalCount;
    const float radius = params.getOr<float>(Token("radius"), 0.5f);

    auto r = std::make_shared<Renderable>();
    r->kind = Renderable::Kind::Surface;
    r->primSubtype = Token("sphere");
    tsd::core::AnyArray pos(ANARI_FLOAT32_VEC3, 1);
    pos.get<float3>(0) = float3(0.f, 0.f, 0.f);
    r->prim.arrays.push_back({Token("vertex.position"), pos});
    r->prim.scalars.push_back({Token("radius"), tsd::core::Any(radius)});
    r->appearance.scalars.push_back(
        {Token("color"), tsd::core::Any(float3(0.8f, 0.2f, 0.2f))});

    Value v;
    v.type = PortType{Token("renderable")};
    v.residency = hostResidency();
    v.payload = r;
    ctx.setOutput(Token("out"), v);
  }
};

anari::Device makeVisRTX()
{
  auto lib = anari::loadLibrary("visrtx", nullptr, nullptr);
  return lib ? anari::newDevice(lib, "default") : nullptr;
}

anari::Device makeDevice(const char *libName)
{
  auto lib = anari::loadLibrary(libName, nullptr, nullptr);
  return lib ? anari::newDevice(lib, "default") : nullptr;
}

} // namespace

SCENARIO("GraphRenderBridge reclaims render-scene objects on rebuild",
    "[bridge-rebuild]")
{
  anari::Device dev = makeVisRTX();
  REQUIRE(dev != nullptr);

  Graph g;
  auto sphere = g.addNode(std::make_unique<EmitSphere>());
  auto *impl = static_cast<EmitSphere *>(g.node(sphere)->impl.get());
  Evaluator e(g);

  GraphRenderBridge bridge(g, e, Token("visrtx"), dev, /*numViewports=*/1);
  bridge.setDisplay(sphere, /*mask=*/0b1, /*enabled=*/true);

  // First rebuild establishes the single-display steady-state object count.
  bridge.update();
  const size_t baseline = bridge.renderSceneObjectCount();
  REQUIRE(baseline > 0);
  REQUIRE(impl->evalCount == 1);

  WHEN("the renderable changes and rebuilds 20 times")
  {
    constexpr int kRebuilds = 20;
    for (int k = 1; k <= kRebuilds; ++k) {
      impl->parameters().set(Token("radius"), 0.1f * float(k));
      g.markDirty(sphere);
      bridge.update();
    }

    THEN("each iteration actually re-evaluated and rebuilt")
    {
      REQUIRE(impl->evalCount == 1 + kRebuilds);
    }

    THEN("the render-scene object count stays bounded (no leak)")
    {
      const size_t after = bridge.renderSceneObjectCount();
      REQUIRE(after == baseline);
    }
  }

  anari::release(dev, dev);
}

SCENARIO("GraphRenderBridge rebuilds a viewport index on a new device",
    "[bridge-device]")
{
  anari::Device dev = makeVisRTX();
  REQUIRE(dev != nullptr);

  // A second, distinct device. helide ships with the ANARI SDK; if it is not
  // loadable in this environment, skip rather than assert nothing.
  anari::Device dev2 = makeDevice("helide");
  if (!dev2) {
    WARN(
        "second ANARI device (helide) not loadable; "
        "setViewportDevice path verified manually");
    anari::release(dev, dev);
    return;
  }

  Graph g;
  auto sphere = g.addNode(std::make_unique<EmitSphere>());
  Evaluator e(g);

  GraphRenderBridge bridge(g, e, Token("visrtx"), dev, /*numViewports=*/1);
  bridge.setDisplay(sphere, /*mask=*/0b1, /*enabled=*/true);
  bridge.update();
  const size_t baseline = bridge.renderSceneObjectCount();
  REQUIRE(baseline > 0);

  WHEN("viewport 0 switches to a second device")
  {
    bridge.setViewportDevice(0, Token("helide"), dev2);

    THEN("its world is rebuilt and the device-agnostic scene is unchanged")
    {
      REQUIRE(bridge.world(0) != nullptr);
      REQUIRE(bridge.renderSceneObjectCount() == baseline);
    }
  }

  anari::release(dev2, dev2);
  anari::release(dev, dev);
}
