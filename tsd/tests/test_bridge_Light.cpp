// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/graph/Evaluator.hpp"
#include "tsd/graph/Renderable.hpp"
#include "tsd/rendering/bridge/GraphRenderBridge.hpp"
// anari
#include <anari/anari_cpp.hpp>
// std
#include <memory>

using tsd::core::Token;
using tsd::graph::EvalContext;
using tsd::graph::Evaluator;
using tsd::graph::Graph;
using tsd::graph::hostResidency;
using tsd::graph::Node;
using tsd::graph::NodeTypeInfo;
using tsd::graph::ParameterList;
using tsd::graph::PortType;
using tsd::graph::Renderable;
using tsd::graph::Value;
using tsd::rendering::GraphRenderBridge;
using float2 = tsd::core::math::float2;
using float3 = tsd::core::math::float3;

namespace {

// Emits a directional-light renderable, mirroring the DisplayLight node.
struct EmitLight : Node
{
  ParameterList params;
  NodeTypeInfo typeInfo() const override
  {
    NodeTypeInfo i;
    i.name = Token("EmitLight");
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
    auto r = std::make_shared<Renderable>();
    r->kind = Renderable::Kind::Light;
    r->primSubtype = Token("directional");
    r->appearance.scalars.push_back(
        {Token("color"), tsd::core::Any(float3(1.f, 1.f, 1.f))});
    r->appearance.scalars.push_back({Token("irradiance"), tsd::core::Any(1.f)});
    r->appearance.scalars.push_back(
        {Token("direction"), tsd::core::Any(float2(0.f, 240.f))});
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

} // namespace

SCENARIO("GraphRenderBridge routes a plain light by its viewport mask",
    "[bridge-light]")
{
  anari::Device dev = makeVisRTX();
  REQUIRE(dev != nullptr);

  Graph g;
  auto light = g.addNode(std::make_unique<EmitLight>());
  Evaluator e(g);

  GraphRenderBridge bridge(g, e, Token("visrtx"), dev, /*numViewports=*/3);
  bridge.setDisplay(light, /*mask=*/0b001, /*enabled=*/true); // viewport 0 only
  bridge.update();

  WHEN("the light is masked to viewport 0 only")
  {
    THEN("only viewport 0 includes its layer; its world is valid")
    {
      REQUIRE(bridge.layersForViewport(0).size() == 1);
      REQUIRE(bridge.layersForViewport(1).empty());
      REQUIRE(bridge.layersForViewport(2).empty());
      REQUIRE(bridge.world(0) != nullptr);
    }
  }

  anari::release(dev, dev);
}
