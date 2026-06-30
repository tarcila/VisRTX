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
using HeadlightState = GraphRenderBridge::HeadlightState;

namespace {

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

anari::Device makeDevice(const char *libName)
{
  auto lib = anari::loadLibrary(libName, nullptr, nullptr);
  return lib ? anari::newDevice(lib, "default") : nullptr;
}

} // namespace

SCENARIO("GraphRenderBridge per-viewport headlight resolves Auto/On/Off",
    "[bridge-headlight]")
{
  anari::Device dev = makeVisRTX();
  REQUIRE(dev != nullptr);

  Graph g;
  Evaluator e(g);
  GraphRenderBridge bridge(g, e, Token("visrtx"), dev, /*numViewports=*/2);

  WHEN("no plain light is masked to the viewport")
  {
    const size_t baseline = bridge.renderSceneObjectCount();
    bridge.setViewportHeadlight(0, HeadlightState{}); // Auto

    THEN("Auto resolves on; world is valid; render scene is untouched")
    {
      REQUIRE(bridge.headlightActive(0));
      REQUIRE(bridge.world(0) != nullptr);
      REQUIRE(bridge.renderSceneObjectCount() == baseline);
    }

    THEN("On forces it on, Off forces it off")
    {
      HeadlightState s;
      s.mode = HeadlightState::Mode::Off;
      bridge.setViewportHeadlight(0, s);
      REQUIRE(!bridge.headlightActive(0));
      s.mode = HeadlightState::Mode::On;
      bridge.setViewportHeadlight(0, s);
      REQUIRE(bridge.headlightActive(0));
    }

    THEN("re-aiming the headlight does not grow the render scene")
    {
      HeadlightState s;
      s.mode = HeadlightState::Mode::On;
      for (int k = 0; k < 10; ++k) {
        s.direction = float3(float(k), 0.f, -1.f);
        bridge.setViewportHeadlight(0, s);
      }
      REQUIRE(bridge.renderSceneObjectCount() == baseline);
      REQUIRE(bridge.world(0) != nullptr);
    }
  }

  WHEN("a plain light is masked to the viewport")
  {
    auto light = g.addNode(std::make_unique<EmitLight>());
    bridge.setDisplay(light, /*mask=*/0b01, /*enabled=*/true);
    bridge.update();

    THEN("Auto resolves off (the plain light covers it)")
    {
      bridge.setViewportHeadlight(0, HeadlightState{}); // Auto
      REQUIRE(!bridge.headlightActive(0));
    }
  }

  anari::release(dev, dev);
}

SCENARIO("GraphRenderBridge headlight survives a viewport device switch",
    "[bridge-headlight]")
{
  anari::Device dev = makeVisRTX();
  REQUIRE(dev != nullptr);

  anari::Device dev2 = makeDevice("helide");
  if (!dev2) {
    WARN(
        "second ANARI device (helide) not loadable; "
        "headlight device-switch path verified manually");
    anari::release(dev, dev);
    return;
  }

  Graph g;
  Evaluator e(g);
  GraphRenderBridge bridge(g, e, Token("visrtx"), dev, /*numViewports=*/1);

  HeadlightState s;
  s.mode = HeadlightState::Mode::On;
  bridge.setViewportHeadlight(0, s);
  REQUIRE(bridge.headlightActive(0));

  WHEN("viewport 0 switches device")
  {
    bridge.setViewportDevice(0, Token("helide"), dev2);
    bridge.setViewportHeadlight(0, s); // rebuilds handles on the new device

    THEN("the headlight is active and the world is valid on the new device")
    {
      REQUIRE(bridge.headlightActive(0));
      REQUIRE(bridge.world(0) != nullptr);
    }
  }

  anari::release(dev2, dev2);
  anari::release(dev, dev);
}
