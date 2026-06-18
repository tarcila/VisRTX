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

namespace {

struct EmitSphere : Node
{
  ParameterList params;
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
    auto r = std::make_shared<Renderable>();
    r->kind = Renderable::Kind::Surface;
    r->primSubtype = Token("sphere");
    r->prim.scalars.push_back({Token("radius"), tsd::core::Any(0.5f)});
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
  if (!lib)
    return nullptr;
  return anari::newDevice(lib, "default");
}

} // namespace

SCENARIO("tsd::rendering::GraphRenderBridge maps viewport masks to layers",
    "[bridge-mask]")
{
  anari::Device dev = makeVisRTX();
  REQUIRE(dev != nullptr);

  Graph g;
  auto a = g.addNode(std::make_unique<EmitSphere>());
  auto b = g.addNode(std::make_unique<EmitSphere>());
  Evaluator e(g);

  GraphRenderBridge bridge(g, e, Token("visrtx"), dev, /*numViewports=*/3);

  bridge.setDisplay(a, /*mask=*/0b011, /*enabled=*/true);
  bridge.setDisplay(b, /*mask=*/0b100, /*enabled=*/true);
  bridge.update();

  WHEN("inspecting per-viewport layer membership")
  {
    THEN("each viewport includes exactly the masked displays' layers")
    {
      REQUIRE(bridge.layersForViewport(0).size() == 1);
      REQUIRE(bridge.layersForViewport(1).size() == 1);
      REQUIRE(bridge.layersForViewport(2).size() == 1);
    }
  }

  WHEN("disabling b and re-updating")
  {
    bridge.setDisplay(b, 0b100, false);
    bridge.update();
    THEN("viewport 2 has no layers; 0 and 1 still have a")
    {
      REQUIRE(bridge.layersForViewport(2).empty());
      REQUIRE(bridge.layersForViewport(0).size() == 1);
    }
  }

  WHEN("removing a")
  {
    bridge.removeDisplay(a);
    bridge.update();
    THEN("viewports 0 and 1 are empty")
    {
      REQUIRE(bridge.layersForViewport(0).empty());
      REQUIRE(bridge.layersForViewport(1).empty());
    }
  }

  anari::release(dev, dev);
}
