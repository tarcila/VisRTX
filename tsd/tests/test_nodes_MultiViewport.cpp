// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/graph/Evaluator.hpp"
#include "tsd/graph/NodeRegistry.hpp"
#include "tsd/graph_nodes/BuiltinNodes.hpp"
#include "tsd/graph_nodes/DemoGraph.hpp"
#include "tsd/graph_nodes/DisplayMask.hpp"
#include "tsd/rendering/bridge/GraphRenderBridge.hpp"
// anari
#include <anari/anari_cpp.hpp>
// std
#include <memory>

using tsd::core::Token;
using namespace tsd::graph;
using tsd::rendering::GraphRenderBridge;

namespace {
anari::Device makeVisRTX()
{
  auto lib = anari::loadLibrary("visrtx", nullptr, nullptr);
  return lib ? anari::newDevice(lib, "default") : nullptr;
}

// Mirror the app's syncDisplays() routing: read masks from the graph, push to
// bridge.
void sync(GraphRenderBridge &bridge, Graph &g)
{
  for (const auto &dm : tsd::graph_nodes::collectDisplayMasks(g))
    bridge.setDisplay(dm.node, dm.mask, /*enabled=*/dm.mask != 0);
  bridge.update();
}
} // namespace

SCENARIO("display viewportMask routes its layer into the masked viewports",
    "[multi-viewport]")
{
  anari::Device dev = makeVisRTX();
  REQUIRE(dev != nullptr);

  NodeRegistry reg;
  tsd::graph_nodes::registerBuiltinNodes(reg);
  Graph g;
  auto d = tsd::graph_nodes::buildVolumeSurfaceDemo(g, reg);
  Evaluator e(g);

  GraphRenderBridge bridge(g, e, Token("visrtx"), dev, /*numViewports=*/3);

  WHEN(
      "the volume display is masked into viewports 0 and 1 (surface stays 0b01)")
  {
    g.node(d.volumeDisplay)
        ->impl->parameters()
        .set(Token("viewportMask"), 0b11);
    sync(bridge, g);
    THEN("vp0 has both displays, vp1 has the volume, vp2 is empty")
    {
      REQUIRE(bridge.layersForViewport(0).size() == 2); // volume + surface
      REQUIRE(bridge.layersForViewport(1).size() == 1); // volume only
      REQUIRE(bridge.layersForViewport(2).empty());
    }
  }

  WHEN("the volume display is masked to no viewports")
  {
    g.node(d.volumeDisplay)->impl->parameters().set(Token("viewportMask"), 0);
    sync(bridge, g);
    THEN("vp0 has only the surface; the volume appears nowhere")
    {
      REQUIRE(bridge.layersForViewport(0).size() == 1); // surface only
      REQUIRE(bridge.layersForViewport(1).empty());
      REQUIRE(bridge.layersForViewport(2).empty());
    }
  }

  anari::release(dev, dev);
}
