// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/graph/Evaluator.hpp"
#include "tsd/graph/NodeRegistry.hpp"
#include "tsd/graph_nodes/BuiltinNodes.hpp"
#include "tsd/graph_nodes/DemoGraph.hpp"
#include "tsd/rendering/bridge/GraphRenderBridge.hpp"
#include "tsd/scene/Layer.hpp"
// anari
#include <anari/anari_cpp.hpp>

using tsd::core::Token;
using namespace tsd::graph;
using tsd::rendering::GraphRenderBridge;
using mat4 = tsd::math::mat4;

namespace {
anari::Device makeVisRTX()
{
  auto lib = anari::loadLibrary("visrtx", nullptr, nullptr);
  return lib ? anari::newDevice(lib, "default") : nullptr;
}
} // namespace

SCENARIO("setDisplayTransform applies to the display's layer-root",
    "[bridge-transform]")
{
  anari::Device dev = makeVisRTX();
  REQUIRE(dev != nullptr);

  NodeRegistry reg;
  tsd::graph_nodes::registerBuiltinNodes(reg);
  Graph g;
  auto d = tsd::graph_nodes::buildVolumeSurfaceDemo(g, reg);
  Evaluator e(g);

  GraphRenderBridge bridge(g, e, Token("visrtx"), dev, /*numViewports=*/1);
  bridge.setDisplay(
      d.volumeDisplay, 0b01, true); // only the volume, in viewport 0
  bridge.setDisplay(d.surfaceDisplay, 0b00, false); // exclude the surface

  mat4 m = tsd::math::IDENTITY_MAT4;
  m[3].x = 5.f; // translate +5 x
  bridge.setDisplayTransform(d.volumeDisplay, m);
  bridge.update();

  WHEN("inspecting the volume display's layer root")
  {
    auto layers = bridge.layersForViewport(0);
    THEN("its root transform is the matrix we set")
    {
      REQUIRE(layers.size() == 1);
      const auto root = layers[0]->root();
      REQUIRE((*root)->getTransform()[3].x == Approx(5.f));
    }
  }

  anari::release(dev, dev);
}
