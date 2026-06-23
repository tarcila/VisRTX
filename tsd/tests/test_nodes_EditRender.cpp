// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/graph/Evaluator.hpp"
#include "tsd/graph_nodes/BuiltinNodes.hpp"
#include "tsd/graph_nodes/DemoGraph.hpp"
#include "tsd/graph_nodes/GraphEditModel.hpp"
#include "tsd/graph_nodes/TransferFunctionNode.hpp"
#include "tsd/rendering/bridge/GraphRenderBridge.hpp"
// anari
#include <anari/anari_cpp/ext/linalg.h>
#include <anari/anari_cpp.hpp>

using tsd::core::Token;
using namespace tsd::graph;
using tsd::rendering::GraphRenderBridge;
using uint2 = anari::math::uint2;
using float3 = anari::math::float3;

namespace {

anari::Device makeVisRTX()
{
  auto lib = anari::loadLibrary("visrtx", nullptr, nullptr);
  return lib ? anari::newDevice(lib, "default") : nullptr;
}

struct Counts
{
  size_t color{0};
  size_t objectId{0};
};

Counts renderCounts(anari::Device d, anari::World world)
{
  auto cam = anari::newObject<anari::Camera>(d, "perspective");
  anari::setParameter(d, cam, "aspect", 1.f);
  anari::setParameter(d, cam, "position", float3(0.f, 0.f, 3.f));
  anari::setParameter(d, cam, "direction", float3(0.f, 0.f, -1.f));
  anari::setParameter(d, cam, "up", float3(0.f, 1.f, 0.f));
  anari::commitParameters(d, cam);
  auto rnd = anari::newObject<anari::Renderer>(d, "default");
  anari::setParameter(d, rnd, "ambientRadiance", 1.f);
  anari::commitParameters(d, rnd);
  auto frame = anari::newObject<anari::Frame>(d);
  uint2 sz{64, 64};
  anari::setParameter(d, frame, "size", sz);
  anari::setParameter(d, frame, "channel.color", ANARI_UFIXED8_RGBA_SRGB);
  anari::setParameter(d, frame, "channel.objectId", ANARI_UINT32);
  anari::setParameter(d, frame, "world", world);
  anari::setParameter(d, frame, "camera", cam);
  anari::setParameter(d, frame, "renderer", rnd);
  anari::commitParameters(d, frame);
  anari::render(d, frame);
  anari::wait(d, frame);
  Counts c;
  auto col = anari::map<uint32_t>(d, frame, "channel.color");
  if (col.data)
    for (uint32_t i = 0; i < col.width * col.height; ++i)
      if ((col.data[i] & 0x00ffffffu) != 0u)
        ++c.color;
  anari::unmap(d, frame, "channel.color");
  auto oid = anari::map<uint32_t>(d, frame, "channel.objectId");
  if (oid.data)
    for (uint32_t i = 0; i < oid.width * oid.height; ++i)
      if (oid.data[i] != ~0u)
        ++c.objectId;
  anari::unmap(d, frame, "channel.objectId");
  anari::release(d, frame);
  anari::release(d, rnd);
  anari::release(d, cam);
  return c;
}

} // namespace

SCENARIO("a programmatic edit keeps the demo renderable", "[edit-render]")
{
  anari::Device dev = makeVisRTX();
  REQUIRE(dev != nullptr);

  tsd::graph::NodeRegistry reg;
  tsd::graph_nodes::registerBuiltinNodes(reg);
  tsd::graph::Graph g;
  auto d = tsd::graph_nodes::buildVolumeSurfaceDemo(g, reg);
  tsd::graph::Evaluator e(g);
  tsd::graph_nodes::GraphEditModel model(g, reg, nullptr);

  tsd::rendering::GraphRenderBridge bridge(
      g, e, tsd::core::Token("visrtx"), dev, /*numViewports=*/2);
  bridge.setDisplay(d.volumeDisplay, 0b01, true);
  bridge.setDisplay(d.surfaceDisplay, 0b10, true);
  bridge.update();

  // Edit: find the TransferFunction node and flip its curve to fully opaque
  // white.
  for (auto id : g.nodeIds()) {
    if (auto *itf = dynamic_cast<tsd::graph_nodes::ITransferFunctionNode *>(
            g.node(id)->impl.get())) {
      itf->tfState().colorPoints = {{0.f, 1.f, 1.f, 1.f}, {1.f, 1.f, 1.f, 1.f}};
      itf->tfState().opacityPoints = {{0.f, 1.f}, {1.f, 1.f}};
      g.markDirty(id);
    }
  }
  bridge.update();

  THEN("the volume viewport still renders color")
  {
    auto c0 = renderCounts(dev, bridge.world(0));
    REQUIRE(c0.color > 0);
  }

  anari::release(dev, dev);
}
