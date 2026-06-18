// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/graph/Evaluator.hpp"
#include "tsd/graph_nodes/BuiltinNodes.hpp"
#include "tsd/rendering/bridge/GraphRenderBridge.hpp"
// anari
#include <anari/anari_cpp/ext/linalg.h>
#include <anari/anari_cpp.hpp>
// std
#include <memory>

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

NodeId add(Graph &g, const char *t)
{
  static NodeRegistry reg = [] {
    NodeRegistry r;
    tsd::graph_nodes::registerBuiltinNodes(r);
    return r;
  }();
  return g.addNode(reg.create(Token(t)));
}

} // namespace

SCENARIO(
    "the 4a catalog drives rendered pixels via the bridge", "[nodes-render]")
{
  anari::Device dev = makeVisRTX();
  REQUIRE(dev != nullptr);

  Graph g;
  auto src = add(g, "GenerateNoiseVolume");
  g.node(src)->impl->parameters().set(
      Token("dims"), tsd::core::math::uint3(24u, 24u, 24u));
  auto sr = add(g, "ScalarRange");
  auto tf = add(g, "TransferFunction");
  auto dv = add(g, "DisplayVolume");
  auto bb = add(g, "BoundingBox");
  auto ds = add(g, "DisplaySurface");
  g.connect(src, Token("out"), sr, Token("in"));
  g.connect(sr, Token("out"), tf, Token("in"));
  g.connect(src, Token("out"), dv, Token("field"));
  g.connect(tf, Token("out"), dv, Token("tf"));
  g.connect(src, Token("out"), bb, Token("in"));
  g.connect(bb, Token("out"), ds, Token("in"));

  Evaluator e(g);
  GraphRenderBridge bridge(g, e, Token("visrtx"), dev, /*numViewports=*/2);
  bridge.setDisplay(dv, /*mask=*/0b01, /*enabled=*/true);
  bridge.setDisplay(ds, /*mask=*/0b10, /*enabled=*/true);
  bridge.update();

  WHEN("rendering both viewports")
  {
    Counts c0 = renderCounts(dev, bridge.world(0));
    Counts c1 = renderCounts(dev, bridge.world(1));
    THEN("vp0 shows the volume (color) and vp1 the box surface (objectId)")
    {
      REQUIRE(c0.color > 0);
      REQUIRE(c1.objectId > 0);
    }
  }

  anari::release(dev, dev);
}
