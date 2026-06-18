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
#include <vector>

using tsd::core::Token;
using namespace tsd::graph;
using tsd::rendering::GraphRenderBridge;
using uint2 = anari::math::uint2;
using float3 = anari::math::float3;

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
    tsd::core::AnyArray pos(ANARI_FLOAT32_VEC3, 1);
    pos.get<float3>(0) = float3(0.f, 0.f, 0.f);
    r->prim.arrays.push_back({Token("vertex.position"), pos});
    r->prim.scalars.push_back({Token("radius"), tsd::core::Any(0.5f)});
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

// Count pixels that hit a surface object.
//
// VisRTX sets the background-miss sentinel to ~0u in channel.objectId; surface
// hits are written with the pool index from AnariHandleCache (0-based). An
// empty world causes a frame reset that fills the entire buffer with ~0u before
// rendering, so background-miss pixels are ~0u regardless.
//
// Primary counter: objectId != ~0u.
// Fallback (in case objectId is unavailable): non-black pixels in channel.color
// — the sphere is red (0.8, 0.2, 0.2) so any pixel with R > 0 is a hit.
size_t countObjectIdHits(anari::Device d, anari::World world)
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

  // Primary: objectId channel — surface hits have id != ~0u (background miss).
  size_t hits = 0;
  {
    auto fb = anari::map<uint32_t>(d, frame, "channel.objectId");
    if (fb.data) {
      for (uint32_t i = 0; i < fb.width * fb.height; ++i)
        if (fb.data[i] != ~0u)
          ++hits;
    }
    anari::unmap(d, frame, "channel.objectId");
  }

  // Fallback: color channel — non-black pixels are sphere hits.
  // Used when objectId is unavailable or all-zero (e.g. no frame reset).
  if (hits == 0) {
    auto fb = anari::map<uint8_t>(d, frame, "channel.color");
    if (fb.data) {
      for (uint32_t i = 0; i < fb.width * fb.height; ++i)
        if (fb.data[i * 4] > 0u) // R channel non-zero → sphere pixel
          ++hits;
    }
    anari::unmap(d, frame, "channel.color");
  }

  anari::release(d, frame);
  anari::release(d, rnd);
  anari::release(d, cam);
  return hits;
}

} // namespace

SCENARIO("GraphRenderBridge renders a masked surface into its viewport only",
    "[bridge-render-surface]")
{
  anari::Device dev = makeVisRTX();
  REQUIRE(dev != nullptr);

  Graph g;
  auto sphere = g.addNode(std::make_unique<EmitSphere>());
  Evaluator e(g);

  GraphRenderBridge bridge(g, e, Token("visrtx"), dev, /*numViewports=*/2);
  bridge.setDisplay(sphere, /*mask=*/0b01, /*enabled=*/true);
  bridge.update();

  WHEN("rendering both viewports")
  {
    size_t hits0 = countObjectIdHits(dev, bridge.world(0));
    size_t hits1 = countObjectIdHits(dev, bridge.world(1));
    THEN("viewport 0 shows the sphere; viewport 1 is empty")
    {
      REQUIRE(hits0 > 0);
      REQUIRE(hits1 == 0);
    }
  }

  WHEN("the mask is swapped to viewport 1")
  {
    bridge.setDisplay(sphere, /*mask=*/0b10, /*enabled=*/true);
    bridge.update();
    THEN("the hit counts swap")
    {
      REQUIRE(countObjectIdHits(dev, bridge.world(0)) == 0);
      REQUIRE(countObjectIdHits(dev, bridge.world(1)) > 0);
    }
  }

  anari::release(dev, dev);
}
