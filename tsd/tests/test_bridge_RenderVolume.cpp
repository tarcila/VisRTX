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
using uint2 = anari::math::uint2;
using float2 = anari::math::float2;
using float3 = anari::math::float3;
using float4 = anari::math::float4;

namespace {

struct EmitVolume : Node
{
  ParameterList params;
  NodeTypeInfo typeInfo() const override
  {
    NodeTypeInfo i;
    i.name = Token("EmitVolume");
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
    r->kind = Renderable::Kind::Volume;
    r->primSubtype = Token("structuredRegular");
    const int N = 8;
    tsd::core::AnyArray data(ANARI_FLOAT32, size_t(N) * N * N);
    for (size_t i = 0; i < data.size(); ++i)
      data.get<float>(i) = float(i) / float(data.size());
    r->prim.arrays.push_back({Token("data"), data});
    r->prim.scalars.push_back({Token("dims"), tsd::core::Any(float3(N, N, N))});
    r->prim.scalars.push_back({Token("origin"), tsd::core::Any(float3(-1.f))});
    r->prim.scalars.push_back(
        {Token("spacing"), tsd::core::Any(float3(2.f / N))});
    tsd::core::AnyArray color(ANARI_FLOAT32_VEC4, 256);
    for (size_t i = 0; i < color.size(); ++i) {
      float t = float(i) / float(color.size());
      color.get<float4>(i) = float4(1.f - t, 0.f, t, t);
    }
    r->appearance.arrays.push_back({Token("color"), color});
    r->appearance.scalars.push_back(
        {Token("valueRange"), tsd::core::Any(float2(0.f, 1.f))});
    Value v;
    v.type = PortType{Token("renderable")};
    v.residency = hostResidency();
    v.payload = r;
    ctx.setOutput(Token("out"), v);
  }
};

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
    // Place sphere to the right of the volume (volume spans [-1,1] on each
    // axis) so it occupies screen columns that the volume does not, giving
    // additive objectId coverage in viewport 1.
    pos.get<float3>(0) = float3(2.5f, 0.f, 0.f);
    r->prim.arrays.push_back({Token("vertex.position"), pos});
    r->prim.scalars.push_back({Token("radius"), tsd::core::Any(0.5f)});
    r->appearance.scalars.push_back(
        {Token("color"), tsd::core::Any(float3(0.2f, 0.8f, 0.2f))});
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

struct Counts
{
  size_t color{0};
  size_t objectId{0};
};

Counts renderCounts(anari::Device d, anari::World world)
{
  auto cam = anari::newObject<anari::Camera>(d, "perspective");
  anari::setParameter(d, cam, "aspect", 1.f);
  // fovy=90° gives half-width tan(45°)*3=3 at z=0, so sphere at x=2.5 is in
  // frame.
  anari::setParameter(d, cam, "fovy", anari::radians(90.f));
  anari::setParameter(d, cam, "position", float3(0.f, 0.f, 3.f));
  anari::setParameter(d, cam, "direction", float3(0.f, 0.f, -1.f));
  anari::setParameter(d, cam, "up", float3(0.f, 1.f, 0.f));
  anari::commitParameters(d, cam);

  auto rnd = anari::newObject<anari::Renderer>(d, "default");
  anari::setParameter(d, rnd, "ambientRadiance", 1.f); // matte surfaces visible
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

SCENARIO(
    "GraphRenderBridge renders a volume and differentiates viewports by mask",
    "[bridge-render-volume]")
{
  anari::Device dev = makeVisRTX();
  REQUIRE(dev != nullptr);

  Graph g;
  auto vol = g.addNode(std::make_unique<EmitVolume>());
  auto sphere = g.addNode(std::make_unique<EmitSphere>());
  Evaluator e(g);

  GraphRenderBridge bridge(g, e, Token("visrtx"), dev, /*numViewports=*/2);
  bridge.setDisplay(vol, /*mask=*/0b11, /*enabled=*/true); // both viewports
  bridge.setDisplay(sphere, /*mask=*/0b10, /*enabled=*/true); // viewport 1 only
  bridge.update();

  WHEN("rendering both viewports")
  {
    Counts c0 = renderCounts(dev, bridge.world(0)); // volume only
    Counts c1 = renderCounts(dev, bridge.world(1)); // volume + sphere
    THEN("both render the shared volume")
    {
      REQUIRE(c0.color > 0);
      REQUIRE(c1.color > 0);
    }
    THEN(
        "viewport 1 has more objectId hits than viewport 0 due to the masked sphere")
    {
      // VisRTX writes non-sentinel objectId for both surfaces and volumes.
      // The sphere at x=2.5 occupies pixels where the volume is not present,
      // so masking it to viewport 1 only makes c1.objectId strictly greater
      // than c0.objectId (which has volume coverage only).
      REQUIRE(c1.objectId > c0.objectId);
    }
  }

  anari::release(dev, dev);
}
