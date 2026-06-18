// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/rendering/bridge/GraphRenderBridge.hpp"
// tsd
#include "tsd/core/Logging.hpp"
#include "tsd/scene/objects/Geometry.hpp"
#include "tsd/scene/objects/Material.hpp"
#include "tsd/scene/objects/SpatialField.hpp"
#include "tsd/scene/objects/Surface.hpp"
#include "tsd/scene/objects/Volume.hpp"
// anari
#include <anari/anari_cpp/ext/linalg.h>
// std
#include <stdexcept>
#include <string>

namespace tsd::rendering {

using namespace tsd::scene;
using tsd::core::Token;
using tsd::graph::NodeId;
using tsd::graph::Renderable;
using tsd::graph::RenderableParams;
using float3 = anari::math::float3;

GraphRenderBridge::GraphRenderBridge(tsd::graph::Graph &graph,
    tsd::graph::Evaluator &eval,
    Token deviceName,
    anari::Device device,
    int numViewports)
    : m_graph(graph), m_eval(eval), m_deviceName(deviceName), m_device(device)
{
  if (!m_device)
    throw std::runtime_error("GraphRenderBridge: null ANARI device");
  if (numViewports < 1 || numViewports > 64)
    throw std::runtime_error("GraphRenderBridge: numViewports must be 1..64");
  for (int i = 0; i < numViewports; ++i) {
    m_indices.push_back(std::make_unique<RenderIndexAllLayers>(
        m_renderScene, m_deviceName, m_device));
  }
}

GraphRenderBridge::~GraphRenderBridge() = default;

void GraphRenderBridge::setDisplay(NodeId node, uint64_t mask, bool enabled)
{
  auto &d = m_displays[node];
  d.mask = mask;
  d.enabled = enabled;
  if (!d.layer) {
    const std::string name = "display_" + std::to_string(node);
    d.layer = m_renderScene.addLayer(Token(name.c_str()));
  }
}

void GraphRenderBridge::removeDisplay(NodeId node)
{
  auto it = m_displays.find(node);
  if (it == m_displays.end())
    return;
  if (it->second.layer)
    m_renderScene.removeLayer(it->second.layer->name());
  m_displays.erase(it);
}

std::vector<const Layer *> GraphRenderBridge::layersForViewport(int i) const
{
  std::vector<const Layer *> out;
  if (i < 0 || i >= int(m_indices.size()))
    return out;
  const uint64_t bit = uint64_t(1) << i;
  for (const auto &kv : m_displays) {
    const Display &d = kv.second;
    if (d.enabled && d.realized && d.layer && (d.mask & bit))
      out.push_back(d.layer);
  }
  return out;
}

void GraphRenderBridge::update()
{
  for (auto &kv : m_displays) {
    Display &d = kv.second;
    if (!d.enabled) {
      d.realized = false;
      continue;
    }
    rebuildLayer(kv.first, d);
  }
  for (int i = 0; i < int(m_indices.size()); ++i) {
    m_indices[i]->setIncludedLayers(layersForViewport(i));
    m_indices[i]->populate();
  }
}

void GraphRenderBridge::rebuildLayer(NodeId node, Display &d)
{
  if (!m_eval.pull(node)) {
    d.realized = false;
    return;
  }
  const auto *out =
      m_eval.output(node, Token("out"), tsd::graph::hostResidency());
  if (!out || !out->payload || out->type.name != Token("renderable")) {
    d.realized = false;
    return;
  }

  if (d.realized && out->version == d.lastVersion)
    return;

  d.layer->clear();

  auto r = std::static_pointer_cast<Renderable>(out->payload);
  if (r->kind == Renderable::Kind::Surface)
    buildSurface(d.layer, *r);
  else
    buildVolume(d.layer, *r);

  d.lastVersion = out->version;
  d.realized = true;
}

void GraphRenderBridge::applyParams(
    tsd::scene::Object &obj, const RenderableParams &p)
{
  for (const auto &s : p.scalars)
    obj.setParameter(s.first, s.second);
  for (const auto &a : p.arrays) {
    auto arr =
        m_renderScene.createArray(a.second.elementType(), a.second.size());
    arr->setData(a.second.data());
    obj.setParameterObject(a.first, *arr);
  }
}

void GraphRenderBridge::buildSurface(Layer *layer, const Renderable &r)
{
  auto geom = m_renderScene.createObject<Geometry>(r.primSubtype);
  applyParams(*geom, r.prim);

  auto mat = m_renderScene.createObject<Material>(tokens::material::matte);
  applyParams(*mat, r.appearance);

  auto surf = m_renderScene.createSurface("renderable", geom, mat);
  m_renderScene.insertChildObjectNode(layer->root(), surf);
}

void GraphRenderBridge::buildVolume(Layer *layer, const Renderable &r)
{
  auto field = m_renderScene.createObject<SpatialField>(r.primSubtype);

  if (r.primSubtype == Token("structuredRegular")) {
    float3 dims(0.f);
    for (const auto &s : r.prim.scalars)
      if (s.first == Token("dims"))
        dims = s.second.get<float3>();
    for (const auto &a : r.prim.arrays) {
      if (a.first == Token("data")) {
        auto arr = m_renderScene.createArray(a.second.elementType(),
            size_t(dims.x),
            size_t(dims.y),
            size_t(dims.z));
        arr->setData(a.second.data());
        field->setParameterObject(Token("data"), *arr);
      }
    }
    for (const auto &s : r.prim.scalars)
      if (s.first != Token("dims"))
        field->setParameter(s.first, s.second);
  } else {
    applyParams(*field, r.prim);
  }

  auto vol =
      m_renderScene.createObject<Volume>(tokens::volume::transferFunction1D);
  vol->setParameterObject("value", *field);
  applyParams(*vol, r.appearance);

  m_renderScene.insertChildObjectNode(layer->root(), vol);
}

anari::World GraphRenderBridge::world(int viewport) const
{
  return m_indices.at(viewport)->world();
}

} // namespace tsd::rendering
