// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/rendering/bridge/GraphRenderBridge.hpp"
// tsd
#include "tsd/core/Logging.hpp"
#include "tsd/scene/objects/Geometry.hpp"
#include "tsd/scene/objects/Light.hpp"
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
  m_viewportDevices.assign(numViewports, m_device);
  m_viewportDeviceNames.assign(numViewports, m_deviceName);
  for (int i = 0; i < numViewports; ++i) {
    m_indices.push_back(std::make_unique<RenderIndexAllLayers>(
        m_renderScene, m_viewportDeviceNames[i], m_viewportDevices[i]));
  }
}

void GraphRenderBridge::setViewportDevice(
    int i, Token deviceName, anari::Device d)
{
  if (i < 0 || i >= int(m_indices.size()) || !d)
    return;
  if (m_viewportDevices[i] == d) // no-op guard: same device
    return;

  m_viewportDeviceNames[i] = deviceName;
  m_viewportDevices[i] = d;
  m_indices[i] =
      std::make_unique<RenderIndexAllLayers>(m_renderScene, deviceName, d);
  m_indices[i]->setIncludedLayers(layersForViewport(i));
}

GraphRenderBridge::~GraphRenderBridge() = default;

void GraphRenderBridge::setDisplay(NodeId node, uint64_t mask, bool enabled)
{
  auto &d = m_displays[node];
  d.mask = mask;
  d.enabled = enabled;
  if (!d.layer)
    d.layer = m_renderScene.addLayer(Token(displayName(node).c_str()));
}

std::string GraphRenderBridge::displayName(NodeId node) const
{
  const auto *gn = m_graph.node(node);
  const std::string base = (gn && gn->impl)
      ? std::string(gn->impl->typeInfo().name.c_str())
      : std::string("display");
  return base + " #" + std::to_string(node);
}

void GraphRenderBridge::setDisplayTransform(
    NodeId node, const tsd::math::mat4 &xfm)
{
  auto it = m_displays.find(node);
  if (it != m_displays.end())
    it->second.transform = xfm;
}

void GraphRenderBridge::removeDisplay(NodeId node)
{
  auto it = m_displays.find(node);
  if (it == m_displays.end())
    return;
  if (it->second.layer) {
    clearLayerObjects(it->second.layer);
    m_renderScene.removeLayer(it->second.layer->name());
  }
  m_displays.erase(it);

  // Drop the freed layer from every viewport index immediately so no index
  // retains a dangling Layer* until the next update().
  for (int i = 0; i < int(m_indices.size()); ++i)
    m_indices[i]->setIncludedLayers(layersForViewport(i));
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
    if (d.layer)
      (*d.layer->root())->setAsTransform(d.transform);
  }
  // setIncludedLayers() triggers the index's full rebuild (populate); calling
  // populate() again here would redundantly tear down + rebuild the cache.
  for (int i = 0; i < int(m_indices.size()); ++i)
    m_indices[i]->setIncludedLayers(layersForViewport(i));
}

void GraphRenderBridge::clearLayerObjects(Layer *layer)
{
  // Layer::clear() only drops the tree nodes; it leaves the referenced Scene
  // objects (and the arrays they reference as parameters) in the pools. Remove
  // each top-level object node with referenced-object deletion, then reclaim
  // the now-orphaned parameter-referenced objects (geometry/material/field and
  // their arrays) whose total use count has dropped to zero.
  std::vector<tsd::scene::LayerNodeRef> children;
  layer->traverse(layer->root(), [&](auto &node, int level) {
    if (level == 1)
      children.push_back(layer->at(node.index()));
    return level == 0; // descend only into the root's direct children
  });

  for (auto &child : children)
    m_renderScene.removeNode(child, /*deleteReferencedObjects=*/true);

  m_renderScene.removeUnusedObjects();
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

  clearLayerObjects(d.layer);

  auto r = std::static_pointer_cast<Renderable>(out->payload);
  const std::string name = displayName(node);
  switch (r->kind) {
  case Renderable::Kind::Surface:
    buildSurface(d.layer, *r, name);
    break;
  case Renderable::Kind::Volume:
    buildVolume(d.layer, *r, name);
    break;
  case Renderable::Kind::Light:
    buildLight(d.layer, *r, name);
    break;
  }

  d.lastVersion = out->version;
  d.realized = true;
  d.isLight = (r->kind == Renderable::Kind::Light);
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

void GraphRenderBridge::buildSurface(
    Layer *layer, const Renderable &r, const std::string &name)
{
  auto geom = m_renderScene.createObject<Geometry>(r.primSubtype);
  geom->setName(r.primSubtype.c_str());
  applyParams(*geom, r.prim);

  auto mat = m_renderScene.createObject<Material>(tokens::material::matte);
  applyParams(*mat, r.appearance);

  // Name the surface after the display node so the layer panel shows the same
  // label as the graph editor (instead of a generic "renderable").
  auto surf = m_renderScene.createSurface(name.c_str(), geom, mat);
  m_renderScene.insertChildObjectNode(layer->root(), surf);
}

void GraphRenderBridge::buildVolume(
    Layer *layer, const Renderable &r, const std::string &name)
{
  auto field = m_renderScene.createObject<SpatialField>(r.primSubtype);
  field->setName(r.primSubtype.c_str());

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
  vol->setName(name.c_str()); // match the graph editor / layer label
  vol->setParameterObject("value", *field);
  applyParams(*vol, r.appearance);

  m_renderScene.insertChildObjectNode(layer->root(), vol);
}

void GraphRenderBridge::buildLight(
    Layer *layer, const Renderable &r, const std::string &name)
{
  // Spec: unknown light subtype → skip + warn (rest of the layer still builds).
  if (r.primSubtype != tokens::light::directional) {
    tsd::core::logWarning(
        "[GraphRenderBridge] unsupported light subtype '%s'; skipping",
        r.primSubtype.c_str());
    return;
  }
  auto light = m_renderScene.createObject<Light>(r.primSubtype);
  light->setName(name.c_str());
  applyParams(*light, r.appearance);
  m_renderScene.insertChildObjectNode(layer->root(), light);
}

anari::World GraphRenderBridge::world(int viewport) const
{
  return m_indices.at(viewport)->world();
}

size_t GraphRenderBridge::renderSceneObjectCount() const
{
  using tsd::scene::ObjectDatabase;
  const ObjectDatabase &db = m_renderScene.objectDB();
  return db.array.size() + db.surface.size() + db.geometry.size()
      + db.material.size() + db.sampler.size() + db.volume.size()
      + db.field.size() + db.light.size();
}

} // namespace tsd::rendering
