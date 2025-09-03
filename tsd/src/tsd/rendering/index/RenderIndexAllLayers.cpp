// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/rendering/index/RenderIndexAllLayers.hpp"

#include "RenderToAnariObjectsVisitor.hpp"
// tsd_core
#include "tsd/core/Logging.hpp"
// std
#include <algorithm>
#include <iterator>

namespace tsd::rendering {

// Helper functions ///////////////////////////////////////////////////////////

static void releaseInstances(
    anari::Device d, const std::vector<anari::Instance> &instances)
{
  for (auto i : instances)
    anari::release(d, i);
}

// RenderIndexAllLayers definitions ///////////////////////////////////////////

RenderIndexAllLayers::RenderIndexAllLayers(
    Context &ctx, anari::Device d, bool alwaysGatherAllLights)
    : RenderIndex(ctx, d), m_forceAllLights(alwaysGatherAllLights)
{
  m_includedLayers = ctx.getActiveLayers();
}

RenderIndexAllLayers::~RenderIndexAllLayers()
{
  releaseAllInstances();
}

bool RenderIndexAllLayers::isFlat() const
{
  return false;
}

void RenderIndexAllLayers::setFilterFunction(RenderIndexFilterFcn f)
{
  m_filter = f;
  m_filterForceUpdate = true;
  signalObjectFilteringChanged();
}

void RenderIndexAllLayers::setIncludedLayers(
    const std::vector<const Layer *> &layers)
{
  m_includedLayers = layers;
  m_customIncludedLayers = !layers.empty();
  signalActiveLayersChanged();
}

void RenderIndexAllLayers::signalArrayUnmapped(const Array *a)
{
  RenderIndex::signalArrayUnmapped(a);
  if (a->elementType() == ANARI_FLOAT32_MAT4)
    updateWorld();
}

void RenderIndexAllLayers::signalLayerAdded(const Layer *l)
{
  syncLayerInstances(l, false, objectMask_all());
  updateWorld();
}

void RenderIndexAllLayers::signalLayerUpdated(const Layer *l)
{
  if (m_instanceCache.contains(l)) {
    syncLayerInstances(l, false, objectMask_all());
    updateWorld();
  }
}

void RenderIndexAllLayers::signalLayerRemoved(const Layer *l)
{
  if (m_instanceCache.contains(l)) {
    releaseInstances(device(), m_instanceCache[l]);
    m_instanceCache.erase(l);
    updateWorld();
  }
}

void RenderIndexAllLayers::signalActiveLayersChanged()
{
  if (!m_customIncludedLayers) {
    if (m_includedLayers.empty()
        && m_ctx->numberOfActiveLayers() == m_ctx->numberOfLayers())
      return;
    m_includedLayers = m_ctx->getActiveLayers();
  }
  signalInvalidateCachedObjects();
}

void RenderIndexAllLayers::signalObjectFilteringChanged()
{
  if (m_filter || m_filterForceUpdate) {
    releaseAllInstances();
    updateWorld();
    m_filterForceUpdate = false;
  }
}

void RenderIndexAllLayers::signalRemoveAllObjects()
{
  releaseAllInstances();
  RenderIndex::signalRemoveAllObjects();
}

void RenderIndexAllLayers::updateWorld()
{
  auto d = device();
  auto w = world();

  if (m_instanceCache.empty()) {
    if (!m_includedLayers.empty()) { // only sync specified layers
      tsd::core::logDebug(
          "[RenderIndexAllLayers] cache empty, "
          "repopulating using specific layers");
      if (m_forceAllLights) {
        // first just the surfaces/volumes from included layers
        for (auto &l : m_includedLayers)
          syncLayerInstances(l, false, objectMask_surfacesAndVolumes());
        // then all lights from all layers
        for (auto &l : m_ctx->layers())
          syncLayerInstances(l.second.ptr.get(), true, objectMask_lights());
      } else {
        for (auto &l : m_includedLayers)
          syncLayerInstances(l, false, objectMask_all());
      }
    } else { // sync everything
      tsd::core::logDebug(
          "[RenderIndexAllLayers] cache empty, "
          "repopulating using all layers");
      for (auto &l : m_ctx->layers())
        syncLayerInstances(l.second.ptr.get(), false, objectMask_all());
    }
  }

  std::vector<anari::Instance> instances;
  instances.reserve(2000);

  for (auto &i : m_instanceCache)
    std::copy(i.second.begin(), i.second.end(), std::back_inserter(instances));

  std::copy(m_externalInstances.begin(),
      m_externalInstances.end(),
      std::back_inserter(instances));

  if (instances.empty())
    anari::unsetParameter(d, w, "instance");
  else {
    anari::setParameterArray1D(
        d, w, "instance", instances.data(), instances.size());
  }

  anari::commitParameters(d, w);
}

void RenderIndexAllLayers::syncLayerInstances(
    const Layer *_layer, bool appendExisting, uint8_t mask)
{
  auto d = device();

  std::vector<anari::Instance> instances;
  instances.reserve(100);

  auto *layer = const_cast<Layer *>(_layer);

  RenderToAnariObjectsVisitor visitor(
      d, m_cache, &instances, m_ctx, mask, m_filter ? &m_filter : nullptr);
  layer->traverse(layer->root(), visitor);

  auto &cached = m_instanceCache[layer];
  if (appendExisting)
    std::copy(instances.begin(), instances.end(), std::back_inserter(cached));
  else {
    releaseInstances(d, cached);
    cached = instances;
  }
}

void RenderIndexAllLayers::releaseAllInstances()
{
  for (auto &i : m_instanceCache)
    releaseInstances(device(), i.second);
  m_instanceCache.clear();
}

} // namespace tsd::rendering
