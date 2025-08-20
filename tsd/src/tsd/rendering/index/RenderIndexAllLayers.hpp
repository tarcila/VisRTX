// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/rendering/index/RenderIndex.hpp"
// std
#include <vector>

namespace tsd::rendering {

struct RenderIndexAllLayers : public RenderIndex
{
  RenderIndexAllLayers(Context &ctx, anari::Device d);
  ~RenderIndexAllLayers() override;

  bool isFlat() const override;

  void setFilterFunction(RenderIndexFilterFcn f) override;

  void signalArrayUnmapped(const Array *a) override;
  void signalLayerAdded(const Layer *l) override;
  void signalLayerUpdated(const Layer *l) override;
  void signalLayerRemoved(const Layer *l) override;
  void signalActiveLayersChanged() override;
  void signalObjectFilteringChanged() override;
  void signalRemoveAllObjects() override;

 private:
  void updateWorld() override;
  void syncLayerInstances(const Layer *layer);
  void releaseAllInstances();

  RenderIndexFilterFcn m_filter;
  std::vector<const Layer *> m_includedLayers;
  bool m_filterForceUpdate{false};

  using InstanceCache = FlatMap<const Layer *, std::vector<anari::Instance>>;
  InstanceCache m_instanceCache;
};

} // namespace tsd::rendering
