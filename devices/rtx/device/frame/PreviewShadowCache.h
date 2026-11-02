// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cuda_runtime_api.h>
#include <array>
#include "gpu/aggregateVolumeData.h"
#include "gpu/previewShadowCacheData.h"
namespace visrtx {
struct PreviewShadowCacheKey
{
  optical_reference::FieldRevision densityRevision;
  box3 bounds, densityDomain;
  float densityScale{0.f}, densityOutside{0.f};
  double worldToObject[3][4]{};
  vec3 direction{0.f};
  double step{0.}, rate{0.};
  std::array<box3, 11> cuts;
  size_t cutCount{0};
};

enum class PreviewShadowBuilder
{
  March,
  Columns
};

class PreviewShadowCache
{
 public:
  explicit PreviewShadowCache(uint32_t resolution = 257,
      PreviewShadowBuilder builder = PreviewShadowBuilder::Columns);
  ~PreviewShadowCache();
  PreviewShadowCache(const PreviewShadowCache &) = delete;
  PreviewShadowCache &operator=(const PreviewShadowCache &) = delete;
  PreviewShadowCacheGPUData prepare(const AggregateVolumeSceneGPUData &scene,
      const FrameGPUData &frame,
      uint64_t lightStamp,
      cudaStream_t stream);
  uint64_t buildCount() const;

 private:
  float *m_values{nullptr};
  PreviewShadowCacheGPUData m_data{};
  uint64_t m_generation{0}, m_lightStamp{0};
  double m_rate{0.};
  bool m_attempted{false};
  uint32_t m_resolution;
  uint64_t m_builds{0};
  PreviewShadowCacheKey m_key;
  bool m_contentValid{false};
  PreviewShadowBuilder m_builder;
};
} // namespace visrtx
