// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <type_traits>
#include "gpu_math.h"
namespace visrtx {
// Nodal density integrals for one directional light and one occurrence.
// Coordinates are cache-local (light-aligned for the column builder), not
// necessarily volume-local. This is never a majorant or a validity mask.
struct PreviewShadowCacheGPUData
{
  const float *density;
  float domainLower[3], domainUpper[3];
  float worldToObject[3][4];
  float extinctionPerDensity[3];
  uint32_t resolution;
  uint32_t lightIndex;
};
static_assert(std::is_trivial_v<PreviewShadowCacheGPUData>,
    "OptiX constant launch parameters must not require dynamic initialization");
} // namespace visrtx
