// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "previewShadowCacheData.h"
#include "volumePreviewMath.h"
namespace visrtx {
VISRTX_HOST_DEVICE bool lookupPreviewShadow(
    const PreviewShadowCacheGPUData &cache,
    vec3 worldPoint,
    uint32_t lightIndex,
    vec3 &transmission)
{
  if (!cache.density || lightIndex != cache.lightIndex)
    return false;
  vec3 p, lower, upper;
  for (int c = 0; c < 3; ++c) {
    p[c] = cache.worldToObject[c][3];
    for (int k = 0; k < 3; ++k)
      p[c] += cache.worldToObject[c][k] * worldPoint[k];
    lower[c] = cache.domainLower[c];
    upper[c] = cache.domainUpper[c];
  }
  if (!preview_math::contains(box3(lower, upper), p))
    return false;
  const vec3 x = (p - lower) / (upper - lower) * float(cache.resolution - 1);
  int lo[3], hi[3];
  float w[3];
  for (int c = 0; c < 3; ++c) {
    lo[c] = int(floorf(x[c]));
    lo[c] = lo[c] < 0
        ? 0
        : (lo[c] >= int(cache.resolution) ? int(cache.resolution) - 1 : lo[c]);
    hi[c] = lo[c] + 1 < int(cache.resolution) ? lo[c] + 1 : lo[c];
    w[c] = fminf(fmaxf(x[c] - float(lo[c]), 0.f), 1.f);
  }
  const size_t n = cache.resolution;
  const auto fetch = [&](int x, int y, int z) {
    return cache.density[size_t(x) + n * (size_t(y) + n * size_t(z))];
  };
  const float a = preview_math::lerp(
      fetch(lo[0], lo[1], lo[2]), fetch(hi[0], lo[1], lo[2]), w[0]);
  const float b = preview_math::lerp(
      fetch(lo[0], hi[1], lo[2]), fetch(hi[0], hi[1], lo[2]), w[0]);
  const float c = preview_math::lerp(
      fetch(lo[0], lo[1], hi[2]), fetch(hi[0], lo[1], hi[2]), w[0]);
  const float d = preview_math::lerp(
      fetch(lo[0], hi[1], hi[2]), fetch(hi[0], hi[1], hi[2]), w[0]);
  const float integral = preview_math::lerp(
      preview_math::lerp(a, b, w[1]), preview_math::lerp(c, d, w[1]), w[2]);
  for (int channel = 0; channel < 3; ++channel)
    transmission[channel] =
        expf(-cache.extinctionPerDensity[channel] * integral);
  return true;
}
} // namespace visrtx
