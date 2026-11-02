// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "gpu/volumePreviewMath.h"

namespace visrtx {
// One clipped span of a single occurrence. Distances remain in world units;
// the local direction must not be renormalized. All ROI cuts are made by the
// aggregate iterator before this request crosses the callable seam.
struct PreviewDensityIntegral
{
  float origin[3], direction[3];
  float lower, upper, anchor, step;
  box3 volumeDomain, fieldDomain;
  float scale, outside;
  bool allowTexture{true};
};
template <typename Sample>
VISRTX_HOST_DEVICE float integratePreviewDensity(
    const PreviewDensityIntegral &q, Sample &sample)
{
  float integral = 0.f;
  float t = q.lower;
  while (t < q.upper) {
    const float index = floorf((t - q.anchor) / q.step);
    float next = q.anchor + (index + 1.f) * q.step;
    if (!(next > t))
      next = q.anchor + (index + 2.f) * q.step;
    if (!(next > t) || next > q.upper)
      next = q.upper;
    const float midpoint = t + (next - t) * .5f;
    const vec3 p(q.origin[0] + midpoint * q.direction[0],
        q.origin[1] + midpoint * q.direction[1],
        q.origin[2] + midpoint * q.direction[2]);
    if (preview_math::contains(q.volumeDomain, p)) {
      const float value =
          preview_math::contains(q.fieldDomain, p) ? sample(p) : q.outside;
      integral += (q.scale * value) * (next - t);
    }
    t = next;
  }
  return integral;
}
} // namespace visrtx
