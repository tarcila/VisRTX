// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "gpu/gpu_math.h"

namespace visrtx {
inline constexpr int PREVIEW_COLD_KNOTS = 64;
struct PreviewColdKnot
{
  float logXYZ[3];
  float derivative[3]; // derivative with respect to log(T)
};
VISRTX_HOST_DEVICE vec3 previewColdLogXYZ(const PreviewColdKnot *table, float t)
{
  constexpr float delta = 1.0986122886681098f / float(PREVIEW_COLD_KNOTS - 1);
  const float u = logf(t / 100.f) / delta;
  const int i =
      int(fminf(fmaxf(floorf(u), 0.f), float(PREVIEW_COLD_KNOTS - 2)));
  const float w = fminf(fmaxf(u - float(i), 0.f), 1.f);
  vec3 result;
  for (int c = 0; c < 3; ++c) {
    const float p0 = table[i].logXYZ[c], p1 = table[i + 1].logXYZ[c];
    const float m0 = delta * table[i].derivative[c];
    const float m1 = delta * table[i + 1].derivative[c];
    result[c] = p0
        + w
            * (m0
                + w
                    * (3.f * (p1 - p0) - 2.f * m0 - m1
                        + w * (2.f * (p0 - p1) + m0 + m1)));
  }
  return result;
}
} // namespace visrtx
