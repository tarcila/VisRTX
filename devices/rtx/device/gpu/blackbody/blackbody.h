// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cmath>
#include "data/tables.h" // separately attributed CC-BY-SA-4.0 numeric data
#include "gpu/gpu_math.h"

namespace visrtx {

struct CalibratedBlackbody
{
  VISRTX_HOST_DEVICE glm::dvec3 operator()(float temperature) const;
};

VISRTX_HOST_DEVICE glm::dvec3 CalibratedBlackbody::operator()(
    float temperature) const
{
  using namespace blackbody_data;
  if (temperature <= 0.f)
    return glm::dvec3(0.);
  double xyz[3]{};
  if (temperature < 300.f || temperature > 100000.f) {
    // Unclamped spectral fallback. Double intermediates preserve cold tails
    // whose exponential alone would underflow in float before weighting.
    for (int i = 0; i < 1881; ++i) {
      const double x = double(spectral[i][0]) / double(temperature);
      const double value = exp(-x) / (-expm1(-x));
      for (int c = 0; c < 3; ++c)
        xyz[c] += double(spectral[i][c + 1]) * value;
    }
  } else {
    constexpr float delta = 5.809142990314028f / 127.f;
    const float u = logf(temperature / 300.f) / delta;
    int i = int(floorf(u));
    i = i < 0 ? 0 : (i > 126 ? 126 : i);
    float t = u - i;
    t = t < 0 ? 0 : (t > 1 ? 1 : t);
    for (int c = 0; c < 3; ++c) {
      const float p0 = hermite[i][c], p1 = hermite[i + 1][c];
      const float m0 = delta * hermite[i][c + 3],
                  m1 = delta * hermite[i + 1][c + 3];
      const float value = p0
          + t
              * (m0
                  + t
                      * (3.f * (p1 - p0) - 2.f * m0 - m1
                          + t * (2.f * (p0 - p1) + m0 + m1)));
      xyz[c] = expf(value);
    }
  }
  double rgb[3]{};
  bool outside = false;
  for (int c = 0; c < 3; ++c) {
    for (int k = 0; k < 3; ++k)
      rgb[c] += double(rgbMatrix[c][k]) * xyz[k];
    outside |= rgb[c] < 0.;
  }
  if (outside) {
    double y = 0.;
    for (int c = 0; c < 3; ++c) {
      rgb[c] = rgb[c] < 0. ? 0. : rgb[c];
      y += double(yWeights[c][0]) * rgb[c];
    }
    const double scale = y > 0. ? xyz[1] / y : 0.;
    for (int c = 0; c < 3; ++c)
      rgb[c] *= scale;
  }
  return glm::dvec3(rgb[0], rgb[1], rgb[2]);
}
} // namespace visrtx
