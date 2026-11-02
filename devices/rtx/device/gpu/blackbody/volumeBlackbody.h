// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "blackbody.h"
#include "previewCold.h"

namespace visrtx {
// A source-only approximation. Never changes A/S or the reference majorant.
// Reconstruct in binary64 so cold radiance remains available for large gains.
struct QualityBlackbody
{
  const PreviewColdKnot *cold{nullptr};
  VISRTX_HOST_DEVICE glm::dvec3 operator()(float temperature) const;
};
VISRTX_HOST_DEVICE glm::dvec3 QualityBlackbody::operator()(
    float temperature) const
{
  // The attributed spectrum has min(h*c/(lambda*k)) >16000 K. For
  // 0<T<=20 every exp(-x) in the reference graph underflows to binary64 zero
  // (x>800). This is an arithmetic identity, not a visibility/temperature
  // clamp.
  if (cold && temperature > 0.f && temperature <= 20.f)
    return glm::dvec3(0.);
  if (!cold || temperature < 25.f || temperature >= 300.f)
    return CalibratedBlackbody{}(temperature);
  constexpr double delta = 2.4849066497880003546 / 255.;
  const double u = log(double(temperature) / 25.) / delta;
  const int i = int(fmin(fmax(floor(u), 0.), 254.));
  const double w = fmin(fmax(u - double(i), 0.), 1.);
  double xyz[3], rgb[3]{};
  for (int c = 0; c < 3; ++c) {
    const double p0 = cold[i].logXYZ[c], p1 = cold[i + 1].logXYZ[c];
    const double m0 = delta * cold[i].derivative[c],
                 m1 = delta * cold[i + 1].derivative[c];
    const double logarithm = p0
        + w
            * (m0
                + w
                    * (3. * (p1 - p0) - 2. * m0 - m1
                        + w * (2. * (p0 - p1) + m0 + m1)));
    xyz[c] = exp(logarithm);
  }
  bool outside = false;
  for (int c = 0; c < 3; ++c) {
    for (int k = 0; k < 3; ++k)
      rgb[c] += double(blackbody_data::rgbMatrix[c][k]) * xyz[k];
    outside |= rgb[c] < 0.;
  }
  if (outside) {
    double y = 0.;
    for (int c = 0; c < 3; ++c) {
      rgb[c] = rgb[c] < 0. ? 0. : rgb[c];
      y += double(blackbody_data::yWeights[c][0]) * rgb[c];
    }
    const double scale = y > 0. ? xyz[1] / y : 0.;
    for (int c = 0; c < 3; ++c)
      rgb[c] *= scale;
  }
  return glm::dvec3(rgb[0], rgb[1], rgb[2]);
}
// Preserve custom/oracle providers and their side effects. Only the Quality
// adapter binds a volume's optional table; absent tables retain the oracle.
template <typename B>
VISRTX_HOST_DEVICE B &bindVolumeBlackbody(B &provider, const PreviewColdKnot *)
{
  return provider;
}
VISRTX_HOST_DEVICE QualityBlackbody bindVolumeBlackbody(
    QualityBlackbody &, const PreviewColdKnot *table)
{
  return {table};
}
} // namespace visrtx
