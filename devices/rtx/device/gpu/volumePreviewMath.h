// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "blackbody/blackbody.h"
#include "blackbody/previewCold.h"
#include "fieldBinding.h"
#include "volumeOptics.h"

namespace visrtx {
namespace preview_math {
// Quadrature arithmetic, deliberately NOT an optical-range certificate.
struct Coefficients
{
  float absorption[3]{};
  float scattering[3]{};
  float emission[3]{};
};
VISRTX_HOST_DEVICE float promote(float x)
{
  return x;
}
VISRTX_HOST_DEVICE float narrow(float x)
{
  return x;
}
VISRTX_HOST_DEVICE float add(float a, float b)
{
  return a + b;
}
VISRTX_HOST_DEVICE float subtract(float a, float b)
{
  return a - b;
}
VISRTX_HOST_DEVICE float multiply(float a, float b)
{
  return a * b;
}
VISRTX_HOST_DEVICE float divide(float a, float b)
{
  return a / b;
}
VISRTX_HOST_DEVICE float lerp(float a, float b, float w)
{
  return fmaf(w, b - a, a);
}
VISRTX_HOST_DEVICE vec3 lerp(vec3 a, vec3 b, float w)
{
  return {lerp(a.x, b.x, w), lerp(a.y, b.y, w), lerp(a.z, b.z, w)};
}
VISRTX_HOST_DEVICE bool nonemptyDomain(const box3 &b)
{
  return b.lower.x < b.upper.x && b.lower.y < b.upper.y
      && b.lower.z < b.upper.z;
}
VISRTX_HOST_DEVICE bool contains(const box3 &b, vec3 p)
{
  return nonemptyDomain(b) && p.x >= b.lower.x && p.x <= b.upper.x
      && p.y >= b.lower.y && p.y <= b.upper.y && p.z >= b.lower.z
      && p.z <= b.upper.z;
}
template <typename T, typename Sampler>
VISRTX_HOST_DEVICE T evaluateBinding(
    const FieldBindingGPUData<T> &b, vec3 p, Sampler &sampler)
{
  return b.field < 0 ? b.scale
                     : b.scale
          * (contains(b.domain, p) ? sampler.template sample<T>(b.field, p)
                                   : b.outside);
}
VISRTX_HOST_DEVICE float extinction(const Coefficients &v, int c)
{
  return v.absorption[c] + v.scattering[c];
}
VISRTX_HOST_DEVICE Coefficients combine(
    const Coefficients &a, const Coefficients &b)
{
  Coefficients result;
  for (int c = 0; c < 3; ++c) {
    result.absorption[c] = a.absorption[c] + b.absorption[c];
    result.scattering[c] = a.scattering[c] + b.scattering[c];
    result.emission[c] = a.emission[c] + b.emission[c];
  }
  return result;
}
VISRTX_HOST_DEVICE Coefficients evaluateOptical(const PrincipledVolumeValues &v,
    const double *constantLog,
    const float *constantExtinction = nullptr)
{
  Coefficients result;
  for (int c = 0; c < 3; ++c) {
    result.emission[c] = v.emission[c];
    if (v.density == 0.f || v.transmissionColor[c] == 1.f)
      continue;
    float e;
    if (constantExtinction)
      e = v.density * constantExtinction[c];
    else {
      const float logValue =
          constantLog ? float(constantLog[c]) : -logf(v.transmissionColor[c]);
      e = (v.density * logValue) / v.transmissionDistance;
    }
    result.absorption[c] = e * (1.f - v.scatteringColor[c]);
    result.scattering[c] = e * v.scatteringColor[c];
  }
  return result;
}
VISRTX_HOST_DEVICE Coefficients evaluateTransfer(vec4 rgba, float distance)
{
  Coefficients result;
  // Retain the established TF classification policy; no new principled clamp.
  const float alpha = fminf(fmaxf(rgba.w, 0.f), 1.f - 0x1p-24f);
  const float e = alpha <= 1e-7f ? 0.f : -logf(1.f - alpha) / distance;
  for (int c = 0; c < 3; ++c) {
    result.absorption[c] = e;
    result.emission[c] = e * rgba[c];
  }
  return result;
}
VISRTX_HOST_DEVICE float phase(const VolumePhaseFunction &p, float cosine)
{
  const float primary = p.anisotropy == 0.f
      ? 1.f / (4.f * kPi)
      : henyeyGreenstein(p.anisotropy, cosine);
  if (p.secondaryWeight == 0.f)
    return primary;
  const float secondary = p.secondaryAnisotropy == 0.f
      ? 1.f / (4.f * kPi)
      : henyeyGreenstein(p.secondaryAnisotropy, cosine);
  return (1.f - p.secondaryWeight) * primary + p.secondaryWeight * secondary;
}
struct SegmentWeights
{
  float transmission, loss, sourceFactor;
};
VISRTX_HOST_DEVICE SegmentWeights segmentWeights(float e, float length)
{
  const float tau = e * length;
  if (tau == 0.f)
    return {1.f, 0.f, length};
  const float transmission = expf(-tau);
  const float loss = -expm1f(-tau);
  return {transmission, loss, tau < 1.f ? length * (loss / tau) : loss / e};
}
VISRTX_HOST_DEVICE void integrateSegment(const Coefficients &v,
    const float (&source)[3],
    float length,
    float (&radiance)[3],
    float (&transmission)[3])
{
  for (int c = 0; c < 3; ++c) {
    const auto weights = segmentWeights(extinction(v, c), length);
    transmission[c] = weights.transmission;
    radiance[c] = source[c] * weights.sourceFactor;
  }
}
// Normal thermal evaluation is FP32. The calibrated spectral tails retain wide
// evaluation/coupling, isolated from the ordinary LUT execution path.
VISRTX_HOST_DEVICE vec3 blackbodyLUT(float temperature)
{
  using namespace blackbody_data;
  constexpr float delta = 5.809142990314028f / 127.f;
  const float u = logf(temperature / 300.f) / delta;
  const int i = int(fminf(fmaxf(floorf(u), 0.f), 126.f));
  const float t = fminf(fmaxf(u - float(i), 0.f), 1.f);
  float xyz[3];
  for (int c = 0; c < 3; ++c) {
    const float p0 = hermite[i][c], p1 = hermite[i + 1][c];
    const float m0 = delta * hermite[i][c + 3];
    const float m1 = delta * hermite[i + 1][c + 3];
    xyz[c] = expf(p0
        + t
            * (m0
                + t
                    * (3.f * (p1 - p0) - 2.f * m0 - m1
                        + t * (2.f * (p0 - p1) + m0 + m1))));
  }
  vec3 rgb(0.f);
  for (int c = 0; c < 3; ++c)
    for (int k = 0; k < 3; ++k)
      rgb[c] += rgbMatrix[c][k] * xyz[k];
  if (rgb.x < 0.f || rgb.y < 0.f || rgb.z < 0.f) {
    float y = 0.f;
    for (int c = 0; c < 3; ++c) {
      rgb[c] = fmaxf(rgb[c], 0.f);
      y += yWeights[c][0] * rgb[c];
    }
    const float normalization = y > 0.f ? xyz[1] / y : 0.f;
    for (int c = 0; c < 3; ++c)
      rgb[c] = rgb[c] * normalization;
  }
  return rgb;
}
VISRTX_HOST_DEVICE void addThermal(Coefficients &result,
    const PrincipledVolumeValues &v,
    const PreviewColdKnot *cold = nullptr)
{
  if (!(v.temperature > 0.f) || v.blackbodyIntensity == 0.f)
    return;
  if (cold && v.temperature < 300.f) {
    // This table is published only when preparation bounds thermal coupling
    // below 2^20. At T<=100, even a 2^30 coupling times a conservative spectral
    // RGB upper bound is below FLT_MIN: zero is the FP32/FTZ result, not a
    // temperature clamp or a chosen visibility threshold.
    if (v.temperature < 100.f)
      return;
    const vec3 logs = previewColdLogXYZ(cold, v.temperature);
    const float xyz[3] = {expf(logs.x - logs.y), 1.f, expf(logs.z - logs.y)};
    vec3 rgb(0.f);
    for (int c = 0; c < 3; ++c)
      for (int k = 0; k < 3; ++k)
        rgb[c] += blackbody_data::rgbMatrix[c][k] * xyz[k];
    if (rgb.x < 0.f || rgb.y < 0.f || rgb.z < 0.f) {
      float y = 0.f;
      for (int c = 0; c < 3; ++c) {
        rgb[c] = fmaxf(rgb[c], 0.f);
        y += blackbody_data::yWeights[c][0] * rgb[c];
      }
      const float normalization = 1.f / y;
      for (int c = 0; c < 3; ++c)
        rgb[c] = rgb[c] * normalization;
    }
    for (int c = 0; c < 3; ++c) {
      const float weight =
          result.absorption[c] * v.blackbodyIntensity * v.blackbodyTint[c];
      if (weight > 0.f && rgb[c] > 0.f)
        result.emission[c] += expf(logs.y + logf(weight) + logf(rgb[c]));
    }
  } else if (v.temperature >= 300.f && v.temperature <= 100000.f) {
    const vec3 radiance = blackbodyLUT(v.temperature);
    for (int c = 0; c < 3; ++c)
      result.emission[c] += result.absorption[c] * v.blackbodyIntensity
          * v.blackbodyTint[c] * radiance[c];
  } else {
    bool active = false;
    for (int c = 0; c < 3; ++c)
      active |= result.absorption[c] > 0.f && v.blackbodyTint[c] > 0.f;
    if (!active)
      return;
    const auto radiance = CalibratedBlackbody{}(v.temperature);
    for (int c = 0; c < 3; ++c)
      result.emission[c] +=
          float(double(result.absorption[c]) * double(v.blackbodyIntensity)
              * double(v.blackbodyTint[c]) * radiance[c]);
  }
}
} // namespace preview_math
} // namespace visrtx
