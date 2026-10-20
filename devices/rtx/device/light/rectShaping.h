/*
 * Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

// Host-only construction of cached quad-light shaping data.
#include "gpu/lightGeometry.h"

namespace visrtx {
namespace detail {

// log(cos(theta)) without losing a narrow angle to cos(theta) rounding to 1.
inline double rectLogCos(double theta)
{
  const double s = sin(0.5 * theta);
  return log1p(-2.0 * s * s);
}

// 2 * integral cone(mu) * mu^(p-1) dmu over the emitting hemisphere.
// The full-strength core is analytic. In the penumbra, integrate in mu^p:
// its Jacobian absorbs the focused lobe, so even arbitrarily large focus is
// resolved without sampling pow(mu, focus) away from the axis and underflowing.
// All angular and power differences stay in double precision until storage.
inline double rectConeFluxIntegral(const RectLightGPUData &rect, double p)
{
  const double halfPi = 1.57079632679489661923;
  const double outer = glm::min(double(rect.coneAngle), halfPi);
  const double inner = glm::min(double(rect.innerAngle), outer);
  if (outer <= 0.0)
    return 0.0;
  const double logOuter = outer >= halfPi ? -INFINITY : rectLogCos(outer);
  if (rect.innerAngle >= rect.coneAngle)
    return (2.0 / p) * -expm1(p * logOuter);

  const double logInner = inner >= halfPi ? -INFINITY : rectLogCos(inner);
  const double core = -expm1(p * logInner);
  if (inner >= outer)
    return (2.0 / p) * core;

  const double width = -expm1(p * (logOuter - logInner));
  const double mass = exp(p * logInner) * width;
  if (mass == 0.0)
    return (2.0 / p) * core;
  constexpr int kSteps = 1024;
  double sum = 0.0;
  for (int i = 0; i < kSteps; ++i) {
    const double u = (double(i) + 0.5) / double(kSteps);
    const double logMu = logInner + log1p(-u * width) / p;
    // acos(exp(logMu)) would again lose tiny angles near the axis.
    const double angle = 2.0 * asin(sqrt(-0.5 * expm1(logMu)));
    const double t = glm::clamp((double(rect.coneAngle) - angle)
            / (double(rect.coneAngle) - double(rect.innerAngle)),
        0.0,
        1.0);
    sum += t * t * (3.0 - 2.0 * t);
  }
  return (2.0 / p) * (core + mass * sum / double(kSteps));
}

// Per-channel surviving Lambertian flux. Tint and focused emission are
// integrated separately with their own power-coordinate transformations.
// Restricting angles to the hemisphere does not disable an obtuse soft cone:
// its penumbra can still attenuate directions within the emitting hemisphere.
inline vec3 rectShapedFluxFraction(const RectLightGPUData &rect)
{
  if (!rectHasShaping(rect))
    return vec3(1.0f);
  const float lambertian = float(rectConeFluxIntegral(rect, 2.0));
  if (!rectHasFocusShaping(rect))
    return vec3(lambertian);
  const float focused =
      float(rectConeFluxIntegral(rect, double(rect.focus) + 2.0));
  return rect.focusTint * lambertian + (vec3(1.0f) - rect.focusTint) * focused;
}

} // namespace detail

// Derive every cached shaping field of a RectLightGPUData from its authored
// coneAngle/coneSoftness and its already-set focus/focusTint.
//
// THE single derivation point. The cosines and the flux fraction are all
// functions of the authored parameters alone, so a producer that set some of
// them by hand would be one edit away from an inconsistent light -- and a stale
// flux fraction is not a visible glitch but a light silently mis-weighted in,
// or missing from, the pick CDF. Taking softness here rather than a
// pre-multiplied innerAngle keeps the one place that knows how softness maps to
// an angle inside this function.
//
// Call after the shaping parameters are set and before the data is uploaded or
// handed to lightPickPower.
inline void rectFinalizeShaping(
    RectLightGPUData &rect, float coneAngle, float coneSoftness)
{
  rect.coneAngle = coneAngle;
  rect.innerAngle = coneAngle * (1.0f - coneSoftness);
  rect.cosOuterAngle = cosf(rect.coneAngle);
  rect.cosInnerAngle = cosf(rect.innerAngle);
  rect.shapedFluxFraction = detail::rectShapedFluxFraction(rect);
}

} // namespace visrtx
