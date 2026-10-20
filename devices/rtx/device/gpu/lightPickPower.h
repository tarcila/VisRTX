/*
 * Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

// Pick Power: a scalar estimate of a light's emitted power (relative flux),
// used only to weight the power-proportional Light Pick. Approximation here
// costs variance, never bias, so the estimates are luminance-weighted flux with
// deliberately coarse handling of transforms and cone falloff. Every light with
// a nonzero contribution must return a strictly positive value.

#include "gpu/gpu_math.h"
#include "gpu/gpu_objects.h"
#include "gpu/lightGeometry.h"

namespace visrtx {

namespace detail {

VISRTX_HOST_DEVICE float pickLuminance(const vec3 &c)
{
  return glm::dot(c, vec3(0.2126f, 0.7152f, 0.0722f));
}

// Isotropic area-scale factor of an affine transform: an object-space area A
// maps to roughly A * |det|^(2/3) in world space. Exact under uniform scale;
// an approximation under shear/non-uniform scale. For Geometry Lights this only
// shifts Pick Power (variance): the sampled pdf is corrected exactly at sample
// time via worldAreaScale. Analytic sphere/rect/ring samplers use object-space
// area with no transform Jacobian, so there the approximation is uncorrected
// (documented in sampleSphereLight).
VISRTX_HOST_DEVICE float affineAreaScale(const mat4 &xfm)
{
  const float det = glm::abs(glm::determinant(mat3(xfm)));
  return powf(det, 2.0f / 3.0f);
}

// log(cos(theta)) without losing a narrow angle to cos(theta) rounding to 1.
VISRTX_HOST_DEVICE double rectLogCos(double theta)
{
  const double s = sin(0.5 * theta);
  return log1p(-2.0 * s * s);
}

// 2 * integral cone(mu) * mu^(p-1) dmu over the emitting hemisphere.
// The full-strength core is analytic. In the penumbra, integrate in mu^p:
// its Jacobian absorbs the focused lobe, so even arbitrarily large focus is
// resolved without sampling pow(mu, focus) away from the axis and underflowing.
// All angular and power differences stay in double precision until storage.
VISRTX_HOST_DEVICE double rectConeFluxIntegral(
    const RectLightGPUData &rect, double p)
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
VISRTX_HOST_DEVICE vec3 rectShapedFluxFraction(const RectLightGPUData &rect)
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

VISRTX_HOST_DEVICE float lightPickPower(
    const LightGPUData &ld, const mat4 &xfm, float sceneRadius)
{
  const float sceneCrossSection = kPi * sceneRadius * sceneRadius;

  switch (ld.type) {
  case LightType::DIRECTIONAL:
    // Irradiance (W/m²) intercepted over the scene cross-section.
    return detail::pickLuminance(ld.color) * ld.distant.irradiance
        * sceneCrossSection;
  case LightType::POINT:
    // Isotropic point light: total flux = 4π · intensity.
    return detail::pickLuminance(ld.color) * ld.point.intensity * 2.0f * kTwoPi;
  case LightType::SPHERE: {
    // Lambertian sphere: flux = L · area · π, area = 4πr².
    const float area = kTwoPi * 2.0f * ld.sphere.radius * ld.sphere.radius
        * detail::affineAreaScale(xfm);
    return detail::pickLuminance(ld.color) * ld.sphere.intensity * area * kPi;
  }
  case LightType::RECT: {
    // Lambertian rectangle: flux = L · area · π, doubled if it emits both
    // sides.
    const float area =
        detail::affineAreaScale(xfm) / glm::max(ld.rect.oneOverArea, 1e-8f);
    // Cone/focus shaping attenuates radiance without renormalizing power, so
    // scale by the surviving flux fraction; both sides shape identically.
    const float sides = float(ld.rect.side.front + ld.rect.side.back);
    return detail::pickLuminance(ld.color
               * (ld.rect.colorMap ? ld.rect.colorMapMean : vec3(1.f))
               * detail::rectShapedFluxFraction(ld.rect))
        * ld.rect.intensity * area * kPi * sides;
  }
  case LightType::SPOT: {
    // Point light restricted to a cone: flux ≈ intensity · solid angle.
    const float coneSolidAngle = kTwoPi * (1.0f - ld.spot.cosOuterAngle);
    return detail::pickLuminance(ld.color) * ld.spot.intensity * coneSolidAngle;
  }
  case LightType::RING: {
    // Lambertian disk annulus; the cone falloff is ignored for the estimate.
    const float area =
        detail::affineAreaScale(xfm) / glm::max(ld.ring.oneOverArea, 1e-8f);
    return detail::pickLuminance(ld.color) * ld.ring.intensity * area * kPi;
  }
  case LightType::HDRI: {
    // Including map brightness allocates NEE samples more effectively between
    // the HDRI and other lights. Pick Power is a variance heuristic, not the
    // sampling pdf: any positive approximation preserves the estimator's mean.
    // Estimate the map's solid-angle mean luminance from pdfWeight, which folds
    // the equirectangular sinθ Jacobian: pdfWeight = (W·H)/(totalLum·2π²), so
    // the mean over the sphere is meanLum = totalLum·π/(2·W·H) =
    // 1/(4π·pdfWeight). A zero-luminance (all-black) map has pdfWeight == 0 and
    // thus zero power — correctly never picked.
    const float meanLuminance = ld.hdri.pdfWeight > 0.0f
        ? 1.0f / (2.0f * kTwoPi * ld.hdri.pdfWeight)
        : 0.0f;
    return detail::pickLuminance(ld.color) * ld.hdri.scale * meanLuminance
        * sceneCrossSection;
  }
  case LightType::GEOMETRY: {
    // Double-sided Lambertian surface: flux = L · area · π, doubled for sides.
    // This diffuse-only assumption is in lockstep with kFaithfulSet
    // (material/EmissionPolicy.h); growing that set requires generalizing this.
    const float area = ld.geometry.area * detail::affineAreaScale(xfm);
    return detail::pickLuminance(ld.geometry.radiance) * area * kPi * 2.0f;
  }
  default:
    return 0.0f;
  }
}

} // namespace visrtx
