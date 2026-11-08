// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <optix_device.h>
#include "gpu/RenderError.h"
#include "gpu/gpu_objects.h"
#include "gpu/gpu_util.h"
#include "gpu/intersectRay.h"
#include "gpu/renderer/shadowTransmittance.h"
#include "gpu/sampleVolumeLight.h"
#include "gpu/sbt.h"

namespace visrtx {
VISRTX_DEVICE void mediumError(
    const ScreenSample &ss, const VolumeHit &hit, RenderErrorCategory category)
{
  recordRenderError(
      ss.frameData->renderError, category, hit.volume->id, hit.instance->id);
}

// FP64 TwoSum with explicit round-to-nearest operations. The residual must
// participate in comparison even when a tiny positive s disappears from t.
VISRTX_DEVICE bool mediumCoefficients(ScreenSample &ss,
    const VolumeHit &hit,
    const vec3 &position,
    glm::dvec3 &extinction,
    glm::dvec3 *effectiveScattering = nullptr)
{
  const auto &material =
      ss.frameData->registry.materials[hit.volume->data.mdl.material];
  if (!material.materialData.mdl.volumeSupported) {
    mediumError(ss, hit, RenderErrorCategory::UNSUPPORTED_MEDIUM);
    return false;
  }
  vec3 absorption, scattering, emission;
  optixDirectCall<void>(material.callableBaseIndex
          + uint32_t(MDLShaderEntryPoints::VolumeCoefficients),
      ss.frameData,
      &hit,
      &position,
      &absorption,
      &scattering,
      &emission);
  bool unsupported = false;
  for (int c = 0; c < 3; ++c) {
    const float a = absorption[c], s = scattering[c], e = emission[c];
    if (!isfinite(a) || !isfinite(s) || a < 0.f || s < 0.f || !isfinite(e)) {
      mediumError(ss, hit, RenderErrorCategory::INVALID_COEFFICIENT);
      return false;
    }
    const double t = __dadd_rn(double(a), double(s));
    const double z = __dsub_rn(t, double(a));
    const double r = __dadd_rn(
        __dsub_rn(double(a), __dsub_rn(t, z)), __dsub_rn(double(s), z));
    const double M = double(hit.volume->data.mdl.extinctionMajorant);
    if (t > M || (t == M && r > 0.)) {
      mediumError(ss, hit, RenderErrorCategory::MAJORANT_VIOLATION);
      return false;
    }
    extinction[c] = t;
    // Preserve the smaller event coefficient, exactly as in the ordered
    // categorical policy. Both primary sources and shadow extinction use t.
    if (effectiveScattering)
      (*effectiveScattering)[c] = s < a ? double(s) : t - double(a);
    unsupported |= e != 0.f;
  }
  if (unsupported) {
    mediumError(ss, hit, RenderErrorCategory::UNSUPPORTED_MEDIUM);
    return false;
  }
  return true;
}

VISRTX_DEVICE float rayMarchMdlVolume(ScreenSample &ss,
    const VolumeHit &hit,
    vec3 &transmittance,
    float inverseSamplingRate,
    vec3 *color = nullptr)
{
  if (hit.volume->data.mdl.extinctionMajorant == 0.f)
    return FLT_MAX; // caller's vacuum guarantee; not a proof of
                    // conservativeness
  const auto &ray = hit.localRay;
  const glm::dvec3 direction(ray.dir);
  const double step = double(hit.volume->stepSize) * double(inverseSamplingRate)
      / sqrt(glm::dot(direction, direction));
  if (!isfinite(step) || step <= 0.) {
    mediumError(ss, hit, RenderErrorCategory::NUMERICAL_FAILURE);
    return FLT_MAX;
  }
  glm::dvec3 opticalDepth(0.);
  glm::dvec3 radiance(0.);
  double t = ray.t.lower;
  float depth = FLT_MAX;
  while (t < double(ray.t.upper)) {
    const double next = fmin(double(ray.t.upper), t + step);
    if (!(next > t)) {
      mediumError(ss, hit, RenderErrorCategory::NUMERICAL_FAILURE);
      break;
    }
    const vec3 position = ray.org + float(t + (next - t) * 0.5) * ray.dir;
    glm::dvec3 extinction, scattering;
    if (!mediumCoefficients(ss, hit, position, extinction, &scattering))
      break;
    if (color && glm::any(glm::greaterThan(scattering, glm::dvec3(0.)))) {
      const vec3 worldPosition =
          xfmPoint(hit.instance->objectToWorld, position);
      const auto light =
          sampleVolumeLight(ss, *ss.frameData, worldPosition, true);
      if (light.pdf > 0.f) {
        // No entry offset: the shadow starts INSIDE this domain, and must
        // include its remaining optical depth. Shadow any-hit invokes this
        // marcher with color=nullptr, so it never recursively samples lights.
        const Ray shadow{worldPosition, light.dir, {0.f, light.dist}};
        const vec3 visibility = surfaceShadowTransmittance(ss, shadow)
            * volumeShadowTransmittance(ss, shadow);
        for (int c = 0; c < 3; ++c) {
          const double h = next - t;
          const double integral = extinction[c] == 0.
              ? h
              : -expm1(-extinction[c] * h) / extinction[c];
          radiance[c] += exp(-opticalDepth[c]) * scattering[c]
              * double(light.radiance[c]) * double(visibility[c])
              / (4. * double(kPi) * double(light.pdf)) * integral;
        }
      }
    }
    if (glm::any(glm::greaterThan(extinction, glm::dvec3(0.))))
      depth = fminf(depth, float(t));
    opticalDepth += extinction * (next - t);
    t = next;
  }
  for (int c = 0; c < 3; ++c) {
    if (color)
      (*color)[c] += transmittance[c] * float(radiance[c]);
    transmittance[c] *= float(exp(-opticalDepth[c]));
  }
  return depth;
}
} // namespace visrtx
