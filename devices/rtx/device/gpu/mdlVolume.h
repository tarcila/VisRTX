// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <optix_device.h>
#include "gpu/RenderError.h"
#include "gpu/gpu_objects.h"
#include "gpu/gpu_util.h"
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
    glm::dvec3 &extinction)
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
    // Validation precedes the temporary transport-support restriction, so a
    // downward-rounded extinction sum can never conceal an unsafe bound.
    unsupported |= s != 0.f || e != 0.f;
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
    float inverseSamplingRate)
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
  double t = ray.t.lower;
  float depth = FLT_MAX;
  while (t < double(ray.t.upper)) {
    const double next = fmin(double(ray.t.upper), t + step);
    if (!(next > t)) {
      mediumError(ss, hit, RenderErrorCategory::NUMERICAL_FAILURE);
      break;
    }
    const vec3 position = ray.org + float(t + (next - t) * 0.5) * ray.dir;
    glm::dvec3 extinction;
    if (!mediumCoefficients(ss, hit, position, extinction))
      break;
    if (glm::any(glm::greaterThan(extinction, glm::dvec3(0.))))
      depth = fminf(depth, float(t));
    opticalDepth += extinction * (next - t);
    t = next;
  }
  for (int c = 0; c < 3; ++c)
    transmittance[c] *= float(exp(-opticalDepth[c]));
  return depth;
}
} // namespace visrtx
