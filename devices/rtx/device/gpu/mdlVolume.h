// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <optix_device.h>
#include "gpu/RenderError.h"
#include "gpu/gpu_objects.h"
#include "gpu/gpu_util.h"
#include "gpu/intersectRay.h"
#include "gpu/mdlTracking.h"
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
    MdlTrackingChannel *channels = nullptr,
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
    MdlTrackingChannel channel;
    if (!makeMdlTrackingChannel(
            a, s, hit.volume->data.mdl.extinctionMajorant, channel)) {
      mediumError(ss, hit, RenderErrorCategory::MAJORANT_VIOLATION);
      return false;
    }
    extinction[c] = channel.extinction;
    if (channels)
      channels[c] = channel;
    if (effectiveScattering)
      (*effectiveScattering)[c] = channel.absorptionFirst
          ? channel.extinction - channel.smaller
          : channel.smaller;
    unsupported |= e != 0.f;
  }
  if (unsupported) {
    mediumError(ss, hit, RenderErrorCategory::UNSUPPORTED_MEDIUM);
    return false;
  }
  return true;
}

// Candidate distances are measured in world-ray t, not object distance.
// An open midpoint variate avoids log endpoints without clamping or a
// positive-probability zero flight; the categorical variate stays half-open.
VISRTX_DEVICE bool advanceMdlCandidate(
    ScreenSample &ss, const VolumeHit &hit, double &t)
{
  const double u = double(pcg_uniform(&ss.rs)) + 0.5 / 16777216.;
  const double step =
      -log1p(-u) / double(hit.volume->data.mdl.extinctionMajorant);
  const double next = t + step;
  if (!isfinite(next) || !(next > t)) {
    mediumError(ss, hit, RenderErrorCategory::NUMERICAL_FAILURE);
    return false;
  }
  t = next;
  return true;
}

VISRTX_DEVICE float sampleMdlDistance(
    ScreenSample &ss, const VolumeHit &hit, bool &didScatter)
{
  const double M = double(hit.volume->data.mdl.extinctionMajorant);
  if (M == 0.)
    return hit.localRay.t.upper;
  double t = hit.localRay.t.lower;
  while (advanceMdlCandidate(ss, hit, t)) {
    if (t >= double(hit.localRay.t.upper))
      break;
    const vec3 position = hit.localRay.org + float(t) * hit.localRay.dir;
    glm::dvec3 extinction;
    MdlTrackingChannel channels[3];
    if (!mediumCoefficients(ss, hit, position, extinction, channels))
      break;
    const uint32_t k = detail::pcg_advance(&ss.rs) >> 8;
    const auto event = mdlTrackingEvent(channels[ss.transportChannel],
        hit.volume->data.mdl.extinctionMajorant,
        k);
    if (event == MdlTrackingEvent::ABSORB) {
      ss.mediumAbsorbed = true;
      return float(t);
    }
    if (event == MdlTrackingEvent::SCATTER) {
      didScatter = true;
      return float(t);
    }
  }
  return hit.localRay.t.upper;
}

VISRTX_DEVICE void ratioTrackMdlVolume(
    ScreenSample &ss, const VolumeHit &hit, vec3 &attenuation)
{
  const double M = double(hit.volume->data.mdl.extinctionMajorant);
  if (M == 0.)
    return;
  double t = hit.localRay.t.lower;
  double transmittance = 1.;
  while (advanceMdlCandidate(ss, hit, t)) {
    if (t >= double(hit.localRay.t.upper))
      break;
    const vec3 position = hit.localRay.org + float(t) * hit.localRay.dir;
    glm::dvec3 extinction;
    if (!mediumCoefficients(ss, hit, position, extinction))
      break;
    transmittance *= 1. - extinction[ss.transportChannel] / M;
    if (transmittance == 0.)
      break;
  }
  attenuation[ss.transportChannel] *= float(transmittance);
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
  const vec3 sides = hit.volume->bounds.upper - hit.volume->bounds.lower;
  const double baseStep = hit.volume->stepSize < 0.f
      ? double(fminf(sides.x, fminf(sides.y, sides.z))) / 128.
      : double(hit.volume->stepSize);
  const double step = baseStep * double(inverseSamplingRate)
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
    if (!mediumCoefficients(
            ss, hit, position, extinction, nullptr, &scattering))
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
