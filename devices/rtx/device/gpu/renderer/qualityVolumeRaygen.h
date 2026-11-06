// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "gpu/evalShading.h"
#include "gpu/renderer/qualityVolume.h"
#include "gpu/renderer/shadowTransmittance.h"
#include "gpu/volumeIntegration.h"
#include "gpu/volumeSourceOwnership.h"

namespace visrtx {
namespace quality_volume {

VISRTX_DEVICE void seed(
    PCGState &state, const ScreenSample &ss, uint32_t ordinal, uint64_t tag)
{
  const uint64_t pixel =
      uint64_t(ss.pixel.x) + uint64_t(ss.pixel.y) * ss.frameData->fb.size.x;
  const uint64_t salt = uint64_t(ss.randomSeed) * 0x9E3779B97F4A7C15ULL;
  pcg_init(&state,
      detail::pcg_mix64(pixel ^ (uint64_t(ordinal) << 32) ^ tag ^ salt),
      detail::pcg_mix64(pixel ^ tag ^ salt));
}
VISRTX_DEVICE glm::dvec3 mediumShadow(
    const FrameGPUData &fd, const Ray &ray, PCGState &random)
{
  if (fd.referenceVolumes->count == 0)
    return glm::dvec3(1.);
  Request q;
  q.operation = Operation::Shadow;
  q.scene = fd.referenceVolumes;
  q.ray = aggregate_volume::worldDistanceRay(ray);
  q.random = random;
  execute(q);
  random = q.random;
  return glm::dvec3(q.transmission[0], q.transmission[1], q.transmission[2]);
}
VISRTX_DEVICE volume_source::Sample sourceSample(
    const FrameGPUData &fd, const vec3 &point, PCGState &random)
{
  Request q;
  q.operation = fd.renderer.params.quality.volumeSourceResampling
      ? Operation::SourceResampled
      : Operation::Source;
  q.scene = fd.referenceVolumes;
  q.random = random;
  for (int c = 0; c < 3; ++c)
    q.ray.origin[c] = optical_reference::promote(point[c]);
  execute(q);
  random = q.random;
  return q.source;
}
VISRTX_DEVICE glm::dvec3 sourceVisibility(ScreenSample &ss,
    const volume_source::Sample &sample,
    float surfaceLower,
    PCGState &random)
{
  const auto &r = sample.ray;
  const Ray ray{vec3(r.origin[0], r.origin[1], r.origin[2]),
      vec3(r.direction[0], r.direction[1], r.direction[2]),
      {surfaceLower, optical_reference::narrow(r.upper)}};
  double holes = 1.;
  if (ray.t.lower < ray.t.upper)
    intersectSurface(ss,
        ray,
        RayType::VOLUME_SOURCE,
        &holes,
        OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT);
  if (holes == 0.)
    return glm::dvec3(0.);
  Request q;
  q.operation = Operation::Shadow;
  q.scene = ss.frameData->referenceVolumes;
  q.ray = r;
  q.random = random;
  execute(q);
  random = q.random;
  return holes
      * glm::dvec3(q.transmission[0], q.transmission[1], q.transmission[2]);
}
VISRTX_DEVICE vec3 surfaceSource(ScreenSample &ss,
    const MaterialShadingState &shading,
    const Ray &incoming,
    const SurfaceHit &hit,
    const volume_tracking::PathState &path,
    float opacity,
    double appearance,
    PCGState &sourceRandom,
    PCGState &shadowRandom)
{
  const auto sample =
      sourceSample(*ss.frameData, shadingHitpoint(hit), sourceRandom);
  if (!sample.valid
      || (sample.weight[0] == 0. && sample.weight[1] == 0.
          && sample.weight[2] == 0.))
    return vec3(0.f);
  const vec3 direction(sample.ray.direction[0],
      sample.ray.direction[1],
      sample.ray.direction[2]);
  const vec3 kernel = materialEvalBsdf(shading, -incoming.dir, direction);
  if (kernel == vec3(0.f))
    return vec3(0.f);
  const auto visibility =
      sourceVisibility(ss, sample, hit.epsilon, shadowRandom);
  vec3 result;
  for (int c = 0; c < 3; ++c)
    result[c] = optical_reference::narrow(path.throughput[c] * double(opacity)
        * appearance * double(kernel[c]) * sample.weight[c] * visibility[c]);
  return result;
}
VISRTX_DEVICE vec3 mediumSource(ScreenSample &ss,
    Request &vertex,
    const vec3 &point,
    double appearance,
    PCGState &sourceRandom,
    PCGState &shadowRandom)
{
  const auto sample = sourceSample(*ss.frameData, point, sourceRandom);
  if (!sample.valid
      || (sample.weight[0] == 0. && sample.weight[1] == 0.
          && sample.weight[2] == 0.))
    return vec3(0.f);
  double cosine = 0.;
  for (int c = 0; c < 3; ++c)
    cosine += vertex.ray.direction[c] * sample.ray.direction[c];
  vertex.cosine = glm::clamp(cosine, -1., 1.);
  vertex.operation = Operation::Phase;
  execute(vertex);
  const auto visibility = sourceVisibility(ss, sample, 0.f, shadowRandom);
  vec3 result(0.f);
  for (int c = 0; c < 3; ++c) {
    const double s = vertex.flight.coefficients.scattering[c];
    if (s > 0.)
      result[c] = optical_reference::narrow(vertex.path.throughput[c]
          * (vertex.phase.kernel[c] / s) * sample.weight[c] * visibility[c]
          * appearance);
  }
  return result;
}

VISRTX_DEVICE vec3 weighted(
    const volume_tracking::PathState &path, const vec3 &radiance, double weight)
{
  vec3 result;
  for (int c = 0; c < 3; ++c)
    result[c] = optical_reference::narrow(
        path.throughput[c] * double(radiance[c]) * weight);
  return result;
}
VISRTX_DEVICE vec3 connection(ScreenSample &ss,
    const Ray &ray,
    PCGState &random,
    const volume_tracking::PathState &path,
    const vec3 &kernel,
    const vec3 &radiance,
    double weight)
{
  const auto surface = surfaceShadowTransmittance(ss, ray);
  const auto medium = mediumShadow(*ss.frameData, ray, random);
  vec3 result;
  for (int c = 0; c < 3; ++c)
    result[c] = optical_reference::narrow(path.throughput[c] * double(kernel[c])
        * double(radiance[c]) * weight * double(surface[c]) * medium[c]);
  return result;
}
VISRTX_DEVICE vec3 throughput(const volume_tracking::PathState &path)
{
  return vec3(optical_reference::narrow(path.throughput[0]),
      optical_reference::narrow(path.throughput[1]),
      optical_reference::narrow(path.throughput[2]));
}
VISRTX_DEVICE bool roulette(
    volume_tracking::PathState &path, PCGState &state, int depth)
{
  double maximum = 0.;
  for (int c = 0; c < 3; ++c)
    maximum = maximum > path.throughput[c] ? maximum : path.throughput[c];
  if (maximum == 0.)
    return true;
  if (depth < 3)
    return false;
  const double p = maximum < .95 ? maximum : .95;
  volume_tracking::Random random{&state};
  if (random.next() >= p)
    return true;
  for (int c = 0; c < 3; ++c)
    path.throughput[c] = optical_reference::divide(path.throughput[c], p);
  return false;
}
struct Coverage
{
  vec3 uncovered{1.f};
  vec3 albedo{0.f};
  vec3 normal{0.f};
  float depth{std::numeric_limits<float>::max()};
  uint32_t primitive{~0u}, object{~0u}, instance{~0u};
};

// Independent whole original-camera-ray query. The ratio-tracking candidate
// weights marginalize the RGB-channel-mixture first-event law: each channel's
// U*t/M loss weights the aggregate local albedo, and volume normals are zero.
// No radiance depth, roulette, transmission/refraction or source flag owns this
// query. The private RGB screen-plate weight is U, never radiance throughput.
VISRTX_DEVICE Coverage coverage(ScreenSample &ss, Ray ray, uint32_t ordinal)
{
  const auto &fd = *ss.frameData;
  Coverage result;
  Request q;
  q.operation = Operation::Coverage;
  q.scene = fd.referenceVolumes;
  q.ray = aggregate_volume::worldDistanceRay(ray);
  seed(q.random, ss, ordinal, 0xf4a519327bULL);
  const auto full = q.ray;
  const double speed = sqrt(double(ray.dir.x) * ray.dir.x
      + double(ray.dir.y) * ray.dir.y + double(ray.dir.z) * ray.dir.z);
  double cursor = full.lower;
  const float end = ray.t.upper;
  result.depth = end;
  while (true) {
    SurfaceHit hit{};
    if (ray.t.lower < end)
      intersectSurface(ss,
          ray,
          RayType::PRIMARY,
          &hit,
          primaryRayOptiXFlags(fd.renderer),
          primaryWithVisibleLightsMask());
    const double far = hit.foundHit ? double(hit.t) * speed : full.upper;
    q.ray = full;
    q.ray.lower = cursor;
    q.ray.upper = far;
    if (cursor < far && q.scene->count > 0)
      execute(q);
    // Retain approximate TF picking independently of physical/AOV transport.
    // Principled picking remains deferred; surface depth arbitration is shared.
    if (ordinal == 0
        && (fd.fb.buffers.depth || fd.fb.buffers.objID || fd.fb.buffers.primID
            || fd.fb.buffers.instID)) {
      ScreenSample picking = ss;
      vec3 color(0.f);
      float opacity = 0.f;
      uint32_t object = ~0u, instance = ~0u;
      const float endPoint = hit.foundHit ? hit.t : end;
      const float depth = rayMarchAllVolumes(picking,
          ray,
          RayType::PRIMARY,
          endPoint,
          fd.renderer.inverseVolumeSamplingRate,
          color,
          opacity,
          object,
          instance);
      if (depth < endPoint && depth < result.depth) {
        result.depth = depth;
        result.primitive = result.object = object;
        result.instance = instance;
      }
    }
    cursor = far;
    if (ordinal == 0 && hit.foundHit && hit.t < result.depth) {
      result.depth = hit.t;
      result.primitive = hit.isLightProxy() ? ~0u : hit.primID;
      result.object = hit.isLightProxy() ? ~0u : hit.objID;
      result.instance = hit.isLightProxy() ? ~0u : hit.instID;
    }
    if (!hit.foundHit)
      break;
    if (hit.isLightProxy()) {
      for (int c = 0; c < 3; ++c)
        q.transmission[c] = 0.;
      break;
    }
    MaterialShadingState shading;
    materialInitShading(&shading, fd, *hit.material, hit);
    const float alpha = materialEvaluateOpacity(shading);
    const vec3 tint = materialEvaluateTint(shading);
    vec3 normal = materialEvaluateNormal(shading);
    const float length = glm::length(normal);
    normal = length > 0.f ? normal / length : vec3(0);
    const double weight = alpha
        * (.2126 * q.transmission[0] + .7152 * q.transmission[1]
            + .0722 * q.transmission[2]);
    for (int c = 0; c < 3; ++c) {
      q.coverageAlbedo[c] += weight * double(tint[c]);
      result.normal[c] += float(weight * double(normal[c]));
      q.transmission[c] *= 1. - double(alpha);
    }
    if (q.transmission[0] == 0. && q.transmission[1] == 0.
        && q.transmission[2] == 0.)
      break;
    ray.t.lower = hit.t + hit.epsilon;
  }
  vec3 ignored;
  const bool background = getBackgroundLight(fd, ray.dir, ignored);
  for (int c = 0; c < 3; ++c) {
    result.uncovered[c] =
        background ? 0.f : optical_reference::narrow(q.transmission[c]);
    result.albedo[c] = optical_reference::narrow(q.coverageAlbedo[c]);
  }
  return result;
}
} // namespace quality_volume
} // namespace visrtx
