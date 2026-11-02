// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "gpu/aggregateVolume.h"
#include "gpu/blackbody/blackbody.h"
#include "gpu/previewShadowCache.h"
#include "gpu/renderer/previewFloatLattice.h"
#include "gpu/renderer/shadowTransmittance.h"
#include "gpu/sampleLight.h"

namespace visrtx {
namespace practical_preview {
namespace av = preview_volume;
namespace ref = preview_math;

VISRTX_DEVICE float luminance(const float (&v)[3])
{
  return .2126f * v[0] + .7152f * v[1] + .0722f * v[2];
}

VISRTX_DEVICE ref::Coefficients opticalOnly(
    const PrincipledVolumeGPUData &volume,
    const vec3 &p,
    VolumePhaseFunction *phase = nullptr,
    bool source = false)
{
  PreviewFieldSampler sampler{volume.fields};
  const auto &b = volume.bindings;
  PrincipledVolumeValues v;
  v.density = ref::evaluateBinding(b.density, p, sampler);
  if (source)
    v.emission = ref::evaluateBinding(b.emission, p, sampler);
  if (v.density == 0.f) {
    if (phase)
      *phase = {};
    ref::Coefficients result;
    for (int c = 0; c < 3; ++c)
      result.emission[c] = v.emission[c];
    return result;
  }
  if (!volume.hasPreviewExtinction) {
    v.transmissionColor = ref::evaluateBinding(b.transmissionColor, p, sampler);
    v.transmissionDistance =
        ref::evaluateBinding(b.transmissionDistance, p, sampler);
  }
  v.scatteringColor = ref::evaluateBinding(b.scatteringColor, p, sampler);
  auto result = ref::evaluateOptical(v,
      volume.hasConstantTransmissionLog ? volume.constantTransmissionLog
                                        : nullptr,
      volume.hasPreviewExtinction ? volume.previewExtinctionPerDensity
                                  : nullptr);
  if (phase) {
    *phase = {};
    if (result.scattering[0] > 0.f || result.scattering[1] > 0.f
        || result.scattering[2] > 0.f)
      *phase = {ref::evaluateBinding(b.anisotropy, p, sampler),
          ref::evaluateBinding(b.secondaryAnisotropy, p, sampler),
          ref::evaluateBinding(b.secondaryWeight, p, sampler)};
  }
  // Exact zero coupling, not a density-only source cutoff: explicit emission
  // above is always sampled, including in vacuum and outside the density ROI.
  if (source
      && (result.absorption[0] > 0.f || result.absorption[1] > 0.f
          || result.absorption[2] > 0.f)) {
    v.blackbodyIntensity =
        ref::evaluateBinding(b.blackbodyIntensity, p, sampler);
    if (v.blackbodyIntensity > 0.f) {
      v.temperature = ref::evaluateBinding(b.temperature, p, sampler);
      v.blackbodyTint = ref::evaluateBinding(b.blackbodyTint, p, sampler);
      ref::addThermal(result, v, volume.previewCold);
    }
  }
  return result;
}

VISRTX_DEVICE ref::Coefficients sampleOccurrence(
    const VolumeOccurrenceGPUData &o, const vec3 &p, bool source)
{
  if (!ref::contains(o.volume.domain, p))
    return {};
  if (o.volume.type == VolumeType::PRINCIPLED)
    return opticalOnly(*o.volume.principled, p, nullptr, source);
  const auto &v = *o.volume.transfer;
  PreviewFieldSampler sampler{&v.field};
  const vec4 classified = av::classify(v, sampler.sample<float>(0, p));
  return ref::evaluateTransfer(
      source ? classified : vec4(0, 0, 0, classified.w), v.unitDistance);
}

struct Sample
{
  ref::Coefficients coefficients;
  VolumePhaseFunction singlePhase;
};
VISRTX_DEVICE Sample evaluate(const AggregateVolumeSceneGPUData &scene,
    const av::Ray &ray,
    av::Interval span,
    float distance,
    const PreparedRay &prepared)
{
  Sample result;
  if (prepared.single) {
    const auto &o = scene.occurrences[0];
    const vec3 p = av::objectPosition(prepared.local, distance);
    if (o.volume.type == VolumeType::PRINCIPLED
        && ref::contains(o.volume.domain, p))
      result.coefficients =
          opticalOnly(*o.volume.principled, p, &result.singlePhase, true);
    else
      result.coefficients = sampleOccurrence(o, p, true);
    return result;
  }
  for (size_t i = 0; i < scene.count; ++i) {
    const auto &o = scene.occurrences[i];
    av::Interval interval;
    if (!av::intersect(o, ray, interval)
        || !(interval.lower <= span.lower && span.lower < interval.upper))
      continue;
    const vec3 p = av::objectPosition(av::objectRay(o, ray), distance);
    if (!ref::contains(o.volume.domain, p))
      continue;
    ref::Coefficients value;
    if (o.volume.type == VolumeType::PRINCIPLED)
      value = opticalOnly(*o.volume.principled, p, nullptr, true);
    else {
      const auto &v = *o.volume.transfer;
      PreviewFieldSampler sampler{&v.field};
      value = ref::evaluateTransfer(
          av::classify(v, sampler.sample<float>(0, p)), v.unitDistance);
    }
    result.coefficients = ref::combine(result.coefficients, value);
  }
  return result;
}

VISRTX_DEVICE ref::Coefficients extinctionAt(
    const AggregateVolumeSceneGPUData &scene,
    const av::Ray &ray,
    av::Interval span,
    float distance)
{
  ref::Coefficients result;
  for (size_t i = 0; i < scene.count; ++i) {
    const auto &o = scene.occurrences[i];
    av::Interval interval;
    if (!av::intersect(o, ray, interval)
        || !(interval.lower <= span.lower && span.lower < interval.upper))
      continue;
    const vec3 p = av::objectPosition(av::objectRay(o, ray), distance);
    ref::Coefficients value;
    if (!ref::contains(o.volume.domain, p))
      continue;
    if (o.volume.type == VolumeType::PRINCIPLED)
      value = opticalOnly(*o.volume.principled, p);
    else {
      const auto &v = *o.volume.transfer;
      PreviewFieldSampler sampler{&v.field};
      const float alpha = av::classify(v, sampler.sample<float>(0, p)).w;
      value = ref::evaluateTransfer(vec4(0, 0, 0, alpha), v.unitDistance);
    }
    result = ref::combine(result, value);
  }
  return result;
}

VISRTX_DEVICE void kernelAt(const AggregateVolumeSceneGPUData &scene,
    const av::Ray &ray,
    av::Interval span,
    float distance,
    float cosine,
    float (&kernel)[3])
{
  for (int c = 0; c < 3; ++c)
    kernel[c] = 0.f;
  for (size_t i = 0; i < scene.count; ++i) {
    const auto &o = scene.occurrences[i];
    av::Interval interval;
    if (o.volume.type != VolumeType::PRINCIPLED
        || !av::intersect(o, ray, interval)
        || !(interval.lower <= span.lower && span.lower < interval.upper))
      continue;
    const vec3 p = av::objectPosition(av::objectRay(o, ray), distance);
    if (!ref::contains(o.volume.domain, p))
      continue;
    VolumePhaseFunction phase;
    const auto value = opticalOnly(*o.volume.principled, p, &phase);
    const float pdf = ref::phase(phase, cosine);
    for (int c = 0; c < 3; ++c)
      kernel[c] = ref::add(kernel[c], ref::multiply(value.scattering[c], pdf));
  }
}

VISRTX_DEVICE vec3 integrateAttenuation(
    const FrameGPUData &fd, const Ray &input)
{
  const auto &scene = *fd.referenceVolumes;
  const auto ray = av::worldDistanceRay(input);
  const auto prepared = prepareRay(scene, ray, float(fd.referenceShadowRate));
  if (prepared.single
      && scene.occurrences[0].volume.type == VolumeType::PRINCIPLED) {
    const auto &volume = *scene.occurrences[0].volume.principled;
    const auto &b = volume.bindings;
    if (b.transmissionColor.field < 0 && b.transmissionDistance.field < 0) {
      float densityIntegral = 0.f;
      float cursor = ray.lower;
      av::Interval span;
      while (av::nextSpan(scene, ray, cursor, span)) {
        if (b.density.field < 0)
          densityIntegral += b.density.scale * (span.upper - span.lower);
        else {
          PreviewDensityIntegral query;
          for (int c = 0; c < 3; ++c) {
            query.origin[c] = prepared.local.origin[c];
            query.direction[c] = prepared.local.direction[c];
          }
          query.lower = span.lower;
          query.upper = span.upper;
          query.anchor = prepared.full.lower;
          query.step = prepared.step;
          query.volumeDomain = b.bounds;
          query.fieldDomain = b.density.domain;
          query.scale = b.density.scale;
          query.outside = b.density.outside;
          query.allowTexture = fd.referenceShadowRate <= 1.;
          const auto &field = volume.fields[b.density.field];
          if (field.previewTexture && query.allowTexture)
            densityIntegral += ref::textureDensityIntegral(field, query);
          else
            densityIntegral += optixDirectCall<float>(
                uint32_t(SbtCallableEntryPoints::ReferenceFieldScalar),
                &field,
                static_cast<const vec3 *>(nullptr),
                true,
                &query);
        }
      }
      vec3 factor, result;
      for (int c = 0; c < 3; ++c) {
        factor[c] = volume.hasPreviewExtinction
            ? volume.previewExtinctionPerDensity[c]
            : (volume.hasConstantTransmissionLog
                      ? float(volume.constantTransmissionLog[c])
                      : -logf(b.transmissionColor.scale[c]))
                / b.transmissionDistance.scale;
      }
      if (factor.x == factor.y && factor.x == factor.z)
        return vec3(expf(-factor.x * densityIntegral));
      for (int c = 0; c < 3; ++c)
        result[c] = expf(-factor[c] * densityIntegral);
      return result;
    }
  }
  float opticalDepth[3]{};
  float cursor = ray.lower;
  av::Interval span;
  while (av::nextSpan(scene, ray, cursor, span)) {
    float t = span.lower;
    while (t < span.upper) {
      const float next = prepared.single
          ? nextPreparedCell(prepared, span, t)
          : nextCell(scene, ray, span, t, float(fd.referenceShadowRate));
      const float midpoint = t + (next - t) * .5f;
      const auto coefficients = prepared.single
          ? sampleOccurrence(scene.occurrences[0],
                av::objectPosition(prepared.local, midpoint),
                false)
          : extinctionAt(scene, ray, span, midpoint);
      for (int c = 0; c < 3; ++c)
        opticalDepth[c] += ref::extinction(coefficients, c) * (next - t);
      t = next;
    }
  }
  return vec3(
      expf(-opticalDepth[0]), expf(-opticalDepth[1]), expf(-opticalDepth[2]));
}

VISRTX_DEVICE vec3 attenuation(const FrameGPUData &fd, const Ray &input)
{
  glm::dvec3 result;
  optixDirectCall<void>(uint32_t(SbtCallableEntryPoints::ReferenceAttenuation),
      &fd,
      &input,
      &result);
  return vec3(result);
}

VISRTX_DEVICE void directSource(ScreenSample &ss,
    const av::Ray &ray,
    av::Interval span,
    float distance,
    float (&source)[3],
    const Sample &mediumSample)
{
  const auto &fd = *ss.frameData;
  const auto &scene = *fd.referenceVolumes;
  const vec3 point(float(ray.origin[0] + distance * ray.direction[0]),
      float(ray.origin[1] + distance * ray.direction[1]),
      float(ray.origin[2] + distance * ray.direction[2]));
  auto connection = [&](LightSample sample,
                        bool geometry,
                        float weight,
                        uint32_t lightIndex) {
    if (!(sample.pdf > 0.f) || !(sample.dist > 0.f))
      return;
    sample.dir = normalize(sample.dir);
    const float cosine = float(sample.dir.x * ray.direction[0]
        + sample.dir.y * ray.direction[1] + sample.dir.z * ray.direction[2]);
    float kernel[3];
    if (scene.count == 1) {
      const float phase = ref::phase(mediumSample.singlePhase, cosine);
      for (int c = 0; c < 3; ++c)
        kernel[c] = mediumSample.coefficients.scattering[c] * phase;
    } else
      kernelAt(scene, ray, span, distance, cosine, kernel);
    if (kernel[0] == 0.f && kernel[1] == 0.f && kernel[2] == 0.f)
      return;
    Ray shadow{point, sample.dir, {0.f, sample.dist}};
    vec3 volumeT;
    if (!lookupPreviewShadow(fd.previewShadowCache, point, lightIndex, volumeT))
      volumeT = attenuation(fd, shadow); // full endpoint, no emitter epsilon
    if (geometry)
      shadow.t.upper *= 1.f - GEOMETRY_LIGHT_SHADOW_EPSILON;
    const vec3 surfaceT = surfaceShadowTransmittance(ss, shadow);
    for (int c = 0; c < 3; ++c)
      source[c] += kernel[c] * float(sample.radiance[c]) * float(volumeT[c])
          * float(surfaceT[c]) * weight / float(sample.pdf);
  };
  auto light = [&](size_t index, float weight) {
    const auto &instance = fd.world.lightInstances[index];
    const auto &data = fd.registry.lights[instance.lightIndex];
    LightSample sample;
    if (data.type == LightType::HDRI) {
      sample.dir = randomDir(ss.rs);
      sample.dist = INFINITY;
      sample.pdf = 1.f / (4.f * kPi);
      sample.radiance =
          sampleHDRI(data, glm::transpose(mat3(instance.xfm)) * sample.dir)
          * data.color;
    } else
      sample = sampleLight(ss,
          point,
          instance.lightIndex,
          instance.xfm,
          instance.surfaceInstanceIndex);
    connection(
        sample, data.type == LightType::GEOMETRY, weight, uint32_t(index));
  };
  const size_t n = fd.world.numLightInstances;
  const int cap = fd.renderer.params.interactive.maxSampledLights;
  if (cap <= 0 || n <= size_t(cap)) {
    for (size_t i = 0; i < n; ++i)
      light(i, 1.f);
  } else {
    for (int i = 0; i < cap; ++i) {
      const size_t pick =
          glm::min(size_t(pcg_uniform(&ss.rs) * float(n)), n - 1);
      light(pick, float(n) / float(cap));
    }
  }
  if (fd.renderer.ambientIntensity > 0.f) {
    LightSample ambient{fd.renderer.ambientColor * fd.renderer.ambientIntensity,
        randomDir(ss.rs),
        INFINITY,
        1.f / (4.f * kPi)};
    connection(ambient, false, 1.f, ~0u);
  }
}

struct CameraIntegral
{
  float color[3]{};
  float fogFree[3]{1, 1, 1};
  float fogOwned[3]{};
  float uncovered[3]{1, 1, 1};
  float albedo[3]{};
};

VISRTX_DEVICE void integrateCameraSegment(ScreenSample &ss,
    const av::Ray &base,
    float begin,
    float end,
    CameraIntegral &integral)
{
  auto ray = base;
  ray.lower = begin;
  ray.upper = end;
  const auto &fd = *ss.frameData;
  const auto &scene = *fd.referenceVolumes;
  const auto prepared = prepareRay(scene, ray, float(fd.referenceVolumeRate));
  float cursor = begin;
  av::Interval span;
  while (av::nextSpan(scene, ray, cursor, span)) {
    float t = span.lower;
    while (t < span.upper) {
      const float next = prepared.single
          ? nextPreparedCell(prepared, span, t)
          : nextCell(scene, ray, span, t, float(fd.referenceVolumeRate));
      const float midpoint = t + (next - t) * .5f;
      const auto sample = evaluate(scene, ray, span, midpoint, prepared);
      bool zero = true;
      for (int c = 0; c < 3; ++c)
        zero &= sample.coefficients.absorption[c] == 0.f
            && sample.coefficients.scattering[c] == 0.f
            && sample.coefficients.emission[c] == 0.f;
      if (zero) {
        // All fields were evaluated for this cell, including independent
        // emission. Skip identity arithmetic only, never future source samples.
        t = next;
        continue;
      }
      float source[3]{sample.coefficients.emission[0],
          sample.coefficients.emission[1],
          sample.coefficients.emission[2]};
      if (sample.coefficients.scattering[0] > 0.f
          || sample.coefficients.scattering[1] > 0.f
          || sample.coefficients.scattering[2] > 0.f)
        directSource(ss, ray, span, midpoint, source, sample);
      float lost[3], transmission[3], radiance[3];
      const float firstExtinction = ref::extinction(sample.coefficients, 0);
      const auto firstWeights = ref::segmentWeights(firstExtinction, next - t);
      for (int c = 0; c < 3; ++c) {
        const float extinction = ref::extinction(sample.coefficients, c);
        const auto weights = extinction == firstExtinction
            ? firstWeights
            : ref::segmentWeights(extinction, next - t);
        transmission[c] = weights.transmission;
        lost[c] = integral.uncovered[c] * weights.loss;
        radiance[c] = source[c] * weights.sourceFactor;
      }
      const float mass = luminance(lost);
      for (int c = 0; c < 3; ++c) {
        const float extinction = ref::extinction(sample.coefficients, c);
        if (fd.fb.buffers.albedo && extinction > 0.f)
          integral.albedo[c] +=
              mass * sample.coefficients.scattering[c] / extinction;
      }
      for (int c = 0; c < 3; ++c) {
        integral.color[c] +=
            (integral.fogFree[c] + integral.fogOwned[c]) * radiance[c];
        integral.fogFree[c] *= transmission[c];
        integral.fogOwned[c] *= transmission[c];
        integral.uncovered[c] *= transmission[c];
      }
      t = next;
    }
  }
}
VISRTX_DEVICE bool eligibleInput(const FrameGPUData &fd, const visrtx::Ray &ray)
{
  const auto &scene = *fd.referenceVolumes;
  if (!scene.previewFloatEligible || fd.referenceVolumeRate > 64.
      || fd.referenceShadowRate > 64.)
    return false;
  float scale = 0.f;
  for (int c = 0; c < 3; ++c) {
    const float d = fabsf(ray.dir[c]);
    const uint32_t bits =
        optical_reference::floatBits(ray.dir[c]) & 0x7fffffffu;
    if ((bits != 0u && bits < 0x00800000u)
        || fabsf(ray.org[c]) > scene.previewOriginLimit || d > 0x1p16f)
      return false;
    scale = fmaxf(scale, d);
  }
  return scale >= 0x1p-20f
      && fabsf(ray.t.lower) * scale * 2.f <= scene.previewOriginLimit;
}
VISRTX_DEVICE bool eligible(
    const FrameGPUData &fd, const aggregate_volume::Ray &ray)
{
  const auto &scene = *fd.referenceVolumes;
  if (!scene.previewFloatEligible || fd.referenceVolumeRate > 64.
      || fd.referenceShadowRate > 64.)
    return false;
  for (int c = 0; c < 3; ++c)
    if (fabs(ray.origin[c]) > scene.previewOriginLimit
        || (ray.direction[c] != 0. && fabs(ray.direction[c]) < 0x1p-126))
      return false;
  return fabs(ray.lower) <= scene.previewOriginLimit;
}
VISRTX_DEVICE av::Ray floatRay(const aggregate_volume::Ray &ray)
{
  av::Ray result;
  for (int c = 0; c < 3; ++c) {
    result.origin[c] = float(ray.origin[c]);
    result.direction[c] = float(ray.direction[c]);
  }
  result.lower = float(ray.lower);
  result.upper = float(ray.upper);
  return result;
}
template <typename A, typename B>
VISRTX_DEVICE void copyIntegral(A &to, const B &from)
{
  for (int c = 0; c < 3; ++c) {
    to.color[c] = from.color[c];
    to.fogFree[c] = from.fogFree[c];
    to.fogOwned[c] = from.fogOwned[c];
    to.uncovered[c] = from.uncovered[c];
    to.albedo[c] = from.albedo[c];
  }
}
VISRTX_DEVICE void cameraSegment(ScreenSample &ss,
    const aggregate_volume::Ray &ray,
    double begin,
    double end,
    CameraIntegral &integral)
{
  optixContinuationCall<void>(
      uint32_t(SbtCallableEntryPoints::ReferenceCameraSegment),
      &ss,
      &ray,
      begin,
      end,
      &integral);
}
} // namespace practical_preview
} // namespace visrtx
