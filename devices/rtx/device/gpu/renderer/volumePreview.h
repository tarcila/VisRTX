// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "gpu/renderer/volumePreviewLattice.h"
#include "gpu/blackbody/blackbody.h"
#include "gpu/renderer/shadowTransmittance.h"
#include "gpu/sampleLight.h"

namespace visrtx {
namespace volume_preview {
namespace av = aggregate_volume;
namespace ref = optical_reference;

VISRTX_DEVICE double luminance(const double (&v)[3])
{
  return .2126 * v[0] + .7152 * v[1] + .0722 * v[2];
}

VISRTX_DEVICE ref::Coefficients opticalOnly(
    const PrincipledVolumeGPUData &volume,
    const vec3 &p,
    VolumePhaseFunction *phase = nullptr)
{
  ReferenceFieldSampler sampler{volume.fields};
  const auto &b = volume.bindings;
  PrincipledVolumeValues v;
  v.density = ref::evaluateBinding(b.density, p, sampler);
  v.transmissionColor = ref::evaluateBinding(b.transmissionColor, p, sampler);
  v.transmissionDistance =
      ref::evaluateBinding(b.transmissionDistance, p, sampler);
  v.scatteringColor = ref::evaluateBinding(b.scatteringColor, p, sampler);
  if (phase)
    *phase = {ref::evaluateBinding(b.anisotropy, p, sampler),
        ref::evaluateBinding(b.secondaryAnisotropy, p, sampler),
        ref::evaluateBinding(b.secondaryWeight, p, sampler)};
  return ref::evaluateOptical(v,
      volume.hasConstantTransmissionLog ? volume.constantTransmissionLog
                                        : nullptr);
}

VISRTX_DEVICE ref::Coefficients extinctionAt(
    const AggregateVolumeSceneGPUData &scene,
    const av::Ray &ray,
    av::Interval span,
    double distance)
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
      ReferenceFieldSampler sampler{&v.field};
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
    double distance,
    float cosine,
    double (&kernel)[3])
{
  for (int c = 0; c < 3; ++c)
    kernel[c] = 0.;
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
    const double pdf = ref::promote(evaluateVolumePhase(phase, cosine));
    for (int c = 0; c < 3; ++c)
      kernel[c] = ref::add(kernel[c], ref::multiply(value.scattering[c], pdf));
  }
}

VISRTX_DEVICE glm::dvec3 integrateAttenuation(
    const FrameGPUData &fd, const Ray &input)
{
  const auto &scene = *fd.referenceVolumes;
  const auto ray = av::worldDistanceRay(input);
  double transmittance[3]{1, 1, 1};
  double cursor = ray.lower;
  av::Interval span;
  while (av::nextSpan(scene, ray, cursor, span)) {
    double t = span.lower;
    while (t < span.upper) {
      const double next = nextCell(scene, ray, span, t, fd.referenceShadowRate);
      const auto coefficients =
          extinctionAt(scene, ray, span, t + (next - t) * .5);
      for (int c = 0; c < 3; ++c)
        transmittance[c] *= exp(-ref::extinction(coefficients, c) * (next - t));
      t = next;
    }
  }
  return glm::dvec3(transmittance[0], transmittance[1], transmittance[2]);
}

VISRTX_DEVICE glm::dvec3 attenuation(const FrameGPUData &fd, const Ray &input)
{
  glm::dvec3 result;
  optixDirectCall<void>(uint32_t(SbtCallableEntryPoints::ReferenceAttenuation),
      &fd,
      &input,
      &result);
  return result;
}

VISRTX_DEVICE void directSource(ScreenSample &ss,
    const av::Ray &ray,
    av::Interval span,
    double distance,
    double (&source)[3])
{
  const auto &fd = *ss.frameData;
  const auto &scene = *fd.referenceVolumes;
  const vec3 point(float(ray.origin[0] + distance * ray.direction[0]),
      float(ray.origin[1] + distance * ray.direction[1]),
      float(ray.origin[2] + distance * ray.direction[2]));
  auto connection = [&](LightSample sample, bool geometry, double weight) {
    if (!(sample.pdf > 0.f) || !(sample.dist > 0.f))
      return;
    sample.dir = normalize(sample.dir);
    const float cosine = float(sample.dir.x * ray.direction[0]
        + sample.dir.y * ray.direction[1] + sample.dir.z * ray.direction[2]);
    double kernel[3];
    kernelAt(scene, ray, span, distance, cosine, kernel);
    if (kernel[0] == 0. && kernel[1] == 0. && kernel[2] == 0.)
      return;
    Ray shadow{point, sample.dir, {0.f, sample.dist}};
    const auto volumeT =
        attenuation(fd, shadow); // full endpoint, no emitter epsilon
    if (geometry)
      shadow.t.upper *= 1.f - GEOMETRY_LIGHT_SHADOW_EPSILON;
    const vec3 surfaceT = surfaceShadowTransmittance(ss, shadow);
    for (int c = 0; c < 3; ++c)
      source[c] += kernel[c] * double(sample.radiance[c]) * double(volumeT[c])
          * double(surfaceT[c]) * weight / double(sample.pdf);
  };
  auto light = [&](size_t index, double weight) {
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
    connection(sample, data.type == LightType::GEOMETRY, weight);
  };
  const size_t n = fd.world.numLightInstances;
  const int cap = fd.renderer.params.interactive.maxSampledLights;
  if (cap <= 0 || n <= size_t(cap)) {
    for (size_t i = 0; i < n; ++i)
      light(i, 1.);
  } else {
    for (int i = 0; i < cap; ++i) {
      const size_t pick =
          glm::min(size_t(pcg_uniform(&ss.rs) * double(n)), n - 1);
      light(pick, double(n) / double(cap));
    }
  }
  if (fd.renderer.ambientIntensity > 0.f) {
    LightSample ambient{fd.renderer.ambientColor * fd.renderer.ambientIntensity,
        randomDir(ss.rs),
        INFINITY,
        1.f / (4.f * kPi)};
    connection(ambient, false, 1.);
  }
}

struct CameraIntegral
{
  double color[3]{};
  double fogFree[3]{1, 1, 1};
  double fogOwned[3]{};
  double uncovered[3]{1, 1, 1};
  double albedo[3]{};
};

VISRTX_DEVICE void integrateCameraSegment(ScreenSample &ss,
    const av::Ray &base,
    double begin,
    double end,
    CameraIntegral &integral)
{
  auto ray = base;
  ray.lower = begin;
  ray.upper = end;
  const auto &fd = *ss.frameData;
  const auto &scene = *fd.referenceVolumes;
  CalibratedBlackbody blackbody;
  double cursor = begin;
  av::Interval span;
  while (av::nextSpan(scene, ray, cursor, span)) {
    double t = span.lower;
    while (t < span.upper) {
      const double next = nextCell(scene, ray, span, t, fd.referenceVolumeRate);
      const double midpoint = t + (next - t) * .5;
      const auto sample =
          av::evaluate(scene, ray, span, midpoint, 0.f, blackbody);
      double source[3]{sample.coefficients.emission[0],
          sample.coefficients.emission[1],
          sample.coefficients.emission[2]};
      if (sample.coefficients.scattering[0] > 0.
          || sample.coefficients.scattering[1] > 0.
          || sample.coefficients.scattering[2] > 0.)
        directSource(ss, ray, span, midpoint, source);
      double lost[3], transmission[3];
      for (int c = 0; c < 3; ++c) {
        const double tau = ref::extinction(sample.coefficients, c) * (next - t);
        transmission[c] = exp(-tau);
        lost[c] = integral.uncovered[c] * (-expm1(-tau));
      }
      const double mass = luminance(lost);
      for (int c = 0; c < 3; ++c) {
        const double extinction = ref::extinction(sample.coefficients, c);
        if (fd.fb.buffers.albedo && extinction > 0.)
          integral.albedo[c] +=
              mass * sample.coefficients.scattering[c] / extinction;
      }
      av::SegmentIntegral cell;
      av::integrateSegment(sample.coefficients, source, next - t, cell);
      for (int c = 0; c < 3; ++c) {
        integral.color[c] +=
            (integral.fogFree[c] + integral.fogOwned[c]) * cell.radiance[c];
        integral.fogFree[c] *= transmission[c];
        integral.fogOwned[c] *= transmission[c];
        integral.uncovered[c] *= transmission[c];
      }
      t = next;
    }
  }
}
} // namespace volume_preview
} // namespace visrtx
