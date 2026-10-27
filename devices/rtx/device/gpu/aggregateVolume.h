// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "aggregateVolumeData.h"
#include "spatial_field/ReferenceSampling.h"

namespace visrtx {
namespace aggregate_volume {
namespace ref = optical_reference;

struct Ray
{
  double origin[3]{};
  double direction[3]{}; // unit world direction
  double lower{0.};
  double upper{0.};
};
struct Interval
{
  double lower{0.};
  double upper{0.};
};

// Preserve the caller's clipped endpoints even for nonunit input directions.
// Preconditions: finite origin/direction, nonzero direction, ordered limits;
// the upper limit may be +infinity for connections to distant illumination.
VISRTX_HOST_DEVICE Ray worldDistanceRay(const visrtx::Ray &input)
{
  Ray result;
  double length2 = 0.;
  for (int c = 0; c < 3; ++c) {
    result.origin[c] = ref::promote(input.org[c]);
    const double d = ref::promote(input.dir[c]);
    length2 = ref::add(length2, ref::multiply(d, d));
  }
  const double length = sqrt(length2);
  for (int c = 0; c < 3; ++c)
    result.direction[c] = ref::divide(ref::promote(input.dir[c]), length);
  result.lower = ref::multiply(ref::promote(input.t.lower), length);
  result.upper = ref::multiply(ref::promote(input.t.upper), length);
  return result;
}

struct ObjectRay
{
  double origin[3]{};
  double direction[3]{};
};

VISRTX_HOST_DEVICE ObjectRay objectRay(
    const VolumeOccurrenceGPUData &occurrence, const Ray &ray)
{
  ObjectRay result;
  for (int row = 0; row < 3; ++row) {
    result.origin[row] = occurrence.worldToObject[row][3];
    for (int col = 0; col < 3; ++col) {
      result.origin[row] = ref::add(result.origin[row],
          ref::multiply(occurrence.worldToObject[row][col], ray.origin[col]));
      result.direction[row] = ref::add(result.direction[row],
          ref::multiply(
              occurrence.worldToObject[row][col], ray.direction[col]));
    }
  }
  return result;
}

VISRTX_HOST_DEVICE bool intersectDomain(const box3 &box,
    const ObjectRay &local,
    double lo,
    double hi,
    Interval &result)
{
  if (!ref::nonemptyDomain(box))
    return false;
  for (int c = 0; c < 3; ++c) {
    const double a = ref::promote(box.lower[c]), b = ref::promote(box.upper[c]);
    if (local.direction[c] == 0.) {
      if (local.origin[c] < a || local.origin[c] > b)
        return false;
    } else {
      double near =
          ref::divide(ref::subtract(a, local.origin[c]), local.direction[c]);
      double far =
          ref::divide(ref::subtract(b, local.origin[c]), local.direction[c]);
      if (far < near) {
        const double swap = near;
        near = far;
        far = swap;
      }
      lo = lo > near ? lo : near;
      hi = hi < far ? hi : far;
    }
    if (!(lo < hi))
      return false;
  }
  result = {lo, hi};
  return true;
}

VISRTX_HOST_DEVICE bool intersect(
    const VolumeOccurrenceGPUData &occurrence, const Ray &ray, Interval &result)
{
  return intersectDomain(occurrence.volume.domain,
      objectRay(occurrence, ray),
      ray.lower,
      ray.upper,
      result);
}

template <typename T>
VISRTX_HOST_DEVICE const box3 *bindingDomain(
    const FieldBindingGPUData<T> &binding)
{
  return binding.field < 0 ? nullptr : &binding.domain;
}

VISRTX_HOST_DEVICE const box3 *bindingDomain(
    const PrincipledVolumeBindingsGPUData &b, int role)
{
  switch (role) {
  case 0:
    return bindingDomain(b.density);
  case 1:
    return bindingDomain(b.scatteringColor);
  case 2:
    return bindingDomain(b.transmissionColor);
  case 3:
    return bindingDomain(b.transmissionDistance);
  case 4:
    return bindingDomain(b.emission);
  case 5:
    return bindingDomain(b.temperature);
  case 6:
    return bindingDomain(b.blackbodyIntensity);
  case 7:
    return bindingDomain(b.blackbodyTint);
  case 8:
    return bindingDomain(b.anisotropy);
  case 9:
    return bindingDomain(b.secondaryAnisotropy);
  case 10:
    return bindingDomain(b.secondaryWeight);
  default:
    return nullptr;
  }
}

VISRTX_HOST_DEVICE void bindingCuts(const box3 &domain,
    const ObjectRay &local,
    Interval volume,
    double cursor,
    double &next)
{
  Interval field;
  if (!intersectDomain(domain, local, volume.lower, volume.upper, field))
    return;
  if (field.lower > cursor && field.lower < next)
    next = field.lower;
  if (field.upper > cursor && field.upper < next)
    next = field.upper;
}

// Constant per-ray storage: replay the complete counted scene, selecting the
// next strictly greater represented endpoint. Equal endpoints are processed
// together; there is no epsilon jump, overlap cap or per-ray hit array.
// Caller requires scene.complete. False also protects against accidental use of
// an incomplete view, but must not be interpreted as a successfully empty
// scene.
VISRTX_HOST_DEVICE bool nextSpan(const AggregateVolumeSceneGPUData &scene,
    const Ray &ray,
    double &cursor,
    Interval &span)
{
  if (!scene.complete)
    return false;
  while (cursor < ray.upper) {
    double next = ray.upper;
    bool active = false;
    for (size_t i = 0; i < scene.count; ++i) {
      Interval interval;
      if (!intersect(scene.occurrences[i], ray, interval))
        continue;
      if (interval.lower > cursor && interval.lower < next)
        next = interval.lower;
      if (interval.upper > cursor && interval.upper < next)
        next = interval.upper;
      active |= interval.lower <= cursor && cursor < interval.upper;
      const auto &occurrence = scene.occurrences[i];
      if (occurrence.volume.type == VolumeType::PRINCIPLED) {
        const auto local = objectRay(occurrence, ray);
        const auto &b = occurrence.volume.principled->bindings;
        // Keep a single slab-intersection body instead of eleven expanded
        // copies in every preview/shadow traversal.
#ifdef __CUDACC__
#pragma unroll 1
#endif
        for (int role = 0; role < 11; ++role) {
          const box3 *domain = bindingDomain(b, role);
          if (domain)
            bindingCuts(*domain, local, interval, cursor, next);
        }
      }
    }
    const double begin = cursor;
    cursor = next;
    if (active) {
      span = {begin, next};
      return true;
    }
  }
  return false;
}

VISRTX_HOST_DEVICE vec3 objectPosition(const ObjectRay &ray, double distance)
{
  vec3 result;
  for (int c = 0; c < 3; ++c)
    result[c] = ref::narrow(
        ref::add(ray.origin[c], ref::multiply(distance, ray.direction[c])));
  return result;
}

VISRTX_HOST_DEVICE vec4 classify(
    const TransferFunctionReferenceGPUData &volume, float value)
{
  const double lo = ref::promote(volume.valueRange.lower);
  const double hi = ref::promote(volume.valueRange.upper);
  const double normalized = ref::divide(
      ref::subtract(ref::promote(value), lo), ref::subtract(hi, lo));
  const double x = ref::clampCoordinate(
      ref::subtract(ref::multiply(normalized, double(volume.colorCount)), .5),
      0.,
      double(volume.colorCount - 1));
  const uint32_t a = uint32_t(x);
  const uint32_t b = a + 1 < volume.colorCount ? a + 1 : a;
  const float weight = ref::narrow(ref::subtract(x, double(a)));
  vec4 result;
  for (int c = 0; c < 4; ++c)
    result[c] =
        ref::canonicalLerp(volume.colors[a][c], volume.colors[b][c], weight);
  return result;
}

// Majorants are metadata-only: no field/source samples are needed to choose a
// candidate rate. The renderer adds any positive source-support proposal rate
// separately; it must not treat a zero extinction bound as absence of emission.
VISRTX_HOST_DEVICE ref::Bounds boundsForSpan(
    const AggregateVolumeSceneGPUData &scene, const Ray &ray, Interval span)
{
  ref::Bounds result;
  for (size_t i = 0; i < scene.count; ++i) {
    const auto &occurrence = scene.occurrences[i];
    Interval interval;
    if (!intersect(occurrence, ray, interval)
        || !(interval.lower <= span.lower && span.lower < interval.upper))
      continue;
    if (occurrence.volume.type == VolumeType::PRINCIPLED)
      result = ref::combineBounds(
          result, occurrence.volume.principled->opticalBounds);
    else if (occurrence.volume.type == VolumeType::TF1D)
      result =
          ref::combineBounds(result, occurrence.volume.transfer->opticalBounds);
  }
  return result;
}

struct Sample
{
  ref::Coefficients coefficients;
  ref::Bounds bounds;
  double phaseKernel[3]{};
  size_t activeCount{0};
};

// Use the span's active set, not an arbitrary selected hit/material or rounded
// field position. Consumers pass distance in [span.lower,span.upper); midpoint
// rounding to an endpoint does not manufacture a new overlap member.
template <typename Blackbody>
VISRTX_HOST_DEVICE Sample evaluate(const AggregateVolumeSceneGPUData &scene,
    const Ray &ray,
    Interval span,
    double distance,
    float cosTheta,
    Blackbody &blackbody)
{
  Sample result;
  for (size_t i = 0; i < scene.count; ++i) {
    const auto &occurrence = scene.occurrences[i];
    Interval interval;
    if (!intersect(occurrence, ray, interval)
        || !(interval.lower <= span.lower && span.lower < interval.upper))
      continue;
    const vec3 p = objectPosition(objectRay(occurrence, ray), distance);
    ref::Coefficients coefficients;
    ref::Bounds bounds;
    if (occurrence.volume.type == VolumeType::PRINCIPLED) {
      const auto &volume = *occurrence.volume.principled;
      ReferenceFieldSampler sampler{volume.fields};
      coefficients =
          ref::evaluateBindings(volume.bindings, p, sampler, blackbody);
      bounds = volume.opticalBounds;
      const VolumePhaseFunction phase{
          ref::evaluateBinding(volume.bindings.anisotropy, p, sampler),
          ref::evaluateBinding(volume.bindings.secondaryAnisotropy, p, sampler),
          ref::evaluateBinding(volume.bindings.secondaryWeight, p, sampler)};
      const double phaseValue =
          ref::promote(evaluateVolumePhase(phase, cosTheta));
      for (int c = 0; c < 3; ++c)
        result.phaseKernel[c] = ref::add(result.phaseKernel[c],
            ref::multiply(coefficients.scattering[c], phaseValue));
    } else if (occurrence.volume.type == VolumeType::TF1D) {
      const auto &volume = *occurrence.volume.transfer;
      ReferenceFieldSampler sampler{&volume.field};
      bounds = volume.opticalBounds;
      if (ref::contains(occurrence.volume.domain, p))
        coefficients = ref::evaluateTransfer(
            classify(volume, sampler.sample<float>(0, p)), volume.unitDistance);
    }
    result.coefficients = ref::combine(result.coefficients, coefficients);
    result.bounds = ref::combineBounds(result.bounds, bounds);
    ++result.activeCount;
  }
  return result;
}

struct SegmentIntegral
{
  double radiance[3]{};
  double transmittance[3]{1., 1., 1.};
};

// Exact in real arithmetic for constant coefficients/source. This is a shared
// segment primitive, not full single/multiple-scattering transport. A preview
// renderer can pass j plus its direct-scattering source estimate as source[].
VISRTX_HOST_DEVICE void integrateSegment(const ref::Coefficients &coefficients,
    const double (&source)[3],
    double worldLength,
    SegmentIntegral &integral)
{
  for (int c = 0; c < 3; ++c) {
    const double t = ref::extinction(coefficients, c);
    const double opticalDepth = ref::multiply(t, worldLength);
    const double transmission = exp(-opticalDepth);
    const double loss = -expm1(-opticalDepth);
    // The zero-product branch also preserves the analytic limit when a
    // positive t*h underflows. It does not change the physical coefficient.
    const double factor = opticalDepth == 0. ? worldLength
        : opticalDepth < 1.
        ? ref::multiply(worldLength, ref::divide(loss, opticalDepth))
        : ref::divide(loss, t);
    integral.radiance[c] = ref::add(integral.radiance[c],
        ref::multiply(
            integral.transmittance[c], ref::multiply(source[c], factor)));
    integral.transmittance[c] =
        ref::multiply(integral.transmittance[c], transmission);
  }
}

} // namespace aggregate_volume
} // namespace visrtx
