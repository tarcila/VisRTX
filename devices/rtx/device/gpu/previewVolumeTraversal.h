// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "aggregateVolumeData.h"
#include "spatial_field/ReferenceSampling.h"

namespace visrtx {
namespace preview_volume {
namespace ref = preview_math;

struct Ray
{
  float origin[3]{};
  float direction[3]{}; // unit world direction
  float lower{0.f};
  float upper{0.f};
};
struct Interval
{
  float lower{0.f};
  float upper{0.f};
};

// Preserve the caller's clipped endpoints even for nonunit input directions.
// Preconditions: finite origin/direction, nonzero direction, ordered limits;
// the upper limit may be +infinity for connections to distant illumination.
VISRTX_HOST_DEVICE Ray worldDistanceRay(const visrtx::Ray &input)
{
  Ray result;
  float length2 = 0.f;
  for (int c = 0; c < 3; ++c) {
    result.origin[c] = ref::promote(input.org[c]);
    const float d = ref::promote(input.dir[c]);
    length2 = ref::add(length2, ref::multiply(d, d));
  }
  const float length = sqrtf(length2);
  for (int c = 0; c < 3; ++c)
    result.direction[c] = ref::divide(ref::promote(input.dir[c]), length);
  result.lower = ref::multiply(ref::promote(input.t.lower), length);
  result.upper = ref::multiply(ref::promote(input.t.upper), length);
  return result;
}

struct ObjectRay
{
  float origin[3]{};
  float direction[3]{};
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
    float lo,
    float hi,
    Interval &result)
{
  if (!ref::nonemptyDomain(box))
    return false;
  for (int c = 0; c < 3; ++c) {
    const float a = ref::promote(box.lower[c]), b = ref::promote(box.upper[c]);
    if (local.direction[c] == 0.f) {
      if (local.origin[c] < a || local.origin[c] > b)
        return false;
    } else {
      float near =
          ref::divide(ref::subtract(a, local.origin[c]), local.direction[c]);
      float far =
          ref::divide(ref::subtract(b, local.origin[c]), local.direction[c]);
      if (far < near) {
        const float swap = near;
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
    float cursor,
    float &next)
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
    float &cursor,
    Interval &span)
{
  if (!scene.complete)
    return false;
  while (cursor < ray.upper) {
    float next = ray.upper;
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
    const float begin = cursor;
    cursor = next;
    if (active) {
      span = {begin, next};
      return true;
    }
  }
  return false;
}

VISRTX_HOST_DEVICE vec3 objectPosition(const ObjectRay &ray, float distance)
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
  const float lo = ref::promote(volume.valueRange.lower);
  const float hi = ref::promote(volume.valueRange.upper);
  const float normalized = ref::divide(
      ref::subtract(ref::promote(value), lo), ref::subtract(hi, lo));
  const float x = preview_math::coordinate(
      ref::subtract(ref::multiply(normalized, float(volume.colorCount)), .5f),
      0.f,
      float(volume.colorCount - 1));
  const uint32_t a = uint32_t(x);
  const uint32_t b = a + 1 < volume.colorCount ? a + 1 : a;
  const float weight = ref::narrow(ref::subtract(x, float(a)));
  vec4 result;
  for (int c = 0; c < 4; ++c)
    result[c] =
        preview_math::lerp(volume.colors[a][c], volume.colors[b][c], weight);
  return result;
}

} // namespace preview_volume
} // namespace visrtx
