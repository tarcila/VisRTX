// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "gpu/previewVolumeTraversal.h"

namespace visrtx {
namespace practical_preview {
// The common single-occurrence path retains fixed ray invariants. Arbitrary
// overlap counts keep the uncapped merged-lattice path below, not a hit cap.
struct PreparedRay
{
  preview_volume::ObjectRay local;
  preview_volume::Interval full, clipped;
  float step{0.f};
  bool single{false}, hit{false};
};
VISRTX_HOST_DEVICE PreparedRay prepareRay(
    const AggregateVolumeSceneGPUData &scene,
    const preview_volume::Ray &ray,
    float rate)
{
  PreparedRay result;
  result.single = scene.count == 1;
  if (!result.single)
    return result;
  const auto &o = scene.occurrences[0];
  result.local = preview_volume::objectRay(o, ray);
  result.hit = preview_volume::intersectDomain(
      o.volume.domain, result.local, ray.lower, ray.upper, result.clipped);
  if (!result.hit)
    return result;
  preview_volume::intersectDomain(
      o.volume.domain, result.local, -INFINITY, INFINITY, result.full);
  const auto &d = result.local.direction;
  const float scale = fmaxf(fabsf(d[0]), fmaxf(fabsf(d[1]), fabsf(d[2])));
  float length2 = 0.f;
  for (int c = 0; c < 3; ++c) {
    const float v = d[c] / scale;
    length2 += v * v;
  }
  result.step = ((float(o.volume.stepSize) / rate) / scale) / sqrtf(length2);
  return result;
}
VISRTX_HOST_DEVICE float nextPreparedCell(
    const PreparedRay &prepared, preview_volume::Interval span, float cursor)
{
  const float index = floorf((cursor - prepared.full.lower) / prepared.step);
  float boundary = prepared.full.lower + (index + 1.f) * prepared.step;
  if (!(boundary > cursor))
    boundary = prepared.full.lower + (index + 2.f) * prepared.step;
  return boundary > cursor && boundary < span.upper ? boundary : span.upper;
}
VISRTX_HOST_DEVICE float nextCell(const AggregateVolumeSceneGPUData &scene,
    const preview_volume::Ray &ray,
    preview_volume::Interval span,
    float cursor,
    float rate)
{
  float next = span.upper;
  for (size_t i = 0; i < scene.count; ++i) {
    const auto &o = scene.occurrences[i];
    preview_volume::Interval clipped;
    if (!preview_volume::intersect(o, ray, clipped)
        || !(clipped.lower <= span.lower && span.lower < clipped.upper))
      continue;
    const auto local = preview_volume::objectRay(o, ray);
    preview_volume::Interval full;
    preview_volume::intersectDomain(
        o.volume.domain, local, -INFINITY, INFINITY, full);
    // An inverse affine transform can have very large finite entries even
    // when the authored float transform is finite and nonsingular. Avoid
    // squaring those entries (or forming rate*speed) before division: that
    // overflow would coarsen a perfectly representable lattice to one cell.
    const float scale = fmaxf(fabsf(local.direction[0]),
        fmaxf(fabsf(local.direction[1]), fabsf(local.direction[2])));
    float length2 = 0.f;
    for (int c = 0; c < 3; ++c) {
      const float d = preview_math::divide(local.direction[c], scale);
      length2 = preview_math::add(length2, preview_math::multiply(d, d));
    }
    const float step = preview_math::divide(
        preview_math::divide(
            preview_math::divide(o.volume.stepSize, rate), scale),
        sqrtf(length2));
    const float index = floorf((cursor - full.lower) / step);
    float boundary = preview_math::add(
        full.lower, preview_math::multiply(index + 1.f, step));
    // A represented tie belongs to the current cell, not a zero-length step.
    if (!(boundary > cursor))
      boundary = preview_math::add(
          full.lower, preview_math::multiply(index + 2.f, step));
    if (boundary > cursor && boundary < next)
      next = boundary;
  }
  // Always integrate the remaining represented span. At extreme magnitudes a
  // finer lattice may be unresolvable; this is quadrature precision loss, not
  // an epsilon jump, discarded source interval or a hidden iteration budget.
  return next;
}

} // namespace practical_preview
} // namespace visrtx
