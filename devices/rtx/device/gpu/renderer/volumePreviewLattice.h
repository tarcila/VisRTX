// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "gpu/aggregateVolume.h"

namespace visrtx {
namespace volume_preview {
VISRTX_HOST_DEVICE double nextCell(const AggregateVolumeSceneGPUData &scene,
    const aggregate_volume::Ray &ray,
    aggregate_volume::Interval span,
    double cursor,
    double rate)
{
  double next = span.upper;
  for (size_t i = 0; i < scene.count; ++i) {
    const auto &o = scene.occurrences[i];
    aggregate_volume::Interval clipped;
    if (!aggregate_volume::intersect(o, ray, clipped)
        || !(clipped.lower <= span.lower && span.lower < clipped.upper))
      continue;
    const auto local = aggregate_volume::objectRay(o, ray);
    aggregate_volume::Interval full;
    aggregate_volume::intersectDomain(
        o.volume.domain, local, -INFINITY, INFINITY, full);
    // An inverse affine transform can have very large finite entries even
    // when the authored float transform is finite and nonsingular. Avoid
    // squaring those entries (or forming rate*speed) before division: that
    // overflow would coarsen a perfectly representable lattice to one cell.
    const double scale = fmax(fabs(local.direction[0]),
        fmax(fabs(local.direction[1]), fabs(local.direction[2])));
    double length2 = 0.;
    for (int c = 0; c < 3; ++c) {
      const double d = optical_reference::divide(local.direction[c], scale);
      length2 =
          optical_reference::add(length2, optical_reference::multiply(d, d));
    }
    const double step = optical_reference::divide(
        optical_reference::divide(
            optical_reference::divide(o.volume.stepSize, rate), scale),
        sqrt(length2));
    const double index = floor((cursor - full.lower) / step);
    double boundary = optical_reference::add(
        full.lower, optical_reference::multiply(index + 1., step));
    // A represented tie belongs to the current cell, not a zero-length step.
    if (!(boundary > cursor))
      boundary = optical_reference::add(
          full.lower, optical_reference::multiply(index + 2., step));
    if (boundary > cursor && boundary < next)
      next = boundary;
  }
  // Always integrate the remaining represented span. At extreme magnitudes a
  // finer lattice may be unresolvable; this is quadrature precision loss, not
  // an epsilon jump, discarded source interval or a hidden iteration budget.
  return next;
}

} // namespace volume_preview
} // namespace visrtx
