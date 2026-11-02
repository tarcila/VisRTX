// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "aggregateVolumeData.h"

namespace visrtx {
namespace volume_source {
VISRTX_HOST_DEVICE bool hasImportance(const VolumeOccurrenceGPUData &o)
{
  // Optional-guide arithmetic envelope: float-world receiver distances
  // squared and <=64^3 cells keep the upper range of qV*r^2 representable.
  // Outside it, retain the complete angular-distance proposal unchanged.
  return o.volume.sourceImportance.cdf
      && o.volume.sourceImportance.resolution > 0
      && o.sourceWorldVolume >= 0x1p-700 && o.sourceWorldVolume <= 0x1p700;
}
VISRTX_HOST_DEVICE uint32_t importanceCell(
    const SourceImportanceGPUData &grid, double u)
{
  uint32_t lo = 0, hi = grid.resolution * grid.resolution * grid.resolution;
  while (lo + 1 < hi) {
    const uint32_t mid = lo + (hi - lo) / 2;
    if (u < grid.cdf[mid])
      hi = mid;
    else
      lo = mid;
  }
  return lo;
}
VISRTX_HOST_DEVICE double importanceDensity(
    const VolumeOccurrenceGPUData &o, const glm::dvec3 &point)
{
  const auto &grid = o.volume.sourceImportance;
  const uint32_t n = grid.resolution;
  uint32_t xyz[3];
  for (int c = 0; c < 3; ++c) {
    const double lo = optical_reference::promote(o.volume.domain.lower[c]);
    const double extent = optical_reference::subtract(
        optical_reference::promote(o.volume.domain.upper[c]), lo);
    const double x = optical_reference::divide(
        optical_reference::multiply(
            optical_reference::subtract(point[c], lo), double(n)),
        extent);
    if (x < 0. || x > double(n))
      return 0.;
    xyz[c] = x >= double(n) ? n - 1 : uint32_t(x);
  }
  const uint32_t cell = (xyz[2] * n + xyz[1]) * n + xyz[0];
  const double mass =
      optical_reference::subtract(grid.cdf[cell + 1], grid.cdf[cell]);
  return optical_reference::divide(
      optical_reference::multiply(mass, double(n) * n * n),
      o.sourceWorldVolume);
}
} // namespace volume_source
} // namespace visrtx
