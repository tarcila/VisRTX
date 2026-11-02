// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cmath>
#include <limits>
#include "gpu/aggregateVolumeData.h"

namespace visrtx {
// Preparation only; bounds are a directional proposal, not an optical majorant.
// A full-sphere mixture in the sampler preserves support independently of cone
// tightness. Unsupported proposal coordinates keep the WHOLE scene
// continuation- owned; the reference optical scene itself remains published and
// unchanged.
inline bool prepareVolumeSource(
    VolumeOccurrenceGPUData &occurrence, const glm::dmat4 &objectToWorld)
{
  const auto &box = occurrence.volume.domain;
  for (int row = 0; row < 3; ++row)
    for (int col = 0; col < 4; ++col)
      occurrence.sourceObjectToWorld[row][col] = objectToWorld[col][row];
  double objectVolume = 1.;
  for (int c = 0; c < 3; ++c)
    objectVolume *= double(box.upper[c]) - double(box.lower[c]);
  const double worldVolume =
      objectVolume * std::abs(glm::determinant(glm::dmat3(objectToWorld)));
  occurrence.sourceWorldVolume =
      std::isfinite(worldVolume) && worldVolume > 0. ? worldVolume : 0.;
  glm::dvec3 lower(std::numeric_limits<double>::infinity()),
      upper(-std::numeric_limits<double>::infinity());
  for (int corner = 0; corner < 8; ++corner)
    for (int row = 0; row < 3; ++row) {
      double lo = objectToWorld[3][row], hi = lo;
      for (int c = 0; c < 3; ++c) {
        const double x =
            (corner & (1 << c)) ? double(box.upper[c]) : double(box.lower[c]);
        const double p = optical_reference::multiply(objectToWorld[c][row], x);
        lo = std::nextafter(
            optical_reference::add(lo, std::nextafter(p, -INFINITY)),
            -INFINITY);
        hi = std::nextafter(
            optical_reference::add(hi, std::nextafter(p, INFINITY)), INFINITY);
      }
      lower[row] = std::min(lower[row], lo);
      upper[row] = std::max(upper[row], hi);
    }
  double radius2 = 0.;
  for (int c = 0; c < 3; ++c) {
    occurrence.sourceCenter[c] = lower[c] * .5 + upper[c] * .5;
    const double extent =
        std::nextafter(std::max(upper[c] - occurrence.sourceCenter[c],
                           occurrence.sourceCenter[c] - lower[c]),
            INFINITY);
    radius2 = optical_reference::addUpper(
        radius2, optical_reference::multiplyUpper(extent, extent));
  }
  occurrence.sourceRadius = std::nextafter(std::sqrt(radius2), INFINITY);
  const double limit = double(std::numeric_limits<float>::max()) / 8.;
  bool ready =
      std::isfinite(occurrence.sourceRadius) && occurrence.sourceRadius > 0.;
  for (int c = 0; c < 3; ++c)
    ready &= std::isfinite(occurrence.sourceCenter[c])
        && std::abs(occurrence.sourceCenter[c]) + occurrence.sourceRadius
            < limit;
  return ready;
}
} // namespace visrtx
