// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <array>
#include "gpu/blackbody/blackbody.h"
#include "gpu/blackbody/previewCold.h"
#include "utility/ScopedReferenceRounding.h"

namespace visrtx {
// Runtime-derived adaptation of the separately attributed CIE numeric data.
// Log-sum evaluation keeps the pilot's tiny XYZ components representable.
inline const std::array<PreviewColdKnot, 256> &qualityColdTable()
{
  static const auto table = [] {
    const ScopedReferenceRounding rounding;
    std::array<PreviewColdKnot, 256> result{};
    for (int k = 0; k < 256; ++k) {
      const double t = 25. * std::exp(std::log(12.) * k / 255.);
      for (int c = 0; c < 3; ++c) {
        double maximum = -INFINITY;
        for (int i = 0; i < 1881; ++i) {
          const double weight = blackbody_data::spectral[i][c + 1];
          if (weight == 0.)
            continue;
          const double x = double(blackbody_data::spectral[i][0]) / t;
          maximum = std::max(
              maximum, std::log(weight) - x - std::log(-std::expm1(-x)));
        }
        double sum = 0., derivative = 0.;
        for (int i = 0; i < 1881; ++i) {
          const double weight = blackbody_data::spectral[i][c + 1];
          if (weight == 0.)
            continue;
          const double x = double(blackbody_data::spectral[i][0]) / t,
                       denominator = -std::expm1(-x);
          const double value =
              std::exp(std::log(weight) - x - std::log(denominator) - maximum);
          sum += value;
          derivative += value * x / denominator;
        }
        result[k].logXYZ[c] = float(maximum + std::log(sum));
        result[k].derivative[c] = float(derivative / sum);
      }
    }
    return result;
  }();
  return table;
}
} // namespace visrtx
