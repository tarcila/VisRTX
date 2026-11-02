// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <array>
#include "gpu/blackbody/blackbody.h"
#include "gpu/blackbody/previewCold.h"
#include "utility/ScopedReferenceRounding.h"

namespace visrtx {
// Derived at runtime from the attributed CIE spectral data already distributed
// with VisRTX. These are not un-attributed replacement numeric tables.
inline const std::array<PreviewColdKnot, PREVIEW_COLD_KNOTS> &previewColdTable()
{
  static const auto table = [] {
    const ScopedReferenceRounding rounding;
    std::array<PreviewColdKnot, PREVIEW_COLD_KNOTS> result{};
    for (int k = 0; k < PREVIEW_COLD_KNOTS; ++k) {
      const double t =
          100. * std::exp(std::log(3.) * k / (PREVIEW_COLD_KNOTS - 1));
      double xyz[3]{}, derivative[3]{};
      for (int i = 0; i < 1881; ++i) {
        const double x = double(blackbody_data::spectral[i][0]) / t;
        const double denominator = -std::expm1(-x);
        const double value = std::exp(-x) / denominator;
        for (int c = 0; c < 3; ++c) {
          const double term =
              double(blackbody_data::spectral[i][c + 1]) * value;
          xyz[c] += term;
          derivative[c] += term * x / denominator;
        }
      }
      for (int c = 0; c < 3; ++c) {
        result[k].logXYZ[c] = float(std::log(xyz[c]));
        result[k].derivative[c] = float(derivative[c] / xyz[c]);
      }
    }
    return result;
  }();
  return table;
}
} // namespace visrtx
