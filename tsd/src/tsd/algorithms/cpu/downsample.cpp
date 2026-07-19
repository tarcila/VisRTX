// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/algorithms/cpu/downsample.hpp"
#include "../math/color.h"
#include "detail/parallel_reduce.h"
// std
#include <algorithm>
#include <cmath>

namespace tsd::algorithms::cpu {

static constexpr float MIN_LUMINANCE = 1e-4f;

// Host sample budget. Cost is resolution-independent: a full-image reduction
// scans every pixel through a transcendental log2 each frame (~23 ms at 4K on
// a serial build), while striding to this many samples estimates the mean to
// within ~0.02 stops — invisible after the exp2 / clamp / temporal smoothing
// the auto-exposure pass applies. Images at or below the budget read fully.
static constexpr uint32_t SAMPLE_COUNT = 16384;

float meanLogLuminance(
    const float *hdrColor, uint32_t width, uint32_t height)
{
  const uint32_t total = width * height;
  if (total == 0)
    return 0.f;

  const uint32_t stride = std::max(1u, total / SAMPLE_COUNT);
  const uint32_t numSamples = (total + stride - 1) / stride;
  const float minLum = MIN_LUMINANCE;
  const float sum = detail::parallel_reduce(
      0u,
      numSamples,
      0.f,
      [=](uint32_t j) -> float {
        const uint32_t idx = j * stride * 4;
        const float lum = std::max(
            math::luminance(
                hdrColor[idx + 0], hdrColor[idx + 1], hdrColor[idx + 2]),
            minLum);
        return std::log2(lum);
      },
      [](float a, float b) -> float { return a + b; });
  return sum / float(numSamples);
}

} // namespace tsd::algorithms::cpu
