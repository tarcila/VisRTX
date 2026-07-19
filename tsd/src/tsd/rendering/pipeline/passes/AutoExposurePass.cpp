// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "AutoExposurePass.h"
// tsd_algorithms
#include "tsd/algorithms/cpu/downsample.hpp"
#ifdef TSD_ALGORITHMS_HAS_CUDA
#include "tsd/algorithms/cuda/downsample.hpp"
#endif
// std
#include <algorithm>
#include <cmath>

namespace tsd::rendering {

namespace {

constexpr float MIN_EXPOSURE = -20.f;
constexpr float MAX_EXPOSURE = 20.f;
constexpr float MID_GRAY = 0.18f;

} // namespace

AutoExposurePass::AutoExposurePass() = default;

AutoExposurePass::~AutoExposurePass() = default;

void AutoExposurePass::setHDREnabled(bool enabled)
{
  if (enabled && !m_hdrEnabled)
    m_hasExposure = false;
  m_hdrEnabled = enabled;
}

float AutoExposurePass::currentExposure() const
{
  return m_currentExposure;
}

void AutoExposurePass::render(ImageBuffers &b, int stageId)
{
  if (stageId == 0 || !m_hdrEnabled)
    return;

  const auto size = getDimensions();
  const uint32_t totalPixels = size.x * size.y;
  if (totalPixels == 0 || !b.hdrColor)
    return;

  // Mean log-luminance via the same API on either backend: the CUDA path is
  // an exact full-image SPD reduction, the host path a strided estimate
  // (bounded per-frame cost; ~0.02-stop accuracy).
  float meanLogLum = 0.f;
#ifdef TSD_ALGORITHMS_HAS_CUDA
  if (b.stream) {
    meanLogLum = tsd::algorithms::cuda::meanLogLuminance(
        b.stream, b.hdrColor, size.x, size.y);
  } else
#endif
  {
    meanLogLum =
        tsd::algorithms::cpu::meanLogLuminance(b.hdrColor, size.x, size.y);
  }

  const float avgLum = std::exp2(meanLogLum);
  const float targetExposure =
      std::clamp(std::log2(MID_GRAY / avgLum), MIN_EXPOSURE, MAX_EXPOSURE);

  if (!m_hasExposure) {
    m_currentExposure = targetExposure;
    m_hasExposure = true;
  } else {
    m_currentExposure += (targetExposure - m_currentExposure) * m_response;
  }

  b.exposure = m_currentExposure;
}

} // namespace tsd::rendering
