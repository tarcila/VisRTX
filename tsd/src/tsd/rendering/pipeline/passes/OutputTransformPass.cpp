// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "OutputTransformPass.h"
// tsd_algorithms
#include "tsd/algorithms/cpu/outputTransform.hpp"
#ifdef TSD_ALGORITHMS_HAS_METAL
#include "tsd/algorithms/metal/outputTransform.hpp"
#endif
#ifdef TSD_ALGORITHMS_HAS_CUDA
#include "tsd/algorithms/cuda/outputTransform.hpp"
#endif
// std
#include <cmath>
#include <limits>

namespace tsd::rendering {

OutputTransformPass::OutputTransformPass() = default;

OutputTransformPass::~OutputTransformPass() = default;

void OutputTransformPass::setColorFormat(anari::DataType format)
{
  m_colorFormat = format;
}

void OutputTransformPass::setGamma(float gamma)
{
  m_gamma = std::max(gamma, 1e-6f);
}

void OutputTransformPass::render(ImageBuffers &b, int stageId)
{
  if (stageId == 0)
    return;

  const auto size = getDimensions();
  const uint32_t totalPixels = size.x * size.y;
  if (totalPixels == 0)
    return;

  const float invGamma = 1.f / m_gamma;

#ifdef TSD_ALGORITHMS_HAS_METAL
  if (b.metalHdrColor) {
    tsd::algorithms::metal::outputTransform(b.metalHdrColor,
        b.metalHdrColor,
        b.metalHdrColor,
        totalPixels,
        invGamma,
        static_cast<uint32_t>(m_colorFormat));
    return;
  }
#endif
  if (!b.color || m_colorFormat == ANARI_UFIXED8_RGBA_SRGB)
    return;
#ifdef TSD_ALGORITHMS_HAS_CUDA
  if (b.stream) {
    tsd::algorithms::cuda::outputTransform(b.stream,
        b.hdrColor,
        b.color,
        b.color,
        totalPixels,
        invGamma,
        m_colorFormat);
    return;
  }
#endif
  tsd::algorithms::cpu::outputTransform(
      b.hdrColor, b.color, b.color, totalPixels, invGamma, m_colorFormat);
}

} // namespace tsd::rendering
