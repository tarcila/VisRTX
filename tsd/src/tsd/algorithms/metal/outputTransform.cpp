// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/algorithms/metal/outputTransform.hpp"
#include "MetalContext.h"

namespace tsd::algorithms::metal {

void outputTransform(MTL::CommandBuffer *cmdBuf,
    MTL::Texture *hdrColor,
    MTL::Texture *colorIn,
    MTL::Texture *colorOut,
    uint32_t totalPixels,
    float invGamma,
    uint32_t colorFormat)
{
  auto &ctx = MetalContext::instance();
  auto *pso = ctx.pipelineState("outputTransformKernel");

  auto *encoder = cmdBuf->computeCommandEncoder();
  encoder->setComputePipelineState(pso);
  encoder->setTexture(hdrColor, 0);
  encoder->setTexture(colorIn, 1);
  encoder->setTexture(colorOut, 2);
  encoder->setBytes(&invGamma, sizeof(invGamma), 0);
  encoder->setBytes(&colorFormat, sizeof(colorFormat), 1);

  auto width = (uint32_t)colorOut->width();
  auto height = (uint32_t)colorOut->height();
  auto tgWidth = pso->threadExecutionWidth();
  auto tgHeight = pso->maxTotalThreadsPerThreadgroup() / tgWidth;
  encoder->dispatchThreads(
      {width, height, 1}, {tgWidth, (NS::UInteger)tgHeight, 1});
  encoder->endEncoding();
}

void outputTransform(MTL::Texture *hdrColor,
    MTL::Texture *colorIn,
    MTL::Texture *colorOut,
    uint32_t totalPixels,
    float invGamma,
    uint32_t colorFormat)
{
  auto &ctx = MetalContext::instance();
  auto *cmdBuf = ctx.defaultQueue()->commandBuffer();
  outputTransform(
      cmdBuf, hdrColor, colorIn, colorOut, totalPixels, invGamma, colorFormat);
  cmdBuf->commit();
  cmdBuf->waitUntilCompleted();
}

} // namespace tsd::algorithms::metal
