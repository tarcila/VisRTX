// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/algorithms/metal/toneMap.hpp"
#include "MetalContext.h"

namespace tsd::algorithms::metal {

void toneMap(MTL::CommandBuffer *cmdBuf,
    MTL::Texture *hdrColor,
    uint32_t numPixels,
    float exposureScale,
    ToneMapOperator op)
{
  auto &ctx = MetalContext::instance();
  auto *pso = ctx.pipelineState("toneMapKernel");

  auto *encoder = cmdBuf->computeCommandEncoder();
  encoder->setComputePipelineState(pso);
  encoder->setTexture(hdrColor, 0);
  encoder->setBytes(&exposureScale, sizeof(exposureScale), 0);
  auto opVal = static_cast<uint32_t>(op);
  encoder->setBytes(&opVal, sizeof(opVal), 1);

  auto width = (uint32_t)hdrColor->width();
  auto height = (uint32_t)hdrColor->height();
  auto tgWidth = pso->threadExecutionWidth();
  auto tgHeight = pso->maxTotalThreadsPerThreadgroup() / tgWidth;
  encoder->dispatchThreads(
      {width, height, 1}, {tgWidth, (NS::UInteger)tgHeight, 1});
  encoder->endEncoding();
}

void toneMap(MTL::Texture *hdrColor,
    uint32_t numPixels,
    float exposureScale,
    ToneMapOperator op)
{
  auto &ctx = MetalContext::instance();
  auto *cmdBuf = ctx.defaultQueue()->commandBuffer();
  toneMap(cmdBuf, hdrColor, numPixels, exposureScale, op);
  cmdBuf->commit();
  cmdBuf->waitUntilCompleted();
}

} // namespace tsd::algorithms::metal
