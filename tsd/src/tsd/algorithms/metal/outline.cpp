// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/algorithms/metal/outline.hpp"
#include "MetalContext.h"

namespace tsd::algorithms::metal {

void outline(MTL::CommandBuffer *cmdBuf,
    MTL::Texture *objectId,
    MTL::Texture *color,
    uint32_t outlineId,
    uint32_t w,
    uint32_t h)
{
  auto &ctx = MetalContext::instance();
  auto *pso = ctx.pipelineState("outlineKernel");

  auto *encoder = cmdBuf->computeCommandEncoder();
  encoder->setComputePipelineState(pso);
  encoder->setTexture(objectId, 0);
  encoder->setTexture(color, 1);
  encoder->setBytes(&outlineId, sizeof(outlineId), 0);

  auto tgWidth = pso->threadExecutionWidth();
  auto tgHeight = pso->maxTotalThreadsPerThreadgroup() / tgWidth;
  encoder->dispatchThreads({w, h, 1}, {tgWidth, (NS::UInteger)tgHeight, 1});
  encoder->endEncoding();
}

void outline(MTL::Texture *objectId,
    MTL::Texture *color,
    uint32_t outlineId,
    uint32_t w,
    uint32_t h)
{
  auto &ctx = MetalContext::instance();
  auto *cmdBuf = ctx.defaultQueue()->commandBuffer();
  outline(cmdBuf, objectId, color, outlineId, w, h);
  cmdBuf->commit();
  cmdBuf->waitUntilCompleted();
}

} // namespace tsd::algorithms::metal
