// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/algorithms/metal/convertColorBuffer.hpp"
#include "MetalContext.h"

namespace tsd::algorithms::metal {

void convertFloatToUint8(MTL::CommandBuffer *cmdBuf,
    MTL::Texture *input,
    MTL::Buffer *output,
    size_t totalSize)
{
  auto &ctx = MetalContext::instance();
  auto *pso = ctx.pipelineState("convertFloatToUint8Kernel");

  auto *encoder = cmdBuf->computeCommandEncoder();
  encoder->setComputePipelineState(pso);
  encoder->setTexture(input, 0);
  encoder->setBuffer(output, 0, 0);

  auto width = (uint32_t)input->width();
  auto height = (uint32_t)input->height();
  auto tgWidth = pso->threadExecutionWidth();
  auto tgHeight = pso->maxTotalThreadsPerThreadgroup() / tgWidth;
  encoder->dispatchThreads(
      {width, height, 1}, {tgWidth, (NS::UInteger)tgHeight, 1});
  encoder->endEncoding();
}

void convertFloatToUint8(
    MTL::Texture *input, MTL::Buffer *output, size_t totalSize)
{
  auto &ctx = MetalContext::instance();
  auto *cmdBuf = ctx.defaultQueue()->commandBuffer();
  convertFloatToUint8(cmdBuf, input, output, totalSize);
  cmdBuf->commit();
  cmdBuf->waitUntilCompleted();
}

void convertFloatToBGRA8(
    MTL::CommandBuffer *cmdBuf, MTL::Texture *input, MTL::Texture *output)
{
  auto &ctx = MetalContext::instance();
  auto *pso = ctx.pipelineState("convertFloatToBGRA8Kernel");

  auto *encoder = cmdBuf->computeCommandEncoder();
  encoder->setComputePipelineState(pso);
  encoder->setTexture(input, 0);
  encoder->setTexture(output, 1);

  auto width = (uint32_t)input->width();
  auto height = (uint32_t)input->height();
  auto tgWidth = pso->threadExecutionWidth();
  auto tgHeight = pso->maxTotalThreadsPerThreadgroup() / tgWidth;
  encoder->dispatchThreads(
      {width, height, 1}, {tgWidth, (NS::UInteger)tgHeight, 1});
  encoder->endEncoding();
}

void convertFloatToBGRA8(MTL::Texture *input, MTL::Texture *output)
{
  auto &ctx = MetalContext::instance();
  auto *cmdBuf = ctx.defaultQueue()->commandBuffer();
  convertFloatToBGRA8(cmdBuf, input, output);
  cmdBuf->commit();
  cmdBuf->waitUntilCompleted();
}

} // namespace tsd::algorithms::metal
