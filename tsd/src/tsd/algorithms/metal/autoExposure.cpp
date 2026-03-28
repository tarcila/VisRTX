// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/algorithms/metal/autoExposure.hpp"
#include "MetalContext.h"

namespace tsd::algorithms::metal {

float sumLogLuminance(MTL::CommandBuffer *cmdBuf,
    MTL::Texture *hdrColor,
    uint32_t numSamples,
    uint32_t stride)
{
  auto &ctx = MetalContext::instance();
  auto *device = ctx.device();

  // Pass 1: partial sums per threadgroup
  auto *pso1 = ctx.pipelineState("sumLogLuminancePass1");
  auto tgSize1 = (uint32_t)pso1->maxTotalThreadsPerThreadgroup();
  auto numGroups = (numSamples + tgSize1 - 1) / tgSize1;

  auto *partials = device->newBuffer(
      numGroups * sizeof(float), MTL::ResourceStorageModeShared);
  auto *result =
      device->newBuffer(sizeof(float), MTL::ResourceStorageModeShared);

  auto texWidth = (uint32_t)hdrColor->width();

  auto *encoder = cmdBuf->computeCommandEncoder();
  encoder->setComputePipelineState(pso1);
  encoder->setTexture(hdrColor, 0);
  encoder->setBuffer(partials, 0, 0);
  encoder->setBytes(&numSamples, sizeof(numSamples), 1);
  encoder->setBytes(&stride, sizeof(stride), 2);
  encoder->setBytes(&texWidth, sizeof(texWidth), 3);
  encoder->setThreadgroupMemoryLength(tgSize1 * sizeof(float), 0);
  encoder->dispatchThreads(
      {(NS::UInteger)numSamples, 1, 1}, {(NS::UInteger)tgSize1, 1, 1});
  encoder->endEncoding();

  // Pass 2: reduce partials into a single value
  auto *pso2 = ctx.pipelineState("sumLogLuminancePass2");
  auto tgSize2 = (uint32_t)pso2->maxTotalThreadsPerThreadgroup();
  // Round up to next power of two for the reduction
  uint32_t reductionSize = 1;
  while (reductionSize < numGroups)
    reductionSize <<= 1;
  if (reductionSize > tgSize2)
    reductionSize = tgSize2;

  auto *encoder2 = cmdBuf->computeCommandEncoder();
  encoder2->setComputePipelineState(pso2);
  encoder2->setBuffer(partials, 0, 0);
  encoder2->setBuffer(result, 0, 1);
  encoder2->setBytes(&numGroups, sizeof(numGroups), 2);
  encoder2->setThreadgroupMemoryLength(reductionSize * sizeof(float), 0);
  encoder2->dispatchThreads(
      {(NS::UInteger)reductionSize, 1, 1}, {(NS::UInteger)reductionSize, 1, 1});
  encoder2->endEncoding();

  cmdBuf->commit();
  cmdBuf->waitUntilCompleted();

  float sum = *static_cast<float *>(result->contents());

  partials->release();
  result->release();

  return sum;
}

float sumLogLuminance(
    MTL::Texture *hdrColor, uint32_t numSamples, uint32_t stride)
{
  auto &ctx = MetalContext::instance();
  auto *cmdBuf = ctx.defaultQueue()->commandBuffer();
  return sumLogLuminance(cmdBuf, hdrColor, numSamples, stride);
}

} // namespace tsd::algorithms::metal
