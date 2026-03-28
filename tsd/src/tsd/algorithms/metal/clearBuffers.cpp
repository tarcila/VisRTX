// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/algorithms/metal/clearBuffers.hpp"
#include "MetalContext.h"

namespace tsd::algorithms::metal {

static void dispatchFill(MTL::CommandBuffer *cmdBuf,
    const char *kernelName,
    MTL::Buffer *buf,
    uint32_t count,
    const void *value,
    size_t valueSize)
{
  auto &ctx = MetalContext::instance();
  auto *pso = ctx.pipelineState(kernelName);

  auto *encoder = cmdBuf->computeCommandEncoder();
  encoder->setComputePipelineState(pso);
  encoder->setBuffer(buf, 0, 0);
  encoder->setBytes(&count, sizeof(count), 1);
  encoder->setBytes(value, valueSize, 2);

  auto tgSize = pso->maxTotalThreadsPerThreadgroup();
  if (tgSize > count)
    tgSize = count;
  encoder->dispatchThreads({count, 1, 1}, {(NS::UInteger)tgSize, 1, 1});
  encoder->endEncoding();
}

void fill(MTL::CommandBuffer *cmdBuf,
    MTL::Buffer *buf,
    uint32_t count,
    uint32_t value)
{
  dispatchFill(cmdBuf, "fillUint32", buf, count, &value, sizeof(value));
}

void fill(MTL::Buffer *buf, uint32_t count, uint32_t value)
{
  auto &ctx = MetalContext::instance();
  auto *cmdBuf = ctx.defaultQueue()->commandBuffer();
  fill(cmdBuf, buf, count, value);
  cmdBuf->commit();
  cmdBuf->waitUntilCompleted();
}

void fill(
    MTL::CommandBuffer *cmdBuf, MTL::Buffer *buf, uint32_t count, float value)
{
  dispatchFill(cmdBuf, "fillFloat", buf, count, &value, sizeof(value));
}

void fill(MTL::Buffer *buf, uint32_t count, float value)
{
  auto &ctx = MetalContext::instance();
  auto *cmdBuf = ctx.defaultQueue()->commandBuffer();
  fill(cmdBuf, buf, count, value);
  cmdBuf->commit();
  cmdBuf->waitUntilCompleted();
}

} // namespace tsd::algorithms::metal
