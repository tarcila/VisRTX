// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/algorithms/metal/visualizeAOV.hpp"
#include "MetalContext.h"

namespace tsd::algorithms::metal {

// 2D dispatch helper
static void dispatch2D(MTL::ComputeCommandEncoder *encoder,
    MTL::ComputePipelineState *pso,
    uint32_t w,
    uint32_t h)
{
  auto tgWidth = pso->threadExecutionWidth();
  auto tgHeight = pso->maxTotalThreadsPerThreadgroup() / tgWidth;
  encoder->dispatchThreads({w, h, 1}, {tgWidth, (NS::UInteger)tgHeight, 1});
}

// Convenience: cmdBuf-less overload pattern
#define METAL_DEFAULT_OVERLOAD(func, ...)                                      \
  {                                                                            \
    auto &ctx = MetalContext::instance();                                      \
    auto *cmdBuf = ctx.defaultQueue()->commandBuffer();                        \
    func(cmdBuf, __VA_ARGS__);                                                 \
    cmdBuf->commit();                                                          \
    cmdBuf->waitUntilCompleted();                                              \
  }

// ID visualizations (objectId, primitiveId, instanceId) ///////////////////////

static void dispatchIdVisualization(MTL::CommandBuffer *cmdBuf,
    const char *kernelName,
    MTL::Texture *idTexture,
    MTL::Texture *color,
    uint32_t w,
    uint32_t h)
{
  auto &ctx = MetalContext::instance();
  auto *pso = ctx.pipelineState(kernelName);

  auto *encoder = cmdBuf->computeCommandEncoder();
  encoder->setComputePipelineState(pso);
  encoder->setTexture(idTexture, 0);
  encoder->setTexture(color, 1);
  dispatch2D(encoder, pso, w, h);
  encoder->endEncoding();
}

void visualizeObjectId(MTL::CommandBuffer *cmdBuf,
    MTL::Texture *objectId,
    MTL::Texture *color,
    uint32_t w,
    uint32_t h)
{
  dispatchIdVisualization(
      cmdBuf, "visualizeObjectIdKernel", objectId, color, w, h);
}

void visualizeObjectId(
    MTL::Texture *objectId, MTL::Texture *color, uint32_t w, uint32_t h)
    METAL_DEFAULT_OVERLOAD(visualizeObjectId, objectId, color, w, h)

        void visualizePrimitiveId(MTL::CommandBuffer *cmdBuf,
            MTL::Texture *primitiveId,
            MTL::Texture *color,
            uint32_t w,
            uint32_t h)
{
  dispatchIdVisualization(
      cmdBuf, "visualizePrimitiveIdKernel", primitiveId, color, w, h);
}

void visualizePrimitiveId(
    MTL::Texture *primitiveId, MTL::Texture *color, uint32_t w, uint32_t h)
    METAL_DEFAULT_OVERLOAD(visualizePrimitiveId, primitiveId, color, w, h)

        void visualizeInstanceId(MTL::CommandBuffer *cmdBuf,
            MTL::Texture *instanceId,
            MTL::Texture *color,
            uint32_t w,
            uint32_t h)
{
  dispatchIdVisualization(
      cmdBuf, "visualizeInstanceIdKernel", instanceId, color, w, h);
}

void visualizeInstanceId(
    MTL::Texture *instanceId, MTL::Texture *color, uint32_t w, uint32_t h)
    METAL_DEFAULT_OVERLOAD(visualizeInstanceId, instanceId, color, w, h)

    // Depth
    // ///////////////////////////////////////////////////////////////////////

    void visualizeDepth(MTL::CommandBuffer *cmdBuf,
        MTL::Texture *depth,
        MTL::Texture *color,
        float minDepth,
        float maxDepth,
        uint32_t w,
        uint32_t h)
{
  auto &ctx = MetalContext::instance();
  auto *pso = ctx.pipelineState("visualizeDepthKernel");

  auto *encoder = cmdBuf->computeCommandEncoder();
  encoder->setComputePipelineState(pso);
  encoder->setTexture(depth, 0);
  encoder->setTexture(color, 1);
  encoder->setBytes(&minDepth, sizeof(minDepth), 0);
  encoder->setBytes(&maxDepth, sizeof(maxDepth), 1);
  dispatch2D(encoder, pso, w, h);
  encoder->endEncoding();
}

void visualizeDepth(MTL::Texture *depth,
    MTL::Texture *color,
    float minDepth,
    float maxDepth,
    uint32_t w,
    uint32_t h)
    METAL_DEFAULT_OVERLOAD(
        visualizeDepth, depth, color, minDepth, maxDepth, w, h)

    // Albedo
    // //////////////////////////////////////////////////////////////////////

    void visualizeAlbedo(MTL::CommandBuffer *cmdBuf,
        MTL::Texture *albedo,
        MTL::Texture *color,
        uint32_t w,
        uint32_t h)
{
  auto &ctx = MetalContext::instance();
  auto *pso = ctx.pipelineState("visualizeAlbedoKernel");

  auto *encoder = cmdBuf->computeCommandEncoder();
  encoder->setComputePipelineState(pso);
  encoder->setTexture(albedo, 0);
  encoder->setTexture(color, 1);
  dispatch2D(encoder, pso, w, h);
  encoder->endEncoding();
}

void visualizeAlbedo(
    MTL::Texture *albedo, MTL::Texture *color, uint32_t w, uint32_t h)
    METAL_DEFAULT_OVERLOAD(visualizeAlbedo, albedo, color, w, h)

    // Normal
    // //////////////////////////////////////////////////////////////////////

    void visualizeNormal(MTL::CommandBuffer *cmdBuf,
        MTL::Texture *normal,
        MTL::Texture *color,
        uint32_t w,
        uint32_t h)
{
  auto &ctx = MetalContext::instance();
  auto *pso = ctx.pipelineState("visualizeNormalKernel");

  auto *encoder = cmdBuf->computeCommandEncoder();
  encoder->setComputePipelineState(pso);
  encoder->setTexture(normal, 0);
  encoder->setTexture(color, 1);
  dispatch2D(encoder, pso, w, h);
  encoder->endEncoding();
}

void visualizeNormal(
    MTL::Texture *normal, MTL::Texture *color, uint32_t w, uint32_t h)
    METAL_DEFAULT_OVERLOAD(visualizeNormal, normal, color, w, h)

    // Edges
    // ///////////////////////////////////////////////////////////////////////

    void visualizeEdges(MTL::CommandBuffer *cmdBuf,
        MTL::Texture *objectId,
        MTL::Texture *color,
        bool invert,
        uint32_t w,
        uint32_t h)
{
  auto &ctx = MetalContext::instance();
  auto *pso = ctx.pipelineState("visualizeEdgesKernel");

  auto *encoder = cmdBuf->computeCommandEncoder();
  encoder->setComputePipelineState(pso);
  encoder->setTexture(objectId, 0);
  encoder->setTexture(color, 1);
  uint32_t invertVal = invert ? 1 : 0;
  encoder->setBytes(&invertVal, sizeof(invertVal), 0);
  dispatch2D(encoder, pso, w, h);
  encoder->endEncoding();
}

void visualizeEdges(MTL::Texture *objectId,
    MTL::Texture *color,
    bool invert,
    uint32_t w,
    uint32_t h)
    METAL_DEFAULT_OVERLOAD(visualizeEdges, objectId, color, invert, w, h)

#undef METAL_DEFAULT_OVERLOAD

} // namespace tsd::algorithms::metal
