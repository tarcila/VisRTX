// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/algorithms/metal/runtime.hpp"
#include "MetalContext.h"

#include <CoreVideo/CoreVideo.h>
#include <IOSurface/IOSurface.h>
#include <Metal/Metal.hpp>

namespace tsd::algorithms::metal {

struct DisplaySurface
{
  IOSurfaceRef ioSurface{nullptr};
  CVPixelBufferRef pixelBuffer{nullptr};
  MTL::Texture *texture{nullptr};
  uint32_t width{0};
  uint32_t height{0};
};

void setSharedQueue(void *commandQueue)
{
  MetalContext::instance().setQueue(
      static_cast<MTL::CommandQueue *>(commandQueue));
}

DisplaySurface *createDisplaySurface(uint32_t width, uint32_t height)
{
  unsigned int w = width;
  unsigned int h = height;
  unsigned int bpe = 4;
  unsigned int bpr = ((w * bpe + 15) / 16) * 16; // 16-byte aligned
  unsigned int allocSize = bpr * h;
  unsigned int pixFmt = kCVPixelFormatType_32BGRA;

  const void *keys[] = {kIOSurfaceWidth,
      kIOSurfaceHeight,
      kIOSurfaceBytesPerElement,
      kIOSurfaceBytesPerRow,
      kIOSurfaceAllocSize,
      kIOSurfacePixelFormat};
  const void *vals[] = {CFNumberCreate(nullptr, kCFNumberIntType, &w),
      CFNumberCreate(nullptr, kCFNumberIntType, &h),
      CFNumberCreate(nullptr, kCFNumberIntType, &bpe),
      CFNumberCreate(nullptr, kCFNumberIntType, &bpr),
      CFNumberCreate(nullptr, kCFNumberIntType, &allocSize),
      CFNumberCreate(nullptr, kCFNumberIntType, &pixFmt)};
  constexpr int nProps = 6;
  auto *props = CFDictionaryCreate(nullptr,
      keys,
      vals,
      nProps,
      &kCFTypeDictionaryKeyCallBacks,
      &kCFTypeDictionaryValueCallBacks);

  auto *surface = new DisplaySurface;
  surface->width = width;
  surface->height = height;
  surface->ioSurface = IOSurfaceCreate(props);

  for (int i = 0; i < nProps; ++i)
    CFRelease(vals[i]);
  CFRelease(props);

  if (!surface->ioSurface) {
    delete surface;
    return nullptr;
  }

  CVReturn cvr = CVPixelBufferCreateWithIOSurface(
      nullptr, surface->ioSurface, nullptr, &surface->pixelBuffer);
  if (cvr != kCVReturnSuccess || !surface->pixelBuffer) {
    CFRelease(surface->ioSurface);
    delete surface;
    return nullptr;
  }

  auto &ctx = MetalContext::instance();
  auto *desc = MTL::TextureDescriptor::texture2DDescriptor(
      MTL::PixelFormatBGRA8Unorm, width, height, false);
  desc->setStorageMode(MTL::StorageModeShared);
  desc->setUsage(MTL::TextureUsageShaderWrite);
  surface->texture = ctx.device()->newTexture(desc, surface->ioSurface, 0);

  if (!surface->texture) {
    CVPixelBufferRelease(surface->pixelBuffer);
    CFRelease(surface->ioSurface);
    delete surface;
    return nullptr;
  }

  return surface;
}

void destroyDisplaySurface(DisplaySurface *surface)
{
  if (!surface)
    return;
  if (surface->texture)
    surface->texture->release();
  if (surface->pixelBuffer)
    CVPixelBufferRelease(surface->pixelBuffer);
  if (surface->ioSurface)
    CFRelease(surface->ioSurface);
  delete surface;
}

void *displaySurfacePixelBuffer(DisplaySurface *surface)
{
  return surface ? surface->pixelBuffer : nullptr;
}

void blitToDisplaySurface(MTL::Texture *input, DisplaySurface *surface)
{
  auto &ctx = MetalContext::instance();
  auto *pso = ctx.pipelineState("convertFloatToBGRA8Kernel");

  auto *cmdBuf = ctx.defaultQueue()->commandBuffer();
  auto *encoder = cmdBuf->computeCommandEncoder();
  encoder->setComputePipelineState(pso);
  encoder->setTexture(input, 0);
  encoder->setTexture(surface->texture, 1);

  auto tgW = pso->threadExecutionWidth();
  auto tgH = pso->maxTotalThreadsPerThreadgroup() / tgW;
  encoder->dispatchThreads(
      {(NS::UInteger)surface->width, (NS::UInteger)surface->height, 1},
      {tgW, (NS::UInteger)tgH, 1});
  encoder->endEncoding();

  cmdBuf->commit();
  cmdBuf->waitUntilCompleted();
}

void compositeByDepth(MTL::Texture *overlayColor,
    MTL::Texture *overlayDepth,
    MTL::Texture *mainColor,
    MTL::Texture *mainDepth)
{
  if (!overlayColor || !overlayDepth || !mainColor || !mainDepth)
    return;

  auto &ctx = MetalContext::instance();
  auto *pso = ctx.pipelineState("compositeByDepthKernel");

  auto *cmdBuf = ctx.defaultQueue()->commandBuffer();
  auto *encoder = cmdBuf->computeCommandEncoder();
  encoder->setComputePipelineState(pso);
  encoder->setTexture(overlayColor, 0);
  encoder->setTexture(overlayDepth, 1);
  encoder->setTexture(mainColor, 2);
  encoder->setTexture(mainDepth, 3);

  auto w = (NS::UInteger)mainColor->width();
  auto h = (NS::UInteger)mainColor->height();
  auto tgW = pso->threadExecutionWidth();
  auto tgH = pso->maxTotalThreadsPerThreadgroup() / tgW;
  encoder->dispatchThreads({w, h, 1}, {tgW, (NS::UInteger)tgH, 1});
  encoder->endEncoding();

  cmdBuf->commit();
  cmdBuf->waitUntilCompleted();
}

} // namespace tsd::algorithms::metal
