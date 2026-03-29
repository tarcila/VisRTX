// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/algorithms/metal/runtime.hpp"
#include "MetalContext.h"

#include <CoreVideo/CoreVideo.h>
#include <IOSurface/IOSurface.h>
#include <Metal/Metal.hpp>

#include <mutex>
#include <string>
#include <unordered_map>

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

// --- Shared-memory buffer management ---

void *newSharedBuffer(size_t bytes)
{
  auto &ctx = MetalContext::instance();
  auto *buf = ctx.device()->newBuffer(bytes, MTL::ResourceStorageModeShared);
  return buf;
}

void releaseBuffer(void *buffer)
{
  if (buffer)
    static_cast<MTL::Buffer *>(buffer)->release();
}

void *bufferContents(void *buffer)
{
  if (!buffer)
    return nullptr;
  return static_cast<MTL::Buffer *>(buffer)->contents();
}

void *newPrivateBuffer(size_t bytes)
{
  auto &ctx = MetalContext::instance();
  auto *buf = ctx.device()->newBuffer(bytes, MTL::ResourceStorageModePrivate);
  return buf;
}

void blitToBuffer(void *src, void *dst, size_t bytes)
{
  if (!src || !dst || bytes == 0)
    return;
  auto &ctx = MetalContext::instance();
  auto *cmdBuf = ctx.defaultQueue()->commandBuffer();
  auto *blit = cmdBuf->blitCommandEncoder();
  blit->copyFromBuffer(static_cast<MTL::Buffer *>(src),
      0,
      static_cast<MTL::Buffer *>(dst),
      0,
      bytes);
  blit->endEncoding();
  cmdBuf->commit();
  cmdBuf->waitUntilCompleted();
}

void *newPrivateBuffer(size_t bytes)
{
    auto &ctx = MetalContext::instance();
    auto *buf = ctx.device()->newBuffer(bytes, MTL::ResourceStorageModePrivate);
    return buf;
}

void blitToBuffer(void *src, void *dst, size_t bytes)
{
    if (!src || !dst || bytes == 0)
        return;
    auto &ctx = MetalContext::instance();
    auto *cmdBuf = ctx.defaultQueue()->commandBuffer();
    auto *blit = cmdBuf->blitCommandEncoder();
    blit->copyFromBuffer(
        static_cast<MTL::Buffer *>(src), 0,
        static_cast<MTL::Buffer *>(dst), 0,
        bytes);
    blit->endEncoding();
    cmdBuf->commit();
    cmdBuf->waitUntilCompleted();
}

// --- Generic compute dispatch ---

// Per-library pipeline state cache (keyed by kernel name)
static std::unordered_map<MTL::Library *,
    std::unordered_map<std::string, MTL::ComputePipelineState *>>
    g_externalPipelines;
static std::mutex g_externalPipelinesMutex;

static MTL::ComputePipelineState *cachedPipelineState(
    MTL::Library *lib, const char *kernelName)
{
  std::lock_guard lock(g_externalPipelinesMutex);
  auto &cache = g_externalPipelines[lib];
  auto it = cache.find(kernelName);
  if (it != cache.end())
    return it->second;

  auto *fn =
      lib->newFunction(NS::String::string(kernelName, NS::ASCIIStringEncoding));
  NS::Error *error = nullptr;
  auto *pso =
      MetalContext::instance().device()->newComputePipelineState(fn, &error);
  fn->release();

  cache[kernelName] = pso;
  return pso;
}

void *compileShaderSource(const char *source)
{
  auto &ctx = MetalContext::instance();
  auto *src = NS::String::string(source, NS::UTF8StringEncoding);
  auto *opts = MTL::CompileOptions::alloc()->init();
  NS::Error *error = nullptr;
  auto *lib = ctx.device()->newLibrary(src, opts, &error);
  opts->release();
  return lib;
}

void releaseLibrary(void *library)
{
  if (!library)
    return;
  auto *lib = static_cast<MTL::Library *>(library);

  std::lock_guard lock(g_externalPipelinesMutex);
  auto it = g_externalPipelines.find(lib);
  if (it != g_externalPipelines.end()) {
    for (auto &[_, pso] : it->second)
      pso->release();
    g_externalPipelines.erase(it);
  }
  lib->release();
}

void dispatchKernel(void *library,
    const char *kernelName,
    void *const *buffers,
    uint32_t numBuffers,
    const void *constants,
    uint32_t constantsSize,
    uint32_t threadCount)
{
  auto *lib = static_cast<MTL::Library *>(library);
  auto *pso = cachedPipelineState(lib, kernelName);

  auto &ctx = MetalContext::instance();
  auto *cmdBuf = ctx.defaultQueue()->commandBuffer();
  auto *encoder = cmdBuf->computeCommandEncoder();
  encoder->setComputePipelineState(pso);

  for (uint32_t i = 0; i < numBuffers; i++)
    encoder->setBuffer(static_cast<MTL::Buffer *>(buffers[i]), 0, i);

  if (constants && constantsSize > 0)
    encoder->setBytes(constants, constantsSize, numBuffers);

  auto tgSize = pso->maxTotalThreadsPerThreadgroup();
  if (tgSize > threadCount)
    tgSize = threadCount;
  encoder->dispatchThreads(
      {(NS::UInteger)threadCount, 1, 1}, {(NS::UInteger)tgSize, 1, 1});
  encoder->endEncoding();

  cmdBuf->commit();
  cmdBuf->waitUntilCompleted();
}

} // namespace tsd::algorithms::metal
