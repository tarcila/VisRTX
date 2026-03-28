// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#if ENABLE_SDL

#include "CopyToSDLTexturePass.h"
#include "tsd/core/Logging.hpp"

#ifdef ENABLE_CUDA
// cuda
#include <SDL3/SDL_opengl.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>
#endif

#ifdef ENABLE_METAL
#include "tsd/algorithms/metal/runtime.hpp"
#endif

namespace tsd::rendering {

struct CopyToSDLTexturePass::CopyToSDLTexturePassImpl
{
  SDL_Renderer *renderer{nullptr};
  SDL_Texture *texture{nullptr};
  bool glInteropAvailable{false};
#ifdef ENABLE_CUDA
  cudaGraphicsResource_t graphicsResource{nullptr};
#endif
#ifdef ENABLE_METAL
  tsd::algorithms::metal::DisplaySurface *displaySurface{nullptr};
#endif
};

CopyToSDLTexturePass::CopyToSDLTexturePass(SDL_Renderer *renderer)
{
  m_impl = new CopyToSDLTexturePassImpl;
  m_impl->renderer = renderer;
  m_impl->glInteropAvailable = checkGLInterop();
}

CopyToSDLTexturePass::~CopyToSDLTexturePass()
{
#ifdef ENABLE_CUDA
  if (m_impl->graphicsResource)
    cudaGraphicsUnregisterResource(m_impl->graphicsResource);
#endif
#ifdef ENABLE_METAL
  tsd::algorithms::metal::destroyDisplaySurface(m_impl->displaySurface);
#endif
  SDL_DestroyTexture(m_impl->texture);
  delete m_impl;
  m_impl = nullptr;
}

SDL_Texture *CopyToSDLTexturePass::getTexture() const
{
  return m_impl->texture;
}

bool CopyToSDLTexturePass::checkGLInterop() const
{
#ifdef ENABLE_CUDA
  unsigned int numDevices = 0;
  int cudaDevices[8];

  cudaError_t err =
      cudaGLGetDevices(&numDevices, cudaDevices, 8, cudaGLDeviceListAll);
  if (err != cudaSuccess) {
    tsd::core::logWarning("[ImagePipeline] failed to get CUDA GL devices");
    cudaGetLastError();
    return false;
  }

  if (numDevices > 0) {
    int currentDevice = 0;
    cudaGetDevice(&currentDevice);
    for (unsigned int i = 0; i < numDevices; ++i) {
      if (currentDevice == cudaDevices[i]) {
        tsd::core::logStatus("[ImagePipeline] using CUDA-GL interop via SDL3");
        return true;
      }
    }
  }
#endif

  tsd::core::logWarning(
      "[ImagePipeline] unable to use CUDA-GL interop via SDL3");
  return false;
}

void CopyToSDLTexturePass::render(ImageBuffers &b, int /*stageId*/)
{
  const auto size = getDimensions();

#ifdef ENABLE_METAL
  if (m_impl->displaySurface && b.metalHdrColor) {
    tsd::algorithms::metal::blitToDisplaySurface(
        b.metalHdrColor, m_impl->displaySurface);
    return;
  }
#endif

#ifdef ENABLE_CUDA
  if (m_impl->glInteropAvailable && m_impl->graphicsResource) {
    cudaGraphicsMapResources(1, &m_impl->graphicsResource);
    cudaArray_t array;
    cudaGraphicsSubResourceGetMappedArray(
        &array, m_impl->graphicsResource, 0, 0);
    cudaMemcpy2DToArray(array,
        0,
        0,
        b.color,
        size.x * sizeof(b.color[0]),
        size.x * sizeof(b.color[0]),
        size.y,
        cudaMemcpyDeviceToDevice);
    cudaGraphicsUnmapResources(1, &m_impl->graphicsResource);
  } else {
#endif
    SDL_UpdateTexture(m_impl->texture,
        nullptr,
        b.color,
        getDimensions().x * sizeof(b.color[0]));
#ifdef ENABLE_CUDA
  }
#endif
}

void CopyToSDLTexturePass::updateSize()
{
#ifdef ENABLE_CUDA
  if (m_impl->graphicsResource) {
    cudaGraphicsUnregisterResource(m_impl->graphicsResource);
    m_impl->graphicsResource = nullptr;
  }
#endif

#ifdef ENABLE_METAL
  tsd::algorithms::metal::destroyDisplaySurface(m_impl->displaySurface);
  m_impl->displaySurface = nullptr;
#endif

  if (m_impl->texture)
    SDL_DestroyTexture(m_impl->texture);

  auto newSize = getDimensions();

#ifdef ENABLE_METAL
  {
    namespace mtl = tsd::algorithms::metal;
    m_impl->displaySurface = mtl::createDisplaySurface(newSize.x, newSize.y);
    if (m_impl->displaySurface) {
      auto *cvpb = mtl::displaySurfacePixelBuffer(m_impl->displaySurface);
      SDL_PropertiesID props = SDL_CreateProperties();
      SDL_SetNumberProperty(
          props, SDL_PROP_TEXTURE_CREATE_FORMAT_NUMBER, SDL_PIXELFORMAT_BGRA32);
      SDL_SetNumberProperty(props,
          SDL_PROP_TEXTURE_CREATE_ACCESS_NUMBER,
          SDL_TEXTUREACCESS_STATIC);
      SDL_SetNumberProperty(
          props, SDL_PROP_TEXTURE_CREATE_WIDTH_NUMBER, newSize.x);
      SDL_SetNumberProperty(
          props, SDL_PROP_TEXTURE_CREATE_HEIGHT_NUMBER, newSize.y);
      SDL_SetPointerProperty(
          props, SDL_PROP_TEXTURE_CREATE_METAL_PIXELBUFFER_POINTER, cvpb);
      m_impl->texture =
          SDL_CreateTextureWithProperties(m_impl->renderer, props);
      SDL_DestroyProperties(props);

      if (m_impl->texture) {
        tsd::core::logStatus(
            "[ImagePipeline] using Metal-SDL interop via IOSurface");
        return;
      }
    }

    // Fallback: destroy surface if SDL texture creation failed
    mtl::destroyDisplaySurface(m_impl->displaySurface);
    m_impl->displaySurface = nullptr;
    tsd::core::logWarning(
        "[ImagePipeline] Metal-SDL interop unavailable, falling back");
  }
#endif

  m_impl->texture = SDL_CreateTexture(m_impl->renderer,
      SDL_PIXELFORMAT_RGBA32,
      SDL_TEXTUREACCESS_STREAMING,
      newSize.x,
      newSize.y);

#ifdef ENABLE_CUDA
  if (m_impl->glInteropAvailable) {
    SDL_PropertiesID propID = SDL_GetTextureProperties(m_impl->texture);
    Sint64 texID = SDL_GetNumberProperty(
        propID, SDL_PROP_TEXTURE_OPENGL_TEXTURE_NUMBER, -1);

    if (texID > 0) {
      cudaGraphicsGLRegisterImage(&m_impl->graphicsResource,
          static_cast<GLuint>(texID),
          GL_TEXTURE_2D,
          cudaGraphicsRegisterFlagsWriteDiscard);
    } else {
      tsd::core::logWarning(
          "[ImagePipeline] could not get SDL texture number!");
    }
  }
#endif
}

} // namespace tsd::rendering

#endif
