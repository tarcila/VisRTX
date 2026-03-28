// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "AnariSceneRenderPass.h"
#include "tsd/core/Logging.hpp"
// tsd_algorithms
#include "tsd/algorithms/cpu/depthCompositeFrame.hpp"
#ifdef TSD_ALGORITHMS_HAS_CUDA
#include "tsd/algorithms/cuda/depthCompositeFrame.hpp"
#endif
// std
#include <algorithm>
#include <cstring>

#ifdef ENABLE_METAL
#include "tsd/algorithms/metal/runtime.hpp"
#endif

namespace tsd::rendering {

// Helper functions ///////////////////////////////////////////////////////////

static bool supportsCUDAFbData(anari::Device d)
{
#ifdef ENABLE_CUDA
  bool supportsCUDA = false;
  auto list = (const char *const *)anariGetObjectInfo(
      d, ANARI_DEVICE, "default", "extension", ANARI_STRING_LIST);

  for (const char *const *i = list; *i != nullptr; ++i) {
    if (std::string(*i) == "ANARI_NV_FRAME_BUFFERS_CUDA") {
      supportsCUDA = true;
      break;
    }
  }

  return supportsCUDA;
#else
  return false;
#endif
}

static bool supportsMetalFbData(anari::Device d)
{
#ifdef ENABLE_METAL
  auto list = (const char *const *)anariGetObjectInfo(
      d, ANARI_DEVICE, "default", "extension", ANARI_STRING_LIST);
  for (const char *const *i = list; *i != nullptr; ++i) {
    if (std::string(*i) == "ANARI_MTL_FRAME_BUFFERS_METAL")
      return true;
  }
#endif
  return false;
}

// AnariSceneRenderPass definitions ///////////////////////////////////////////

AnariSceneRenderPass::AnariSceneRenderPass(anari::Device d) : m_device(d)
{
  anari::retain(d, d);
  m_frame = anari::newObject<anari::Frame>(d);
  anari::setParameter(d, m_frame, "channel.color", ANARI_UFIXED8_RGBA_SRGB);
  anari::setParameter(d, m_frame, "channel.depth", ANARI_FLOAT32);
  anari::setParameter(d, m_frame, "accumulation", true);

  m_deviceSupportsCUDAFrames = supportsCUDAFbData(d);

  if (m_deviceSupportsCUDAFrames)
    tsd::core::logStatus("[ImagePipeline] using CUDA-mapped fb channels");

#ifdef ENABLE_METAL
  m_deviceSupportsMetalFrames = supportsMetalFbData(d);
  if (m_deviceSupportsMetalFrames) {
    tsd::core::logStatus("[ImagePipeline] using Metal-mapped fb channels");
    MTL::CommandQueue *queue = nullptr;
    anariGetProperty(d,
        d,
        "mtl.commandQueue",
        ANARI_VOID_POINTER,
        &queue,
        sizeof(queue),
        ANARI_WAIT);
    m_metalQueue = queue;
    if (queue)
      tsd::algorithms::metal::setSharedQueue(queue);

    anari::setParameter(d, m_frame, "channel.colorMTL", ANARI_FLOAT32_VEC4);
    anari::setParameter(d, m_frame, "channel.depthMTL", ANARI_FLOAT32);
  }
#endif

  if (!m_deviceSupportsCUDAFrames
#ifdef ENABLE_METAL
      && !m_deviceSupportsMetalFrames
#endif
  )
    tsd::core::logStatus("[ImagePipeline] using host-mapped fb channels");
}

AnariSceneRenderPass::~AnariSceneRenderPass()
{
  cleanup();

  anari::discard(m_device, m_frame);
  anari::wait(m_device, m_frame);

  anari::release(m_device, m_frame);
  anari::release(m_device, m_camera);
  anari::release(m_device, m_renderer);
  anari::release(m_device, m_world);
  anari::release(m_device, m_device);
}

void AnariSceneRenderPass::setCamera(anari::Camera c)
{
  if (c)
    anari::retain(m_device, c);
  anari::setParameter(m_device, m_frame, "camera", c);
  anari::commitParameters(m_device, m_frame);
  anari::release(m_device, m_camera);
  m_camera = c;
  if (m_camera) {
    auto size = getDimensions();
    anari::setParameter(m_device, m_camera, "aspect", size.x / float(size.y));
    anari::commitParameters(m_device, m_camera);
  }
}

void AnariSceneRenderPass::setRenderer(anari::Renderer r)
{
  if (r)
    anari::retain(m_device, r);
  anari::setParameter(m_device, m_frame, "renderer", r);
  anari::commitParameters(m_device, m_frame);
  anari::release(m_device, m_renderer);
  m_renderer = r;
}

void AnariSceneRenderPass::setWorld(anari::World w)
{
  if (w)
    anari::retain(m_device, w);
  anari::setParameter(m_device, m_frame, "world", w);
  anari::commitParameters(m_device, m_frame);
  anari::release(m_device, m_world);
  m_world = w;
}

void AnariSceneRenderPass::setColorFormat(anari::DataType t)
{
  m_format = t;
  anari::setParameter(m_device, m_frame, "channel.color", t);
  anari::commitParameters(m_device, m_frame);
}

void AnariSceneRenderPass::setEnableIDs(bool on)
{
  if (on == m_enableIDs)
    return;

  m_enableIDs = on;

  if (on) {
    tsd::core::logInfo("[ImagePipeline] enabling objectId frame channel");

    anari::discard(m_device, m_frame);
    anari::wait(m_device, m_frame);

    anari::setParameter(m_device, m_frame, "channel.objectId", ANARI_UINT32);
#ifdef ENABLE_METAL
    if (m_deviceSupportsMetalFrames)
      anari::setParameter(
          m_device, m_frame, "channel.objectIdMTL", ANARI_UINT32);
#endif
    anari::commitParameters(m_device, m_frame);

    anari::render(m_device, m_frame);
    anari::wait(m_device, m_frame);
  } else {
    tsd::core::logInfo("[ImagePipeline] disabling objectId frame channel");
    anari::unsetParameter(m_device, m_frame, "channel.objectId");
#ifdef ENABLE_METAL
    if (m_deviceSupportsMetalFrames)
      anari::unsetParameter(m_device, m_frame, "channel.objectIdMTL");
#endif
    anari::commitParameters(m_device, m_frame);

    auto size = getDimensions();
    const size_t totalSize = size_t(size.x) * size_t(size.y);
    std::fill(m_buffers.objectId, m_buffers.objectId + totalSize, ~0u);
  }
}

void AnariSceneRenderPass::setEnablePrimitiveId(bool on)
{
  if (on == m_enablePrimitiveId)
    return;

  m_enablePrimitiveId = on;

  if (on) {
    tsd::core::logInfo("[ImagePipeline] enabling primitiveId frame channel");

    anari::discard(m_device, m_frame);
    anari::wait(m_device, m_frame);

    anari::setParameter(m_device, m_frame, "channel.primitiveId", ANARI_UINT32);
#ifdef ENABLE_METAL
    if (m_deviceSupportsMetalFrames)
      anari::setParameter(
          m_device, m_frame, "channel.primitiveIdMTL", ANARI_UINT32);
#endif
    anari::commitParameters(m_device, m_frame);

    anari::render(m_device, m_frame);
    anari::wait(m_device, m_frame);
  } else {
    tsd::core::logInfo("[ImagePipeline] disabling primitiveId frame channel");
    anari::unsetParameter(m_device, m_frame, "channel.primitiveId");
#ifdef ENABLE_METAL
    if (m_deviceSupportsMetalFrames)
      anari::unsetParameter(m_device, m_frame, "channel.primitiveIdMTL");
#endif
    anari::commitParameters(m_device, m_frame);

    auto size = getDimensions();
    const size_t totalSize = size_t(size.x) * size_t(size.y);
    std::fill(m_buffers.primitiveId, m_buffers.primitiveId + totalSize, ~0u);
  }
}

void AnariSceneRenderPass::setEnableInstanceId(bool on)
{
  if (on == m_enableInstanceId)
    return;

  m_enableInstanceId = on;

  if (on) {
    tsd::core::logInfo("[ImagePipeline] enabling instanceId frame channel");

    anari::discard(m_device, m_frame);
    anari::wait(m_device, m_frame);

    anari::setParameter(m_device, m_frame, "channel.instanceId", ANARI_UINT32);
#ifdef ENABLE_METAL
    if (m_deviceSupportsMetalFrames)
      anari::setParameter(
          m_device, m_frame, "channel.instanceIdMTL", ANARI_UINT32);
#endif
    anari::commitParameters(m_device, m_frame);

    anari::render(m_device, m_frame);
    anari::wait(m_device, m_frame);
  } else {
    tsd::core::logInfo("[ImagePipeline] disabling instanceId frame channel");
    anari::unsetParameter(m_device, m_frame, "channel.instanceId");
#ifdef ENABLE_METAL
    if (m_deviceSupportsMetalFrames)
      anari::unsetParameter(m_device, m_frame, "channel.instanceIdMTL");
#endif
    anari::commitParameters(m_device, m_frame);

    auto size = getDimensions();
    const size_t totalSize = size_t(size.x) * size_t(size.y);
    std::fill(m_buffers.instanceId, m_buffers.instanceId + totalSize, ~0u);
  }
}

void AnariSceneRenderPass::setEnableAlbedo(bool on)
{
  if (on == m_enableAlbedo)
    return;

  m_enableAlbedo = on;

  if (on) {
    tsd::core::logInfo("[ImagePipeline] enabling albedo frame channel");

    anari::discard(m_device, m_frame);
    anari::wait(m_device, m_frame);

    anari::setParameter(
        m_device, m_frame, "channel.albedo", ANARI_FLOAT32_VEC3);
#ifdef ENABLE_METAL
    if (m_deviceSupportsMetalFrames)
      anari::setParameter(
          m_device, m_frame, "channel.albedoMTL", ANARI_FLOAT32_VEC4);
#endif
    anari::commitParameters(m_device, m_frame);

    anari::render(m_device, m_frame);
    anari::wait(m_device, m_frame);
  } else {
    tsd::core::logInfo("[ImagePipeline] disabling albedo frame channel");
    anari::unsetParameter(m_device, m_frame, "channel.albedo");
#ifdef ENABLE_METAL
    if (m_deviceSupportsMetalFrames)
      anari::unsetParameter(m_device, m_frame, "channel.albedoMTL");
#endif
    anari::commitParameters(m_device, m_frame);
  }
}

void AnariSceneRenderPass::setEnableNormals(bool on)
{
  if (on == m_enableNormals)
    return;

  m_enableNormals = on;

  if (on) {
    tsd::core::logInfo("[ImagePipeline] enabling normal frame channel");

    anari::discard(m_device, m_frame);
    anari::wait(m_device, m_frame);

    anari::setParameter(
        m_device, m_frame, "channel.normal", ANARI_FLOAT32_VEC3);
#ifdef ENABLE_METAL
    if (m_deviceSupportsMetalFrames)
      anari::setParameter(
          m_device, m_frame, "channel.normalMTL", ANARI_FLOAT32_VEC4);
#endif
    anari::commitParameters(m_device, m_frame);

    anari::render(m_device, m_frame);
    anari::wait(m_device, m_frame);
  } else {
    tsd::core::logInfo("[ImagePipeline] disabling normal frame channel");
    anari::unsetParameter(m_device, m_frame, "channel.normal");
#ifdef ENABLE_METAL
    if (m_deviceSupportsMetalFrames)
      anari::unsetParameter(m_device, m_frame, "channel.normalMTL");
#endif
    anari::commitParameters(m_device, m_frame);
  }
}

void AnariSceneRenderPass::setRunAsync(bool on)
{
  m_runAsync = on;
}

anari::Frame AnariSceneRenderPass::getFrame() const
{
  return m_frame;
}

void AnariSceneRenderPass::updateSize()
{
  cleanup();
  auto size = getDimensions();
  anari::setParameter(m_device, m_frame, "size", size);
  anari::commitParameters(m_device, m_frame);

  if (m_camera) {
    anari::setParameter(m_device, m_camera, "aspect", size.x / float(size.y));
    anari::commitParameters(m_device, m_camera);
  }

  const size_t totalSize = size_t(size.x) * size_t(size.y);
  m_buffers.color = detail::allocate<uint32_t>(totalSize);
  m_buffers.hdrColor = detail::allocate<float>(totalSize * 4);
  m_buffers.depth = detail::allocate<float>(totalSize);
  m_buffers.objectId = detail::allocate<uint32_t>(totalSize);
  m_buffers.primitiveId = detail::allocate<uint32_t>(totalSize);
  m_buffers.instanceId = detail::allocate<uint32_t>(totalSize);
  m_buffers.albedo = detail::allocate<tsd::math::float3>(totalSize);
  m_buffers.normal = detail::allocate<tsd::math::float3>(totalSize);
}

void AnariSceneRenderPass::render(ImageBuffers &b, int stageId)
{
  m_buffers.stream = b.stream;

  if (m_firstFrame)
    anari::render(m_device, m_frame);

  if (m_firstFrame || !m_runAsync) {
    anari::wait(m_device, m_frame);
    m_firstFrame = false;
  }

  if (anari::isReady(m_device, m_frame)) {
    copyFrameData();
    anari::render(m_device, m_frame);
  }

  composite(b, stageId);
}

void AnariSceneRenderPass::copyFrameData()
{
#ifdef ENABLE_METAL
  if (m_deviceSupportsMetalFrames) {
    auto color = anari::map<void>(m_device, m_frame, "channel.colorMTL");
    m_buffers.metalHdrColor =
        reinterpret_cast<MTL::Texture *>(const_cast<void *>(color.data));

    auto depth = anari::map<void>(m_device, m_frame, "channel.depthMTL");
    m_buffers.metalDepth =
        reinterpret_cast<MTL::Texture *>(const_cast<void *>(depth.data));

    if (m_enableIDs) {
      auto oid = anari::map<void>(m_device, m_frame, "channel.objectIdMTL");
      m_buffers.metalObjectId =
          reinterpret_cast<MTL::Texture *>(const_cast<void *>(oid.data));
    }
    if (m_enablePrimitiveId) {
      auto pid = anari::map<void>(m_device, m_frame, "channel.primitiveIdMTL");
      m_buffers.metalPrimitiveId =
          reinterpret_cast<MTL::Texture *>(const_cast<void *>(pid.data));
    }
    if (m_enableInstanceId) {
      auto iid = anari::map<void>(m_device, m_frame, "channel.instanceIdMTL");
      m_buffers.metalInstanceId =
          reinterpret_cast<MTL::Texture *>(const_cast<void *>(iid.data));
    }
    if (m_enableAlbedo) {
      auto alb = anari::map<void>(m_device, m_frame, "channel.albedoMTL");
      m_buffers.metalAlbedo =
          reinterpret_cast<MTL::Texture *>(const_cast<void *>(alb.data));
    }
    if (m_enableNormals) {
      auto nrm = anari::map<void>(m_device, m_frame, "channel.normalMTL");
      m_buffers.metalNormal =
          reinterpret_cast<MTL::Texture *>(const_cast<void *>(nrm.data));
    }
    return;
  }
#endif

  const char *colorChannel =
      m_deviceSupportsCUDAFrames ? "channel.colorCUDA" : "channel.color";
  const char *depthChannel =
      m_deviceSupportsCUDAFrames ? "channel.depthCUDA" : "channel.depth";
  const char *objectIdChannel =
      m_deviceSupportsCUDAFrames ? "channel.objectIdCUDA" : "channel.objectId";
  const char *primitiveIdChannel = m_deviceSupportsCUDAFrames
      ? "channel.primitiveIdCUDA"
      : "channel.primitiveId";
  const char *instanceIdChannel = m_deviceSupportsCUDAFrames
      ? "channel.instanceIdCUDA"
      : "channel.instanceId";
  const char *albedoChannel =
      m_deviceSupportsCUDAFrames ? "channel.albedoCUDA" : "channel.albedo";
  const char *normalChannel =
      m_deviceSupportsCUDAFrames ? "channel.normalCUDA" : "channel.normal";

  auto color = anari::map<void>(m_device, m_frame, colorChannel);
  auto depth = anari::map<float>(m_device, m_frame, depthChannel);

  const tsd::math::uint2 size(getDimensions());
  const size_t totalSize = size.x * size.y;
  if (totalSize > 0 && size.x == color.width && size.y == color.height) {
    if (color.pixelType == ANARI_FLOAT32_VEC4) {
      detail::copy(m_buffers.hdrColor, (float *)color.data, totalSize * 4);
      detail::convertFloatColorBuffer_(m_buffers.stream,
          (const float *)color.data,
          (uint8_t *)m_buffers.color,
          totalSize * 4);
    } else
      detail::copy(m_buffers.color, (uint32_t *)color.data, totalSize);

    detail::copy(m_buffers.depth, depth.data, totalSize);
    if (m_enableIDs) {
      auto objectId = anari::map<uint32_t>(m_device, m_frame, objectIdChannel);
      if (objectId.data)
        detail::copy(m_buffers.objectId, objectId.data, totalSize);
    }
    if (m_enablePrimitiveId) {
      auto primitiveId =
          anari::map<uint32_t>(m_device, m_frame, primitiveIdChannel);
      if (primitiveId.data)
        detail::copy(m_buffers.primitiveId, primitiveId.data, totalSize);
    }
    if (m_enableInstanceId) {
      auto instanceId =
          anari::map<uint32_t>(m_device, m_frame, instanceIdChannel);
      if (instanceId.data)
        detail::copy(m_buffers.instanceId, instanceId.data, totalSize);
    }
    if (m_enableAlbedo) {
      auto albedo =
          anari::map<tsd::math::float3>(m_device, m_frame, albedoChannel);
      if (albedo.data)
        detail::copy(m_buffers.albedo, albedo.data, totalSize);
    }
    if (m_enableNormals) {
      auto normal =
          anari::map<tsd::math::float3>(m_device, m_frame, normalChannel);
      if (normal.data)
        detail::copy(m_buffers.normal, normal.data, totalSize);
    }
  }

  anari::unmap(m_device, m_frame, colorChannel);
  anari::unmap(m_device, m_frame, depthChannel);
  if (m_enableIDs)
    anari::unmap(m_device, m_frame, objectIdChannel);
  if (m_enablePrimitiveId)
    anari::unmap(m_device, m_frame, primitiveIdChannel);
  if (m_enableInstanceId)
    anari::unmap(m_device, m_frame, instanceIdChannel);
  if (m_enableAlbedo)
    anari::unmap(m_device, m_frame, albedoChannel);
  if (m_enableNormals)
    anari::unmap(m_device, m_frame, normalChannel);
}

void AnariSceneRenderPass::composite(ImageBuffers &b, int stageId)
{
  const bool firstPass = stageId == 0;
  const tsd::math::uint2 size(getDimensions());
  const size_t totalSize = size.x * size.y;

#ifdef ENABLE_METAL
  if (m_deviceSupportsMetalFrames) {
    if (firstPass) {
      b.metalHdrColor = m_buffers.metalHdrColor;
      b.metalDepth = m_buffers.metalDepth;
      b.metalObjectId = m_buffers.metalObjectId;
      b.metalPrimitiveId = m_buffers.metalPrimitiveId;
      b.metalInstanceId = m_buffers.metalInstanceId;
      b.metalAlbedo = m_buffers.metalAlbedo;
      b.metalNormal = m_buffers.metalNormal;
      b.stream = m_metalQueue;
    } else if (b.metalHdrColor && m_buffers.metalHdrColor) {
      tsd::algorithms::metal::compositeByDepth(m_buffers.metalHdrColor,
          m_buffers.metalDepth,
          b.metalHdrColor,
          b.metalDepth);
    }
    return;
  }
#endif

  if (firstPass) {
    detail::copy(b.color, m_buffers.color, totalSize);
    if (m_format == ANARI_FLOAT32_VEC4)
      detail::copy(b.hdrColor, m_buffers.hdrColor, totalSize * 4);
    detail::copy(b.depth, m_buffers.depth, totalSize);
    detail::copy(b.objectId, m_buffers.objectId, totalSize);
    if (m_enablePrimitiveId)
      detail::copy(b.primitiveId, m_buffers.primitiveId, totalSize);
    if (m_enableInstanceId)
      detail::copy(b.instanceId, m_buffers.instanceId, totalSize);
    if (m_enableAlbedo)
      detail::copy(b.albedo, m_buffers.albedo, totalSize);
    if (m_enableNormals)
      detail::copy(b.normal, m_buffers.normal, totalSize);
  } else {
    const uint32_t totalPixels = uint32_t(size.x) * uint32_t(size.y);
#ifdef TSD_ALGORITHMS_HAS_CUDA
    if (b.stream) {
      tsd::algorithms::cuda::depthCompositeFrame(b.stream,
          b.color,
          b.depth,
          b.objectId,
          m_buffers.color,
          m_buffers.depth,
          m_buffers.objectId,
          totalPixels,
          firstPass);
      return;
    }
#endif
    tsd::algorithms::cpu::depthCompositeFrame(b.color,
        b.depth,
        b.objectId,
        m_buffers.color,
        m_buffers.depth,
        m_buffers.objectId,
        totalPixels,
        firstPass);
  }
}

void AnariSceneRenderPass::cleanup()
{
  detail::free(m_buffers.color);
  detail::free(m_buffers.hdrColor);
  detail::free(m_buffers.depth);
  detail::free(m_buffers.objectId);
  detail::free(m_buffers.primitiveId);
  detail::free(m_buffers.instanceId);
  detail::free(m_buffers.albedo);
  detail::free(m_buffers.normal);
}

} // namespace tsd::rendering
