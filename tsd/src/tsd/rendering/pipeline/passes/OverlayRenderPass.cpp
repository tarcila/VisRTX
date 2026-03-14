// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "OverlayRenderPass.h"
#include "tsd/core/Logging.hpp"
// std
#include <algorithm>
#include <cstring>

namespace tsd::rendering {

static void statusFunc(
    const void *, ANARIDevice, ANARIObject, ANARIDataType, ANARIStatusSeverity,
    ANARIStatusCode, const char *message)
{
  tsd::core::logStatus("[vector2d] %s", message);
}

// OverlayRenderPass definitions //////////////////////////////////////////////

OverlayRenderPass::OverlayRenderPass()
{
  m_library = anari::loadLibrary("vector2d", statusFunc);
  if (!m_library) {
    tsd::core::logError("[OverlayRenderPass] failed to load vector2d library");
    setEnabled(false);
    return;
  }

  m_device = anari::newDevice(m_library, "default");
  if (!m_device) {
    tsd::core::logError("[OverlayRenderPass] failed to create vector2d device");
    setEnabled(false);
    return;
  }

  m_camera = anari::newObject<anari::Camera>(m_device, "perspective");
  anari::commitParameters(m_device, m_camera);

  m_renderer = anari::newObject<anari::Renderer>(m_device, "default");
  tsd::math::float4 bg(0.f, 0.f, 0.f, 0.f);
  anari::setParameter(m_device, m_renderer, "background", bg);
  anari::commitParameters(m_device, m_renderer);

  m_world = anari::newObject<anari::World>(m_device);
  anari::commitParameters(m_device, m_world);

  m_frame = anari::newObject<anari::Frame>(m_device);
  anari::setParameter(m_device, m_frame, "channel.color", ANARI_FLOAT32_VEC4);
  anari::setParameter(m_device, m_frame, "camera", m_camera);
  anari::setParameter(m_device, m_frame, "renderer", m_renderer);
  anari::setParameter(m_device, m_frame, "world", m_world);
  anari::commitParameters(m_device, m_frame);

  tsd::core::logStatus("[OverlayRenderPass] vector2d device loaded");
}

OverlayRenderPass::~OverlayRenderPass()
{
  if (m_device) {
    anari::release(m_device, m_frame);
    anari::release(m_device, m_camera);
    anari::release(m_device, m_renderer);
    anari::release(m_device, m_world);
    anari::release(m_device, m_device);
  }
  if (m_library)
    anari::unloadLibrary(m_library);
}

void OverlayRenderPass::setCamera(tsd::math::float3 pos,
    tsd::math::float3 dir,
    tsd::math::float3 up,
    float fovy,
    float aspect)
{
  if (!m_device)
    return;
  anari::setParameter(m_device, m_camera, "position", pos);
  anari::setParameter(m_device, m_camera, "direction", dir);
  anari::setParameter(m_device, m_camera, "up", up);
  anari::setParameter(m_device, m_camera, "fovy", fovy);
  anari::setParameter(m_device, m_camera, "aspect", aspect);
  anari::commitParameters(m_device, m_camera);
}

void OverlayRenderPass::setWorld(anari::World w)
{
  if (!m_device)
    return;
  anari::setParameter(m_device, m_frame, "world", w);
  anari::commitParameters(m_device, m_frame);
  anari::release(m_device, m_world);
  m_world = w;
  if (w)
    anari::retain(m_device, w);
  m_firstFrame = true;
}

anari::Device OverlayRenderPass::device() const
{
  return m_device;
}

void OverlayRenderPass::updateSize()
{
  if (!m_device)
    return;
  auto size = getDimensions();
  anari::setParameter(m_device, m_frame, "size", size);
  anari::commitParameters(m_device, m_frame);
  m_firstFrame = true;
}

void OverlayRenderPass::render(ImageBuffers &b, int stageId)
{
  if (!m_device)
    return;

  anari::render(m_device, m_frame);
  anari::wait(m_device, m_frame);

  auto color = anari::map<tsd::math::float4>(m_device, m_frame, "channel.color");

  auto size = getDimensions();
  const size_t totalPixels = size_t(size.x) * size_t(size.y);

  if (color.data && totalPixels > 0
      && size.x == color.width && size.y == color.height) {
    for (size_t i = 0; i < totalPixels; i++) {
      auto oc = color.data[i];
      if (oc.w <= 0.f)
        continue;

      auto sc = helium::cvt_color_to_float4(b.color[i]);
      float invA = 1.f - oc.w;
      tsd::math::float4 blended(
          oc.x + sc.x * invA,
          oc.y + sc.y * invA,
          oc.z + sc.z * invA,
          oc.w + sc.w * invA);

      b.color[i] = helium::cvt_color_to_uint32(blended);
    }
  }

  anari::unmap(m_device, m_frame, "channel.color");
}

} // namespace tsd::rendering
