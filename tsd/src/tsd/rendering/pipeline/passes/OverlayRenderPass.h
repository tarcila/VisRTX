// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ImagePass.h"
// anari
#include <anari/anari_cpp.hpp>

namespace tsd::rendering {

struct OverlayRenderPass : public ImagePass
{
  OverlayRenderPass();
  ~OverlayRenderPass() override;

  void setCamera(tsd::math::float3 pos,
      tsd::math::float3 dir,
      tsd::math::float3 up,
      float fovy,
      float aspect);

  void setWorld(anari::World w);

  anari::Device device() const;

 private:
  void updateSize() override;
  void render(ImageBuffers &b, int stageId) override;

  anari::Library m_library{nullptr};
  anari::Device m_device{nullptr};
  anari::Camera m_camera{nullptr};
  anari::Renderer m_renderer{nullptr};
  anari::World m_world{nullptr};
  anari::Frame m_frame{nullptr};

  bool m_dirty{true};
  bool m_deviceSupportsCUDAFrames{false};
};

} // namespace tsd::rendering
