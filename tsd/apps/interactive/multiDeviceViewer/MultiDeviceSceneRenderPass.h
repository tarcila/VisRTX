// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// tsd_rendering
#include <tsd/rendering/pipeline/passes/RenderPass.h>
// anari
#include <anari/anari_cpp.hpp>
// std
#include <functional>

namespace tsd::dpt {

struct MultiDeviceSceneRenderPass : public tsd::rendering::RenderPass
{
  MultiDeviceSceneRenderPass(const std::vector<anari::Device> &devices);
  ~MultiDeviceSceneRenderPass() override;

  size_t numDevices() const;

  void setCamera(size_t i, anari::Camera c);
  void setRenderer(size_t i, anari::Renderer r);
  void setWorld(size_t i, anari::World w);
  void setColorFormat(anari::DataType t);

  // default' true' -- if 'false', then anari::wait() on each pass
  void setRunAsync(bool on);

 private:
  void foreach_frame(
      const std::function<void(anari::Device, anari::Frame)> &func) const;

  void updateSize() override;
  void render(tsd::rendering::RenderBuffers &b, int stageId) override;
  void copyFrameData();
  void composite(tsd::rendering::RenderBuffers &b, int stageId);
  void cleanup();

  tsd::rendering::RenderBuffers m_buffers;

  bool m_firstFrame{true};
  bool m_runAsync{true};

  std::vector<anari::Device> m_devices;
  std::vector<anari::Frame> m_frames;
};

} // namespace tsd::dpt
