// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/rendering/bridge/GraphRenderBridge.hpp"
#include "tsd/rendering/pipeline/ImagePipeline.h"
#include "tsd/rendering/pipeline/passes/AnariSceneRenderPass.h"
#include "tsd/rendering/pipeline/passes/CopyToSDLTexturePass.h"
#include "tsd/rendering/view/Manipulator.hpp"
#include "tsd/ui/imgui/windows/Window.h"
// anari
#include <anari/anari_cpp.hpp>

namespace tsd::ui::imgui {

// Standalone viewport that renders one of a GraphRenderBridge's per-viewport
// ANARI worlds. Owns its own ANARI camera/renderer + manipulator + pipeline; it
// does NOT use BaseViewport (which is bound to TSD scene cameras/renderers).
struct GraphViewport : public Window
{
  GraphViewport(Application *app,
      tsd::rendering::GraphRenderBridge *bridge,
      int viewportIndex,
      anari::Device device,
      const char *name = "Viewport");
  ~GraphViewport() override;

  void buildUI() override;

 private:
  void handleNavigation();

  tsd::rendering::GraphRenderBridge *m_bridge{nullptr};
  int m_viewportIndex{0};
  anari::Device m_device{nullptr};
  anari::Camera m_camera{nullptr};
  anari::Renderer m_renderer{nullptr};
  tsd::rendering::Manipulator m_manip;
  tsd::rendering::UpdateToken m_manipToken{0};
  tsd::rendering::ImagePipeline m_pipeline;
  tsd::rendering::AnariSceneRenderPass *m_anariPass{nullptr};
  tsd::rendering::CopyToSDLTexturePass *m_outputPass{nullptr};
  tsd::math::int2 m_size{0, 0};

  // Mouse-navigation state (mirrors BaseViewport's normalized-delta model).
  tsd::math::float2 m_prevMouse{-1.f};
  bool m_manipulating{false};
  bool m_rotating{false};
};

} // namespace tsd::ui::imgui
