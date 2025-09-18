// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// tsd_app
#include <tsd/app/Core.h>
// tsd_ui_imgui
#include <tsd/ui/imgui/windows/Window.h>
// tsd_core
#include <tsd/core/Timer.hpp>
// std
#include <string>
#include <vector>

namespace tsd::demo {

struct AnimationControls : public tsd::ui::imgui::Window
{
  AnimationControls(tsd::ui::imgui::Application *app,
      const char *name = "Animation Controls");

  void buildUI() override;

 private:
  void buildUI_incrementAnimation();
  void buildUI_fileSelection();
  void buildUI_animationControls();
  void importAnimation();
  void setTimeStepArray();

  // Data //

  std::string m_filename;
  tsd::core::Timer m_timer;
  bool m_playing{false};
  float m_targetFps{24.f};
  tsd::core::LayerNodeRef m_surfaceNode;
  tsd::core::ObjectUsePtr<tsd::core::Geometry> m_geometry;
  std::vector<tsd::core::ObjectUsePtr<tsd::core::Array>> m_timeSteps;
  int m_currentTimeStep{0};
};

} // namespace tsd::demo
