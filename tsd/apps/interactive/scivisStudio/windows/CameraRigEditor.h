// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ProjectContext.h"
#include "tsd/ui/imgui/windows/Window.h"

#include <string>

namespace tsd::scivis_studio {

struct CameraRigEditor : public tsd::ui::imgui::Window
{
  CameraRigEditor(
      tsd::ui::imgui::Application *app, ProjectContext *projectContext);
  ~CameraRigEditor() override;

  void buildUI() override;

 private:
  bool inputText(const char *label, std::string &value, size_t capacity = 512);
  void syncSelectionToActiveShot();
  void buildUI_rigControls();
  void buildUI_keyframes(CameraRig &cameraRig);

  ProjectContext *m_projectContext{nullptr};
  int m_selectedRig{0};
  int m_selectedKeyframe{-1};
  ShotID m_lastActiveShotId;
  CameraRigID m_pendingDeleteRig;
};

} // namespace tsd::scivis_studio
