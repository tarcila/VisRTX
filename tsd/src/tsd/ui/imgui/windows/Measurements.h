// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Window.h"
#include "tsd/ui/imgui/tools/MeasureTool.h"

namespace tsd::ui::imgui {

struct Measurements : public Window
{
  Measurements(Application *app);
  void buildUI() override;
  void setMeasureTool(MeasureTool *tool);

 private:
  MeasureTool *m_tool{nullptr};
};

} // namespace tsd::ui::imgui
