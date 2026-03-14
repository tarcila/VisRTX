// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "Measurements.h"
#include "imgui.h"

namespace tsd::ui::imgui {

Measurements::Measurements(Application *app)
    : Window(app, "Measurements")
{}

void Measurements::buildUI()
{
  if (m_tool)
    m_tool->buildUI();
  else
    ImGui::TextDisabled("No measurement tool available.");
}

void Measurements::setMeasureTool(MeasureTool *tool)
{
  m_tool = tool;
}

} // namespace tsd::ui::imgui
