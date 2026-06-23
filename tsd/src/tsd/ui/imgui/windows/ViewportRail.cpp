// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/ui/imgui/windows/ViewportRail.hpp"
// imgui
#include "imgui.h"
// std
#include <cstdio>

namespace tsd::ui::imgui {

ViewportRail::ViewportRail(
    Application *app, std::vector<Window *> viewports, const char *name)
    : Window(app, name), m_viewports(std::move(viewports))
{}

void ViewportRail::buildUI()
{
  for (size_t i = 0; i < m_viewports.size(); ++i) {
    ImGui::PushID(int(i));
    bool *vis = m_viewports[i]->visiblePtr();
    char lbl[8];
    std::snprintf(lbl, sizeof(lbl), "%zu", i + 1);
    // Stacked vertically (no SameLine) → a slim vertical rail.
    if (ImGui::Selectable(
            lbl, *vis, ImGuiSelectableFlags_None, ImVec2(28.f, 28.f)))
      *vis = !*vis;
    ImGui::PopID();
  }
}

} // namespace tsd::ui::imgui
