// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/ui/imgui/windows/Window.h"
// std
#include <vector>

namespace tsd::ui::imgui {

// A slim strip of toggle cells, one per supplied window: a cell is highlighted
// when its window is visible; clicking it flips visibility. Borrows the window
// pointers (owned by the app's WindowArray).
struct ViewportRail : public Window
{
  ViewportRail(Application *app,
      std::vector<Window *> viewports,
      const char *name = "Viewports");
  void buildUI() override;

 private:
  std::vector<Window *> m_viewports;
};

} // namespace tsd::ui::imgui
