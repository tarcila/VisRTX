// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/ColorMapUtil.hpp"
// SDL
#include <SDL3/SDL.h>

namespace tsd::ui::imgui {

class Application;

// Embeddable panel (not a Window). Edits a core::TransferFunction's color and
// opacity control points interactively, with a colormap preview strip.
class TFCurveEditor
{
 public:
  explicit TFCurveEditor(Application *app);
  ~TFCurveEditor();

  // Renders the panel for `tf`/`samples`. Sets changed=true if the user edited.
  void draw(tsd::core::TransferFunction &tf, int &samples, bool &changed);

 private:
  void drawPresetCombo(tsd::core::TransferFunction &tf, bool &changed);
  void drawOpacityCurve(tsd::core::TransferFunction &tf, bool &changed);
  void drawColorStops(tsd::core::TransferFunction &tf, bool &changed);
  void refreshPreview(const tsd::core::TransferFunction &tf, int samples);

  Application *m_app{nullptr};
  SDL_Texture *m_preview{nullptr};
  int m_previewWidth{0};
};

} // namespace tsd::ui::imgui
