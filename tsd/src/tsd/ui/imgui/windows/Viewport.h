// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Window.h"

#include "tsd/ui/imgui/tsd_ui_imgui.h"
// tsd_core
#include "tsd/core/scene/Object.hpp"
#include "tsd/core/scene/UpdateDelegate.hpp"
// tsd_rendering
#include "tsd/rendering/index/RenderIndex.hpp"
#include "tsd/rendering/pipeline/RenderPipeline.h"
#include "tsd/rendering/view/Manipulator.hpp"
// std
#include <array>
#include <functional>
#include <future>
#include <limits>

namespace tsd::ui::imgui {

using ViewportDeviceChangeCb = std::function<void(const std::string &)>;

struct Viewport : public Window
{
  Viewport(Application *app,
      tsd::rendering::Manipulator *m,
      const char *name = "Viewport");
  ~Viewport();

  void buildUI() override;
  void setManipulator(tsd::rendering::Manipulator *m);
  void resetView(bool resetAzEl = true);
  void centerView();
  void setLibrary(const std::string &libName, bool doAsync = true);
  void setDeviceChangeCb(ViewportDeviceChangeCb cb);
  void setExternalInstances(
      const anari::Instance *instances = nullptr, size_t count = 0);

 private:
  void saveSettings(tsd::core::DataNode &thisWindowRoot) override;
  void loadSettings(tsd::core::DataNode &thisWindowRoot) override;

  void loadANARIRendererParameters(anari::Device d);
  void updateAllRendererParameters(anari::Device d);

  void setupRenderPipeline();
  void teardownDevice();
  void reshape(tsd::math::int2 newWindowSize);
  void pick(tsd::math::int2 location, bool selectObject);
  void setSelectionVisibilityFilterEnabled(bool enabled);

  void updateFrame();
  void updateCamera(bool force = false);
  void updateImage();

  void echoCameraConfig();
  void ui_menubar();
  void ui_handleInput();
  bool ui_picking();
  void ui_overlay();

  int windowFlags() const override; // anari_viewer::Window

  // Data /////////////////////////////////////////////////////////////////////

  ViewportDeviceChangeCb m_deviceChangeCb;
  float m_timeToLoadDevice{0.f};
  std::future<void> m_initFuture;
  bool m_deviceReadyToUse{false};
  std::string m_libName;
  tsd::rendering::RenderIndex *m_rIdx{nullptr};

  tsd::math::float2 m_previousMouse{-1.f, -1.f};
  bool m_mouseRotating{false};
  bool m_manipulating{false};
  bool m_frameCancelled{false};
  bool m_saveNextFrame{false};
  bool m_echoCameraConfig{false};
  int m_screenshotIndex{0};

  bool m_showOverlay{true};
  bool m_showCameraInfo{false};
  bool m_highlightSelection{true};
  bool m_showOnlySelected{false};
  int m_frameSamples{0};

  bool m_visualizeDepth{false};
  bool m_showAxes{true};
  float m_depthVisualMaximum{1.f};

  float m_fov{40.f};

  // Picking state //

  bool m_selectObjectNextPick{false};
  tsd::math::int2 m_pickCoord{0, 0};
  float m_pickedDepth{0.f};

  // ANARI objects //

  anari::DataType m_format{ANARI_UFIXED8_RGBA_SRGB};

  anari::Extensions m_extensions{};
  anari::Device m_device{nullptr};
  anari::Camera m_currentCamera{nullptr};
  anari::Camera m_perspCamera{nullptr};
  anari::Camera m_orthoCamera{nullptr};
  anari::Camera m_omniCamera{nullptr};

  std::vector<anari::Renderer> m_renderers;
  std::vector<tsd::core::Object> m_rendererObjects;
  int m_currentRenderer{0};

  struct RendererUpdateDelegate : public tsd::core::EmptyUpdateDelegate
  {
    void signalParameterUpdated(
        const tsd::core::Object *o, const tsd::core::Parameter *p) override;
    anari::Device d{nullptr};
    anari::Renderer r{nullptr};
  } m_rud;

  // camera manipulator

  int m_arcballUp{1};
  tsd::rendering::Manipulator m_localArcball;
  tsd::rendering::Manipulator *m_arcball{nullptr};
  tsd::rendering::UpdateToken m_cameraToken{0};
  float m_apertureRadius{0.f};
  float m_focusDistance{1.f};

  // display

  tsd::rendering::RenderPipeline m_pipeline;
  tsd::rendering::AnariSceneRenderPass *m_anariPass{nullptr};
  tsd::rendering::PickPass *m_pickPass{nullptr};
  tsd::rendering::VisualizeDepthPass *m_visualizeDepthPass{nullptr};
  tsd::rendering::OutlineRenderPass *m_outlinePass{nullptr};
  tsd::rendering::AnariAxesRenderPass *m_axesPass{nullptr};
  tsd::rendering::CopyToSDLTexturePass *m_outputPass{nullptr};

  tsd::math::int2 m_viewportSize{0, 0};
  tsd::math::int2 m_renderSize{0, 0};
  float m_resolutionScale{1.f};

  float m_latestFL{0.f};
  float m_latestAnariFL{0.f};
  float m_minFL{std::numeric_limits<float>::max()};
  float m_maxFL{-std::numeric_limits<float>::max()};

  std::string m_overlayWindowName;
};

} // namespace tsd::ui::imgui
