// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/ui/imgui/windows/GraphViewport.hpp"
#include "tsd/rendering/view/ManipulatorToAnari.hpp"
#include "tsd/ui/imgui/Application.h"
// imgui
#include "imgui.h"
// anari
#include <anari/anari_cpp/ext/linalg.h>

namespace tsd::ui::imgui {

using float2 = anari::math::float2;
using float3 = anari::math::float3;

// Frame the demo content (field spans ~[-1,1]) at a comfortable distance.
constexpr float kInitialViewDistance = 3.f;
// Distance change per scroll-wheel notch is m_speed * this (m_speed default
// 0.25), kept gentle so a notch nudges rather than jumps the view.
constexpr float kWheelZoomScale = 1.f;

GraphViewport::GraphViewport(Application *app,
    tsd::rendering::GraphRenderBridge *bridge,
    int viewportIndex,
    anari::Device device,
    const char *name)
    : Window(app, name),
      m_bridge(bridge),
      m_viewportIndex(viewportIndex),
      m_device(device)
{
  // Camera + renderer on the bridge's device.
  m_camera = anari::newObject<anari::Camera>(m_device, "perspective");
  anari::commitParameters(m_device, m_camera);
  m_renderer = anari::newObject<anari::Renderer>(m_device, "default");
  anari::setParameter(m_device, m_renderer, "ambientRadiance", 1.f);
  anari::commitParameters(m_device, m_renderer);

  m_manip.setConfig(float3(0.f, 0.f, 0.f), kInitialViewDistance);

  // Pipeline: render the (external) world, then copy to an SDL texture.
  m_anariPass =
      m_pipeline.emplace_back<tsd::rendering::AnariSceneRenderPass>(m_device);
  m_anariPass->setCamera(m_camera);
  m_anariPass->setRenderer(m_renderer);
  m_anariPass->setRunAsync(false);
  m_outputPass = m_pipeline.emplace_back<tsd::rendering::CopyToSDLTexturePass>(
      m_app->sdlRenderer());
}

GraphViewport::~GraphViewport()
{
  if (m_camera)
    anari::release(m_device, m_camera);
  if (m_renderer)
    anari::release(m_device, m_renderer);
  // m_pipeline owns its passes; m_device is owned by the app/bridge.
}

void GraphViewport::buildUI()
{
  const ImVec2 avail = ImGui::GetContentRegionAvail();
  const tsd::math::int2 size{int(avail.x), int(avail.y)};
  if (size.x > 0 && size.y > 0 && (size.x != m_size.x || size.y != m_size.y)) {
    m_size = size;
    m_pipeline.setDimensions(uint32_t(size.x), uint32_t(size.y));
    anari::setParameter(
        m_device, m_camera, "aspect", float(size.x) / float(size.y));
    anari::commitParameters(m_device, m_camera);
  }
  if (m_size.x <= 0 || m_size.y <= 0)
    return;

  // Drive the camera from the current world + manipulator.
  if (m_manip.hasChanged(m_manipToken)) {
    tsd::rendering::updateCameraParametersPerspective(
        m_device, m_camera, m_manip);
    anari::commitParameters(m_device, m_camera);
  }
  m_anariPass->setWorld(m_bridge->world(m_viewportIndex));
  m_pipeline.render();

  // Reserve the viewport area with an invisible button BEFORE blitting: while
  // held it owns ImGui's ActiveId, so drags are consumed as navigation rather
  // than moving the dock/window. The rendered texture is then drawn into the
  // same rect via the window draw list.
  const ImVec2 pos = ImGui::GetCursorScreenPos();
  const ImVec2 imgSize(float(m_size.x), float(m_size.y));
  ImGui::InvisibleButton("##viewport",
      imgSize,
      ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight
          | ImGuiButtonFlags_MouseButtonMiddle);
  ImGui::GetWindowDrawList()->AddImage((ImTextureID)m_outputPass->getTexture(),
      pos,
      ImVec2(pos.x + imgSize.x, pos.y + imgSize.y),
      ImVec2(0, 1),
      ImVec2(1, 0));

  handleNavigation();
}

// Left-drag orbits, right-drag (or Left+Shift) dollies, middle-drag (or
// Left+Alt) pans, scroll wheel zooms. Mouse deltas are normalized to
// screen-fraction units before reaching the Manipulator, matching BaseViewport
// — raw pixel deltas would be ~100x too fast (Manipulator::rotate scales by
// 100).
void GraphViewport::handleNavigation()
{
  const bool hovered = ImGui::IsItemHovered();
  ImGuiIO &io = ImGui::GetIO();

  const bool dolly = ImGui::IsMouseDown(ImGuiMouseButton_Right)
      || (ImGui::IsMouseDown(ImGuiMouseButton_Left)
          && ImGui::IsKeyDown(ImGuiKey_LeftShift));
  const bool pan = ImGui::IsMouseDown(ImGuiMouseButton_Middle)
      || (ImGui::IsMouseDown(ImGuiMouseButton_Left)
          && ImGui::IsKeyDown(ImGuiKey_LeftAlt));
  const bool orbit =
      ImGui::IsMouseDown(ImGuiMouseButton_Left) && !dolly && !pan;

  const bool anyMovement = dolly || pan || orbit;
  if (!anyMovement) {
    m_manipulating = false;
    m_prevMouse = float2(-1.f);
  } else if (hovered && !m_manipulating) {
    m_manipulating = true;
  }
  if (m_rotating && !orbit)
    m_rotating = false;

  if (m_manipulating) {
    const float2 mouse(io.MousePos.x, io.MousePos.y);
    if (m_prevMouse != float2(-1.f)) {
      const float2 delta = (mouse - m_prevMouse) * 2.f / float2(m_size);
      if (delta != float2(0.f)) {
        if (orbit) {
          if (!m_rotating) {
            m_manip.startNewRotation();
            m_rotating = true;
          }
          m_manip.rotate(delta);
        } else if (dolly)
          m_manip.zoom(delta.y);
        else if (pan)
          m_manip.pan(delta);
      }
    }
    m_prevMouse = mouse;
  }

  if (hovered && io.MouseWheel != 0.f)
    m_manip.zoom(io.MouseWheel * kWheelZoomScale);
}

} // namespace tsd::ui::imgui
