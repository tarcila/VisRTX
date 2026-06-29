// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/ui/imgui/windows/GraphViewport.hpp"
#include "tsd/graph_nodes/TransformableNode.hpp"
#include "tsd/rendering/view/ManipulatorToAnari.hpp"
#include "tsd/ui/imgui/Application.h"
#include "tsd/ui/imgui/tsd_ui_imgui.h"
// imgui
#include "imgui.h"
// anari
#include <anari/anari_cpp/ext/linalg.h>
// std
#include <algorithm>

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
    tsd::core::Token deviceName,
    tsd::graph::Graph *graph,
    tsd::graph::NodeId *selected,
    bool *graphDirty,
    const char *name)
    : Window(app, name),
      m_bridge(bridge),
      m_viewportIndex(viewportIndex),
      m_device(device),
      m_deviceName(deviceName),
      m_graph(graph),
      m_selected(selected),
      m_graphDirty(graphDirty)
{
  // Camera + renderer on the bridge's device.
  m_camera = anari::newObject<anari::Camera>(m_device, "perspective");
  anari::commitParameters(m_device, m_camera);
  rebuildRendererObject(); // introspected defaults; no hard-coded
                           // ambientRadiance

  m_manip.setConfig(float3(0.f, 0.f, 0.f), kInitialViewDistance);

  // Pipeline: render the (external) world, then copy to an SDL texture.
  m_anariPass =
      m_pipeline.emplace_back<tsd::rendering::AnariSceneRenderPass>(m_device);
  m_anariPass->setCamera(m_camera);
  m_anariPass->setRunAsync(false);
  reifyRenderer(); // builds the live renderer + sets it on m_anariPass
  m_outputPass = m_pipeline.emplace_back<tsd::rendering::CopyToSDLTexturePass>(
      m_app->sdlRenderer());
}

int GraphViewport::windowFlags() const
{
  return Window::windowFlags() | ImGuiWindowFlags_MenuBar;
}

void GraphViewport::rebuildRendererObject()
{
  m_rendererObj = tsd::scene::parseANARIObjectInfo(
      m_device, ANARI_RENDERER, m_rendererSubtype.c_str());
  // Point the delegate at this viewport's live device/renderer members (their
  // addresses are stable; values are refreshed on each reify / device switch).
  m_rud.device = &m_device;
  m_rud.renderer = &m_renderer;
  m_rendererObj.setUpdateDelegate(&m_rud);
}

void GraphViewport::reifyRenderer()
{
  if (m_renderer)
    anari::release(m_device, m_renderer);
  m_renderer =
      anari::newObject<anari::Renderer>(m_device, m_rendererSubtype.c_str());
  m_rendererObj.updateAllANARIParameters(m_device, m_renderer);
  anari::commitParameters(m_device, m_renderer);
  if (m_anariPass)
    m_anariPass->setRenderer(m_renderer);
}

void GraphViewport::RendererUpdateDelegate::signalParameterUpdated(
    const tsd::scene::Object *o, const tsd::scene::Parameter *p)
{
  if (!device || !renderer || !*renderer)
    return;
  o->updateANARIParameter(*device, *renderer, *p, p->name().c_str());
  anari::commitParameters(*device, *renderer);
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
  if (ImGui::BeginMenuBar()) {
    ui_menu_Device();
    ui_menu_Renderer();
    ImGui::EndMenuBar();
  }

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

  const ImVec2 pos = ImGui::GetCursorScreenPos();
  const ImVec2 imgSize(float(m_size.x), float(m_size.y));
  // Blit the rendered texture first. AddImage is draw-list only: it does not
  // advance the cursor or claim input, so the cursor stays at pos for the
  // InvisibleButton below.
  ImGui::GetWindowDrawList()->AddImage((ImTextureID)m_outputPass->getTexture(),
      pos,
      ImVec2(pos.x + imgSize.x, pos.y + imgSize.y),
      ImVec2(0, 1),
      ImVec2(1, 0));

  // Run the gizmo first; ImGuizmo hit-tests and captures the mouse internally.
  const bool gizmoActive = drawGizmo(pos, imgSize);

  // Camera drag-nav only when the gizmo is not hot. The InvisibleButton owns
  // ImGui's ActiveId only when a press lands inside it, so window-decoration
  // drags and out-of-rect presses never manipulate; and when the gizmo is hot
  // no button is submitted to steal its press.
  if (!gizmoActive) {
    ImGui::InvisibleButton("##viewport",
        imgSize,
        ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight
            | ImGuiButtonFlags_MouseButtonMiddle);
    handleNavigation();
  } else {
    m_orbiting = false; // gizmo owns the frame; drop any orbit latch
  }

  // Wheel zoom works anywhere over the viewport (ImGuizmo ignores the wheel),
  // suppressed only during an active gizmo drag.
  ImGuiIO &io = ImGui::GetIO();
  if (ImGui::IsWindowHovered() && !ImGuizmo::IsUsing() && io.MouseWheel != 0.f)
    m_manip.zoom(io.MouseWheel * kWheelZoomScale);
}

bool GraphViewport::drawGizmo(const ImVec2 &imgPos, const ImVec2 &imgSize)
{
  if (!m_selected || *m_selected == tsd::graph::INVALID_NODE || !m_graph)
    return false;
  auto *gn = m_graph->node(*m_selected);
  if (!gn || !gn->impl)
    return false;
  auto *itf =
      dynamic_cast<tsd::graph_nodes::ITransformableNode *>(gn->impl.get());
  if (!itf)
    return false;
  // Only show the gizmo if this display is masked into this viewport.
  const int mask =
      gn->impl->parameters().getOr<int>(tsd::core::Token("viewportMask"), 0);
  if (!((mask >> m_viewportIndex) & 1))
    return false;

  const float3 eye = m_manip.eye(), at = m_manip.at(), up = m_manip.up();
  const auto view = linalg::lookat_matrix(eye, at, up);
  const float aspect = float(m_size.x) / float(m_size.y);
  constexpr float kFovy =
      1.04719755f; // π/3 — the ANARI/VisRTX perspective default
  const float focusDist = std::max(linalg::length(at - eye), 1e-4f);
  const float near = std::max(0.01f * focusDist, 1e-3f);
  const float far = 100.f * focusDist + 10.f;
  const float oneOverTanFov = 1.f / std::tan(kFovy * 0.5f);
  // Column-major perspective (matches BaseViewport's manual construction).
  const tsd::math::mat4 proj{
      {oneOverTanFov / aspect, 0.f, 0.f, 0.f},
      {0.f, oneOverTanFov, 0.f, 0.f},
      {0.f, 0.f, -(far + near) / (far - near), -1.f},
      {0.f, 0.f, -2.f * far * near / (far - near), 0.f},
  };

  ImGuizmo::BeginFrame();
  ImGuizmo::SetOrthographic(false);
  ImGuizmo::SetDrawlist();
  ImGuizmo::SetRect(imgPos.x, imgPos.y, imgSize.x, imgSize.y);

  tsd::math::mat4 m = itf->transform();
  if (ImGuizmo::Manipulate(
          &view[0].x, &proj[0].x, m_gizmoOp, m_gizmoMode, &m[0].x)) {
    itf->transform() = m;
    *m_graphDirty = true;
  }
  return ImGuizmo::IsUsing() || ImGuizmo::IsOver();
}

void GraphViewport::ui_menu_Renderer()
{
  if (!ImGui::BeginMenu("Renderer"))
    return;

  const auto subtypes =
      tsd::scene::getANARIObjectSubtypes(m_device, ANARI_RENDERER);
  if (subtypes.size() > 1) {
    ImGui::Text("Subtype:");
    for (size_t i = 0; i < subtypes.size(); ++i) {
      ImGui::PushID(int(i));
      const bool selected =
          m_rendererSubtype == tsd::core::Token(subtypes[i].c_str());
      if (ImGui::RadioButton(subtypes[i].c_str(), selected) && !selected) {
        m_rendererSubtype = tsd::core::Token(subtypes[i].c_str());
        rebuildRendererObject();
        reifyRenderer();
      }
      ImGui::PopID();
    }
    ImGui::Separator();
  }

  ImGui::Text("Parameters:");
  // The scene arg only resolves object-reference params; the standalone
  // renderer object has none, but buildUI_object requires a scene handle.
  tsd::ui::buildUI_object(
      m_rendererObj, m_bridge->renderScene(), /*useTable=*/true);

  ImGui::EndMenu();
}

void GraphViewport::ui_menu_Device()
{
  if (!ImGui::BeginMenu("Device"))
    return;

  for (const auto &name : appContext()->anari.libraryList()) {
    const bool selected = m_deviceName == tsd::core::Token(name.c_str());
    if (ImGui::RadioButton(name.c_str(), selected) && !selected) {
      // loadDevice ALWAYS retains (both create and cache-hit paths), so this
      // call returns a +1 handle. switchDevice's pass + bridge index each take
      // their own retain; release this transient menu ref afterward (mirrors
      // ANARIDeviceManager::loadDeviceExtensions). The manager keeps the device
      // cached/alive regardless.
      anari::Device d = appContext()->anari.loadDevice(name); // may be null
      if (d && d != m_device) {
        switchDevice(tsd::core::Token(name.c_str()), d);
        anari::release(d, d);
      } else if (d) {
        anari::release(d, d); // already our device: drop the transient ref
      }
    }
  }

  ImGui::EndMenu();
}

void GraphViewport::switchDevice(tsd::core::Token name, anari::Device d)
{
  // 1) Release this viewport's own handles on the old device.
  if (m_camera)
    anari::release(m_device, m_camera);
  if (m_renderer)
    anari::release(m_device, m_renderer);
  m_camera = nullptr;
  m_renderer = nullptr;

  // 2) Adopt the new device.
  m_device = d;
  m_deviceName = name;

  // 3) Rebuild the pipeline on the new device.
  m_pipeline.clear();
  m_anariPass =
      m_pipeline.emplace_back<tsd::rendering::AnariSceneRenderPass>(m_device);
  m_anariPass->setRunAsync(false);
  m_outputPass = m_pipeline.emplace_back<tsd::rendering::CopyToSDLTexturePass>(
      m_app->sdlRenderer());
  if (m_size.x > 0 && m_size.y > 0)
    m_pipeline.setDimensions(uint32_t(m_size.x), uint32_t(m_size.y));

  // 4) Recreate camera + renderer on the new device.
  m_camera = anari::newObject<anari::Camera>(m_device, "perspective");
  anari::commitParameters(m_device, m_camera);
  m_anariPass->setCamera(m_camera);
  rebuildRendererObject();
  reifyRenderer();

  // 5) Tell the bridge to rebuild this viewport's world on the new device.
  m_bridge->setViewportDevice(m_viewportIndex, m_deviceName, m_device);

  // 6) Force the buildUI size block to re-apply aspect/dimensions and the
  //    manipulator to re-push camera params against the fresh camera.
  m_manipToken = 0;
  m_size = tsd::math::int2(0, 0);
}

void GraphViewport::handleNavigation()
{
  // Manipulate only while the viewport's InvisibleButton is held — i.e. the
  // press landed inside the viewport rect. Title-bar drags and presses that
  // started elsewhere never set this item active, so they never manipulate.
  if (!ImGui::IsItemActive()) {
    m_orbiting = false;
    return;
  }

  ImGuiIO &io = ImGui::GetIO();
  const bool dolly = ImGui::IsMouseDown(ImGuiMouseButton_Right)
      || (ImGui::IsMouseDown(ImGuiMouseButton_Left)
          && ImGui::IsKeyDown(ImGuiKey_LeftShift));
  const bool pan = ImGui::IsMouseDown(ImGuiMouseButton_Middle)
      || (ImGui::IsMouseDown(ImGuiMouseButton_Left)
          && ImGui::IsKeyDown(ImGuiKey_LeftAlt));
  const bool orbit =
      ImGui::IsMouseDown(ImGuiMouseButton_Left) && !dolly && !pan;

  // Re-baseline rotation on the rising edge of orbit (including resume after a
  // dolly/pan interlude in the same held drag), or the view jumps.
  if (orbit && !m_orbiting)
    m_manip.startNewRotation();
  m_orbiting = orbit;

  const float2 delta =
      float2(io.MouseDelta.x, io.MouseDelta.y) * 2.f / float2(m_size);
  if (delta == float2(0.f))
    return;
  if (orbit)
    m_manip.rotate(delta);
  else if (dolly)
    m_manip.zoom(delta.y);
  else if (pan)
    m_manip.pan(delta);
}

} // namespace tsd::ui::imgui
