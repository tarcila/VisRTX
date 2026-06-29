// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/graph/Graph.hpp"
#include "tsd/rendering/bridge/GraphRenderBridge.hpp"
#include "tsd/rendering/pipeline/ImagePipeline.h"
#include "tsd/rendering/pipeline/passes/AnariSceneRenderPass.h"
#include "tsd/rendering/pipeline/passes/CopyToSDLTexturePass.h"
#include "tsd/rendering/view/Manipulator.hpp"
#include "tsd/ui/imgui/windows/Window.h"
// tsd
#include "tsd/core/Token.hpp"
#include "tsd/scene/Object.hpp"
#include "tsd/scene/UpdateDelegate.hpp" // EmptyUpdateDelegate
// anari
#include <anari/anari_cpp.hpp>
// imguizmo
#include "ImGuizmo.h"

namespace tsd::ui::imgui {

// Standalone viewport that renders one of a GraphRenderBridge's per-viewport
// ANARI worlds. Owns its own ANARI camera/renderer + manipulator + pipeline; it
// does NOT use BaseViewport (which is bound to TSD scene cameras/renderers).
struct GraphViewport : public Window
{
  GraphViewport(Application *app,
      tsd::rendering::GraphRenderBridge *bridge,
      int viewportIndex,
      anari::Device device,
      tsd::core::Token deviceName,
      tsd::graph::Graph *graph,
      tsd::graph::NodeId *selected,
      bool *graphDirty,
      const char *name = "Viewport");
  ~GraphViewport() override;

  void buildUI() override;

 private:
  void handleNavigation();
  bool drawGizmo(const ImVec2 &imgPos, const ImVec2 &imgSize);

  int windowFlags() const override;

  void ui_menu_Renderer();
  void
  rebuildRendererObject(); // (re)introspect m_rendererObj + attach delegate
  void reifyRenderer(); // full rebuild: new handle + push all params + commit

  void ui_menu_Device();
  void switchDevice(tsd::core::Token name, anari::Device d);

  // Pushes a single edited renderer param onto the live anari::Renderer.
  // Mirrors MultiDeviceViewport::RendererUpdateDelegate. Subclass
  // EmptyUpdateDelegate (NOT BaseUpdateDelegate, whose methods are pure
  // virtual) so only signalParameterUpdated must be overridden.
  struct RendererUpdateDelegate : public tsd::scene::EmptyUpdateDelegate
  {
    anari::Device *device{nullptr};
    anari::Renderer *renderer{nullptr};
    void signalParameterUpdated(
        const tsd::scene::Object *o, const tsd::scene::Parameter *p) override;
  };

  tsd::rendering::GraphRenderBridge *m_bridge{nullptr};
  int m_viewportIndex{0};
  anari::Device m_device{nullptr};
  tsd::core::Token m_deviceName;
  anari::Camera m_camera{nullptr};
  anari::Renderer m_renderer{nullptr};
  tsd::core::Token m_rendererSubtype{tsd::core::Token("default")};
  tsd::scene::Object m_rendererObj; // editable renderer mirror
  RendererUpdateDelegate m_rud; // reifies edits onto m_renderer
  tsd::rendering::Manipulator m_manip;
  tsd::rendering::UpdateToken m_manipToken{0};
  tsd::rendering::ImagePipeline m_pipeline;
  tsd::rendering::AnariSceneRenderPass *m_anariPass{nullptr};
  tsd::rendering::CopyToSDLTexturePass *m_outputPass{nullptr};
  tsd::math::int2 m_size{0, 0};

  // Camera-navigation state: a single rising-edge latch so startNewRotation()
  // re-arms whenever orbit resumes within one held drag (e.g. orbit → Shift to
  // dolly → release back to orbit).
  bool m_orbiting{false};

  tsd::graph::Graph *m_graph{nullptr};
  tsd::graph::NodeId *m_selected{nullptr};
  bool *m_graphDirty{nullptr};
  ImGuizmo::OPERATION m_gizmoOp{ImGuizmo::TRANSLATE};
  ImGuizmo::MODE m_gizmoMode{ImGuizmo::WORLD};
};

} // namespace tsd::ui::imgui
