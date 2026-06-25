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
      tsd::graph::Graph *graph,
      tsd::graph::NodeId *selected,
      bool *graphDirty,
      const char *name = "Viewport");
  ~GraphViewport() override;

  void buildUI() override;

 private:
  void handleNavigation();
  bool drawGizmo(const ImVec2 &imgPos, const ImVec2 &imgSize);

  tsd::rendering::GraphRenderBridge *m_bridge{nullptr};
  int m_viewportIndex{0};
  anari::Device m_device{nullptr};
  anari::Camera m_camera{nullptr};
  anari::Renderer m_renderer{nullptr};
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
