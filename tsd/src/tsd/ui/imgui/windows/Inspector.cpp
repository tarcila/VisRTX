// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/ui/imgui/windows/Inspector.hpp"
#include "tsd/graph_nodes/DisplayMask.hpp"
#include "tsd/graph_nodes/TransferFunctionNode.hpp"
#include "tsd/graph_nodes/TransformableNode.hpp"
#include "tsd/ui/imgui/Application.h"
// imgui
#include "ImGuizmo.h"
#include "imgui.h"
// anari
#include <anari/anari.h>
// std
#include <string>

namespace tsd::ui::imgui {

using namespace tsd::graph;
using tsd::core::Token;

Inspector::Inspector(Application *app,
    Graph *graph,
    NodeId *selected,
    bool *graphDirty,
    const char *name)
    : Window(app, name),
      m_graph(graph),
      m_selected(selected),
      m_graphDirty(graphDirty)
{}

void Inspector::drawParameters(NodeId id)
{
  auto *gn = m_graph->node(id);
  if (!gn || !gn->impl)
    return;
  auto &params = gn->impl->parameters();

  // Iterate the live items() list; write back via set on change.
  for (const auto &p : params.items()) {
    const Token name = p.name;
    const auto t = p.value.type();
    ImGui::PushID(name.c_str());
    if (name == tsd::core::Token("viewportMask")) {
      int mask = p.value.get<int>();
      ImGui::TextUnformatted("Viewports");
      bool changed = false;
      for (int i = 0; i < tsd::graph_nodes::kMaxViewports; ++i) {
        ImGui::PushID(i);
        if (i != 0)
          ImGui::SameLine();
        const bool on = (mask >> i) & 1;
        char lbl[8];
        std::snprintf(lbl, sizeof(lbl), "%d", i + 1);
        if (ImGui::Selectable(
                lbl, on, ImGuiSelectableFlags_None, ImVec2(24.f, 0.f))) {
          mask ^= (1 << i); // Selectable returns true on click → flip the bit
          changed = true;
        }
        ImGui::PopID();
      }
      if (changed) {
        params.set(name, mask);
        *m_graphDirty = true;
      }
    } else if (t == ANARI_BOOL) {
      bool v = p.value.get<bool>();
      if (ImGui::Checkbox(name.c_str(), &v)) {
        params.set(name, v);
        *m_graphDirty = true;
      }
    } else if (t == ANARI_FLOAT32) {
      float v = p.value.get<float>();
      if (ImGui::InputFloat(name.c_str(),
              &v,
              0.f,
              0.f,
              "%.4f",
              ImGuiInputTextFlags_EnterReturnsTrue)) {
        params.set(name, v);
        *m_graphDirty = true;
      }
    } else if (t == ANARI_INT32) {
      int v = p.value.get<int>();
      if (ImGui::InputInt(
              name.c_str(), &v, 1, 10, ImGuiInputTextFlags_EnterReturnsTrue)) {
        params.set(name, v);
        *m_graphDirty = true;
      }
    } else if (t == ANARI_STRING) {
      std::string s = p.value.getString();
      char buf[256];
      std::snprintf(buf, sizeof(buf), "%s", s.c_str());
      if (ImGui::InputText(name.c_str(),
              buf,
              sizeof(buf),
              ImGuiInputTextFlags_EnterReturnsTrue)) {
        params.set(name, static_cast<const char *>(buf));
        *m_graphDirty = true;
      }
    } else {
      ImGui::Text("%s (unsupported type)", name.c_str());
    }
    ImGui::PopID();
  }
}

void Inspector::buildUI()
{
  if (*m_selected == INVALID_NODE) {
    ImGui::TextDisabled("No selection");
    return;
  }
  auto *gn = m_graph->node(*m_selected);
  if (!gn || !gn->impl) {
    ImGui::TextDisabled("No selection");
    return;
  }
  ImGui::Text("%s", gn->impl->typeInfo().name.c_str());
  ImGui::Separator();
  if (auto *itf = dynamic_cast<tsd::graph_nodes::ITransferFunctionNode *>(
          gn->impl.get())) {
    if (!m_tfEditor)
      m_tfEditor = std::make_unique<TFCurveEditor>(m_app);
    bool changed = false;
    m_tfEditor->draw(itf->tfState(), itf->samples(), changed);
    if (changed) {
      m_graph->markDirty(*m_selected);
      *m_graphDirty = true;
    }
  } else {
    drawParameters(*m_selected);
  }
  if (auto *it = dynamic_cast<tsd::graph_nodes::ITransformableNode *>(
          gn->impl.get())) {
    ImGui::Separator();
    ImGui::TextUnformatted("Transform");
    tsd::core::math::mat4 &m = it->transform();
    float t[3], r[3], s[3];
    ImGuizmo::DecomposeMatrixToComponents(&m[0].x, t, r, s);
    bool changed = false;
    changed |= ImGui::DragFloat3("Translate", t, 0.01f);
    changed |= ImGui::DragFloat3("Rotate", r, 0.5f);
    changed |= ImGui::DragFloat3("Scale", s, 0.01f);
    if (changed) {
      ImGuizmo::RecomposeMatrixFromComponents(t, r, s, &m[0].x);
      *m_graphDirty = true; // NOTE: NO m_graph->markDirty — transform is
                            // render-routing, not node data
    }
    if (ImGui::Button("Reset##transform")) {
      m = tsd::core::math::IDENTITY_MAT4;
      *m_graphDirty = true;
    }
  }
}

} // namespace tsd::ui::imgui
