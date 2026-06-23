// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/ui/imgui/windows/Inspector.hpp"
#include "tsd/ui/imgui/Application.h"
// imgui
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

  // Iterate a snapshot of names+types; write back via set on change.
  for (const auto &p : params.items()) {
    const Token name = p.name;
    const auto t = p.value.type();
    ImGui::PushID(name.c_str());
    if (t == ANARI_BOOL) {
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
  drawParameters(*m_selected);
}

} // namespace tsd::ui::imgui
