// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/ui/imgui/windows/LayerDebug.hpp"
#include "tsd/scene/Layer.hpp"
#include "tsd/scene/LayerNodeData.hpp"
#include "tsd/scene/Object.hpp"
#include "tsd/ui/imgui/tsd_ui_imgui.h"
// imgui
#include "imgui.h"
// anari
#include <anari/anari_cpp.hpp>
// std
#include <cstdio>
#include <string>

namespace tsd::ui::imgui {

namespace {

std::string stripAnariPrefix(const char *s)
{
  std::string t(s ? s : "");
  const std::string p = "ANARI_";
  if (t.rfind(p, 0) == 0)
    t = t.substr(p.size());
  return t;
}

std::string nodeLabel(const tsd::scene::LayerNodeData &n)
{
  if (n.isObject()) {
    auto *o = n.getObject();
    const std::string sub = o ? std::string(o->subtype().c_str()) : std::string();
    const std::string ty = stripAnariPrefix(anari::toString(n.type()));
    return sub.empty() ? ty : (ty + " : " + sub);
  }
  const std::string &nm = n.name();
  return (nm.empty() ? std::string("node") : nm) + " (transform)";
}

} // namespace

LayerDebug::LayerDebug(
    Application *app, tsd::rendering::GraphRenderBridge *bridge, const char *name)
    : Window(app, name), m_bridge(bridge)
{}

void LayerDebug::buildUI()
{
  if (!m_bridge) {
    ImGui::TextDisabled("No bridge");
    return;
  }

  tsd::scene::Object *selObj = nullptr;

  for (int i = 0; i < m_bridge->numViewports(); ++i) {
    const auto layers = m_bridge->layersForViewport(i);
    if (layers.empty())
      continue;

    char hdr[32];
    std::snprintf(hdr, sizeof(hdr), "Viewport %d", i + 1);
    if (!ImGui::CollapsingHeader(hdr, ImGuiTreeNodeFlags_DefaultOpen))
      continue;

    ImGui::PushID(i);
    for (int L = 0; L < int(layers.size()); ++L) {
      const tsd::scene::Layer *layer = layers[std::size_t(L)];
      if (!layer)
        continue;
      ImGui::PushID(L);
      layer->traverse_const(
          layer->root(), [&](auto &node, int level) -> bool {
            if (level > 0) {
              const std::string label = nodeLabel(*node);
              const bool isSel = (i == m_selViewport && L == m_selLayer
                  && node.index() == m_selNodeIndex);
              ImGui::Indent(INDENT_AMOUNT * level);
              ImGui::PushID(int(node.index()));
              if (ImGui::Selectable(label.c_str(), isSel)) {
                m_selViewport = i;
                m_selLayer = L;
                m_selNodeIndex = node.index();
              }
              ImGui::PopID();
              ImGui::Unindent(INDENT_AMOUNT * level);
              if (isSel && node->isObject())
                selObj = node->getObject();
            }
            return true; // descend into children
          });
      ImGui::PopID();
    }
    ImGui::PopID();
  }

  ImGui::Separator();
  if (selObj) {
    ImGui::BeginDisabled(true);
    tsd::ui::buildUI_object(*selObj, m_bridge->renderScene(), /*useTable=*/true);
    ImGui::EndDisabled();
  } else {
    ImGui::TextDisabled("No object selected");
  }
}

} // namespace tsd::ui::imgui
