// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "AppSettingsDialog.h"
// tsd_ui
#include "tsd/ui/imgui/tsd_ui_imgui.h"

namespace tsd::ui::imgui {

AppSettingsDialog::AppSettingsDialog(Application *app)
    : Modal(app, "AppSettings")
{
  auto *core = appCore();
  if (core->offline.renderer.activeRenderer < 0)
    core->setOfflineRenderingLibrary(core->commandLine.libraryList[0]);
}

void AppSettingsDialog::buildUI()
{
  buildUI_applicationSettings();
  ImGui::Separator();
  buildUI_offlineRenderSettings();
  ImGui::Separator();
  ImGui::NewLine();

  if (ImGui::Button("close") || ImGui::IsKeyDown(ImGuiKey_Escape))
    this->hide();
}

void AppSettingsDialog::applySettings()
{
  auto *core = appCore();

  ImGuiIO &io = ImGui::GetIO();
  io.FontGlobalScale = core->windows.fontScale;

  ImGuiStyle &style = ImGui::GetStyle();
  style.WindowRounding = core->windows.uiRounding;
  style.ChildRounding = core->windows.uiRounding;
  style.FrameRounding = core->windows.uiRounding;
  style.ScrollbarRounding = core->windows.uiRounding;
  style.GrabRounding = core->windows.uiRounding;
  style.PopupRounding = core->windows.uiRounding;
}

void AppSettingsDialog::buildUI_applicationSettings()
{
  auto *core = appCore();

  ImGui::Text("Application Settings:");
  ImGui::Indent(tsd::ui::INDENT_AMOUNT);

  bool doUpdate = false;

  doUpdate |=
      ImGui::DragFloat("font size", &core->windows.fontScale, 0.01f, 0.5f, 4.f);

  doUpdate |=
      ImGui::DragFloat("rounding", &core->windows.uiRounding, 0.01f, 0.f, 12.f);

  bool useFlat = core->anari.useFlatRenderIndex();
  if (ImGui::Checkbox("use flat render index", &useFlat))
    core->anari.setUseFlatRenderIndex(useFlat);

  if (ImGui::IsItemHovered())
    ImGui::SetTooltip("Check this option to bypass instancing of objects.");

  if (doUpdate)
    applySettings();

  ImGui::Unindent(tsd::ui::INDENT_AMOUNT);
}

void AppSettingsDialog::buildUI_offlineRenderSettings()
{
  auto *core = appCore();

  ImGui::Text("Offline Render Settings (tsdRender):");
  ImGui::Indent(tsd::ui::INDENT_AMOUNT);

  // Frame //

  ImGui::Text("Frame:");
  ImGui::DragInt("##width", (int *)&core->offline.frame.width, 1, 10, 10000);
  ImGui::SameLine();
  ImGui::Text("x");
  ImGui::SameLine();
  ImGui::DragInt("##height", (int *)&core->offline.frame.height, 1, 10, 10000);
  ImGui::SameLine();
  ImGui::Text("size");

  ImGui::DragInt("samples",
      (int *)&core->offline.frame.samples,
      1,
      1,
      std::numeric_limits<int>::max());

  // Depth of Field //

  ImGui::Separator();
  ImGui::Text("Depth-of-Field:");
  ImGui::DragFloat("apertureRadius",
      &core->offline.camera.apertureRadius,
      1,
      0.f,
      std::numeric_limits<float>::max());
  ImGui::DragFloat("focusDistance",
      &core->offline.camera.focusDistance,
      1,
      0.f,
      std::numeric_limits<float>::max());

  // Renderer //

  ImGui::Separator();
  ImGui::Text("Renderer:");

  if (ImGui::InputText("##ANARI library",
          &core->offline.renderer.libraryName,
          ImGuiInputTextFlags_EnterReturnsTrue)) {
    core->setOfflineRenderingLibrary(core->offline.renderer.libraryName);
  }

  ImGui::SameLine();
  if (ImGui::BeginCombo("##library_combo", "", ImGuiComboFlags_NoPreview)) {
    for (size_t n = 0; n < core->commandLine.libraryList.size(); n++) {
      if (ImGui::Selectable(core->commandLine.libraryList[n].c_str(), false))
        core->setOfflineRenderingLibrary(core->commandLine.libraryList[n]);
    }
    ImGui::EndCombo();
  }

  ImGui::SameLine();
  ImGui::Text("ANARI library");

  auto comboGetRendererSubtype = [](void *data, int n) -> const char * {
    auto &renderers = *(std::vector<tsd::core::Object> *)data;
    return renderers[n].name().c_str();
  };

  ImGui::Combo("renderer",
      &core->offline.renderer.activeRenderer,
      comboGetRendererSubtype,
      &core->offline.renderer.rendererObjects,
      core->offline.renderer.rendererObjects.size());

  {
    ImGui::Indent(tsd::ui::INDENT_AMOUNT);
    auto &activeRenderer = core->offline.renderer.activeRenderer;
    tsd::ui::buildUI_object(
        core->offline.renderer.rendererObjects[activeRenderer],
        core->tsd.ctx,
        false);
    ImGui::Unindent(tsd::ui::INDENT_AMOUNT);
  }

  ImGui::Unindent(tsd::ui::INDENT_AMOUNT);
}

} // namespace tsd::ui::imgui
