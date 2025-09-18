// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "AnimationControls.h"
// tsd_core
#include <tsd/core/Logging.hpp>
// tsd_io
#include <tsd/io/importers/detail/importer_common.hpp>
// tsd_ui_imgui
#include <tsd/ui/imgui/Application.h>
#include <tsd/ui/imgui/modals/BlockingTaskModal.h>

namespace tsd::demo {

AnimationControls::AnimationControls(
    tsd::ui::imgui::Application *app, const char *name)
    : tsd::ui::imgui::Window(app, name)
{}

void AnimationControls::buildUI()
{
  buildUI_incrementAnimation();

  buildUI_fileSelection();
  ImGui::Separator();
  buildUI_animationControls();
}

void AnimationControls::buildUI_incrementAnimation()
{
  if (!m_playing)
    return;

  m_timer.end();
  if (auto fps = m_timer.perSecond(); m_targetFps >= fps) {
    m_currentTimeStep++;
    m_currentTimeStep = int(m_currentTimeStep % m_timeSteps.size());
    setTimeStepArray();
  }
}

void AnimationControls::buildUI_fileSelection()
{
  constexpr int MAX_LENGTH = 2000;
  m_filename.reserve(MAX_LENGTH);

  static std::string outPath;
  if (ImGui::Button("...")) {
    outPath.clear();
    m_app->getFilenameFromDialog(outPath);
  }

  if (!outPath.empty()) {
    m_filename = outPath;
    outPath.clear();
  }

  ImGui::SameLine();

  auto text_cb = [](ImGuiInputTextCallbackData *cbd) {
    auto &fname = *(std::string *)cbd->UserData;
    fname.resize(cbd->BufTextLen);
    return 0;
  };

  ImGui::InputText("##filename",
      m_filename.data(),
      MAX_LENGTH,
      ImGuiInputTextFlags_CallbackEdit,
      text_cb,
      &m_filename);

  const bool readyToLoad = !m_filename.empty() && m_timeSteps.empty();

  ImGui::BeginDisabled(!readyToLoad);
  if (ImGui::Button("load animation")) {
    auto doLoad = [&]() { importAnimation(); };
    if (!appCore()->windows.taskModal)
      doLoad();
    else {
      appCore()->windows.taskModal->activate(
          doLoad, "Please Wait: Importing Data...");
    }
  }
  ImGui::EndDisabled();

  ImGui::SameLine();

  ImGui::BeginDisabled(m_timeSteps.empty());
  if (ImGui::Button("clear")) {
    m_timeSteps.clear();
    if (m_surfaceNode)
      (*m_surfaceNode)->setEmpty();
    m_currentTimeStep = 0;
    m_geometry.reset();
    auto &scene = appCore()->tsd.scene;
    scene.signalLayerChange(scene.defaultLayer());
    scene.cleanupScene();
  }
  ImGui::EndDisabled();
}

void AnimationControls::buildUI_animationControls()
{
  ImGui::BeginDisabled(m_playing || m_timeSteps.empty());

  if (m_timeSteps.empty()) {
    ImGui::Text("no animation loaded");
    ImGui::EndDisabled();
    return;
  }

  int numTimeSteps = static_cast<int>(m_timeSteps.size());
  if (ImGui::SliderInt("time step", &m_currentTimeStep, 0, numTimeSteps - 1))
    setTimeStepArray();

  ImGui::SameLine();
  ImGui::Text("/ %d", numTimeSteps - 1);

  if (ImGui::Button("play")) {
    m_timer.start();
    m_playing = true;
  }
  ImGui::EndDisabled();

  ImGui::BeginDisabled(!m_playing || m_timeSteps.empty());
  ImGui::SameLine();
  if (ImGui::Button("stop"))
    m_playing = false;
  ImGui::EndDisabled();

  ImGui::SameLine();
  ImGui::DragFloat("fps", &m_targetFps, 0.01f, 1.f, 120.f);
}

void AnimationControls::importAnimation()
{
  auto *core = appCore();
  auto &scene = core->tsd.scene;
  auto *layer = core->tsd.scene.defaultLayer();
  auto importRoot = layer->root();

  // load particle data from file //

  auto *fp = std::fopen(m_filename.c_str(), "rb");
  if (!fp) {
    tsd::core::logError(
        "[import_axyz] could not open file %s", m_filename.c_str());
    return;
  }

  std::string file = tsd::io::fileOf(m_filename);

  uint64_t numTimeSteps = 0;
  uint64_t numParticles = 0;
  auto r = std::fread(&numTimeSteps, sizeof(numTimeSteps), 1, fp);
  r = std::fread(&numParticles, sizeof(numParticles), 1, fp);

  if (numTimeSteps == 0 || numParticles == 0) {
    tsd::core::logError(
        "[import_axyz] animation has no points in '%s'", m_filename.c_str());
    std::fclose(fp);
    return;
  }

  tsd::core::logInfo(
      "[import_axyz] loading [%zu] time steps containing [%zu]"
      " points each from %s",
      numTimeSteps,
      numParticles,
      m_filename.c_str());

  for (int t = 0; t < numTimeSteps; ++t) {
    auto positionsArray = scene.createArray(ANARI_FLOAT32_VEC3, numParticles);
    positionsArray->setName((file + '_' + std::to_string(t)).c_str());
    positionsArray->setData(fp);
    m_timeSteps.emplace_back(positionsArray);
  }

  std::fclose(fp);

  // create TSD objects //

  auto axyz_root = scene.insertChildNode(layer->root());
  (*axyz_root)->name() = "axyz_transform";

  // geometry + material

  auto geom = scene.createObject<tsd::core::Geometry>(
      tsd::core::tokens::geometry::sphere);
  geom->setName("axyz_geometry");
  geom->setParameter("radius", 0.1f); // TODO: something smarter
  geom->setParameterObject("vertex.position", *m_timeSteps[0]);
  m_geometry = geom;

  auto mat = scene.createObject<tsd::core::Material>(
      tsd::core::tokens::material::matte);
  mat->setName("axyz_material");
  mat->setParameter("color", tsd::math::float3(0.8f, 0.8f, 0.8f));

  // surface

  auto surface = scene.createSurface("axyz_surface", geom, mat);
  m_surfaceNode = scene.insertChildObjectNode(axyz_root, surface);

  // Notify scene changed //

  scene.signalLayerChange(layer);
}

void AnimationControls::setTimeStepArray()
{
  if (m_geometry) {
    m_geometry->setParameterObject(
        "vertex.position", *m_timeSteps[m_currentTimeStep]);
  }
}

} // namespace tsd::demo
