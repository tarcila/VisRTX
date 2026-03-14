// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "MeasureTool.h"
#include "imgui.h"
// std
#include <algorithm>
#include <cmath>
#include <cstdio>

namespace tsd::ui::imgui {

// vector2d vendor extension type
static constexpr ANARIDataType ANARI_RASTER = (ANARIDataType)0x7F00;

static constexpr float LINE_WIDTH = 2.f;
static constexpr float TEXT_HEIGHT = 16.f;
static constexpr float TEXT_OFFSET_X = 5.f;
static constexpr float TEXT_OFFSET_Y = -8.f;

static ANARIObject newRaster(ANARIDevice d)
{
  return anariNewObject(d, "raster", "default");
}

static float length(tsd::math::float3 v)
{
  return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

// MeasureTool definitions ////////////////////////////////////////////////////

MeasureTool::MeasureTool(anari::Device overlayDevice)
    : m_device(overlayDevice)
{
  anari::retain(m_device, m_device);

  m_segmentGeo = anariNewGeometry(m_device, "segment");
  m_textGeo = anariNewGeometry(m_device, "text");
  float height = TEXT_HEIGHT;
  anariSetParameter(m_device, m_textGeo, "height", ANARI_FLOAT32, &height);
  float offset[] = {TEXT_OFFSET_X, TEXT_OFFSET_Y};
  anariSetParameter(m_device, m_textGeo, "offset", ANARI_FLOAT32_VEC2, offset);

  m_segmentRaster = newRaster(m_device);
  anariSetParameter(
      m_device, m_segmentRaster, "geometry", ANARI_GEOMETRY, &m_segmentGeo);
  anariCommitParameters(m_device, m_segmentRaster);

  m_textRaster = newRaster(m_device);
  anariSetParameter(
      m_device, m_textRaster, "geometry", ANARI_GEOMETRY, &m_textGeo);
  anariCommitParameters(m_device, m_textRaster);

  m_group = anariNewObject(m_device, "group", nullptr);
  ANARIObject rasters[] = {m_segmentRaster, m_textRaster};
  auto rasterArray = anariNewArray1D(
      m_device, rasters, nullptr, nullptr, ANARI_RASTER, 2);
  anariSetParameter(m_device, m_group, "raster", ANARI_ARRAY1D, &rasterArray);
  anariCommitParameters(m_device, m_group);
  anariRelease(m_device, rasterArray);

  m_instance = anariNewObject(m_device, "instance", nullptr);
  anariSetParameter(m_device, m_instance, "group", ANARI_GROUP, &m_group);
  anariCommitParameters(m_device, m_instance);

  m_world = anari::newObject<anari::World>(m_device);
  auto instanceArray = anariNewArray1D(
      m_device, &m_instance, nullptr, nullptr, ANARI_INSTANCE, 1);
  anariSetParameter(
      m_device, m_world, "instance", ANARI_ARRAY1D, &instanceArray);
  anariCommitParameters(m_device, m_world);
  anariRelease(m_device, instanceArray);
}

MeasureTool::~MeasureTool()
{
  anariRelease(m_device, m_world);
  anariRelease(m_device, m_instance);
  anariRelease(m_device, m_group);
  anariRelease(m_device, m_textRaster);
  anariRelease(m_device, m_segmentRaster);
  anariRelease(m_device, m_textGeo);
  anariRelease(m_device, m_segmentGeo);
  anariRelease(m_device, m_device);
}

void MeasureTool::setPointA(tsd::math::float3 pos)
{
  m_pendingA = pos;
  m_state = State::PICKED_A;
  rebuildWorld();
}

void MeasureTool::setPointB(tsd::math::float3 pos)
{
  Measurement m;
  char name[16];
  std::snprintf(name, sizeof(name), "M%d", m_nextId++);
  m.name = name;
  m.pointA = m_pendingA;
  m.pointB = pos;
  m.distance = length(pos - m_pendingA);
  m_measurements.push_back(std::move(m));

  m_state = State::IDLE;
  rebuildWorld();
}

void MeasureTool::cancelPick()
{
  m_state = State::IDLE;
  rebuildWorld();
}

void MeasureTool::remove(size_t index)
{
  if (index < m_measurements.size()) {
    m_measurements.erase(m_measurements.begin() + index);
    rebuildWorld();
  }
}

void MeasureTool::removeAll()
{
  m_measurements.clear();
  m_state = State::IDLE;
  rebuildWorld();
}

MeasureTool::State MeasureTool::state() const
{
  return m_state;
}

tsd::math::float3 MeasureTool::pendingPointA() const
{
  return m_pendingA;
}

const std::vector<Measurement> &MeasureTool::measurements() const
{
  return m_measurements;
}

anari::World MeasureTool::world() const
{
  return m_world;
}

void MeasureTool::buildUI()
{
  for (size_t i = 0; i < m_measurements.size(); i++) {
    auto &m = m_measurements[i];
    ImGui::PushID(static_cast<int>(i));

    bool open = ImGui::TreeNodeEx(m.name.c_str(),
        ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_AllowOverlap);

    ImGui::SameLine();
    ImGui::TextColored(ImVec4(1.f, 1.f, 0.f, 1.f), "%.4f", m.distance);

    ImGui::SameLine(ImGui::GetContentRegionAvail().x - 20.f);
    if (ImGui::SmallButton("X")) {
      remove(i);
      if (open)
        ImGui::TreePop();
      ImGui::PopID();
      break;
    }

    if (open) {
      ImGui::Text("A: (%.3f, %.3f, %.3f)",
          m.pointA.x, m.pointA.y, m.pointA.z);
      ImGui::Text("B: (%.3f, %.3f, %.3f)",
          m.pointB.x, m.pointB.y, m.pointB.z);
      ImGui::TreePop();
    }

    ImGui::PopID();
  }

  if (m_measurements.empty() && m_state == State::IDLE) {
    ImGui::TextDisabled("No measurements. Click to measure.");
  } else if (m_state == State::PICKED_A) {
    ImGui::TextColored(
        ImVec4(0.f, 1.f, 1.f, 1.f), "Click to set point B...");
  }

  ImGui::Separator();
  if (ImGui::Button("Clear All") && !m_measurements.empty())
    removeAll();
}

void MeasureTool::rebuildWorld()
{
  float color[] = {0.f, 1.f, 1.f, 1.f};

  // Count total segments: one per completed measurement + one for pending pick
  size_t segCount = m_measurements.size();
  bool hasPending = (m_state == State::PICKED_A);
  if (hasPending)
    segCount++;

  if (segCount == 0) {
    anariUnsetParameter(m_device, m_segmentGeo, "vertex.position");
    anariCommitParameters(m_device, m_segmentGeo);
    anariUnsetParameter(m_device, m_textGeo, "vertex.position");
    anariUnsetParameter(m_device, m_textGeo, "text");
    anariCommitParameters(m_device, m_textGeo);
    anariCommitParameters(m_device, m_segmentRaster);
    anariCommitParameters(m_device, m_textRaster);
    anariCommitParameters(m_device, m_group);
    anariCommitParameters(m_device, m_world);
    return;
  }

  // Build segment positions (2 vertices per segment)
  std::vector<float> segPositions;
  segPositions.reserve(segCount * 6);
  for (auto &m : m_measurements) {
    segPositions.push_back(m.pointA.x);
    segPositions.push_back(m.pointA.y);
    segPositions.push_back(m.pointA.z);
    segPositions.push_back(m.pointB.x);
    segPositions.push_back(m.pointB.y);
    segPositions.push_back(m.pointB.z);
  }
  if (hasPending) {
    // Zero-length segment at pending point A
    segPositions.push_back(m_pendingA.x);
    segPositions.push_back(m_pendingA.y);
    segPositions.push_back(m_pendingA.z);
    segPositions.push_back(m_pendingA.x);
    segPositions.push_back(m_pendingA.y);
    segPositions.push_back(m_pendingA.z);
  }

  auto segPosArray = anariNewArray1D(m_device,
      segPositions.data(), nullptr, nullptr,
      ANARI_FLOAT32_VEC3, segCount * 2);
  anariSetParameter(
      m_device, m_segmentGeo, "vertex.position", ANARI_ARRAY1D, &segPosArray);
  float w = LINE_WIDTH;
  anariSetParameter(m_device, m_segmentGeo, "width", ANARI_FLOAT32, &w);
  anariSetParameter(m_device, m_segmentGeo, "color", ANARI_FLOAT32_VEC4, color);
  anariCommitParameters(m_device, m_segmentGeo);
  anariRelease(m_device, segPosArray);

  // Build text labels at midpoints (one per completed measurement)
  size_t textCount = m_measurements.size();
  if (textCount > 0) {
    std::vector<float> textPositions;
    textPositions.reserve(textCount * 3);
    std::vector<std::string> labelStrings;
    labelStrings.reserve(textCount);

    for (auto &m : m_measurements) {
      auto mid = (m.pointA + m.pointB) * 0.5f;
      textPositions.push_back(mid.x);
      textPositions.push_back(mid.y);
      textPositions.push_back(mid.z);

      char buf[128];
      std::snprintf(buf, sizeof(buf), "%s: %.3f", m.name.c_str(), m.distance);
      labelStrings.emplace_back(buf);
    }

    auto textPosArray = anariNewArray1D(m_device,
        textPositions.data(), nullptr, nullptr,
        ANARI_FLOAT32_VEC3, textCount);
    anariSetParameter(
        m_device, m_textGeo, "vertex.position", ANARI_ARRAY1D, &textPosArray);

    // Build null-terminated string list
    std::vector<const char *> labelPtrs;
    labelPtrs.reserve(textCount + 1);
    for (auto &s : labelStrings)
      labelPtrs.push_back(s.c_str());
    labelPtrs.push_back(nullptr);

    anariSetParameter(
        m_device, m_textGeo, "text", ANARI_STRING_LIST, labelPtrs.data());
    anariSetParameter(
        m_device, m_textGeo, "color", ANARI_FLOAT32_VEC4, color);
    anariCommitParameters(m_device, m_textGeo);
    anariRelease(m_device, textPosArray);
  } else {
    anariUnsetParameter(m_device, m_textGeo, "vertex.position");
    anariUnsetParameter(m_device, m_textGeo, "text");
    anariCommitParameters(m_device, m_textGeo);
  }

  anariCommitParameters(m_device, m_segmentRaster);
  anariCommitParameters(m_device, m_textRaster);
  anariCommitParameters(m_device, m_group);
  anariCommitParameters(m_device, m_world);
}

} // namespace tsd::ui::imgui
