// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "MeasureTool.h"
// std
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

// MeasureTool definitions ////////////////////////////////////////////////////

MeasureTool::MeasureTool(anari::Device overlayDevice)
    : m_device(overlayDevice)
{
  anari::retain(m_device, m_device);

  // Segment geometry (line between A and B)
  m_segmentGeo = anariNewGeometry(m_device, "segment");

  // Text geometry (distance label at midpoint)
  m_textGeo = anariNewGeometry(m_device, "text");
  float height = TEXT_HEIGHT;
  anariSetParameter(m_device, m_textGeo, "height", ANARI_FLOAT32, &height);
  float offset[] = {TEXT_OFFSET_X, TEXT_OFFSET_Y};
  anariSetParameter(m_device, m_textGeo, "offset", ANARI_FLOAT32_VEC2, offset);

  // Rasters wrapping each geometry
  m_segmentRaster = newRaster(m_device);
  anariSetParameter(
      m_device, m_segmentRaster, "geometry", ANARI_GEOMETRY, &m_segmentGeo);
  anariCommitParameters(m_device, m_segmentRaster);

  m_textRaster = newRaster(m_device);
  anariSetParameter(
      m_device, m_textRaster, "geometry", ANARI_GEOMETRY, &m_textGeo);
  anariCommitParameters(m_device, m_textRaster);

  // Group -> Instance -> World
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
  m_pointA = pos;
  m_state = State::PICKED_A;
  rebuildGeometry();
}

void MeasureTool::setPointB(tsd::math::float3 pos)
{
  m_pointB = pos;
  auto d = m_pointB - m_pointA;
  m_distance = std::sqrt(d.x * d.x + d.y * d.y + d.z * d.z);
  m_state = State::MEASURED;
  rebuildGeometry();
}

void MeasureTool::clear()
{
  m_state = State::IDLE;
  m_distance = 0.f;

  // Clear geometry by unsetting positions
  anariUnsetParameter(m_device, m_segmentGeo, "vertex.position");
  anariCommitParameters(m_device, m_segmentGeo);
  anariUnsetParameter(m_device, m_textGeo, "vertex.position");
  anariUnsetParameter(m_device, m_textGeo, "text");
  anariCommitParameters(m_device, m_textGeo);
  anariCommitParameters(m_device, m_segmentRaster);
  anariCommitParameters(m_device, m_textRaster);
  anariCommitParameters(m_device, m_group);
  anariCommitParameters(m_device, m_world);
}

MeasureTool::State MeasureTool::state() const
{
  return m_state;
}

float MeasureTool::distance() const
{
  return m_distance;
}

tsd::math::float3 MeasureTool::pointA() const
{
  return m_pointA;
}

tsd::math::float3 MeasureTool::pointB() const
{
  return m_pointB;
}

anari::World MeasureTool::world() const
{
  return m_world;
}

void MeasureTool::rebuildGeometry()
{
  // Measurement line color (cyan)
  float color[] = {0.f, 1.f, 1.f, 1.f};

  if (m_state == State::PICKED_A) {
    // Single point — show just point A as a zero-length segment
    float positions[] = {
        m_pointA.x, m_pointA.y, m_pointA.z,
        m_pointA.x, m_pointA.y, m_pointA.z,
    };
    auto posArray = anariNewArray1D(
        m_device, positions, nullptr, nullptr, ANARI_FLOAT32_VEC3, 2);
    anariSetParameter(
        m_device, m_segmentGeo, "vertex.position", ANARI_ARRAY1D, &posArray);
    float w = LINE_WIDTH;
    anariSetParameter(m_device, m_segmentGeo, "width", ANARI_FLOAT32, &w);
    anariSetParameter(
        m_device, m_segmentGeo, "color", ANARI_FLOAT32_VEC4, color);
    anariCommitParameters(m_device, m_segmentGeo);
    anariRelease(m_device, posArray);

    // No text yet
    anariUnsetParameter(m_device, m_textGeo, "vertex.position");
    anariUnsetParameter(m_device, m_textGeo, "text");
    anariCommitParameters(m_device, m_textGeo);
  } else if (m_state == State::MEASURED) {
    // Segment from A to B
    float positions[] = {
        m_pointA.x, m_pointA.y, m_pointA.z,
        m_pointB.x, m_pointB.y, m_pointB.z,
    };
    auto posArray = anariNewArray1D(
        m_device, positions, nullptr, nullptr, ANARI_FLOAT32_VEC3, 2);
    anariSetParameter(
        m_device, m_segmentGeo, "vertex.position", ANARI_ARRAY1D, &posArray);
    float w = LINE_WIDTH;
    anariSetParameter(m_device, m_segmentGeo, "width", ANARI_FLOAT32, &w);
    anariSetParameter(
        m_device, m_segmentGeo, "color", ANARI_FLOAT32_VEC4, color);
    anariCommitParameters(m_device, m_segmentGeo);
    anariRelease(m_device, posArray);

    // Text label at midpoint
    auto mid = (m_pointA + m_pointB) * 0.5f;
    float textPos[] = {mid.x, mid.y, mid.z};
    auto textPosArray = anariNewArray1D(
        m_device, textPos, nullptr, nullptr, ANARI_FLOAT32_VEC3, 1);
    anariSetParameter(
        m_device, m_textGeo, "vertex.position", ANARI_ARRAY1D, &textPosArray);

    char label[64];
    std::snprintf(label, sizeof(label), "%.3f", m_distance);
    const char *labels[] = {label, nullptr};
    anariSetParameter(m_device, m_textGeo, "text", ANARI_STRING_LIST, labels);
    anariSetParameter(
        m_device, m_textGeo, "color", ANARI_FLOAT32_VEC4, color);
    anariCommitParameters(m_device, m_textGeo);
    anariRelease(m_device, textPosArray);
  }

  anariCommitParameters(m_device, m_segmentRaster);
  anariCommitParameters(m_device, m_textRaster);
  anariCommitParameters(m_device, m_group);
  anariCommitParameters(m_device, m_world);
}

} // namespace tsd::ui::imgui
