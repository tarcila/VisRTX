// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// tsd_core
#include "tsd/core/TSDMath.hpp"
// anari
#include <anari/anari_cpp.hpp>

namespace tsd::ui::imgui {

struct MeasureTool
{
  enum class State { IDLE, PICKED_A, MEASURED };

  MeasureTool(anari::Device overlayDevice);
  ~MeasureTool();

  void setPointA(tsd::math::float3 pos);
  void setPointB(tsd::math::float3 pos);
  void clear();

  State state() const;
  float distance() const;
  tsd::math::float3 pointA() const;
  tsd::math::float3 pointB() const;
  anari::World world() const;

 private:
  void rebuildGeometry();

  State m_state{State::IDLE};
  tsd::math::float3 m_pointA{0.f};
  tsd::math::float3 m_pointB{0.f};
  float m_distance{0.f};

  anari::Device m_device{nullptr};
  anari::World m_world{nullptr};
  ANARIObject m_segmentGeo{nullptr};
  ANARIObject m_textGeo{nullptr};
  ANARIObject m_segmentRaster{nullptr};
  ANARIObject m_textRaster{nullptr};
  ANARIObject m_group{nullptr};
  ANARIObject m_instance{nullptr};
};

} // namespace tsd::ui::imgui
