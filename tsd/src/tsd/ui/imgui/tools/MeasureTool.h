// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// tsd_core
#include "tsd/core/TSDMath.hpp"
// anari
#include <anari/anari_cpp.hpp>
// std
#include <string>
#include <vector>

namespace tsd::ui::imgui {

struct Measurement
{
  std::string name;
  tsd::math::float3 pointA{0.f};
  tsd::math::float3 pointB{0.f};
  float distance{0.f};
};

struct MeasureTool
{
  enum class State { IDLE, PICKED_A };

  MeasureTool(anari::Device overlayDevice);
  ~MeasureTool();

  MeasureTool(const MeasureTool &) = delete;
  MeasureTool &operator=(const MeasureTool &) = delete;

  void setPointA(tsd::math::float3 pos);
  void setPointB(tsd::math::float3 pos);
  void cancelPick();

  void remove(size_t index);
  void removeAll();

  State state() const;
  tsd::math::float3 pendingPointA() const;
  const std::vector<Measurement> &measurements() const;
  anari::World world() const;

  void buildUI();

 private:
  void rebuildWorld();

  State m_state{State::IDLE};
  tsd::math::float3 m_pendingA{0.f};
  int m_nextId{1};

  std::vector<Measurement> m_measurements;

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
