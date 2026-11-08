// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "Volume.h"
#include "material/MDL.h"

namespace visrtx {
struct MDLVolume : Volume
{
  MDLVolume(DeviceGlobalState *d);
  void commitParameters() override;
  void finalize() override;
  bool isValid() const override;
  bool isMDL() const override;
  const char *renderError() const override;

 private:
  VolumeGPUData gpuData() const override;
  helium::ChangeObserverPtr<Material> m_material;
  box3 m_extent{vec3(-1.f), vec3(1.f)};
  float m_majorant{0.f};
  float m_step{0.f};
  std::string m_error;
};
} // namespace visrtx
