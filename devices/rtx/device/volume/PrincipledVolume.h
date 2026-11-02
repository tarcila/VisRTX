// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "Volume.h"
#include "gpu/principledVolumeData.h"
#include "spatial_field/SpatialField.h"

#include <map>
#include <memory>
#include <vector>

namespace visrtx {

struct PrincipledVolume : public Volume
{
  PrincipledVolume(DeviceGlobalState *d);
  ~PrincipledVolume() override = default;

  void commitParameters() override;
  void finalize() override;
  void markFinalized() override;
  bool isValid() const override;
  void *deviceData() const override;
  ReferenceVolumeGPUData referenceVolume() override;
  bool getProperty(const std::string_view &name,
      ANARIDataType type,
      void *ptr,
      uint64_t size,
      uint32_t flags) override;

  // Internal committed-state seam, not a new ANARI parameter/property. Its
  // pointers are borrowed until the next supported finalize/update boundary.
  const PrincipledVolumeGPUData *committedData() const;

 private:
  VolumeGPUData gpuData() const override;
  bool checkType(
      const std::string &name, ANARIDataType type, bool required = false);
  template <typename T>
  void readBinding(const char *name, FieldBindingGPUData<T> &binding);
  void updateObservers();
  void reportPreparation(
      const optical_reference::PreparationResult &result) const;

  bool m_parametersValid{false};
  bool m_valid{false};
  bool m_previewTexturePrepared{false};
  double m_referenceStep{0.};
  PrincipledVolumeBindingsGPUData m_bindings;
  PrincipledVolumeGPUData m_committed;
  optical_reference::PreparedVolumeBindings m_prepared;
  // Nonowning index table. Exactly one stable observer owner per distinct field
  // retains every referenced field even while the volume is invalid.
  std::vector<SpatialField *> m_fields;
  std::map<SpatialField *,
      std::unique_ptr<helium::ChangeObserverPtr<SpatialField>>>
      m_observers;
  DeviceBuffer m_fieldViews;
  DeviceBuffer m_payload;
  DeviceBuffer m_previewCold;
  DeviceBuffer m_qualityCold;
};

// Inlined definitions ////////////////////////////////////////////////////////

inline const PrincipledVolumeGPUData *PrincipledVolume::committedData() const
{
  return m_valid ? &m_committed : nullptr;
}

} // namespace visrtx
