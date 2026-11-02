// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "blackbody/previewCold.h"
#include "spatial_field/StoredFieldReference.h"

namespace visrtx {

// Separate payload: never alias this as TF1D's field/texture data. Field
// indices in bindings are local to this volume's compact retained snapshot
// table.
struct PrincipledVolumeGPUData
{
  PrincipledVolumeBindingsGPUData bindings;
  optical_reference::Bounds opticalBounds;
  const StoredFieldReferenceGPUData *fields{nullptr};
  uint32_t fieldCount{0};
  // Published only for an unbound transmissionColor, after metadata validation.
  // Default-off also preserves uncached synthetic/reference test payloads.
  double constantTransmissionLog[3]{};
  bool hasConstantTransmissionLog{false};
  bool previewFloatEligible{false};
  const PreviewColdKnot *previewCold{nullptr};
  const PreviewColdKnot *qualityCold{nullptr};
  bool previewShadowCacheEligible{false};
  optical_reference::FieldRevision densityRevision;
  float previewExtinctionPerDensity[3]{};
  bool hasPreviewExtinction{false};
};

static_assert(std::is_trivially_copyable_v<PrincipledVolumeGPUData>);

} // namespace visrtx
