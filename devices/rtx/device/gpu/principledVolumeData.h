// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

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
};

static_assert(std::is_trivially_copyable_v<PrincipledVolumeGPUData>);

} // namespace visrtx
