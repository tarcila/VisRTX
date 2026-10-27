/*
 * Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include "gpu/gpu_objects.h"

namespace visrtx {

VISRTX_HOST_DEVICE bool supportedNanoVDBFieldType(nanovdb::GridType type)
{
  switch (type) {
  case nanovdb::GridType::Float:
  case nanovdb::GridType::Fp4:
  case nanovdb::GridType::Fp8:
  case nanovdb::GridType::Fp16:
  case nanovdb::GridType::FpN:
  case nanovdb::GridType::Vec3f:
    return true;
  default:
    return false;
  }
}

// Host-callable finalization check, deliberately separate from CUDA samplers.
// Custom fields are not implicitly opted into the stored-field interface.
VISRTX_HOST_DEVICE bool supportsStoredFieldSampling(
    const SpatialFieldGPUData &field, FieldValueType type)
{
  if (field.valueType != type)
    return false;
  switch (field.storageType) {
  case StoredFieldType::StructuredRegular:
  case StoredFieldType::StructuredRectilinear:
    return true;
  case StoredFieldType::NanoVDB:
  case StoredFieldType::NanoVDBRectilinear: {
    const auto gridType = field.storageType == StoredFieldType::NanoVDB
        ? field.data.nvdbRegular.gridType
        : field.data.nvdbRectilinear.gridType;
    return supportedNanoVDBFieldType(gridType)
        && ((gridType == nanovdb::GridType::Vec3f)
            == (type == FieldValueType::RGB));
  }
  default:
    return false;
  }
}

} // namespace visrtx
