// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "gpu_objects.h"
#include "principledVolumeData.h"

namespace visrtx {

// Separate from TF1D's legacy texture/classification payload. The common
// optical path uses canonical interpolation and emission-absorption, without
// altering current renderer dispatch until that path is explicitly enabled.
struct TransferFunctionReferenceGPUData
{
  StoredFieldReferenceGPUData field;
  const vec4 *colors{nullptr};
  uint32_t colorCount{0};
  box1 valueRange;
  float unitDistance{1.f};
  optical_reference::Bounds opticalBounds;
};

struct ReferenceVolumeGPUData
{
  VolumeType type{VolumeType::UNKNOWN};
  box3 domain;
  const PrincipledVolumeGPUData *principled{nullptr};
  const TransferFunctionReferenceGPUData *transfer{nullptr};
  double stepSize{0.}; // object-space resolution hint; not an optical length
  DeviceObjectIndex registryIndex{-1}; // for frame-local legacy picking eligibility
};

struct VolumeOccurrenceGPUData
{
  // Double inverse of the committed affine transform. Ray directions must NOT
  // be renormalized after this transform: interval t remains world distance.
  double worldToObject[3][4]{};
  ReferenceVolumeGPUData volume;
  size_t placement{0};
  size_t groupSlot{0};
};

struct AggregateVolumeSceneGPUData
{
  const VolumeOccurrenceGPUData *occurrences{nullptr};
  size_t count{0};
  bool complete{false};
  uint64_t generation{0}; // identity for this publication, not a lifetime lease
};

static_assert(std::is_trivially_copyable_v<TransferFunctionReferenceGPUData>);
static_assert(std::is_trivially_copyable_v<VolumeOccurrenceGPUData>);

} // namespace visrtx
