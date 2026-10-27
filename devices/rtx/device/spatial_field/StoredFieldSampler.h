/*
 * Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include "NvdbRectilinearSamplerInline.h"
#include "NvdbRegularSamplerInline.h"
#include "StoredFieldTypes.h"
#include "StructuredRectilinearSamplerInline.h"
#include "StructuredRegularSamplerInline.h"

#include <type_traits>

namespace visrtx {

namespace detail {

template <typename State>
union StoredSamplerStateBox
{
  static_assert(std::is_trivially_destructible_v<State>);
  State state;
  VISRTX_DEVICE StoredSamplerStateBox() {}
  VISRTX_DEVICE ~StoredSamplerStateBox() {}
};

template <typename T, typename ValueType>
VISRTX_DEVICE T sampleStoredNanoVDB(
    const SpatialFieldGPUData &field, const vec3 &p)
{
  if (field.storageType == StoredFieldType::NanoVDB) {
    StoredSamplerStateBox<NvdbRegularSamplerState<ValueType>> box;
    initNvdbSampler(box.state, &field);
    if constexpr (std::is_same_v<T, vec3>)
      return sampleRGB(box.state, field, p);
    else
      return sampleValue(box.state, field, p);
  } else {
    StoredSamplerStateBox<NvdbRectilinearSamplerState<ValueType>> box;
    initNvdbRectilinearSampler(box.state, &field);
    if constexpr (std::is_same_v<T, vec3>)
      return sampleRGB(box.state, field, p);
    else
      return sampleValue(box.state, field, p);
  }
}

} // namespace detail

// Adapter for evaluateFieldBinding(). Bindings must first pass the host-side
// capability/arity check and bake their field bounds intersect ROI. No data
// checks, implicit broadcasting, or outside-value filtering occur in this
// fetch.
struct StoredFieldSampler
{
  const SpatialFieldGPUData *fields;

  template <typename T>
  VISRTX_DEVICE T sample(int32_t index, const vec3 &p) const;
};

// Inlined definitions ////////////////////////////////////////////////////////

template <typename T>
VISRTX_DEVICE T StoredFieldSampler::sample(int32_t index, const vec3 &p) const
{
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, vec3>);
  const auto &field = fields[index];
  switch (field.storageType) {
  case StoredFieldType::StructuredRegular: {
    StructuredRegularSamplerState state;
    initStructuredRegularSampler(state, &field);
    if constexpr (std::is_same_v<T, vec3>)
      return sampleRGB(state, field, p);
    else
      return sampleValue(state, field, p);
  }
  case StoredFieldType::StructuredRectilinear: {
    StructuredRectilinearSamplerState state;
    initStructuredRectilinearSampler(state, &field);
    if constexpr (std::is_same_v<T, vec3>)
      return sampleRGB(state, field, p);
    else
      return sampleValue(state, field, p);
  }
  case StoredFieldType::NanoVDB:
  case StoredFieldType::NanoVDBRectilinear:
    if constexpr (std::is_same_v<T, vec3>) {
      return detail::sampleStoredNanoVDB<T, nanovdb::Vec3f>(field, p);
    } else {
      const auto type = field.storageType == StoredFieldType::NanoVDB
          ? field.data.nvdbRegular.gridType
          : field.data.nvdbRectilinear.gridType;
      switch (type) {
      case nanovdb::GridType::Fp4:
        return detail::sampleStoredNanoVDB<T, nanovdb::Fp4>(field, p);
      case nanovdb::GridType::Fp8:
        return detail::sampleStoredNanoVDB<T, nanovdb::Fp8>(field, p);
      case nanovdb::GridType::Fp16:
        return detail::sampleStoredNanoVDB<T, nanovdb::Fp16>(field, p);
      case nanovdb::GridType::FpN:
        return detail::sampleStoredNanoVDB<T, nanovdb::FpN>(field, p);
      case nanovdb::GridType::Float:
        return detail::sampleStoredNanoVDB<T, float>(field, p);
      default:
        break;
      }
    }
    break;
  default:
    break;
  }
  // Unsupported metadata violates the finalization-time binding contract.
  return T(0.f);
}

} // namespace visrtx
