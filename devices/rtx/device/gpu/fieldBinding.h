/*
 * Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include "gpu_math.h"

#include <glm/common.hpp>

#include <type_traits>

namespace visrtx {

// Separate from the existing scalar field payload/callable ABI. Finalization
// checks arity once; sampling never implicitly broadcasts or reduces channels.
enum class FieldValueType : uint8_t
{
  Scalar,
  RGB
};

template <typename T>
constexpr FieldValueType fieldValueType()
{
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, vec3>,
      "A field binding must be scalar float or linear RGB");
  return std::is_same_v<T, float> ? FieldValueType::Scalar
                                  : FieldValueType::RGB;
}

template <typename T>
constexpr bool fieldBindingAccepts(FieldValueType type)
{
  return type == fieldValueType<T>();
}

VISRTX_HOST_DEVICE box3 fieldBindingDomain(const box3 &bounds, const box3 &roi)
{
  return {glm::max(bounds.lower, roi.lower), glm::min(bounds.upper, roi.upper)};
}

// An empty intersection (including a zero-width axis) has no samples. Nonempty
// domains include their boundary; decide membership before backend filtering.
VISRTX_HOST_DEVICE bool containsFieldSample(const box3 &domain, const vec3 &p)
{
  return domain.lower.x < domain.upper.x && domain.lower.y < domain.upper.y
      && domain.lower.z < domain.upper.z && p.x >= domain.lower.x
      && p.x <= domain.upper.x && p.y >= domain.lower.y && p.y <= domain.upper.y
      && p.z >= domain.lower.z && p.z <= domain.upper.z;
}

template <typename T>
struct FieldBindingGPUData
{
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, vec3>,
      "A field binding must be scalar float or linear RGB");

  // Registry index, matching DeviceObjectIndex without pulling OptiX into this
  // header. -1 means constant; domains are baked as field bounds intersect ROI.
  int32_t field{-1};
  box3 domain;
  T scale{1.f};
  T outside{1.f};

  VISRTX_HOST_DEVICE FieldBindingGPUData() = default;
  VISRTX_HOST_DEVICE FieldBindingGPUData(const T &value, const T &outsideValue);
};

// Inlined definitions ////////////////////////////////////////////////////////

template <typename T>
VISRTX_HOST_DEVICE FieldBindingGPUData<T>::FieldBindingGPUData(
    const T &value, const T &outsideValue)
    : scale(value), outside(outsideValue)
{}

// Samplers provide sample<T>(registryIndex, objectPosition) with an exact T
// result. Existing scalar callables are not changed by this interface. The
// caller guarantees valid indices, matching arity and valid field contents.
template <typename T, typename Sampler>
VISRTX_HOST_DEVICE T evaluateFieldBinding(
    const FieldBindingGPUData<T> &binding, const vec3 &p, Sampler &sampler)
{
  static_assert(
      std::is_same_v<decltype(sampler.template sample<T>(binding.field, p)), T>,
      "Field sampler result must exactly match the binding type");
  if (binding.field == -1)
    return binding.scale;
  if (!containsFieldSample(binding.domain, p))
    return binding.scale * binding.outside;
  return binding.scale * sampler.template sample<T>(binding.field, p);
}

} // namespace visrtx
