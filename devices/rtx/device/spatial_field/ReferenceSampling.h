// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "StoredFieldReferenceSampler.h"

#if defined(VISRTX_REFERENCE_OPTIX_CALLS) && defined(__CUDA_ARCH__)
#include <optix_device.h>
#include "gpu/sbt.h"
#endif

namespace visrtx {
#if defined(VISRTX_REFERENCE_OPTIX_CALLS) && defined(__CUDA_ARCH__)
struct OptixReferenceFieldSampler
{
  const StoredFieldReferenceGPUData *fields;
  template <typename T>
  VISRTX_DEVICE T sample(int32_t index, const vec3 &p) const;
};
template <typename T>
VISRTX_DEVICE T OptixReferenceFieldSampler::sample(
    int32_t index, const vec3 &p) const
{
  constexpr auto slot = std::is_same_v<T, float>
      ? SbtCallableEntryPoints::ReferenceFieldScalar
      : SbtCallableEntryPoints::ReferenceFieldRGB;
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, vec3>);
  return optixDirectCall<T>(uint32_t(slot), &fields[index], &p);
}
using ReferenceFieldSampler = OptixReferenceFieldSampler;
#else
using ReferenceFieldSampler = StoredFieldReferenceSampler;
#endif
} // namespace visrtx
