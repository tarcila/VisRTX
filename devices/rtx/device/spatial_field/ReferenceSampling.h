// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "StoredFieldPreviewSampler.h"
#include "StoredFieldReferenceSampler.h"

#if defined(VISRTX_REFERENCE_OPTIX_CALLS) && defined(__CUDA_ARCH__)
#include <optix_device.h>
#include "gpu/sbt.h"
#endif

namespace visrtx {
#if defined(VISRTX_REFERENCE_OPTIX_CALLS) && defined(__CUDA_ARCH__)
template <bool Preview>
struct OptixReferenceFieldSampler
{
  const StoredFieldReferenceGPUData *fields;
  template <typename T>
  VISRTX_DEVICE T sample(int32_t index, const vec3 &p) const;
};
template <bool Preview>
template <typename T>
VISRTX_DEVICE T OptixReferenceFieldSampler<Preview>::sample(
    int32_t index, const vec3 &p) const
{
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, vec3>);
  if constexpr (Preview) {
    // Share the FP32 sampler as a regular noinline device function, avoiding
    // an OptiX callable transition for every camera sample.
    if (!fields[index].customProvider)
      return StoredFieldPreviewSampler{fields}.sample<T>(index, p);
  }
  constexpr auto slot = std::is_same_v<T, float>
      ? SbtCallableEntryPoints::ReferenceFieldScalar
      : SbtCallableEntryPoints::ReferenceFieldRGB;
  return optixDirectCall<T>(uint32_t(slot),
      &fields[index],
      &p,
      false,
      static_cast<const PreviewDensityIntegral *>(nullptr));
}
using ReferenceFieldSampler = OptixReferenceFieldSampler<false>;
using PreviewFieldSampler = OptixReferenceFieldSampler<true>;
#else
using ReferenceFieldSampler = StoredFieldReferenceSampler;
using PreviewFieldSampler = StoredFieldPreviewSampler;
#endif
} // namespace visrtx
