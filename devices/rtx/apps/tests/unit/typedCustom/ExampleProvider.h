// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
// A standalone configured manifest, independent of built-in test fixtures.
#include "spatial_field/TypedCustomField.h"
namespace typed_custom_example {
struct RGB
{
  using Value = visrtx::vec3;
  struct Payload
  {
    visrtx::vec3 value;
  };
  static constexpr uint32_t ID = 0x4558414du;
  static constexpr uint64_t REVISION = 1;
  static constexpr bool CUDA_SAMPLING = false, RESOURCE_LEASE = false;
  static VISRTX_DEVICE Value sampleOptix(const Payload &, const visrtx::vec3 &);
};
#if defined(__CUDACC__) && defined(VISRTX_TYPED_CUSTOM_OPTIX)
VISRTX_DEVICE RGB::Value RGB::sampleOptix(
    const Payload &s, const visrtx::vec3 &)
{
  return s.value;
}
#endif
} // namespace typed_custom_example
#define VISRTX_TYPED_CUSTOM_PROVIDERS(X) X(typed_custom_example::RGB)
