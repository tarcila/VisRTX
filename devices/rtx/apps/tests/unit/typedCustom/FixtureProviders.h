// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "spatial_field/TypedCustomField.h"

namespace visrtx {
namespace typed_custom_test {
inline constexpr uint32_t LEGACY_ID = 0x54434600u;
struct Payload
{
  vec3 value{1.f};
  float slope{0.f};
  box3 domain{vec3(0.f), vec3(1.f)};
  const vec3 *resource{nullptr};
};
struct Scalar
{
  using Value = float;
  using Payload = typed_custom_test::Payload;
  static constexpr uint32_t ID = 0x54434601u;
  static constexpr uint64_t REVISION = 1;
  static constexpr bool CUDA_SAMPLING = true, RESOURCE_LEASE = false;
  static VISRTX_DEVICE Value sampleOptix(const Payload &, const vec3 &);
  static VISRTX_DEVICE Value sampleCuda(const Payload &, const vec3 &);
};
struct RgbOptix
{
  using Value = vec3;
  using Payload = typed_custom_test::Payload;
  static constexpr uint32_t ID = 0x54434602u;
  static constexpr uint64_t REVISION = 1;
  static constexpr bool CUDA_SAMPLING = false, RESOURCE_LEASE = false;
  static VISRTX_DEVICE Value sampleOptix(const Payload &, const vec3 &);
};
struct RgbCuda
{
  using Value = vec3;
  using Payload = typed_custom_test::Payload;
  static constexpr uint32_t ID = 0x54434603u;
  static constexpr uint64_t REVISION = 1;
  static constexpr bool CUDA_SAMPLING = true, RESOURCE_LEASE = false;
  static VISRTX_DEVICE Value sampleOptix(const Payload &, const vec3 &);
  static VISRTX_DEVICE Value sampleCuda(const Payload &, const vec3 &);
};
struct RgbOwned
{
  using Value = vec3;
  using Payload = typed_custom_test::Payload;
  static constexpr uint32_t ID = 0x54434604u;
  static constexpr uint64_t REVISION = 1;
  static constexpr bool CUDA_SAMPLING = false, RESOURCE_LEASE = true;
  static VISRTX_DEVICE Value sampleOptix(const Payload &, const vec3 &);
};
struct RgbOwnedCuda
{
  using Value = vec3;
  using Payload = typed_custom_test::Payload;
  static constexpr uint32_t ID = 0x54434605u;
  static constexpr uint64_t REVISION = 1;
  static constexpr bool CUDA_SAMPLING = true, RESOURCE_LEASE = true;
  static VISRTX_DEVICE Value sampleOptix(const Payload &, const vec3 &);
  static VISRTX_DEVICE Value sampleCuda(const Payload &, const vec3 &);
};
struct ScalarOptix
{
  using Value = float;
  using Payload = typed_custom_test::Payload;
  static constexpr uint32_t ID = 0x54434606u;
  static constexpr uint64_t REVISION = 1;
  static constexpr bool CUDA_SAMPLING = false, RESOURCE_LEASE = false;
  static VISRTX_DEVICE Value sampleOptix(const Payload &, const vec3 &);
};
#if defined(__CUDACC__)
VISRTX_DEVICE vec3 value(const Payload &s, const vec3 &p, bool poison = true)
{
  // Outside the advertised domain is intentionally not a valid typed sample.
  // Binding membership must handle those points before invoking the provider.
  if (poison && !optical_reference::contains(s.domain, p))
    return vec3(NAN);
  return (s.resource ? *s.resource : s.value) * (1.f + s.slope * p.x);
}
VISRTX_DEVICE float Scalar::sampleCuda(const Payload &s, const vec3 &p)
{
  return value(s, p).x;
}
VISRTX_DEVICE vec3 RgbCuda::sampleCuda(const Payload &s, const vec3 &p)
{
  return value(s, p);
}
VISRTX_DEVICE vec3 RgbOwnedCuda::sampleCuda(const Payload &s, const vec3 &p)
{
  return value(s, p);
}
#ifdef VISRTX_TYPED_CUSTOM_OPTIX
VISRTX_DEVICE float ScalarOptix::sampleOptix(const Payload &s, const vec3 &p)
{
  return value(s, p).x;
}
VISRTX_DEVICE vec3 RgbOwnedCuda::sampleOptix(const Payload &s, const vec3 &p)
{
  return value(s, p);
}
VISRTX_DEVICE float Scalar::sampleOptix(const Payload &s, const vec3 &p)
{
  return value(s, p).x;
}
VISRTX_DEVICE vec3 RgbOptix::sampleOptix(const Payload &s, const vec3 &p)
{
  return value(s, p);
}
VISRTX_DEVICE vec3 RgbCuda::sampleOptix(const Payload &s, const vec3 &p)
{
  return value(s, p);
}
VISRTX_DEVICE vec3 RgbOwned::sampleOptix(const Payload &s, const vec3 &p)
{
  return value(s, p);
}
#endif
#endif
} // namespace typed_custom_test
} // namespace visrtx
#define VISRTX_TYPED_CUSTOM_TEST_LIST(X)                                       \
  X(visrtx::typed_custom_test::Scalar)                                         \
  X(visrtx::typed_custom_test::RgbOptix)                                       \
  X(visrtx::typed_custom_test::RgbCuda)                                        \
  X(visrtx::typed_custom_test::RgbOwned)                                       \
  X(visrtx::typed_custom_test::RgbOwnedCuda)                                   \
  X(visrtx::typed_custom_test::ScalarOptix)
