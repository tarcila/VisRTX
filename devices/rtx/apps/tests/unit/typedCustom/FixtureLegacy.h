// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "FixtureProviders.h"
#include "gpu/gpu_objects.h"
namespace visrtx {
namespace typed_custom_test {
VISRTX_HOST_DEVICE bool legacyFixture(uint32_t id)
{
  return id >= LEGACY_ID && id <= ScalarOptix::ID;
}
VISRTX_DEVICE float legacyValue(const CustomFieldData &data, const vec3 &p)
{
  const auto &s = *reinterpret_cast<const Payload *>(data.fieldData);
  return value(s, p, false).x;
}
VISRTX_DEVICE box1 legacyRange(const CustomFieldData &data)
{
  const auto &s = *reinterpret_cast<const Payload *>(data.fieldData);
  const float v = s.value.x;
  if (s.slope == 0.f)
    return {v, v};
  const float magnitude = fabsf(v) * (1.f + s.slope) * 1.0001f;
  return v < 0.f ? box1{-magnitude, 0.f} : box1{0.f, magnitude};
}
} // namespace typed_custom_test
} // namespace visrtx
