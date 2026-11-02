// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#include <cstdio>
#include <stdexcept>
#include "spatial_field/TypedCustomField.h"
using namespace visrtx;
struct Provider
{
  using Value = float;
  struct Payload
  {
    float value;
  };
  static constexpr uint32_t ID = 7;
  static constexpr uint64_t REVISION = 19;
  static constexpr bool CUDA_SAMPLING = false, RESOURCE_LEASE = true;
  static float sampleOptix(const Payload &, const vec3 &);
};
int main()
{
  const auto compiled = typedCustomDescriptor<Provider>();
  auto check = [&](TypedCustomDescriptor request,
                   TypedCustomError expected,
                   bool range = true,
                   bool resources = true,
                   bool known = true) {
    if (validateTypedCustom(request,
            known ? &compiled : nullptr,
            sizeof(Provider::Payload),
            alignof(Provider::Payload),
            FieldValueType::Scalar,
            range,
            resources)
        != expected)
      throw std::runtime_error("typed custom negotiation mismatch");
  };
  check(compiled, TypedCustomError::None);
  auto d = compiled;
  d.structSize = 4;
  check(d, TypedCustomError::DescriptorSize);
  d = compiled;
  d.version = 2;
  check(d, TypedCustomError::Version);
  check(compiled, TypedCustomError::MissingProvider, true, true, false);
  d = compiled;
  d.capabilities |= 0x80000000u;
  check(d, TypedCustomError::Capabilities);
  d = compiled;
  ++d.payloadRevision;
  check(d, TypedCustomError::Layout);
  d = compiled;
  ++d.payloadSize;
  check(d, TypedCustomError::Layout);
  d = compiled;
  ++d.payloadAlignment;
  check(d, TypedCustomError::Layout);
  d = compiled;
  d.valueType = FieldValueType::RGB;
  check(d, TypedCustomError::Arity);
  check(compiled, TypedCustomError::MissingRange, false);
  check(compiled, TypedCustomError::MissingResources, true, false);
  std::puts("Typed custom metadata negotiation controls passed");
}
