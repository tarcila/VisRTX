// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#include <stdexcept>
#include "spatial_field/TypedCustomField.h"
int main()
{
  const auto *provider = visrtx::compiledTypedCustomProvider(0x4558414du);
#ifdef EXPECT_TYPED_PROVIDER
  if (!provider || provider->valueType != visrtx::FieldValueType::RGB
      || provider->capabilities != visrtx::TypedCustomOptix
      || provider->payloadSize != sizeof(visrtx::vec3))
    throw std::runtime_error(
        "configured manifest did not match compiled provider");
#else
  if (provider)
    throw std::runtime_error("empty build unexpectedly exposes a provider");
#endif
  if (visrtx::compiledTypedCustomProvider(0))
    throw std::runtime_error("invalid provider sentinel exposed");
}
