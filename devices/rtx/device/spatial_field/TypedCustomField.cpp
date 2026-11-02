// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#include "TypedCustomFieldDispatch.h"

namespace visrtx {
namespace {
constexpr TypedCustomDescriptor providers[]{TypedCustomDescriptor{},
#define VISRTX_TYPED_DESCRIPTOR(P) typedCustomDescriptor<P>(),
    VISRTX_ALL_TYPED_CUSTOM_PROVIDERS(VISRTX_TYPED_DESCRIPTOR)
#undef VISRTX_TYPED_DESCRIPTOR
};
constexpr bool uniqueProviders()
{
  for (size_t i = 1; i < sizeof(providers) / sizeof(providers[0]); ++i) {
    if (providers[i].provider == 0)
      return false;
    for (size_t j = 1; j < i; ++j)
      if (providers[i].provider == providers[j].provider)
        return false;
  }
  return true;
}
static_assert(
    uniqueProviders(), "typed custom provider IDs must be nonzero and unique");
} // namespace
const TypedCustomDescriptor *compiledTypedCustomProvider(uint32_t id)
{
  for (size_t i = 1; i < sizeof(providers) / sizeof(providers[0]); ++i)
    if (providers[i].provider == id)
      return &providers[i];
  return nullptr;
}
const char *typedCustomErrorString(TypedCustomError error)
{
  switch (error) {
  case TypedCustomError::None:
    return "ready";
  case TypedCustomError::DescriptorSize:
    return "short typed descriptor";
  case TypedCustomError::Version:
    return "unsupported typed contract version";
  case TypedCustomError::MissingProvider:
    return "typed provider is not compiled into this device";
  case TypedCustomError::Capabilities:
    return "typed execution/resource capabilities mismatch";
  case TypedCustomError::Layout:
    return "typed payload revision/size/alignment mismatch";
  case TypedCustomError::Arity:
    return "typed scalar/RGB arity mismatch";
  case TypedCustomError::MissingRange:
    return "missing conservative global raw range";
  case TypedCustomError::InvalidRange:
    return "invalid conservative raw range metadata";
  case TypedCustomError::MissingResources:
    return "missing immutable resource generation lease";
  case TypedCustomError::InvalidDomain:
    return "nonfinite typed field domain";
  case TypedCustomError::MissingPayload:
    return "missing typed payload";
  }
  return "unknown typed contract error";
}
} // namespace visrtx
