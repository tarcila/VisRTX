// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "TypedCustomField.h"

// The configured header declares provider traits and defines
// VISRTX_TYPED_CUSTOM_PROVIDERS(X) as X(ProviderType) entries. OptiX-only
// implementation headers must be guarded by VISRTX_TYPED_CUSTOM_OPTIX.
#ifdef VISRTX_TYPED_CUSTOM_PROVIDER_HEADER
#include VISRTX_TYPED_CUSTOM_PROVIDER_HEADER
#endif
#ifndef VISRTX_TYPED_CUSTOM_PROVIDERS
#define VISRTX_TYPED_CUSTOM_PROVIDERS(X)
#endif
#ifdef VISRTX_TYPED_CUSTOM_TEST_PROVIDERS
#include "typedCustom/FixtureProviders.h"
#else
#define VISRTX_TYPED_CUSTOM_TEST_LIST(X)
#endif
#define VISRTX_ALL_TYPED_CUSTOM_PROVIDERS(X)                                   \
  VISRTX_TYPED_CUSTOM_PROVIDERS(X) VISRTX_TYPED_CUSTOM_TEST_LIST(X)

#if defined(__CUDA_ARCH__)
namespace visrtx {
// Backend is part of the symbol identity. Macro-dependent inline/noinline
// bodies can otherwise be coalesced across OptiX modules with the wrong
// backend.
template <typename Provider, typename T, bool Optix>
VISRTX_DEVICE T invokeTypedCustom(const void *payload, const vec3 &position)
{
  if constexpr (std::is_same_v<T, typename Provider::Value>) {
    const auto &state =
        *static_cast<const typename Provider::Payload *>(payload);
    if constexpr (Optix)
      return Provider::sampleOptix(state, position);
    else if constexpr (Provider::CUDA_SAMPLING)
      return Provider::sampleCuda(state, position);
  }
  // Internal dispatch-contract violation, never a missing-capability zero.
  // Preparation rejects these states before a launch.
  __trap();
  return T(0.f);
}
template <typename T, bool Optix = false>
VISRTX_DEVICE T sampleTypedCustom(
    uint32_t provider, const void *payload, const vec3 &position)
{
  switch (provider) {
#define VISRTX_TYPED_CASE(P)                                                   \
  case P::ID:                                                                  \
    return invokeTypedCustom<P, T, Optix>(payload, position);
    VISRTX_ALL_TYPED_CUSTOM_PROVIDERS(VISRTX_TYPED_CASE)
#undef VISRTX_TYPED_CASE
  }
  __trap();
  return T(0.f);
}
} // namespace visrtx
#endif
