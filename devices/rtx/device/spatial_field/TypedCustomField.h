// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include "gpu/referenceFieldBinding.h"

namespace visrtx {
// Source-build semantic contract, not a binary plugin ABI. Provider and device
// must be rebuilt compatibly. The manifest is shared by host metadata and GPU
// dispatch; factory registration alone never enables typed sampling.
inline constexpr uint32_t TYPED_CUSTOM_CONTRACT = 1;
enum TypedCustomCapability : uint32_t
{
  TypedCustomOptix = 1,
  TypedCustomCuda = 2,
  TypedCustomResources = 4
};
struct TypedCustomDescriptor
{
  uint32_t structSize{sizeof(TypedCustomDescriptor)};
  uint32_t version{TYPED_CUSTOM_CONTRACT};
  uint32_t provider{0};
  FieldValueType valueType{FieldValueType::Scalar};
  uint64_t payloadRevision{0};
  uint64_t payloadSize{0};
  uint64_t payloadAlignment{0};
  uint32_t capabilities{0};
};
enum class TypedCustomError
{
  None,
  DescriptorSize,
  Version,
  MissingProvider,
  Capabilities,
  Layout,
  Arity,
  MissingRange,
  InvalidRange,
  MissingResources,
  InvalidDomain,
  MissingPayload
};
// Provider traits: Value(float/vec3), Payload(trivially copyable), ID,
// REVISION, CUDA_SAMPLING, RESOURCE_LEASE, and exact static sampleOptix(
// const Payload &, const vec3 &) returning Value. Optional sampleCuda has the
// same signature. Device-only implementations may be conditionally defined;
// declarations must be visible to the host's contract checks.
template <typename Provider>
constexpr TypedCustomDescriptor typedCustomDescriptor()
{
  using T = typename Provider::Value;
  using P = typename Provider::Payload;
  static_assert(Provider::ID > 0 && uint64_t(Provider::ID) <= UINT32_MAX,
      "typed provider ID must be a nonzero uint32");
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, vec3>);
  static_assert(std::is_trivially_copyable_v<P>);
  static_assert(
      alignof(P) <= 256, "payload alignment exceeds cudaMalloc guarantee");
  static_assert(std::is_same_v<decltype(&Provider::sampleOptix),
      T (*)(const P &, const vec3 &)>);
  if constexpr (Provider::CUDA_SAMPLING)
    static_assert(std::is_same_v<decltype(&Provider::sampleCuda),
        T (*)(const P &, const vec3 &)>);
  TypedCustomDescriptor d;
  d.provider = Provider::ID;
  d.payloadRevision = Provider::REVISION;
  d.payloadSize = sizeof(P);
  d.payloadAlignment = alignof(P);
  d.valueType =
      std::is_same_v<T, float> ? FieldValueType::Scalar : FieldValueType::RGB;
  d.capabilities = TypedCustomOptix
      | (Provider::CUDA_SAMPLING ? TypedCustomCuda : 0)
      | (Provider::RESOURCE_LEASE ? TypedCustomResources : 0);
  return d;
}
// Pure metadata negotiation, before allocation or sampler dispatch. A lease
// retains an immutable resource generation transitively, not a mutable field.
inline TypedCustomError validateTypedCustom(
    const TypedCustomDescriptor &request,
    const TypedCustomDescriptor *compiled,
    size_t bytes,
    size_t alignment,
    FieldValueType arity,
    bool hasRange,
    bool hasResources)
{
  if (request.structSize < sizeof(TypedCustomDescriptor))
    return TypedCustomError::DescriptorSize;
  if (request.version != TYPED_CUSTOM_CONTRACT)
    return TypedCustomError::Version;
  if (!compiled)
    return TypedCustomError::MissingProvider;
  if (request.capabilities != compiled->capabilities)
    return TypedCustomError::Capabilities;
  if (request.provider != compiled->provider
      || request.payloadRevision != compiled->payloadRevision
      || request.payloadSize != compiled->payloadSize
      || request.payloadAlignment != compiled->payloadAlignment
      || bytes != compiled->payloadSize
      || alignment != compiled->payloadAlignment)
    return TypedCustomError::Layout;
  if (request.valueType != compiled->valueType || arity != compiled->valueType)
    return TypedCustomError::Arity;
  if (!hasRange)
    return TypedCustomError::MissingRange;
  if ((compiled->capabilities & TypedCustomResources) && !hasResources)
    return TypedCustomError::MissingResources;
  return TypedCustomError::None;
}
const TypedCustomDescriptor *compiledTypedCustomProvider(uint32_t id);
const char *typedCustomErrorString(TypedCustomError error);
} // namespace visrtx
