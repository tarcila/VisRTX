// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "gpu/referenceFieldBinding.h"
#include "utility/ScopedReferenceRounding.h"

namespace visrtx {
namespace optical_reference {

// Identity is stable field ownership, NOT registry slot/address identity.
// revision changes whenever sample payload, domain/ROI, mapping/filter, decode
// or range data changes. contract identifies compatible sampling
// arithmetic/ABI. Producers publish immutable descriptors during supported
// prepare/render use.
struct FieldRevision
{
  uint64_t identity{0};
  uint64_t revision{0};
  uint64_t contract{0};
};

inline bool operator==(const FieldRevision &a, const FieldRevision &b)
{
  return a.identity == b.identity && a.revision == b.revision
      && a.contract == b.contract;
}

struct FieldSampleDescriptor
{
  FieldRevision key;
  FieldValueType valueType{FieldValueType::Scalar};
  box3 domain; // finalized bounds intersect ROI, possibly empty
};

template <typename T>
struct RawRangeCertificate
{
  FieldRevision key;
  RawRange<T> range; // global enclosure over descriptor.domain, including edges
};

struct FieldDependency
{
  int32_t field{-1};
  FieldSampleDescriptor sample;
};

// A prepared value is a metadata snapshot, not an allocation owner or lease.
// The field/volume owners must retain the matching sampler payload through use.
// Fixed capacity comes from the eleven public optical binding roles, not an
// overlap/occurrence cap. Duplicate bindings to one field share one dependency.
struct PreparedVolumeBindings
{
  PrincipledVolumeBindingsGPUData bindings;
  Bounds bounds;
  FieldDependency dependencies[11];
  int dependencyCount{0};
};

enum class PreparationError
{
  None,
  InvalidMediumDomain,
  FieldUnavailable,
  TypeMismatch,
  InvalidFieldMetadata,
  RevisionMismatch,
  RangeUnavailable,
  InvalidRangeMetadata,
  OpticalRangeUnavailable
};

struct PreparationResult
{
  PreparationError error{PreparationError::None};
  const char *property{nullptr};
  int32_t field{-1};
};

inline bool sameDescriptor(
    const FieldSampleDescriptor &a, const FieldSampleDescriptor &b)
{
  if (!(a.key == b.key) || a.valueType != b.valueType)
    return false;
  for (int c = 0; c < 3; ++c)
    if (floatBits(a.domain.lower[c]) != floatBits(b.domain.lower[c])
        || floatBits(a.domain.upper[c]) != floatBits(b.domain.upper[c]))
      return false;
  return true;
}

// Provider interface (host preparation only):
// bool describe(int32_t, FieldSampleDescriptor&) const;
// template<class T> bool range(int32_t, RawRangeCertificate<T>&) const;
// A successful describe promises a typed sampler even for sample-only roles.
// A range is requested only for d/C/D where the field domain is reachable.
// Never obtain these records by guessing from a legacy hardware texture hull.
template <typename T, typename Provider>
inline PreparationError prepareBinding(const FieldBindingGPUData<T> &input,
    const box3 &mediumDomain,
    const Provider &provider,
    bool needsRange,
    PreparedVolumeBindings &candidate,
    FieldBindingGPUData<T> &output,
    RawRange<T> &effective)
{
  output = input;
  if (input.field == -1) {
    effective = {input.scale, input.scale, false};
    return PreparationError::None;
  }
  FieldSampleDescriptor sample;
  if (input.field < 0 || !provider.describe(input.field, sample))
    return PreparationError::FieldUnavailable;
  if (!fieldBindingAccepts<T>(sample.valueType))
    return PreparationError::TypeMismatch;
  if (!sample.key.identity || !sample.key.revision || !sample.key.contract
      || !validDomainMetadata(sample.domain))
    return PreparationError::InvalidFieldMetadata;
  output.domain = sample.domain;

  bool known = false;
  for (int i = 0; i < candidate.dependencyCount; ++i) {
    const auto &dependency = candidate.dependencies[i];
    if (dependency.field != input.field)
      continue;
    if (!sameDescriptor(dependency.sample, sample))
      return PreparationError::RevisionMismatch;
    known = true;
    break;
  }
  if (!known)
    candidate.dependencies[candidate.dependencyCount++] = {input.field, sample};

  if (!needsRange)
    return PreparationError::None;
  RawRange<T> raw;
  if (domainsIntersect(mediumDomain, sample.domain)) {
    RawRangeCertificate<T> certificate;
    if (!provider.template range<T>(input.field, certificate))
      return PreparationError::RangeUnavailable;
    if (!(certificate.key == sample.key))
      return PreparationError::RevisionMismatch;
    if (!validRawRange(certificate.range))
      return PreparationError::InvalidRangeMetadata;
    raw = certificate.range;
  }
  effective = bindingRange(output, mediumDomain, raw);
  if (!validRawRange(effective))
    return PreparationError::InvalidRangeMetadata;
  return PreparationError::None;
}

// Check once at preparation/publication, not at samples. This detects revision,
// slot reuse, arity and domain changes, but is not synchronization with a
// racing producer. Caller uses existing supported lifecycle ordering and
// retained owners.
template <typename Provider>
inline bool matchesPreparedFields(
    const PreparedVolumeBindings &prepared, const Provider &provider)
{
  for (int i = 0; i < prepared.dependencyCount; ++i) {
    const auto &dependency = prepared.dependencies[i];
    FieldSampleDescriptor current;
    if (!provider.describe(dependency.field, current)
        || !sameDescriptor(current, dependency.sample))
      return false;
  }
  return true;
}

// Transactional: output is untouched on any failure, including final recheck.
// The returned property/index can feed existing preparation diagnostics; this
// is not a new frame-result/status mechanism.
template <typename Provider>
inline PreparationResult prepareVolumeBindings(
    const PrincipledVolumeBindingsGPUData &input,
    const Provider &provider,
    PreparedVolumeBindings &output)
{
  const ScopedReferenceRounding rounding;
  if (!validDomainMetadata(input.bounds) || !nonemptyDomain(input.bounds))
    return {PreparationError::InvalidMediumDomain, "bounds", -1};
  PreparedVolumeBindings candidate;
  candidate.bindings = input;
  RawRange<float> density, distance, unusedScalar;
  RawRange<vec3> transmission, unusedRGB;
  PreparationResult result;
  auto prepare = [&](const auto &binding,
                     auto &target,
                     auto &range,
                     bool needsRange,
                     const char *property) {
    if (result.error != PreparationError::None)
      return;
    const auto error = prepareBinding(
        binding, input.bounds, provider, needsRange, candidate, target, range);
    if (error != PreparationError::None)
      result = {error, property, binding.field};
  };
  auto &b = candidate.bindings;
  prepare(input.density, b.density, density, true, "density");
  prepare(input.transmissionColor,
      b.transmissionColor,
      transmission,
      true,
      "transmissionColor");
  prepare(input.transmissionDistance,
      b.transmissionDistance,
      distance,
      true,
      "transmissionDistance");
  prepare(input.scatteringColor,
      b.scatteringColor,
      unusedRGB,
      false,
      "scatteringColor");
  prepare(input.anisotropy, b.anisotropy, unusedScalar, false, "anisotropy");
  prepare(input.secondaryAnisotropy,
      b.secondaryAnisotropy,
      unusedScalar,
      false,
      "secondaryAnisotropy");
  prepare(input.secondaryWeight,
      b.secondaryWeight,
      unusedScalar,
      false,
      "secondaryWeight");
  prepare(input.emission, b.emission, unusedRGB, false, "emission");
  prepare(input.temperature, b.temperature, unusedScalar, false, "temperature");
  prepare(input.blackbodyIntensity,
      b.blackbodyIntensity,
      unusedScalar,
      false,
      "blackbodyIntensity");
  prepare(
      input.blackbodyTint, b.blackbodyTint, unusedRGB, false, "blackbodyTint");
  if (result.error != PreparationError::None)
    return result;

  const EffectiveRanges ranges{{density.lower, density.upper},
      {{transmission.lower.x, transmission.upper.x},
          {transmission.lower.y, transmission.upper.y},
          {transmission.lower.z, transmission.upper.z}},
      {distance.lower, distance.upper}};
  // Report the relevant property when a loose range cannot establish physical
  // validity. This does not assert that actual samples violate the contract.
  const double maxFloat = promote(std::numeric_limits<float>::max());
  if (!validRange(ranges.density, 0., maxFloat, false))
    return {
        PreparationError::OpticalRangeUnavailable, "density", b.density.field};
  if (!validRange(ranges.distance, 0., maxFloat, true))
    return {PreparationError::OpticalRangeUnavailable,
        "transmissionDistance",
        b.transmissionDistance.field};
  for (int c = 0; c < 3; ++c)
    if (!validRange(ranges.transmission[c], 0., 1., true))
      return {PreparationError::OpticalRangeUnavailable,
          "transmissionColor",
          b.transmissionColor.field};
  if (!buildBounds(ranges, candidate.bounds))
    return {PreparationError::OpticalRangeUnavailable, nullptr, -1};
  if (!matchesPreparedFields(candidate, provider))
    return {PreparationError::RevisionMismatch, nullptr, -1};
  output = candidate;
  return {};
}

} // namespace optical_reference
} // namespace visrtx
