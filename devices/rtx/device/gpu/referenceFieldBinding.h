// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "volumeOpticsReference.h"

#include <cstddef>

namespace visrtx {
namespace optical_reference {

// A global raw hull encloses returned samples, not just active sparse voxels.
// Producers must use matching decoding/filtering arithmetic. Empty is distinct
// from a singleton zero. These structures do not certify legacy tex3D results.
template <typename T>
struct RawRange
{
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, vec3>);
  T lower{};
  T upper{};
  bool empty{true};
};

VISRTX_HOST_DEVICE float component(float value, int)
{
  return value;
}

VISRTX_HOST_DEVICE float component(const vec3 &value, int channel)
{
  return value[channel];
}

VISRTX_HOST_DEVICE void setComponent(float &value, int, float x)
{
  value = x;
}

VISRTX_HOST_DEVICE void setComponent(vec3 &value, int channel, float x)
{
  value[channel] = x;
}

template <typename T>
constexpr int channels()
{
  return std::is_same_v<T, float> ? 1 : 3;
}

// Range construction assumes finite raw values. This is not a validation scan
// and deliberately does not promise to detect invalid samples/NaNs.
template <typename T>
VISRTX_HOST_DEVICE RawRange<T> extendRange(RawRange<T> range, const T &value)
{
  if (range.empty)
    return {value, value, false};
  for (int c = 0; c < channels<T>(); ++c) {
    const float v = component(value, c);
    if (promote(v) < promote(component(range.lower, c)))
      setComponent(range.lower, c, v);
    if (promote(v) > promote(component(range.upper, c)))
      setComponent(range.upper, c, v);
  }
  return range;
}

template <typename T>
VISRTX_HOST_DEVICE RawRange<T> mergeRanges(RawRange<T> a, const RawRange<T> &b)
{
  if (b.empty)
    return a;
  return extendRange(extendRange(a, b.lower), b.upper);
}

template <typename T>
VISRTX_HOST_DEVICE RawRange<T> canonicalRange(const T *values, size_t count)
{
  RawRange<T> result;
  for (size_t i = 0; i < count; ++i)
    result = extendRange(result, values[i]);
  return result;
}

// Binary32 endpoint products fit exactly in binary64 before the final RN32.
// Sign-aware endpoint evaluation therefore encloses this same sampled product.
template <typename T>
VISRTX_HOST_DEVICE T bindingProduct(const T &scale, const T &sample)
{
  T result{};
  for (int c = 0; c < channels<T>(); ++c)
    setComponent(result,
        c,
        narrow(multiply(
            promote(component(scale, c)), promote(component(sample, c)))));
  return result;
}

// Controlled convex interpolation for canonical float endpoints, w in [0,1].
// The double expression is monotone in the endpoints and RN32 reproduces equal
// endpoints exactly. Thus its returned value stays in their float32 hull, even
// for opposite-sign values whose float32 difference would overflow.
template <typename T>
VISRTX_HOST_DEVICE T canonicalLerp(const T &a, const T &b, float weight)
{
  const double w = promote(weight);
  const double other = subtract(1., w);
  T result{};
  for (int c = 0; c < channels<T>(); ++c)
    setComponent(result,
        c,
        narrow(add(multiply(other, promote(component(a, c))),
            multiply(w, promote(component(b, c))))));
  return result;
}

VISRTX_HOST_DEVICE bool nonemptyDomain(const box3 &domain)
{
  for (int c = 0; c < 3; ++c)
    if (!(promote(domain.lower[c]) < promote(domain.upper[c])))
      return false;
  return true;
}

// Membership comparisons use the same no-FTZ promotion as the optical inputs.
// No sample-content or revision checks occur in the evaluation path.
VISRTX_HOST_DEVICE bool contains(const box3 &domain, const vec3 &p)
{
  if (!nonemptyDomain(domain))
    return false;
  for (int c = 0; c < 3; ++c)
    if (!(promote(p[c]) >= promote(domain.lower[c])
            && promote(p[c]) <= promote(domain.upper[c])))
      return false;
  return true;
}

// Each operand is a nonempty closed domain. Touching faces/edges still contain
// possible endpoint samples even though their intersection has zero measure.
VISRTX_HOST_DEVICE bool domainsIntersect(const box3 &a, const box3 &b)
{
  if (!nonemptyDomain(a) || !nonemptyDomain(b))
    return false;
  for (int c = 0; c < 3; ++c)
    if (promote(a.upper[c]) < promote(b.lower[c])
        || promote(b.upper[c]) < promote(a.lower[c]))
      return false;
  return true;
}

VISRTX_HOST_DEVICE bool domainContains(const box3 &outer, const box3 &inner)
{
  if (!nonemptyDomain(outer) || !nonemptyDomain(inner))
    return false;
  for (int c = 0; c < 3; ++c)
    if (promote(outer.lower[c]) > promote(inner.lower[c])
        || promote(outer.upper[c]) < promote(inner.upper[c]))
      return false;
  return true;
}

template <typename T, typename Sampler>
VISRTX_HOST_DEVICE T evaluateBinding(
    const FieldBindingGPUData<T> &binding, const vec3 &p, Sampler &sampler)
{
  static_assert(
      std::is_same_v<decltype(sampler.template sample<T>(binding.field, p)),
          T>);
  if (binding.field == -1)
    return binding.scale;
  return bindingProduct(binding.scale,
      contains(binding.domain, p) ? sampler.template sample<T>(binding.field, p)
                                  : binding.outside);
}

// This is raw-to-effective construction, not validity of the physical property.
// Precondition: raw is nonempty and encloses the actual sampler whenever the
// medium and field domains intersect. Preparation enforces its availability.
// Overflowed effective products fail the subsequent metadata/optics check.
template <typename T>
VISRTX_HOST_DEVICE RawRange<T> bindingRange(
    const FieldBindingGPUData<T> &binding,
    const box3 &mediumDomain,
    const RawRange<T> &raw)
{
  if (binding.field == -1)
    return {binding.scale, binding.scale, false};
  RawRange<T> result;
  if (domainsIntersect(mediumDomain, binding.domain) && !raw.empty) {
    result = extendRange(result, bindingProduct(binding.scale, raw.lower));
    result = extendRange(result, bindingProduct(binding.scale, raw.upper));
  }
  if (!domainContains(binding.domain, mediumDomain))
    result =
        extendRange(result, bindingProduct(binding.scale, binding.outside));
  return result;
}

// All bound properties use the controlled multiplier, including sample-only
// source/phase fields. This overload is disconnected from renderer dispatch.
template <typename Sampler, typename Blackbody>
VISRTX_HOST_DEVICE Coefficients evaluateBindings(
    const PrincipledVolumeBindingsGPUData &b,
    const vec3 &p,
    Sampler &sampler,
    Blackbody &blackbody)
{
  if (!contains(b.bounds, p))
    return {};
  PrincipledVolumeValues v;
  v.density = evaluateBinding(b.density, p, sampler);
  v.scatteringColor = evaluateBinding(b.scatteringColor, p, sampler);
  v.transmissionColor = evaluateBinding(b.transmissionColor, p, sampler);
  v.transmissionDistance = evaluateBinding(b.transmissionDistance, p, sampler);
  v.emission = evaluateBinding(b.emission, p, sampler);
  v.temperature = evaluateBinding(b.temperature, p, sampler);
  v.blackbodyIntensity = evaluateBinding(b.blackbodyIntensity, p, sampler);
  v.blackbodyTint = evaluateBinding(b.blackbodyTint, p, sampler);
  // Phase is evaluated separately by consumers using evaluateBinding; no source
  // or extinction term depends on it, so don't sample it just for coefficients.
  return evaluate(v, blackbody);
}

VISRTX_HOST_DEVICE bool validDomainMetadata(const box3 &domain)
{
  const double limit = promote(std::numeric_limits<float>::max());
  for (int c = 0; c < 3; ++c) {
    const double lo = promote(domain.lower[c]);
    const double hi = promote(domain.upper[c]);
    if (!(lo >= -limit && lo <= limit && hi >= -limit && hi <= limit))
      return false;
  }
  // Reversed or zero-width field domains are valid empty ROI intersections.
  return true;
}

template <typename T>
VISRTX_HOST_DEVICE bool validRawRange(const RawRange<T> &range)
{
  if (range.empty)
    return false;
  const double limit = promote(std::numeric_limits<float>::max());
  for (int c = 0; c < channels<T>(); ++c)
    if (!validRange({component(range.lower, c), component(range.upper, c)},
            -limit,
            limit,
            false))
      return false;
  return true;
}

} // namespace optical_reference
} // namespace visrtx
