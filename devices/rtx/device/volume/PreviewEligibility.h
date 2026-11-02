// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include "spatial_field/StoredFieldReference.h"

namespace visrtx {
namespace preview_preparation {
inline bool ordinaryOperand(float value)
{
  const double magnitude = std::abs(optical_reference::promote(value));
  return magnitude <= 0x1p40
      && (magnitude == 0. || magnitude >= std::numeric_limits<float>::min());
}
// Optimization eligibility only. These are deliberately broad dynamic-range
// margins, NOT certificates for FP32 tracking and NOT public scene limits.
template <typename T>
inline bool range(const FieldBindingGPUData<T> &binding,
    const box3 &domain,
    const StoredFieldReferenceProvider &provider,
    double &lo,
    double &hi)
{
  for (int c = 0; c < optical_reference::channels<T>(); ++c)
    if (!ordinaryOperand(optical_reference::component(binding.scale, c)))
      return false;
  optical_reference::RawRange<T> raw;
  if (binding.field >= 0) {
    optical_reference::RawRangeCertificate<T> certificate;
    if (!provider.range(binding.field, certificate)
        || !provider.fields[binding.field]->supportsFloatPreview())
      return false;
    raw = certificate.range;
    for (int c = 0; c < optical_reference::channels<T>(); ++c)
      if (!ordinaryOperand(optical_reference::component(raw.lower, c))
          || !ordinaryOperand(optical_reference::component(raw.upper, c))
          || (!optical_reference::domainContains(binding.domain, domain)
              && !ordinaryOperand(
                  optical_reference::component(binding.outside, c))))
        return false;
  }
  const auto effective = optical_reference::bindingRange(binding, domain, raw);
  lo = INFINITY;
  hi = -INFINITY;
  for (int c = 0; c < optical_reference::channels<T>(); ++c) {
    if (!ordinaryOperand(optical_reference::component(effective.lower, c))
        || !ordinaryOperand(optical_reference::component(effective.upper, c)))
      return false;
    lo = std::min(lo, double(optical_reference::component(effective.lower, c)));
    hi = std::max(hi, double(optical_reference::component(effective.upper, c)));
  }
  return lo >= -0x1p40 && hi <= 0x1p40;
}
inline bool eligible(const PrincipledVolumeBindingsGPUData &b,
    const optical_reference::Bounds &bounds,
    const StoredFieldReferenceProvider &provider)
{
  double lo, hi, emission, intensity, tint, temperature;
  if (!range(b.density, b.bounds, provider, lo, hi)
      || !range(b.transmissionColor, b.bounds, provider, lo, hi) || lo < 0x1p-20
      || hi > 1. || !range(b.transmissionDistance, b.bounds, provider, lo, hi)
      || lo < 0x1p-20 || !range(b.scatteringColor, b.bounds, provider, lo, hi)
      || !range(b.anisotropy, b.bounds, provider, lo, hi)
      || !range(b.secondaryAnisotropy, b.bounds, provider, lo, hi)
      || !range(b.secondaryWeight, b.bounds, provider, lo, hi)
      || !range(b.emission, b.bounds, provider, lo, emission)
      || !range(b.temperature, b.bounds, provider, lo, temperature)
      || !range(b.blackbodyIntensity, b.bounds, provider, lo, intensity)
      || !range(b.blackbodyTint, b.bounds, provider, lo, tint))
    return false;
  const double extinction = optical_reference::majorant(bounds);
  // Broad headroom for source sums/products. Higher temperatures/intensities
  // keep the wide path; they are not clamped or rejected. Bound gain separately
  // too: tiny absorption must not hide a large amplification of cold/subnormal
  // intermediates.
  return extinction < 0x1p60 && temperature <= 100000.
      && intensity * tint <= 0x1p20
      && emission + extinction * intensity * tint * 0x1p60 < 0x1p80;
}
} // namespace preview_preparation
} // namespace visrtx
