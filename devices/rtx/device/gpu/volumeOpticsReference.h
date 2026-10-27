// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "volumeOptics.h"

#include <cmath>
#include <cstring>
#include <limits>

namespace visrtx {
namespace optical_reference {

static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559
    && std::numeric_limits<float>::digits == 24);
static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559
    && std::numeric_limits<double>::digits == 53);

// This seam is not used by renderer dispatch yet. Inputs are valid effective
// float32 properties, not raw field ranges. Host callers require IEEE binary64,
// round-to-nearest, gradual underflow and no unsafe-math/reassociation flags.
// CUDA operations explicitly retain their rounding under --use_fast_math.
VISRTX_HOST_DEVICE uint32_t floatBits(float value)
{
#ifdef __CUDA_ARCH__
  return __float_as_uint(value);
#else
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
#endif
}

VISRTX_HOST_DEVICE double doubleFromBits(uint64_t bits)
{
#ifdef __CUDA_ARCH__
  return __longlong_as_double(static_cast<long long>(bits));
#else
  double value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
#endif
}

// Bit expansion prevents an implicit fast-math float conversion from flushing
// a valid subnormal input. This also preserves signed zero.
VISRTX_HOST_DEVICE double promote(float value)
{
  const uint32_t bits = floatBits(value);
  const uint64_t sign = uint64_t(bits >> 31) << 63;
  uint32_t fraction = bits & 0x7fffffu;
  const uint32_t exponent = (bits >> 23) & 255u;
  if (exponent == 255u)
    return doubleFromBits(
        sign | (uint64_t(2047) << 52) | (uint64_t(fraction) << 29));
  if (exponent != 0u)
    return doubleFromBits(
        sign | (uint64_t(exponent + 896u) << 52) | (uint64_t(fraction) << 29));
  if (fraction == 0u)
    return doubleFromBits(sign);
  int unbiased = -126;
  while ((fraction & 0x800000u) == 0u) {
    fraction <<= 1;
    --unbiased;
  }
  return doubleFromBits(sign | (uint64_t(unbiased + 1023) << 52)
      | (uint64_t(fraction & 0x7fffffu) << 29));
}

// Thermal providers may retain double radiance until absorption coupling;
// narrowing a cold B(T) first can lose a representable sigma_a*B(T) product.
VISRTX_HOST_DEVICE double promote(double value)
{
  return value;
}

VISRTX_HOST_DEVICE float narrow(double value)
{
#ifdef __CUDA_ARCH__
  float result;
  asm("cvt.rn.f32.f64 %0, %1;" : "=f"(result) : "d"(value));
  return result;
#else
  volatile float result = static_cast<float>(value);
  return result;
#endif
}

VISRTX_HOST_DEVICE double add(double a, double b)
{
#ifdef __CUDA_ARCH__
  return __dadd_rn(a, b);
#else
  volatile double result = a + b;
  return result;
#endif
}

VISRTX_HOST_DEVICE double subtract(double a, double b)
{
#ifdef __CUDA_ARCH__
  return __dsub_rn(a, b);
#else
  volatile double result = a - b;
  return result;
#endif
}

VISRTX_HOST_DEVICE double multiply(double a, double b)
{
#ifdef __CUDA_ARCH__
  return __dmul_rn(a, b);
#else
  volatile double result = a * b;
  return result;
#endif
}

VISRTX_HOST_DEVICE double divide(double a, double b)
{
#ifdef __CUDA_ARCH__
  return __ddiv_rn(a, b);
#else
  volatile double result = a / b;
  return result;
#endif
}

// Upward helpers below are for nonnegative operands only. The host deliberately
// expands RN by one ULP; CUDA uses RU directly. Bounds need not be
// bit-identical. Exact zero identities avoid manufacturing extinction in
// transparent inputs.
VISRTX_HOST_DEVICE double addUpper(double a, double b)
{
  if (a == 0.)
    return b;
  if (b == 0.)
    return a;
#ifdef __CUDA_ARCH__
  return __dadd_ru(a, b);
#else
  return std::nextafter(add(a, b), std::numeric_limits<double>::infinity());
#endif
}

VISRTX_HOST_DEVICE double multiplyUpper(double a, double b)
{
  if (a == 0. || b == 0.)
    return 0.;
#ifdef __CUDA_ARCH__
  return __dmul_ru(a, b);
#else
  return std::nextafter(
      multiply(a, b), std::numeric_limits<double>::infinity());
#endif
}

VISRTX_HOST_DEVICE double divideUpper(double a, double b)
{
  if (a == 0.)
    return 0.;
#ifdef __CUDA_ARCH__
  return __ddiv_ru(a, b);
#else
  return std::nextafter(divide(a, b), std::numeric_limits<double>::infinity());
#endif
}

// C = m * 2^-k, .5 <= m < 1. For positive float32 C<=1,
// subtraction/addition in (1-m)/(1+m) are exact binary64 operations.
struct LogReduction
{
  double m;
  int k;
};

VISRTX_HOST_DEVICE LogReduction reduceLog(uint32_t bits)
{
  int exponent = int((bits >> 23) & 255u) - 127;
  uint32_t fraction = bits & 0x7fffffu;
  if ((bits >> 23) == 0u) {
    exponent = -126;
    while ((fraction & 0x800000u) == 0u) {
      fraction <<= 1;
      --exponent;
    }
    fraction &= 0x7fffffu;
  }
  return {doubleFromBits((uint64_t(1022) << 52) | (uint64_t(fraction) << 29)),
      -exponent - 1};
}

template <bool Upper>
VISRTX_HOST_DEVICE double logPolynomial(double z)
{
  const double z2 = Upper ? multiplyUpper(z, z) : multiply(z, z);
  double power = z;
  double sum = 0.;
  for (int j = 0; j < 20; ++j) {
    const double term = Upper ? divideUpper(power, double(2 * j + 1))
                              : divide(power, double(2 * j + 1));
    sum = Upper ? addUpper(sum, term) : add(sum, term);
    if (j != 19)
      power = Upper ? multiplyUpper(power, z2) : multiply(power, z2);
  }
  return Upper ? multiplyUpper(2., sum) : multiply(2., sum);
}

template <bool Upper>
VISRTX_HOST_DEVICE VISRTX_NOINLINE double negativeLogBits(uint32_t bits)
{
  if (bits == 0x3f800000u)
    return 0.;
  const auto reduction = reduceLog(bits);
  const double numerator = subtract(1., reduction.m);
  const double denominator = add(1., reduction.m);
  const double z = Upper ? divideUpper(numerator, denominator)
                         : divide(numerator, denominator);
  const double ln2 =
      logPolynomial<Upper>(Upper ? divideUpper(1., 3.) : divide(1., 3.));
  const double p = logPolynomial<Upper>(z);
  return Upper ? addUpper(multiplyUpper(double(reduction.k), ln2), p)
               : add(multiply(double(reduction.k), ln2), p);
}

// Approximation to the physical -ln(C), not a correctly-rounded libm function.
// Accuracy and enclosure of the *executed graph* are separate: see the test
// oracle and docs/volume-optics-reference.md. Precondition: 0<C<=1.
VISRTX_HOST_DEVICE double negativeLog(float transmission)
{
  return negativeLogBits<false>(floatBits(transmission));
}

struct EffectiveRange
{
  float lower;
  float upper;
};

// RN may change by an ULP at exponent boundaries. Enclose each bin separately
// rather than assume global monotonicity of the polynomial implementation.
VISRTX_HOST_DEVICE double negativeLogUpper(EffectiveRange range)
{
  const uint32_t lo = floatBits(range.lower);
  const uint32_t hi = floatBits(range.upper);
  double result = 0.;
  for (int k = 0; k <= 148; ++k) {
    const uint32_t begin =
        k <= 125 ? uint32_t(126 - k) << 23 : uint32_t(1) << (148 - k);
    const uint32_t end = k <= 125 ? begin + 0x7fffffu : 2 * begin - 1;
    const uint32_t first = lo > begin ? lo : begin;
    const uint32_t last = hi < end ? hi : end;
    if (first <= last) {
      const double bound = negativeLogBits<true>(first);
      result = result > bound ? result : bound;
    }
  }
  return result;
}

struct Coefficients
{
  double absorption[3]{};
  double scattering[3]{};
  double emission[3]{};
};

// Bounds contain A and S separately to match the sampled reduction graph.
// No emission bound is required by the all-candidate source estimator.
struct Bounds
{
  double absorption[3]{};
  double scattering[3]{};
};

struct EffectiveRanges
{
  EffectiveRange density;
  EffectiveRange transmission[3];
  EffectiveRange distance;
};

// Extinction/partition and explicit source only. Use the provider overload for
// temperature-dependent source; the phase parameters remain unchanged in v.
VISRTX_HOST_DEVICE Coefficients evaluate(const PrincipledVolumeValues &v)
{
  Coefficients result;
  const double density = promote(v.density);
  const double distance = promote(v.transmissionDistance);
  for (int c = 0; c < 3; ++c) {
    result.emission[c] = promote(v.emission[c]);
    if (density == 0. || floatBits(v.transmissionColor[c]) == 0x3f800000u)
      continue;
    const double e = divide(
        multiply(negativeLog(v.transmissionColor[c]), density), distance);
    const double albedo = promote(v.scatteringColor[c]);
    result.scattering[c] = multiply(e, albedo);
    result.absorption[c] = multiply(e, subtract(1., albedo));
  }
  return result;
}

// Provider returns calibrated RGB, as in the existing optical seam. Thermal
// coupling retains double A; no narrowing before multiplication by blackbody.
template <typename Blackbody>
VISRTX_HOST_DEVICE Coefficients evaluate(
    const PrincipledVolumeValues &v, Blackbody &blackbody)
{
  Coefficients result = evaluate(v);
  double weights[3];
  bool hasThermal = false;
  for (int c = 0; c < 3; ++c) {
    weights[c] =
        multiply(multiply(result.absorption[c], promote(v.blackbodyTint[c])),
            promote(v.blackbodyIntensity));
    hasThermal |= weights[c] > 0.;
  }
  if (promote(v.temperature) > 0. && hasThermal) {
    const auto radiance = blackbody(v.temperature);
    for (int c = 0; c < 3; ++c)
      result.emission[c] =
          add(result.emission[c], multiply(weights[c], promote(radiance[c])));
  }
  return result;
}

// Construction metadata check, not per-sample validation. False means this
// interval certificate is unavailable, not that field contents are invalid.
VISRTX_HOST_DEVICE bool validRange(
    EffectiveRange r, double minimum, double maximum, bool strictMinimum)
{
  const double lo = promote(r.lower);
  const double hi = promote(r.upper);
  return (strictMinimum ? lo > minimum : lo >= minimum) && hi >= lo
      && hi <= maximum;
}

VISRTX_HOST_DEVICE bool buildBounds(
    const EffectiveRanges &ranges, Bounds &output)
{
  const double maxFloat = promote(std::numeric_limits<float>::max());
  if (!validRange(ranges.density, 0., maxFloat, false)
      || !validRange(ranges.distance, 0., maxFloat, true))
    return false;
  for (int c = 0; c < 3; ++c)
    if (!validRange(ranges.transmission[c], 0., 1., true))
      return false;
  Bounds result;
  for (int c = 0; c < 3; ++c) {
    const double e =
        divideUpper(multiplyUpper(negativeLogUpper(ranges.transmission[c]),
                        promote(ranges.density.upper)),
            promote(ranges.distance.lower));
    // Effective albedo is in [0,1]. Each rounded partition is <= E. This
    // intentionally loose bound needs no albedo-range provider capability.
    result.absorption[c] = e;
    result.scattering[c] = e;
  }
  output = result;
  return true;
}

VISRTX_HOST_DEVICE Coefficients combine(
    const Coefficients &a, const Coefficients &b)
{
  Coefficients result;
  for (int c = 0; c < 3; ++c) {
    result.absorption[c] = add(a.absorption[c], b.absorption[c]);
    result.scattering[c] = add(a.scattering[c], b.scattering[c]);
    result.emission[c] = add(a.emission[c], b.emission[c]);
  }
  return result;
}

VISRTX_HOST_DEVICE Bounds combineBounds(const Bounds &a, const Bounds &b)
{
  Bounds result;
  for (int c = 0; c < 3; ++c) {
    result.absorption[c] = addUpper(a.absorption[c], b.absorption[c]);
    result.scattering[c] = addUpper(a.scattering[c], b.scattering[c]);
  }
  return result;
}

VISRTX_HOST_DEVICE double extinction(const Coefficients &v, int channel)
{
  return add(v.absorption[channel], v.scattering[channel]);
}

VISRTX_HOST_DEVICE double majorant(const Bounds &v)
{
  double result = 0.;
  for (int c = 0; c < 3; ++c) {
    const double t = addUpper(v.absorption[c], v.scattering[c]);
    result = result > t ? result : t;
  }
  return result;
}

// Retain TF's existing alpha ceiling/epsilon; distance replaces the old float
// reciprocal so the new path cannot overflow before a representable quotient.
VISRTX_HOST_DEVICE float transferTransmission(float opacity)
{
  double alpha = promote(opacity);
  alpha = alpha < 0. ? 0. : (alpha > 1. - 0x1p-24 ? 1. - 0x1p-24 : alpha);
  if (alpha <= promote(1e-7f))
    return 1.f;
  return narrow(subtract(1., alpha));
}

VISRTX_HOST_DEVICE Coefficients evaluateTransfer(
    const vec4 &colorOpacity, float unitDistance)
{
  Coefficients result;
  if (!(promote(unitDistance) > 0.))
    return result;
  const double e = divide(
      negativeLog(transferTransmission(colorOpacity.w)), promote(unitDistance));
  for (int c = 0; c < 3; ++c) {
    result.absorption[c] = e;
    result.emission[c] = multiply(e, promote(colorOpacity[c]));
  }
  return result;
}

VISRTX_HOST_DEVICE bool buildTransferBounds(
    EffectiveRange opacity, EffectiveRange distance, Bounds &output)
{
  const double maxFloat = promote(std::numeric_limits<float>::max());
  if (!validRange(opacity, -maxFloat, maxFloat, false)
      || !validRange(distance, 0., maxFloat, true))
    return false;
  const EffectiveRange transmission{
      transferTransmission(opacity.upper), transferTransmission(opacity.lower)};
  const double e =
      divideUpper(negativeLogUpper(transmission), promote(distance.lower));
  Bounds result;
  for (int c = 0; c < 3; ++c)
    result.absorption[c] = e;
  output = result;
  return true;
}

} // namespace optical_reference
} // namespace visrtx
