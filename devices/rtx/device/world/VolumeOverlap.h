// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include "gpu/gpu_decl.h"

namespace visrtx::detail {

// Host-only conservative SAT for affine boxes (including shear). Every scalar
// encloses the exact result for the committed FP32 inputs. TwoSum and FMA
// residuals avoid enlarging EXACT arithmetic, so established contact is legal.
// There is no spatial epsilon. Retry uncertain predicates in long double;
// platforms without extra precision can still return Indeterminate safely.
enum class VolumeOverlap
{
  Separate,
  Overlap,
  Indeterminate
};

template <typename T>
struct Enclosure
{
  T lo{}, hi{};
};

template <typename T>
T outward(T value, T residual, bool upper)
{
  if ((upper && residual > 0) || (!upper && residual < 0))
    return std::nextafter(value,
        upper ? std::numeric_limits<T>::infinity()
              : -std::numeric_limits<T>::infinity());
  return value;
}

template <typename T>
T addBound(T a, T b, bool upper)
{
  const T sum = a + b;
  const T bv = sum - a;
  const T residual = (a - (sum - bv)) + (b - bv);
  return outward(sum, residual, upper);
}

template <typename T>
T multiplyBound(T a, T b, bool upper)
{
  const T product = a * b;
  // A rounded-to-zero FMA residual cannot certify a subnormal product.
  if (a != 0 && b != 0 && std::abs(product) < std::numeric_limits<T>::min())
    return std::nextafter(product,
        upper ? std::numeric_limits<T>::infinity()
              : -std::numeric_limits<T>::infinity());
  return outward(product, std::fma(a, b, -product), upper);
}

template <typename T>
Enclosure<T> operator+(Enclosure<T> a, Enclosure<T> b)
{
  return {addBound(a.lo, b.lo, false), addBound(a.hi, b.hi, true)};
}

template <typename T>
Enclosure<T> operator-(Enclosure<T> a, Enclosure<T> b)
{
  return a + Enclosure<T>{-b.hi, -b.lo};
}

template <typename T>
Enclosure<T> operator*(Enclosure<T> a, Enclosure<T> b)
{
  T lo = std::numeric_limits<T>::infinity();
  T hi = -lo;
  for (T x : {a.lo, a.hi})
    for (T y : {b.lo, b.hi}) {
      lo = std::min(lo, multiplyBound(x, y, false));
      hi = std::max(hi, multiplyBound(x, y, true));
    }
  return {lo, hi};
}

template <typename T>
using EnclosedVector = std::array<Enclosure<T>, 3>;

template <typename T>
EnclosedVector<T> cross(const EnclosedVector<T> &a, const EnclosedVector<T> &b)
{
  return {a[1] * b[2] - a[2] * b[1],
      a[2] * b[0] - a[0] * b[2],
      a[0] * b[1] - a[1] * b[0]};
}

template <typename T>
Enclosure<T> dot(const EnclosedVector<T> &a, const EnclosedVector<T> &b)
{
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

template <typename T>
struct EnclosedBox
{
  std::array<EnclosedVector<T>, 8> corners;
  std::array<EnclosedVector<T>, 3> directions;
};

template <typename T>
EnclosedBox<T> encloseBox(const box3 &bounds, const mat4x3 &xfm)
{
  EnclosedBox<T> result;
  for (int c = 0; c < 3; ++c)
    for (int r = 0; r < 3; ++r)
      result.directions[c][r] = {T(xfm[c][r]), T(xfm[c][r])};
  for (int i = 0; i < 8; ++i)
    for (int r = 0; r < 3; ++r) {
      Enclosure<T> p{T(xfm[3][r]), T(xfm[3][r])};
      for (int c = 0; c < 3; ++c) {
        const T v = (i & (1 << c)) ? bounds.upper[c] : bounds.lower[c];
        p = p + result.directions[c][r] * Enclosure<T>{v, v};
      }
      result.corners[i][r] = p;
    }
  return result;
}

template <typename T>
Enclosure<T> project(
    const EnclosedBox<T> &box, const EnclosedVector<T> &axis, bool minimum)
{
  auto range = dot(box.corners[0], axis);
  for (int i = 1; i < 8; ++i) {
    const auto p = dot(box.corners[i], axis);
    if (minimum) {
      range.lo = std::min(range.lo, p.lo);
      range.hi = std::min(range.hi, p.hi);
    } else {
      range.lo = std::max(range.lo, p.lo);
      range.hi = std::max(range.hi, p.hi);
    }
  }
  return range;
}

template <typename T>
VolumeOverlap classifyBoxes(
    const box3 &a, const mat4x3 &aXfm, const box3 &b, const mat4x3 &bXfm)
{
  const auto ea = encloseBox<T>(a, aXfm);
  const auto eb = encloseBox<T>(b, bXfm);
  bool uncertain = false;
  auto separates = [&](const EnclosedVector<T> &axis) {
    bool zero = true;
    for (const auto v : axis)
      zero &= v.lo == 0 && v.hi == 0;
    if (zero)
      return false; // parallel edge directions produce no SAT axis
    const auto ab = project(eb, axis, true) - project(ea, axis, false);
    const auto ba = project(ea, axis, true) - project(eb, axis, false);
    // A certified nonnegative gap excludes positive-volume overlap. This
    // includes exact contact; an uncertain sign never licenses omission.
    if (ab.lo >= 0 || ba.lo >= 0)
      return true;
    uncertain |= !(ab.hi < 0 && ba.hi < 0);
    return false;
  };
  // Conservative world-AABB broad phase. Its overlap is NOT proof of overlap.
  for (int c = 0; c < 3; ++c) {
    EnclosedVector<T> axis{};
    axis[c] = {1, 1};
    if (separates(axis))
      return VolumeOverlap::Separate;
  }
  uncertain = false;
  // Parallelepiped SAT: six face normals and nine edge cross-products.
  for (int c = 0; c < 3; ++c) {
    if (separates(cross(ea.directions[(c + 1) % 3], ea.directions[(c + 2) % 3]))
        || separates(
            cross(eb.directions[(c + 1) % 3], eb.directions[(c + 2) % 3])))
      return VolumeOverlap::Separate;
    for (int d = 0; d < 3; ++d)
      if (separates(cross(ea.directions[c], eb.directions[d])))
        return VolumeOverlap::Separate;
  }
  return uncertain ? VolumeOverlap::Indeterminate : VolumeOverlap::Overlap;
}

inline VolumeOverlap volumeOverlap(
    const box3 &a, const mat4x3 &aXfm, const box3 &b, const mat4x3 &bXfm)
{
  auto result = classifyBoxes<double>(a, aXfm, b, bXfm);
  if (result == VolumeOverlap::Indeterminate)
    result = classifyBoxes<long double>(a, aXfm, b, bXfm);
  return result;
}
} // namespace visrtx::detail
