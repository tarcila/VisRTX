// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cmath>
#include "fixtures.h"
namespace native_test::oracle {
using D = glm::dvec3;
constexpr double PI = 3.1415926535897932384626433832795;
inline double fifth(double x)
{
  return x * x * x * x * x;
}
inline double lum(D v)
{
  return .2126 * v.x + .7152 * v.y + .0722 * v.z;
}
inline double fc(double x)
{
  return .04 + .96 * fifth(1 - x);
}
inline double distribution(double nh, double a)
{
  const double t = 1 + (a * a - 1) * nh * nh;
  return a * a / (PI * t * t);
}
inline double lambda(double c, double a)
{
  const double c2 = std::max(c * c, 1e-8);
  return .5 * (std::sqrt(1 + a * a * (1 - c2) / c2) - 1);
}
inline double smith(double v, double l, double a)
{
  return 1 / (1 + lambda(v, a) + lambda(l, a));
}
inline D fresnel(const PhysicallyBasedShadingState &s, double c)
{
  const double f = std::pow((1 - double(s.eta)) / (1 + double(s.eta)), 2);
  const D f0 = (1 - double(s.metallic)) * glm::min(D(1), D(s.specularColor) * f)
          * double(s.specular)
      + double(s.metallic) * D(s.baseColor);
  const D f90(
      (1 - double(s.metallic)) * double(s.specular) + double(s.metallic));
  return f0 + (f90 - f0) * fifth(1 - std::abs(c));
}
inline D transmission(const PhysicallyBasedShadingState &s)
{
  // Frozen fixtures have zero thickness; no copied production helper is used.
  return D(s.baseColor) * ((1 - double(s.metallic)) * s.transmission);
}
inline bool fold(const PhysicallyBasedShadingState &s, D v, D h)
{
  if (!(lum(transmission(s)) > 0))
    return false;
  const double vh = glm::dot(v, h), eta = s.eta,
               k = 1 - eta * eta * (1 - vh * vh);
  if (k < 0)
    return true;
  const D t = -eta * v + (eta * vh - std::sqrt(k)) * h;
  return glm::length(t) < 1e-6 || glm::dot(t, D(s.viewAdjustedNormal)) >= 0;
}
inline double coatPick(const PhysicallyBasedShadingState &s, D v)
{
  return std::clamp(double(s.clearcoat)
          * fc(std::max(0., glm::dot(D(s.viewAdjustedClearcoatNormal), v))),
      0.,
      1.);
}
inline D kernel(const Fixture &f, D l)
{
  const auto &s = f.state;
  const D v(f.view), ng(s.geometricNormal), n(s.viewAdjustedNormal),
      nc(s.viewAdjustedClearcoatNormal);
  if (glm::dot(ng, v) <= 0 || glm::dot(ng, l) <= 0)
    return D(0);
  const double nv = glm::dot(n, v), nl = glm::dot(n, l), cv = glm::dot(nc, v),
               cl = glm::dot(nc, l);
  D result(0);
  if (!(nv > 0 && nl > 0) && !(s.clearcoat > 0 && cv > 0 && cl > 0))
    return result;
  const D h = glm::normalize(v + l);
  const double vh = std::max(0., glm::dot(v, h));
  if (nv > 0 && nl > 0) {
    const double vs = std::max(nv, 1e-6), ls = std::max(nl, 1e-6),
                 nh = std::max(0., glm::dot(n, h));
    const double a = std::max(double(s.roughness) * s.roughness, 1e-4);
    const D ed = glm::max(D(0), D(1) - fresnel(s, vs)) * D(s.baseColor)
        * ((1 - double(s.metallic)) * (1 - double(s.transmission))
            * s.occlusion);
    const D er = fold(s, v, h) ? D(1) : fresnel(s, vh);
    const double out = s.clearcoat == 0
        ? 1
        : std::clamp(1 - double(s.clearcoat) * fc(std::abs(cl)), 0., 1.);
    result =
        (ed * (nl / PI)
            + er
                * (distribution(nh, a) * smith(vs, ls, a) * nl / (4 * vs * ls)))
        * ((1 - coatPick(s, v)) * out);
    if (s.sheenColor != vec3(0)) {
      const double as =
          std::max(double(s.sheenRoughness) * s.sheenRoughness, 1e-4);
      const double ds = (2 + 1 / as)
          * std::pow(std::max(0., 1 - nh * nh), .5 / as) / (2 * PI);
      result += D(s.sheenColor) * (ds * nl / (4 * (vs + ls - vs * ls) + 1e-6));
    }
  }
  if (s.clearcoat > 0 && cv > 0 && cl > 0) {
    const double vs = std::max(cv, 1e-6), ls = std::max(cl, 1e-6);
    const double a =
        std::max(double(s.clearcoatRoughness) * s.clearcoatRoughness, 1e-4);
    result += D(double(s.clearcoat) * fc(vh)
        * distribution(std::max(0., glm::dot(nc, h)), a) * smith(vs, ls, a) * cl
        / (4 * vs * ls));
  }
  return result;
}
inline double reflectedDensity(D n, D v, D l, double a)
{
  const double nv = glm::dot(n, v);
  if (nv <= 0 || glm::dot(n, l) <= 0)
    return 0;
  const double nh = glm::dot(n, glm::normalize(v + l));
  // Independently retain q_h and its reflection Jacobian, rather than DUT's
  // algebraically cancelled density. Visible-normal G1 has no target floor.
  const D h = glm::normalize(v + l);
  const double vh = glm::dot(v, h);
  const double g = 2 * nv / (nv + std::sqrt(a * a + (1 - a * a) * nv * nv));
  return (distribution(nh, a) * g * vh / nv) / (4 * std::abs(vh));
}
inline double pdf(const Fixture &f, D l)
{
  const auto &s = f.state;
  const D v(f.view), n(s.viewAdjustedNormal), nc(s.viewAdjustedClearcoatNormal);
  if (glm::dot(D(s.geometricNormal), v) <= 0
      || glm::dot(D(s.geometricNormal), l) <= 0)
    return 0;
  const double pc = coatPick(s, v), nv = glm::dot(n, v), nl = glm::dot(n, l);
  double q = pc
      * reflectedDensity(nc,
          v,
          l,
          std::max(double(s.clearcoatRoughness) * s.clearcoatRoughness, 1e-4));
  if (nv > 0 && nl > 0) {
    const D fv = fresnel(s, nv), complement = glm::max(D(0), D(1) - fv),
            tf = transmission(s);
    const double sw =
        std::max(0., lum(fv)) + std::max(0., lum(complement * tf));
    const double dw = std::max(0.,
        lum(complement * D(s.baseColor)
            * ((1 - double(s.metallic)) * (1 - double(s.transmission))
                * s.occlusion)));
    if (sw + dw > 0) {
      const D h = glm::normalize(v + l), fh = fresnel(s, glm::dot(v, h));
      const bool tir = fold(s, v, h);
      const double r = tir ? 1 : std::max(0., lum(fh)),
                   t = tir ? 0
                           : std::max(0., lum(glm::max(D(0), D(1) - fh) * tf));
      q += (1 - pc)
          * (dw / (sw + dw) * nl / PI
              + sw / (sw + dw) * (r + t > 0 ? r / (r + t) : 1)
                  * reflectedDensity(n,
                      v,
                      l,
                      std::max(double(s.roughness) * s.roughness, 1e-4)));
    }
  }
  return .875 * q + .125 / (2 * PI);
}
} // namespace native_test::oracle
