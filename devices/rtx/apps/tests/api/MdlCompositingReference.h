// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "MdlScatteringReference.h"

namespace compositing_reference {
// Independent analog solution for two finite boxes and a diffuse triangle.
// Physical free flights, analog absorption, isotropic phase and Lambertian
// continuation only: no null collisions, NEE, MIS or renderer evaluators.
inline scattering_reference::V3 mixedSurface()
{
  using namespace scattering_reference;
  const std::vector<Medium> media{
      {{-2, -2, 2}, {2, 2, 4}, {.1f, .2f, 0}, {.3f, .1f, 0}},
      {{-2, -2, 5},
          {2, 2, 7},
          {float(-std::log(.75)), float(-std::log(.75)), float(-std::log(.75))},
          {0, 0, 0}}};
  constexpr V3 environment{.2f, .3f, .4f}, tint{.2f, .5f, .8f};
  constexpr int samples = 1048576, maxDepth = 12;
  V3 result{};
  for (int c = 0; c < 3; ++c) {
    std::mt19937_64 rng(91731 + c);
    auto u = [&] { return (double(rng() >> 11) + .5) / 9007199254740992.; };
    double sum = 0., sum2 = 0.;
    for (int n = 0; n < samples; ++n) {
      D3 p{.1 * (u() - .5), .1 * (u() - .5), 0}, d{0, 0, 1};
      double weight = 1., value = 0.;
      int depth = 0;
      for (;;) {
        double entry = 1e100, exit = 0.;
        const Medium *medium = nullptr;
        for (const auto &m : media) {
          double a, b;
          if (Reference::interval(p, d, m, a, b) && a < entry) {
            entry = a;
            exit = b;
            medium = &m;
          }
        }
        double plane = 1e100;
        if (d[2] > 0.) {
          const double t = (8. - p[2]) / d[2];
          const double x = p[0] + t * d[0], y = p[1] + t * d[1];
          if (t > 0. && y >= -4. && y <= 4. && std::abs(x) <= 2. - .5 * y)
            plane = t;
        }
        if (plane < entry) {
          if (depth++ == maxDepth)
            break;
          for (int k = 0; k < 3; ++k)
            p[k] += plane * d[k];
          p[2] -= 1e-7;
          weight *= tint[c];
          const double r = std::sqrt(u()), phi = 2. * 3.141592653589793 * u();
          d = {r * std::cos(phi), r * std::sin(phi), -std::sqrt(1. - r * r)};
          continue;
        }
        if (!medium) {
          value = weight * environment[c];
          break;
        }
        const double extinction = double(medium->a[c]) + medium->s[c];
        const double flight =
            extinction == 0. ? 1e100 : -std::log(u()) / extinction;
        if (entry + flight >= exit) {
          for (int k = 0; k < 3; ++k)
            p[k] += (exit + 1e-7) * d[k];
          continue;
        }
        if (depth == maxDepth || u() * extinction < medium->a[c])
          break;
        ++depth;
        for (int k = 0; k < 3; ++k)
          p[k] += (entry + flight) * d[k];
        const double z = 2. * u() - 1., phi = 2. * 3.141592653589793 * u();
        const double r = std::sqrt(1. - z * z);
        d = {r * std::cos(phi), r * std::sin(phi), z};
      }
      sum += value;
      sum2 += value * value;
    }
    result[c] = float(sum / samples);
    const double se = std::sqrt(
        std::max(0., sum2 / samples - double(result[c]) * result[c]) / samples);
    std::printf("CPU mixed diffuse c%d mean %.9g SE %.9g\n", c, result[c], se);
    if (se > .0005)
      throw std::runtime_error("inconclusive CPU mixed reference");
  }
  return result;
}
} // namespace compositing_reference
