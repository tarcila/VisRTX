// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>

// Test-only, independent scalar analog reference. Physical exponential free
// flights: no GPU evaluator, majorant, null tracking, NEE or production
// sampler.
namespace scattering_reference {
using V3 = std::array<float, 3>;
using D3 = std::array<double, 3>;
struct Medium
{
  D3 lo, hi;
  V3 a, s;
};
struct Plane
{
  double z, half;
  V3 emission;
  double opacity = 1.;
};
struct Reference
{
  std::vector<Medium> media{
      {{-1, -1, 2}, {1, 1, 4}, {0, 0, 0}, {0, .5f, 1.5f}}};
  std::vector<Plane> planes;
  V3 environment{1, 1, 1};
  int depth = 32;
  static bool interval(D3 p, D3 d, const Medium &m, double &lo, double &hi)
  {
    lo = 0.;
    hi = 1e100;
    for (int k = 0; k < 3; ++k) {
      if (d[k] == 0.) {
        if (p[k] < m.lo[k] || p[k] > m.hi[k])
          return false;
      } else {
        double a = (m.lo[k] - p[k]) / d[k], b = (m.hi[k] - p[k]) / d[k];
        if (a > b)
          std::swap(a, b);
        lo = std::max(lo, a);
        hi = std::min(hi, b);
      }
    }
    return hi > lo;
  }
  V3 analog() const
  {
    constexpr int samples = 1048576;
    V3 result{};
    for (int c = 0; c < 3; ++c) {
      std::mt19937_64 rng(18027 + c);
      auto u = [&] { return (double(rng() >> 11) + .5) / 9007199254740992.; };
      double sum = 0., sum2 = 0.;
      for (int n = 0; n < samples; ++n) {
        D3 p{0, 0, 0}, d{0, 0, 1};
        int events = 0;
        double value = 0.;
        for (;;) {
          double nearest = 1e100, exit = 0.;
          const Medium *medium = nullptr;
          for (const auto &m : media) {
            double a, b;
            if (interval(p, d, m, a, b) && a < nearest) {
              nearest = a;
              exit = b;
              medium = &m;
            }
          }
          const Plane *plane = nullptr;
          double hit = 1e100;
          for (const auto &q : planes) {
            if (d[2] == 0.)
              continue;
            double t = (q.z - p[2]) / d[2];
            if (t > 0. && t < hit && std::abs(p[0] + t * d[0]) <= q.half
                && std::abs(p[1] + t * d[1]) <= q.half) {
              hit = t;
              plane = &q;
            }
          }
          if (plane && hit < nearest) {
            if (u() < plane->opacity) {
              value = plane->emission[c];
              break;
            }
            for (int k = 0; k < 3; ++k)
              p[k] += hit * d[k];
            p[2] = std::nextafter(p[2], p[2] + d[2]);
            continue;
          }
          if (!medium) {
            value = environment[c];
            break;
          }
          const double a = medium->a[c], s = medium->s[c], t = a + s;
          double flight = t > 0. ? -std::log(u()) / t : 1e100;
          if (nearest + flight < exit) {
            if (u() * t < a || events == depth)
              break;
            for (int k = 0; k < 3; ++k)
              p[k] += (nearest + flight) * d[k];
            ++events;
            double z = 2 * u() - 1., phi = 6.283185307179586 * u();
            double r = std::sqrt(1 - z * z);
            d = {r * std::cos(phi), r * std::sin(phi), z};
          } else {
            for (int k = 0; k < 3; ++k) {
              p[k] += exit * d[k];
              if (d[k] != 0.)
                p[k] = std::nextafter(p[k], p[k] + d[k]);
            }
          }
        }
        sum += value;
        sum2 += value * value;
      }
      result[c] = float(sum / samples);
      double se = std::sqrt(
          std::max(0., sum2 / samples - std::pow(sum / samples, 2)) / samples);
      printf("CPU analog depth%d c%d: %.8f SE %.8f\n", depth, c, result[c], se);
      if (se >= .0005)
        throw std::runtime_error("CPU reference uncertainty exceeds allowance");
    }
    return result;
  }
};
// Deterministic midpoint integration over distance, mu and azimuth. Halving
// all steps certifies the zero-plus-one reference independently of the GPU.
inline double zeroPlusOne(double s, int resolution)
{
  const double pi = 3.141592653589793;
  double integral = 0.;
  Medium box{{-1, -1, 2}, {1, 1, 4}, {}, {}};
  for (int iz = 0; iz < resolution; ++iz) {
    double z = 2. * (iz + .5) / resolution, angular = 0.;
    for (int im = 0; im < resolution; ++im) {
      double mu = 2. * (im + .5) / resolution - 1., r = std::sqrt(1. - mu * mu);
      for (int ip = 0; ip < 2 * resolution; ++ip) {
        double phi = 2 * pi * (ip + .5) / (2 * resolution), lo, hi;
        Reference::interval({0, 0, 2 + z},
            {r * std::cos(phi), r * std::sin(phi), mu},
            box,
            lo,
            hi);
        angular += std::exp(-s * hi);
      }
    }
    integral += s * std::exp(-s * z) * angular / (2 * resolution * resolution)
        * 2. / resolution;
  }
  return std::exp(-2 * s) + integral;
}
} // namespace scattering_reference
