// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/graph_nodes/Descriptors.hpp"
// std
#include <cmath>
#include <cstdint>

// Shared helpers for procedural volume source nodes: a unit-cube field
// allocator and dependency-free value/fBm noise. Keeps GenerateNoiseVolume,
// GenerateGyroid, GenerateTurbulence, and GenerateMetaballs DRY.

namespace tsd::graph_nodes {

constexpr float kPi = 3.14159265358979323846f;

// A field spanning the unit cube [-1, 1]^3 with float32 storage allocated.
inline Field makeUnitField(tsd::core::math::uint3 dims)
{
  Field f;
  f.dims = dims;
  f.origin = tsd::core::math::float3(-1.f, -1.f, -1.f);
  f.spacing = tsd::core::math::float3(
      2.f / float(dims.x), 2.f / float(dims.y), 2.f / float(dims.z));
  f.data = tsd::core::AnyArray(ANARI_FLOAT32, size_t(dims.x) * dims.y * dims.z);
  return f;
}

// Normalized voxel coordinate in [-1, 1] along an axis of size n.
inline float normCoord(uint32_t i, uint32_t n)
{
  return (float(i) / float(n > 1 ? n - 1 : 1)) * 2.f - 1.f;
}

inline float clamp01(float v)
{
  return v < 0.f ? 0.f : (v > 1.f ? 1.f : v);
}

// Integer-lattice hash → float in [0, 1].
inline float hashLattice(int x, int y, int z, int seed)
{
  uint32_t h = uint32_t(x) * 73856093u ^ uint32_t(y) * 19349663u
      ^ uint32_t(z) * 83492791u ^ uint32_t(seed) * 2654435761u;
  h ^= h >> 13;
  h *= 0x85ebca6bu;
  h ^= h >> 16;
  return float(h) / float(0xffffffffu);
}

inline float smootherstep(float t)
{
  return t * t * t * (t * (t * 6.f - 15.f) + 10.f);
}

// Trilinearly-interpolated value noise, period-free, in [0, 1].
inline float valueNoise(float x, float y, float z, int seed)
{
  const float fx = std::floor(x), fy = std::floor(y), fz = std::floor(z);
  const int xi = int(fx), yi = int(fy), zi = int(fz);
  const float u = smootherstep(x - fx);
  const float v = smootherstep(y - fy);
  const float w = smootherstep(z - fz);

  auto lerp = [](float a, float b, float t) { return a + (b - a) * t; };
  const float c000 = hashLattice(xi, yi, zi, seed);
  const float c100 = hashLattice(xi + 1, yi, zi, seed);
  const float c010 = hashLattice(xi, yi + 1, zi, seed);
  const float c110 = hashLattice(xi + 1, yi + 1, zi, seed);
  const float c001 = hashLattice(xi, yi, zi + 1, seed);
  const float c101 = hashLattice(xi + 1, yi, zi + 1, seed);
  const float c011 = hashLattice(xi, yi + 1, zi + 1, seed);
  const float c111 = hashLattice(xi + 1, yi + 1, zi + 1, seed);

  const float x00 = lerp(c000, c100, u);
  const float x10 = lerp(c010, c110, u);
  const float x01 = lerp(c001, c101, u);
  const float x11 = lerp(c011, c111, u);
  return lerp(lerp(x00, x10, v), lerp(x01, x11, v), w);
}

// Fractional Brownian motion: sum of `octaves` value-noise octaves. Result is
// normalized to ~[0, 1] by the sum of amplitudes.
inline float fbm(float x,
    float y,
    float z,
    int octaves,
    float lacunarity,
    float gain,
    int seed)
{
  float sum = 0.f, amp = 0.5f, freq = 1.f, norm = 0.f;
  const int n = octaves < 1 ? 1 : (octaves > 12 ? 12 : octaves);
  for (int i = 0; i < n; ++i) {
    sum += amp * valueNoise(x * freq, y * freq, z * freq, seed + i);
    norm += amp;
    freq *= lacunarity;
    amp *= gain;
  }
  return norm > 0.f ? sum / norm : 0.f;
}

} // namespace tsd::graph_nodes
