/*
 * Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include "gpu/gpu_decl.h"

#include <vector_types.h> // ::float4
#include <cstdint>

// Halton low-discrepancy sequence for the top of the path. Pattern: PCG
// everywhere, QMC at the camera and first-bounce cosine-hemisphere NEE.
//   dims 0..1 (bases 2, 3): pixel x/y sub-pixel jitter
//   dims 2..3 (bases 5, 7): lens u/v for DoF
//   dims 4..5 (bases 11, 13): first-bounce cosine hemisphere (env NEE /
//   ambient)
//
// Pixel decorrelation: additive offset = haltonPixelHash(pixel). Not full
// Owen scrambling — adequate at 6D, full scrambling is more code for
// negligible win. Do not reuse dims 0–3 for NEE: that correlates lighting
// with AA/DoF and produces structured artifacts.

namespace visrtx {

// Per-pixel scramble hash — PCG-style integer mixer, additive offset on
// the Halton sample index.
VISRTX_DEVICE uint32_t haltonPixelHash(uint32_t x, uint32_t y)
{
  // Salt breaks the (0, 0) zero fixed point — otherwise pixel (0, 0) would
  // share its Halton subsequence with the origin.
  uint32_t v = x + y * 0x9E3779B9u + 0xB5297A4Du;
  v ^= v >> 16;
  v *= 0x7feb352du;
  v ^= v >> 15;
  v *= 0x846ca68bu;
  v ^= v >> 16;
  return v;
}

// Radical inverse Φ_Base(i): digits of i in base Base, reversed across the
// decimal point. Generic loop covers bases 2/3/5/7; Base=2 could specialise
// to __brev but the saving is marginal at this dimensionality.
template <uint32_t Base>
VISRTX_DEVICE float radicalInverse(uint32_t i)
{
  static_assert(Base >= 2u, "radicalInverse requires Base >= 2");
  constexpr float invBase = 1.0f / float(Base);
  float result = 0.0f;
  float scale = 1.0f;
  while (i > 0u) {
    scale *= invBase;
    result += float(i % Base) * scale;
    i /= Base;
  }
  // FP rounding can push the sum to exactly 1.0f; callers need [0, 1).
  return fminf(result, 0x1.fffffep-1f);
}

// Halton 4D point (Φ_2, Φ_3, Φ_5, Φ_7), each in [0, 1). Combine sampleIdx
// with haltonPixelHash for per-pixel decorrelation.
VISRTX_DEVICE ::float4 halton4D(uint32_t sampleIdx)
{
  return ::float4{
      radicalInverse<2>(sampleIdx),
      radicalInverse<3>(sampleIdx),
      radicalInverse<5>(sampleIdx),
      radicalInverse<7>(sampleIdx),
  };
}

// First-bounce cosine-hemisphere NEE uniforms (Φ_11, Φ_13), each in [0, 1).
// Same sampleIdx + haltonPixelHash as the camera so the extra dimensions
// share the per-pixel scramble without colliding with AA/DoF.
VISRTX_DEVICE ::float2 haltonHemi2D(uint32_t sampleIdx)
{
  return ::float2{radicalInverse<11>(sampleIdx), radicalInverse<13>(sampleIdx)};
}

} // namespace visrtx
