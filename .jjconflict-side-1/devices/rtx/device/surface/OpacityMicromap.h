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

#include "optix_visrtx.h"
#include "utility/SinglePassDownsampler.h"

namespace visrtx {

struct Geometry;
struct Material;

// Min/max pyramid over a sampler's transformed alpha (float2 = {min, max}
// per texel), built once per (sampler content, transform, channel) with the
// SPD downsampler and cached in DeviceGlobalState. The bake queries a level
// matched to each microtriangle's UV footprint instead of scanning texels.
struct OmmAlphaPyramid
{
  DeviceBuffer storage; // backs every level plus the SPD tile counter
  spd::MipChainView<float2> chain;
  uint2 srcDims{};
};

// Conservative Opacity Micromap for one Surface (ADR 0009). The bake
// classifies each micro-triangle of every base triangle against the
// material's Opacity Function and emits, by default, two Opacity States:
//   TRANSPARENT     — alpha provably 0 over the whole footprint; traversal
//                     skips the hit entirely (no any-hit, no closest-hit)
//   UNKNOWN_OPAQUE  — everything else; the existing exact any-hit /
//                     transparency-loop paths run unchanged
// (The experimental `ommOpaqueStates` toggle additionally emits hard OPAQUE
// where alpha is provably committing — default off, see ADR 0009.)
// Not emitting hard OPAQUE keeps every surviving hit on today's shading
// paths, so every path that evaluates alpha exactly (primary Transparency
// Loop, shadow/AO any-hit) renders pixel-identical by construction. Paths
// that never evaluated alpha diverge by design once provably transparent
// microtris stop being hit at all: DISABLE_ANYHIT bounce rays pass through
// cutouts they used to treat as opaque, pick IDs/depth report the surface
// behind a fully transparent hit, and removed alpha=0 loop iterations shift
// the RNG stream (different, equally unbiased noise). See ADR 0009's
// determinism-scope consequence.
struct OpacityMicromapBuffers
{
  DeviceBuffer micromapArray; // optixOpacityMicromapArrayBuild output
  DeviceBuffer indexBuffer; // one int32 per base triangle
  OptixOpacityMicromapUsageCount usage[1] = {};
  uint32_t numUsage{0};
  bool attached{false}; // true when buildInput() should reference these

  void reset();
};

// Resolved bake plan (opaque outside OpacityMicromap.cu). Produced by
// computeOpacityMicromapKey, consumed by bakeOpacityMicromaps — resolving is
// not free (registry lookups, factor resolution, domain-range queries), so a
// cache miss must not do it twice.
struct OmmBakeSetup;

// Bakes OMMs for the resolved plan into `out`. Returns false (with `out`
// reset) when the bake would win nothing (no provably transparent region).
// Non-fatal on failure: the surface just renders without OMM acceleration.
bool bakeOpacityMicromaps(OpacityMicromapBuffers &out,
    OmmBakeSetup &setup,
    Object *reporter);

// Content-addressed dedup key: hashes everything a bake would read (index /
// attribute buffers, factor spec, sampler content stamps), so surfaces whose
// host objects differ but whose bake inputs are byte-identical share one
// micromap. False when the pair is OMM-ineligible (no key exists; `setup`
// is cleared). On success `setup` holds the resolved plan for the bake.
bool computeOpacityMicromapKey(uint64_t &key,
    std::shared_ptr<OmmBakeSetup> &setup,
    Geometry *geometry,
    const Material *material,
    Object *reporter);

// Inlined definitions ////////////////////////////////////////////////////////

inline void OpacityMicromapBuffers::reset()
{
  micromapArray.reset();
  indexBuffer.reset();
  numUsage = 0;
  attached = false;
}

} // namespace visrtx
