/*
 * Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted, provided that the following conditions are met:
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

// Device-side shade helpers shared by the builtin wavefront shade-emit stage
// (WavefrontLaunch.cu) and the per-material MDL shade shell
// (WavefrontMdlShell_ptx.cu). Both run in plain CUDA and produce the same
// WavefrontShadeRecord contract for the material-agnostic shadow-trace +
// resolve stages, so the light-pick / pass-through-bounce / hit-id shapes they
// have in common live here once instead of being copy-pasted across the two
// translation units (which are linked into separate cubins, so a shared header
// is the only way to share device code between them).

#pragma once

#include "gpu/gpu_decl.h" // VISRTX_DEVICE
#include "gpu/gpu_objects.h" // FrameGPUData, Wavefront* records, ScreenSample
#include "gpu/sampleLight.h" // sampleLight, LightSample

namespace visrtx {

// Result of picking one light for next-event estimation: the sampled light, the
// env flag (the HDRI env is also reachable by the BSDF escape, so its NEE
// deposit needs balance-heuristic MIS), and the chosen light index.
struct WavefrontLightPick
{
  LightSample ls;
  bool isEnv;
  uint32_t k;
};

// Pick ONE of the `n` light instances uniformly, sample it toward `org`, and
// advance `rng` by the pick + sample. Returns the sample, whether the picked
// light is the HDRI env, and the picked index. The caller owns the n == 0
// guard (it needs `n` for the uniform-pick reweight anyway) and the BSDF
// evaluation / MIS weighting, which differ per material subtype.
VISRTX_DEVICE WavefrontLightPick wavefrontPickLight(
    const FrameGPUData &fd, RandState &rng, const vec3 &org, uint32_t n)
{
  WavefrontLightPick p{};
  ScreenSample ss;
  ss.frameData = &fd;
  ss.rs = rng;
  ss.shadowContribWeight = 1.0f;
  uint32_t k = uint32_t(pcg_uniform(&ss.rs) * float(n));
  if (k >= n)
    k = n - 1u;
  const InstanceLightGPUData &li = fd.world.lightInstances[k];
  p.ls = sampleLight(ss, org, li.lightIndex, li.xfm, li.surfaceInstanceIndex);
  p.isEnv = fd.registry.lights[li.lightIndex].type == LightType::HDRI;
  p.k = k;
  rng = ss.rs;
  return p;
}

// Stochastic-cutout pass-through continuation: the surface is treated as
// absent for this sample, so the path continues straight through the hit with
// no attenuation. A delta pass-through (pdf +inf) so the escape owns the
// environment and NEE never double-counts it. The continuation origin is
// offset to the side the ray is travelling so it does not self-intersect.
VISRTX_DEVICE void wavefrontWritePassThroughBounce(
    WavefrontShadeRecord &sr, const WavefrontHitRecord &rec)
{
  sr.hasSampledBounce = 1u;
  sr.bounceDir = rec.rayDir; // straight through, unchanged direction
  sr.bounceWeight = vec3(1.f); // no attenuation
  sr.bouncePdf = INFINITY; // delta pass-through: escape owns the env (no NEE)
  const float side = glm::dot(rec.rayDir, rec.hit.Ng) >= 0.f ? 1.f : -1.f;
  sr.bounceOrg = rec.hit.hitpoint + rec.hit.Ng * (rec.hit.epsilon * side);
}

// Copy the first-hit AOV identity fields (depth + object/instance/prim ids)
// from a hit record into the shade record. The albedo/normal/opacity terms
// differ per branch, so they stay at the call site.
VISRTX_DEVICE void wavefrontFillHitIds(
    WavefrontShadeRecord &sr, const WavefrontHitRecord &rec)
{
  sr.depth = rec.hit.t;
  sr.primID = rec.hit.primID;
  sr.objID = rec.hit.objID;
  sr.instID = rec.hit.instID;
}

} // namespace visrtx
