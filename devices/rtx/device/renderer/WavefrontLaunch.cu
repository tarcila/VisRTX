/*
 * Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "WavefrontLaunch.h"

#include "gpu/evalMaterialParameters.h" // getMaterialParameter, adjustedMaterialOpacity
#include "gpu/gpu_objects.h" // FrameGPUData, WavefrontHitRecord, MaterialGPUData
#include "gpu/gpu_util.h" // accumPixelSample, setPixelIds
#include "gpu/renderer/common.h" // getBackgroundLight
#include "gpu/sampleLight.h" // sampleLight, LightSample

namespace visrtx {

namespace {

constexpr int kThreadsPerBlock = 256;

__global__ void wavefrontRegenerateKernel(WavefrontPathSlot *slots,
    uint64_t waveBase,
    uint32_t numPixels,
    uint64_t totalSamples,
    uint32_t liveSlots)
{
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= liveSlots)
    return;

  const uint64_t sampleId = waveBase + i;
  if (sampleId >= totalSamples) {
    slots[i].alive = 0;
    return;
  }

  slots[i].pixel = uint32_t(sampleId % numPixels);
  slots[i].sampleIdx = uint32_t(sampleId / numPixels);
  slots[i].alive = 1;
}

// Statically-evaluated builtin surface appearance: base color, opacity and
// emission read directly from the material data (no optixDirectCall). Covers
// the native Matte and PhysicallyBased materials; MDL is not on the static path
// (it needs the linking spike) and renders as a neutral surface for now.
struct BuiltinAppearance
{
  vec3 baseColor{0.8f};
  vec3 emission{0.f};
  float opacity{1.f};
};

__device__ BuiltinAppearance evalBuiltinAppearance(
    const FrameGPUData &fd, const SurfaceHit &hit)
{
  BuiltinAppearance a;
  if (!hit.material)
    return a;

  if (hit.material->callableBaseIndex
      == uint32_t(SbtCallableEntryPoints::Matte)) {
    const auto &md = hit.material->materialData.matte;
    const vec4 color = getMaterialParameter(fd, md.color, hit);
    const float op = getMaterialParameter(fd, md.opacity, hit).x;
    a.baseColor = vec3(color);
    a.opacity = adjustedMaterialOpacity(color.w * op, md.alphaMode, md.cutoff);
  } else if (hit.material->callableBaseIndex
      == uint32_t(SbtCallableEntryPoints::PBR)) {
    const auto &md = hit.material->materialData.physicallyBased;
    const vec4 color = getMaterialParameter(fd, md.baseColor, hit);
    const float op = getMaterialParameter(fd, md.opacity, hit).x;
    a.baseColor = vec3(color);
    a.opacity = adjustedMaterialOpacity(color.w * op, md.alphaMode, md.cutoff);
    a.emission = vec3(getMaterialParameter(fd, md.emissive, hit));
  }
  return a;
}

// Sample one analytic light. Deliberately does NOT call the shared sampleLight()
// dispatcher: its GEOMETRY branch reaches evaluateSurfaceEmission (an
// optixDirectCall) which ptxas cannot resolve in a plain CUDA translation unit.
// Dispatching only the analytic types here keeps the geometry path out of
// codegen; Geometry Lights get their own emission-eval stage (ticket 09).
__device__ LightSample sampleAnalyticLight(
    const LightGPUData &ld, const mat4 &xfm, const vec3 &origin, RandState &rs)
{
  switch (ld.type) {
  case LightType::DIRECTIONAL:
    return detail::sampleDirectionalLight(ld, xfm);
  case LightType::POINT:
    return detail::samplePointLight(ld, xfm, origin);
  case LightType::SPHERE:
    return detail::sampleSphereLight(ld, xfm, origin, rs);
  case LightType::RECT:
    return detail::sampleRectLight(ld, xfm, origin, rs);
  case LightType::SPOT:
    return detail::sampleSpotLight(ld, xfm, origin);
  case LightType::RING:
    return detail::sampleRingLight(ld, xfm, origin, rs);
  case LightType::HDRI:
    return detail::sampleHDRILight(ld, xfm, rs);
  default: // GEOMETRY and anything else: not handled on the static path yet
    return LightSample{vec3(0.f), vec3(0.f), 0.f, 0.f};
  }
}

// Shade-emit: evaluate the surface, pick ONE light for next-event estimation,
// and stash the deferred shading state — the unshadowed part (ambient +
// emission, or background on a miss) applied unconditionally, the shadowable
// direct-light contribution, and a shadow ray toward the picked light. The
// shadow trace stage fills in visibility and the resolve stage combines them.
// One light per slot keeps the shadow queue the size of the pool (Lambertian,
// VisRTX's albedo*E convention). Geometry Lights are skipped (ticket 09).
__global__ void wavefrontShadeEmitKernel(
    const FrameGPUData *fd, uint32_t liveSlots)
{
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= liveSlots)
    return;
  const WavefrontPathSlot slot = fd->wavefrontSlots[i];
  if (!slot.alive)
    return;

  const WavefrontHitRecord &rec = fd->wavefrontHits[i];
  WavefrontShadeRecord &sr = fd->wavefrontShade[i];

  sr.directContrib = vec3(0.f);
  sr.shadowDist = 0.f; // no shadow ray unless a light is picked
  sr.visibility = 1.f;
  sr.hasHit = rec.hit.foundHit ? 1u : 0u;

  if (!rec.hit.foundHit) {
    vec3 hdri;
    if (getBackgroundLight(*fd, rec.rayDir, hdri)) {
      sr.unshadowed = hdri;
      sr.opacity = 1.f;
    } else {
      sr.unshadowed = vec3(0.f);
      sr.opacity = 0.f;
    }
    sr.albedo = vec3(0.f);
    sr.normal = vec3(0.f);
    sr.depth = 1e30f;
    sr.primID = ~0u;
    sr.objID = ~0u;
    sr.instID = ~0u;
    return;
  }

  const BuiltinAppearance a = evalBuiltinAppearance(*fd, rec.hit);
  vec3 N = rec.hit.Ns;
  if (!(glm::dot(N, N) > 1e-12f))
    N = rec.hit.Ng;

  const vec3 ambient =
      a.baseColor * fd->renderer.ambientIntensity * fd->renderer.ambientColor;
  sr.unshadowed = ambient * a.opacity + a.emission;
  sr.albedo = a.baseColor * a.opacity;
  sr.normal = N;
  sr.opacity = a.opacity;
  sr.depth = rec.hit.t;
  sr.primID = rec.hit.primID;
  sr.objID = rec.hit.objID;
  sr.instID = rec.hit.instID;

  const uint32_t n = uint32_t(fd->world.numLightInstances);
  if (n == 0)
    return;

  RandState rng = rec.rng;
  uint32_t k = uint32_t(pcg_uniform(&rng) * float(n));
  if (k >= n)
    k = n - 1u;
  const InstanceLightGPUData &li = fd->world.lightInstances[k];
  const LightGPUData &ld = fd->registry.lights[li.lightIndex];
  const vec3 origin = rec.hit.hitpoint + rec.hit.Ng * rec.hit.epsilon;
  const LightSample ls = sampleAnalyticLight(ld, li.xfm, origin, rng);
  if (ls.pdf > 0.f && ls.dist > 0.f) {
    const float ndotl = fmaxf(0.f, glm::dot(N, ls.dir));
    // Uniform pick over n lights has probability 1/n, so reweight by n.
    sr.directContrib =
        a.baseColor * ndotl * ls.radiance / ls.pdf * float(n) * a.opacity;
    sr.shadowOrg = origin;
    sr.shadowDir = ls.dir;
    sr.shadowDist = ls.dist;
  }
}

// Atomic scatter-add accumulation for concurrent same-pixel deposits (multiple
// pool slots of one pixel shading in the same wave). Only the per-sample
// firefly modes are handled here: NONE and TONEMAP have no per-pixel state, so
// atomicAdd composes correctly. CLAMP/TRIM keep per-pixel running statistics
// (Welford / top-k) that a scatter-add would corrupt, so for those modes the
// host keeps the pool capped to one slot per pixel per wave and the shade stage
// uses the ordinary accumPixelSample instead.
__device__ void accumPixelSampleAtomic(const FrameGPUData &fd,
    const uvec2 &pixel,
    const vec4 &color,
    const vec3 &albedo,
    const vec3 &normal)
{
  const auto &fb = fd.fb;
  const uint32_t idx = detail::pixelIndex(fb, pixel);

  const vec4 c = fd.renderer.fireflyFilterMode == FireflyFilterMode::TONEMAP
      ? detail::tonemap(color)
      : color;

  vec4 *ca = fb.buffers.colorAccumulation;
  atomicAdd(&ca[idx].x, c.x);
  atomicAdd(&ca[idx].y, c.y);
  atomicAdd(&ca[idx].z, c.z);
  atomicAdd(&ca[idx].w, c.w);
  if (fb.buffers.albedo) {
    atomicAdd(&fb.buffers.albedo[idx].x, albedo.x);
    atomicAdd(&fb.buffers.albedo[idx].y, albedo.y);
    atomicAdd(&fb.buffers.albedo[idx].z, albedo.z);
  }
  if (fb.buffers.normal) {
    atomicAdd(&fb.buffers.normal[idx].x, normal.x);
    atomicAdd(&fb.buffers.normal[idx].y, normal.y);
    atomicAdd(&fb.buffers.normal[idx].z, normal.z);
  }
}

__device__ bool fireflyModeIsAtomicSafe(FireflyFilterMode mode)
{
  return mode == FireflyFilterMode::NONE || mode == FireflyFilterMode::TONEMAP;
}

// Resolve: fold the shadow-ray visibility into the deferred shade record and
// accumulate. The unshadowed term (ambient + emission / background) always
// lands; the direct-light term is gated by how much of the shadow ray reached
// the light.
__global__ void wavefrontResolveKernel(
    const FrameGPUData *fd, uint32_t liveSlots)
{
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= liveSlots)
    return;

  const WavefrontPathSlot slot = fd->wavefrontSlots[i];
  if (!slot.alive)
    return;

  const WavefrontShadeRecord &sr = fd->wavefrontShade[i];
  const uvec2 pixel = {slot.pixel % fd->fb.size.x, slot.pixel / fd->fb.size.x};
  const bool isVeryFirstRay = slot.sampleIdx == 0 && fd->fb.frameID == 0;

  const vec3 color = sr.unshadowed + sr.visibility * sr.directContrib;

  // First-hit AOVs (ids, depth) are written by exactly one slot per pixel — the
  // frame's first sample — so no atomics are needed even under concurrency.
  if (isVeryFirstRay)
    setPixelIds(fd->fb, pixel, sr.depth, sr.primID, sr.objID, sr.instID);

  // Atomic scatter-add when the pool may run several samples of this pixel
  // concurrently (the host only lifts the one-slot-per-pixel cap for the
  // atomic-safe firefly modes); otherwise the ordinary accumulate is fine and
  // keeps CLAMP/TRIM working.
  if (fireflyModeIsAtomicSafe(fd->renderer.fireflyFilterMode))
    accumPixelSampleAtomic(*fd, pixel, vec4(color, sr.opacity), sr.albedo, sr.normal);
  else
    accumPixelSample(*fd, pixel, vec4(color, sr.opacity), sr.albedo, sr.normal);
}

} // namespace

void wavefrontRegenerate(cudaStream_t stream,
    WavefrontPathSlot *slots,
    uint64_t waveBase,
    uint32_t numPixels,
    uint64_t totalSamples,
    uint32_t liveSlots)
{
  if (liveSlots == 0)
    return;
  const uint32_t blocks = (liveSlots + kThreadsPerBlock - 1) / kThreadsPerBlock;
  wavefrontRegenerateKernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      slots, waveBase, numPixels, totalSamples, liveSlots);
}

void wavefrontShadeEmit(
    cudaStream_t stream, const FrameGPUData *frameData, uint32_t liveSlots)
{
  if (liveSlots == 0)
    return;
  const uint32_t blocks = (liveSlots + kThreadsPerBlock - 1) / kThreadsPerBlock;
  wavefrontShadeEmitKernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      frameData, liveSlots);
}

void wavefrontResolve(
    cudaStream_t stream, const FrameGPUData *frameData, uint32_t liveSlots)
{
  if (liveSlots == 0)
    return;
  const uint32_t blocks = (liveSlots + kThreadsPerBlock - 1) / kThreadsPerBlock;
  wavefrontResolveKernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      frameData, liveSlots);
}

} // namespace visrtx
