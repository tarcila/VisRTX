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

// Direct-visibility shade: albedo lit by the renderer's ambient term plus the
// material's own emission. The |N.V| factor is a stand-in that gives shape
// without a light in the scene; real direct lighting (NEE) and shadows arrive
// with the shadow-loop and bounce tickets.
__device__ vec3 shadeHitStatic(const FrameGPUData &fd,
    const SurfaceHit &hit,
    const vec3 &rayDir,
    vec3 &albedoOut,
    vec3 &normalOut,
    float &opacityOut)
{
  const BuiltinAppearance a = evalBuiltinAppearance(fd, hit);

  // Fall back to the geometric normal if the shading normal is degenerate.
  vec3 shadingNormal = hit.Ns;
  if (!(glm::dot(shadingNormal, shadingNormal) > 1e-12f))
    shadingNormal = hit.Ng;

  const float ndotv = glm::abs(glm::dot(rayDir, shadingNormal));
  const vec3 lit = a.baseColor * (ndotv * fd.renderer.ambientIntensity)
      * fd.renderer.ambientColor;

  albedoOut = a.baseColor * a.opacity;
  normalOut = shadingNormal;
  opacityOut = a.opacity;
  return lit * a.opacity + a.emission;
}

__global__ void wavefrontShadeKernel(const FrameGPUData *fd, uint32_t liveSlots)
{
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= liveSlots)
    return;

  const WavefrontPathSlot slot = fd->wavefrontSlots[i];
  if (!slot.alive)
    return;

  const WavefrontHitRecord &rec = fd->wavefrontHits[i];
  const uvec2 pixel = {slot.pixel % fd->fb.size.x, slot.pixel / fd->fb.size.x};
  const bool isVeryFirstRay = slot.sampleIdx == 0 && fd->fb.frameID == 0;

  vec3 color(0.f);
  vec3 albedo(0.f);
  vec3 normal(0.f);
  float opacity = 0.f;
  float depth = 1e30f;
  uint32_t primID = ~0u;
  uint32_t objID = ~0u;
  uint32_t instID = ~0u;

  if (rec.hit.foundHit) {
    color = shadeHitStatic(*fd, rec.hit, rec.rayDir, albedo, normal, opacity);
    depth = rec.hit.t;
    primID = rec.hit.primID;
    objID = rec.hit.objID;
    instID = rec.hit.instID;
  } else if (vec3 hdri; getBackgroundLight(*fd, rec.rayDir, hdri)) {
    color = hdri;
    opacity = 1.f;
  }

  if (isVeryFirstRay)
    setPixelIds(fd->fb, pixel, depth, primID, objID, instID);
  accumPixelSample(*fd, pixel, vec4(color, opacity), albedo, normal);
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

void wavefrontShade(
    cudaStream_t stream, const FrameGPUData *frameData, uint32_t liveSlots)
{
  if (liveSlots == 0)
    return;
  const uint32_t blocks = (liveSlots + kThreadsPerBlock - 1) / kThreadsPerBlock;
  wavefrontShadeKernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      frameData, liveSlots);
}

} // namespace visrtx
