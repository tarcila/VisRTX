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

// The wavefront shade stage runs in plain CUDA and cannot dispatch the material
// emission callable, so Geometry-Light NEE uses the light's mean radiance
// (exact for constant emitters). This keeps sampleLight()'s GEOMETRY branch out
// of ptxas codegen; without it the optixDirectCall is an unresolved symbol.
// Must be defined before ANY include, since sampleLight.h is also pulled
// transitively (with #pragma once, a later define would be too late).
#define VISRTX_STATIC_GEOMETRY_LIGHT_EMISSION

#include "gpu/evalMaterialParameters.h" // getMaterialParameter, adjustedMaterialOpacity
#include "gpu/gpu_objects.h" // FrameGPUData, WavefrontHitRecord, MaterialGPUData
#include "gpu/gpu_util.h" // accumPixelSample, setPixelIds
#include "gpu/physicallyBasedBsdf.h" // pbrInitState, pbrEvalNEE, pbrSampleNextRay
#include "gpu/renderer/common.h" // getBackgroundLight
#include "gpu/sampleLight.h" // sampleLight, LightSample
#include "gpu/sbt.h" // SbtCallableEntryPoints

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
// the native Matte and PhysicallyBased materials; MDL hits take the
// per-material CUDA shade kernel instead (see isMdlMaterial /
// wavefrontMdlShade).
struct BuiltinAppearance
{
  vec3 baseColor{0.8f};
  vec3 emission{0.f};
  float opacity{1.f};
};

// MDL materials get a per-compiled-material callableBaseIndex at or above the
// static callable block; the builtin static path only knows Matte and PBR, so
// MDL hits are shaded by the wavefront MDL kernel instead (ticket 10).
__device__ bool isMdlMaterial(const MaterialGPUData *m)
{
  return m && m->callableBaseIndex >= uint32_t(SbtCallableEntryPoints::Last);
}

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
  WavefrontPathState &path = fd->wavefrontPaths[i];
  if (!path.alive)
    return;

  const WavefrontHitRecord &rec = fd->wavefrontHits[i];
  WavefrontShadeRecord &sr = fd->wavefrontShade[i];

  sr.directContrib = vec3(0.f);
  sr.shadowDist = 0.f; // no shadow ray unless a light is picked
  sr.visibility = 1.f;
  sr.hasHit = rec.hit.foundHit ? 1u : 0u;
  sr.hasSampledBounce = 0u; // builtin uses the diffuse fallback in resolve

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

  vec3 N = rec.hit.Ns;
  if (!(glm::dot(N, N) > 1e-12f))
    N = rec.hit.Ng;

  // MDL hit: write a geometry-only placeholder (black appearance, valid ids /
  // normal / continuation origin) and let the per-material MDL kernel overwrite
  // the appearance + NEE. If no kernel is built for it, the slot renders black
  // rather than reading stale data.
  if (isMdlMaterial(rec.hit.material)) {
    sr.unshadowed = vec3(0.f);
    sr.albedo = vec3(0.f);
    sr.normal = N;
    sr.opacity = 1.f;
    sr.depth = rec.hit.t;
    sr.primID = rec.hit.primID;
    sr.objID = rec.hit.objID;
    sr.instID = rec.hit.instID;
    sr.shadowOrg = rec.hit.hitpoint + rec.hit.Ng * rec.hit.epsilon;
    return;
  }

  const BuiltinAppearance a = evalBuiltinAppearance(*fd, rec.hit);

  RandState rng = path.rng;

  // Stochastic cutout (matte BLEND/MASK, PBR opacity): draw once against the
  // material opacity. A transparent draw treats the surface as absent for this
  // sample — no shading, no coverage — and continues the ray straight through
  // so what's behind shows through holes. Averaged over samples this yields the
  // authored alpha, unlike an opacity*shading dimming that never reveals the
  // background. Reuses the sampled-bounce continuation.
  if (pcg_uniform(&rng) >= a.opacity) {
    sr.unshadowed = vec3(0.f);
    sr.albedo = vec3(0.f);
    sr.normal = N;
    sr.opacity = 0.f;
    sr.depth = rec.hit.t;
    sr.primID = rec.hit.primID;
    sr.objID = rec.hit.objID;
    sr.instID = rec.hit.instID;
    sr.hasSampledBounce = 1u;
    sr.bounceDir = rec.rayDir; // straight through, unchanged direction
    sr.bounceWeight = vec3(1.f); // no attenuation
    const float side = glm::dot(rec.rayDir, rec.hit.Ng) >= 0.f ? 1.f : -1.f;
    sr.bounceOrg = rec.hit.hitpoint + rec.hit.Ng * (rec.hit.epsilon * side);
    path.rng = rng;
    return;
  }

  // PhysicallyBased: evaluate the full glTF BRDF (GGX specular, metallic,
  // Fresnel, transmission) via the shared header — the SAME code the
  // interactive renderer's callables run. NEE uses pbrEvalNEE; the continuation
  // bounce is importance-sampled by pbrSampleNextRay (reflection/refraction),
  // so metallic and glass render, not just the diffuse albedo. Matte stays on
  // the diffuse fallback below (its BSDF is Lambertian and its own nextRay is a
  // dead ray; the resolve stage's cosine bounce gives it correct GI).
  if (rec.hit.material->callableBaseIndex
      == uint32_t(SbtCallableEntryPoints::PBR)) {
    PhysicallyBasedShadingState st;
    pbrInitState(
        &st, fd, &rec.hit, &rec.hit.material->materialData.physicallyBased);
    const vec3 wo = -rec.rayDir;

    sr.albedo = st.baseColor;
    sr.normal = st.normal;
    sr.opacity = 1.f;
    sr.unshadowed =
        st.baseColor * fd->renderer.ambientIntensity * fd->renderer.ambientColor
        + st.emission;
    sr.depth = rec.hit.t;
    sr.primID = rec.hit.primID;
    sr.objID = rec.hit.objID;
    sr.instID = rec.hit.instID;
    sr.shadowOrg = rec.hit.hitpoint + rec.hit.Ng * rec.hit.epsilon;

    const uint32_t nPbr = uint32_t(fd->world.numLightInstances);
    if (nPbr > 0) {
      ScreenSample ssPbr;
      ssPbr.frameData = fd;
      ssPbr.rs = rng;
      ssPbr.shadowContribWeight = 1.0f;
      uint32_t kPbr = uint32_t(pcg_uniform(&ssPbr.rs) * float(nPbr));
      if (kPbr >= nPbr)
        kPbr = nPbr - 1u;
      const InstanceLightGPUData &liPbr = fd->world.lightInstances[kPbr];
      const LightSample lsPbr = sampleLight(ssPbr,
          sr.shadowOrg,
          liPbr.lightIndex,
          liPbr.xfm,
          liPbr.surfaceInstanceIndex);
      rng = ssPbr.rs;
      if (lsPbr.pdf > 0.f && lsPbr.dist > 0.f) {
        // pbrEvalNEE already folds NdotL and /pdf; reweight by nPbr for the
        // uniform 1/n light pick.
        sr.directContrib = pbrEvalNEE(&st, &rec.hit, &lsPbr, &wo) * float(nPbr);
        sr.shadowDir = lsPbr.dir;
        sr.shadowDist = lsPbr.dist;
      }
    }

    // Importance-sampled continuation (reflection or refraction).
    const Ray ray{rec.hit.hitpoint, rec.rayDir};
    const NextRay nr = pbrSampleNextRay(&st, &ray, &rng);
    sr.hasSampledBounce = 1u;
    sr.bounceDir = nr.direction;
    sr.bounceWeight = nr.contributionWeight;
    const float side = glm::dot(nr.direction, rec.hit.Ng) >= 0.f ? 1.f : -1.f;
    sr.bounceOrg = rec.hit.hitpoint + rec.hit.Ng * (rec.hit.epsilon * side);
    path.rng = rng;
    return;
  }

  // Matte, opaque this sample: full Lambertian shading, full coverage — opacity
  // is folded stochastically above, so no per-term opacity factor here.
  const vec3 ambient =
      a.baseColor * fd->renderer.ambientIntensity * fd->renderer.ambientColor;
  sr.unshadowed = ambient + a.emission;
  sr.albedo = a.baseColor;
  sr.normal = N;
  sr.opacity = 1.f;
  sr.depth = rec.hit.t;
  sr.primID = rec.hit.primID;
  sr.objID = rec.hit.objID;
  sr.instID = rec.hit.instID;
  // Offset hit point: shadow-ray origin AND the continuation-ray origin.
  sr.shadowOrg = rec.hit.hitpoint + rec.hit.Ng * rec.hit.epsilon;

  const uint32_t n = uint32_t(fd->world.numLightInstances);
  if (n == 0) {
    path.rng = rng;
    return;
  }

  // sampleLight() handles every light type, including Geometry Lights (their
  // emission is read as the light's mean radiance here — see the
  // VISRTX_STATIC_GEOMETRY_LIGHT_EMISSION note above).
  ScreenSample ss;
  ss.frameData = fd;
  ss.rs = rng;
  ss.shadowContribWeight = 1.0f;
  uint32_t k = uint32_t(pcg_uniform(&ss.rs) * float(n));
  if (k >= n)
    k = n - 1u;
  const InstanceLightGPUData &li = fd->world.lightInstances[k];
  const LightSample ls = sampleLight(
      ss, sr.shadowOrg, li.lightIndex, li.xfm, li.surfaceInstanceIndex);
  path.rng = ss.rs; // the light pick + sample consumed the path RNG
  if (ls.pdf > 0.f && ls.dist > 0.f) {
    const float ndotl = fmaxf(0.f, glm::dot(N, ls.dir));
    // Uniform pick over n lights has probability 1/n, so reweight by n.
    sr.directContrib = a.baseColor * ndotl * ls.radiance / ls.pdf * float(n);
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

VISRTX_DEVICE void wavefrontAccumulate(const FrameGPUData &fd,
    const uvec2 &pixel,
    const vec4 &color,
    const vec3 &albedo,
    const vec3 &normal)
{
  if (fireflyModeIsAtomicSafe(fd.renderer.fireflyFilterMode))
    accumPixelSampleAtomic(fd, pixel, color, albedo, normal);
  else
    accumPixelSample(fd, pixel, color, albedo, normal);
}

// Resolve one bounce: deposit this bounce's throughput-weighted radiance and,
// if the path continues, sample a diffuse continuation ray. The pixel divisor
// counts one sample per PATH, so every bounce adds to the same pixel while only
// the first bounce contributes the coverage/AOV channels. The path terminates
// on a miss, at max depth, or when throughput collapses.
__global__ void wavefrontResolveKernel(const FrameGPUData *fd,
    uint32_t liveSlots,
    uint32_t bounce,
    uint32_t maxDepth)
{
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= liveSlots)
    return;
  const WavefrontPathSlot slot = fd->wavefrontSlots[i];
  if (!slot.alive)
    return;
  WavefrontPathState &path = fd->wavefrontPaths[i];
  if (!path.alive)
    return;

  const WavefrontShadeRecord &sr = fd->wavefrontShade[i];
  const uvec2 pixel = {slot.pixel % fd->fb.size.x, slot.pixel / fd->fb.size.x};

  const vec3 bounceRadiance = sr.unshadowed + sr.visibility * sr.directContrib;
  const vec3 color = path.throughput * bounceRadiance;

  if (bounce == 0) {
    // First bounce owns the coverage/AOV channels and the first-hit ids.
    if (slot.sampleIdx == 0 && fd->fb.frameID == 0)
      setPixelIds(fd->fb, pixel, sr.depth, sr.primID, sr.objID, sr.instID);
    wavefrontAccumulate(
        *fd, pixel, vec4(color, sr.opacity), sr.albedo, sr.normal);
  } else {
    // Later bounces add radiance only — coverage and AOVs already counted.
    wavefrontAccumulate(*fd, pixel, vec4(color, 0.f), vec3(0.f), vec3(0.f));
  }

  // Continue the path, or terminate.
  if (!sr.hasHit || bounce + 1u >= maxDepth) {
    path.alive = 0;
    return;
  }
  if (sr.hasSampledBounce) {
    // A per-material kernel (MDL) importance-sampled its own BSDF: take its
    // direction, BSDF-over-pdf throughput factor, and side-aware origin (which
    // sits past the surface for a transmission lobe) directly.
    path.throughput *= sr.bounceWeight;
    path.nextDir = sr.bounceDir;
    path.nextOrg = sr.bounceOrg;
  } else {
    // Builtin cosine-weighted Lambertian: the cos/pdf and 1/pi fold to albedo.
    RandState rng = path.rng;
    path.nextDir = sampleHemisphere(rng, sr.normal);
    path.rng = rng;
    path.throughput *= sr.albedo;
    path.nextOrg = sr.shadowOrg; // +Ng-offset hit point recorded by shade-emit
  }
  // Kill paths whose contribution can no longer matter.
  if (fmaxf(path.throughput.x, fmaxf(path.throughput.y, path.throughput.z))
      < 1.0e-4f)
    path.alive = 0;
}

// Linear scan over the (small) set of registered MDL materials to find the
// bucket for a hit's callableBaseIndex. Returns -1 for non-MDL / unregistered.
__device__ int mdlBucketOf(const uint32_t *baseIndices,
    uint32_t numMaterials,
    uint32_t callableBaseIndex)
{
  for (uint32_t b = 0; b < numMaterials; ++b) {
    if (baseIndices[b] == callableBaseIndex)
      return int(b);
  }
  return -1;
}

// True for a live pool slot whose trace hit a surface with a registered MDL
// material. Shared gate for the count and scatter passes so both partition
// identically.
__device__ bool mdlSlotBucket(const FrameGPUData &fd,
    const uint32_t *baseIndices,
    uint32_t numMaterials,
    uint32_t i,
    int &bucket)
{
  if (!fd.wavefrontSlots[i].alive || !fd.wavefrontPaths[i].alive)
    return false;
  const WavefrontHitRecord &rec = fd.wavefrontHits[i];
  if (!rec.hit.foundHit || !rec.hit.material)
    return false;
  bucket = mdlBucketOf(
      baseIndices, numMaterials, rec.hit.material->callableBaseIndex);
  return bucket >= 0;
}

// Gather surviving (slot, path) pairs into a dense prefix of the destination
// buffers via an atomic append cursor. A slot survives if it was live this
// bounce and its path is still alive after resolve.
__global__ void wavefrontCompactAliveKernel(const WavefrontPathSlot *srcSlots,
    const WavefrontPathState *srcPaths,
    WavefrontPathSlot *dstSlots,
    WavefrontPathState *dstPaths,
    uint32_t inCount,
    uint32_t *outCount)
{
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= inCount)
    return;
  if (!srcSlots[i].alive || !srcPaths[i].alive)
    return;
  const uint32_t pos = atomicAdd(outCount, 1u);
  dstSlots[pos] = srcSlots[i];
  dstPaths[pos] = srcPaths[i];
}

// Single-pass material-sorted scatter. Each material owns a fixed-stride region
// packed[bucket * stride ..]; a per-material atomic cursor gives the append
// position AND doubles as that material's final slot count (no separate count
// pass or prefix sum). One full-pool read instead of two — profiling showed the
// old count+offset+scatter compaction was ~16% of GPU time in an MDL scene.
__global__ void wavefrontMdlScatterKernel(const FrameGPUData *fd,
    const uint32_t *baseIndices,
    uint32_t numMaterials,
    uint32_t liveSlots,
    uint32_t stride,
    uint32_t *cursor,
    uint32_t *packed)
{
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= liveSlots)
    return;
  int bucket = -1;
  if (mdlSlotBucket(*fd, baseIndices, numMaterials, i, bucket)) {
    const uint32_t pos = atomicAdd(&cursor[bucket], 1u);
    packed[uint32_t(bucket) * stride + pos] = i;
  }
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

void wavefrontResolve(cudaStream_t stream,
    const FrameGPUData *frameData,
    uint32_t liveSlots,
    uint32_t bounce,
    uint32_t maxDepth)
{
  if (liveSlots == 0)
    return;
  const uint32_t blocks = (liveSlots + kThreadsPerBlock - 1) / kThreadsPerBlock;
  wavefrontResolveKernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      frameData, liveSlots, bounce, maxDepth);
}

void wavefrontCompactAlive(cudaStream_t stream,
    const WavefrontPathSlot *srcSlots,
    const WavefrontPathState *srcPaths,
    WavefrontPathSlot *dstSlots,
    WavefrontPathState *dstPaths,
    uint32_t inCount,
    uint32_t *outCount)
{
  cudaMemsetAsync(outCount, 0, sizeof(uint32_t), stream);
  if (inCount == 0)
    return;
  const uint32_t blocks = (inCount + kThreadsPerBlock - 1) / kThreadsPerBlock;
  wavefrontCompactAliveKernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      srcSlots, srcPaths, dstSlots, dstPaths, inCount, outCount);
}

void wavefrontMdlCompact(cudaStream_t stream,
    const FrameGPUData *frameData,
    const uint32_t *baseIndices,
    uint32_t numMaterials,
    uint32_t liveSlots,
    uint32_t stride,
    uint32_t *cursor,
    uint32_t *packed)
{
  if (numMaterials == 0 || liveSlots == 0)
    return;
  const uint32_t blocks = (liveSlots + kThreadsPerBlock - 1) / kThreadsPerBlock;
  // cursor doubles as the per-material count; zero it, then a single scatter
  // pass appends each MDL slot into its material's fixed-stride region.
  cudaMemsetAsync(cursor, 0, size_t(numMaterials) * sizeof(uint32_t), stream);
  wavefrontMdlScatterKernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      frameData, baseIndices, numMaterials, liveSlots, stride, cursor, packed);
}

} // namespace visrtx
