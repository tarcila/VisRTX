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

#include "OpacityMicromap.h"

#include "geometry/Geometry.h"
#include "gpu/evalAttributes.h"
#include "material/Material.h"

#include <optix_micromap.h>

#include <texture_indirect_functions.h>

#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace visrtx {

// Bake tuning ////////////////////////////////////////////////////////////////

// 4^6 microtris per triangle: carves a 64-per-edge grid, plenty for foliage
// cutouts while keeping bakes and micromap memory small.
constexpr int OMM_MAX_LEVEL = 6;
// Attribute-only alpha is linear per triangle; a shallow level already carves
// the zero half-plane.
constexpr int OMM_ATTRIBUTE_ONLY_LEVEL = 3;
constexpr float OMM_TARGET_TEXELS_PER_MICROTRI = 4.f; // ~2x2 texel blocks
// Per-microtri texel-scan cap; costlier footprints classify as unknown.
constexpr uint32_t OMM_MAX_TEXELS_PER_SCAN = 1024;
// Scratch ceiling for the un-compacted per-triangle state slots; the level is
// lowered until the bake fits.
constexpr size_t OMM_MAX_SCRATCH_BYTES = size_t(64) << 20;

constexpr uint32_t TRI_HAS_TRANSPARENT = 1;
constexpr uint32_t TRI_HAS_UNKNOWN = 2;
constexpr uint32_t TRI_HAS_OPAQUE = 4;

// Bake kernels ///////////////////////////////////////////////////////////////

namespace {

struct OmmFactorView
{
  MaterialParameterType type{MaterialParameterType::VALUE};
  float value{1.f};
  MaterialAttribute attribute{MaterialAttribute::UNKNOWN};
  SamplerGPUData sampler{};
  int channel{0};
  // MDL's mono float lookup has no single defined channel here; bound over
  // all of them — a superset interval is conservative for any convention.
  bool allChannels{false};
};

struct OmmBakeParams
{
  TriangleGeometryData tri;
  AttributeData primAttr[5];
  vec4 attrUniform[5];
  OmmFactorView factors[2];
  AlphaMode mode;
  float cutoff;
  // Emit hard OPAQUE states (experimental; requires provably-zero
  // transmission and no backface culling — see resolveBakeSetup).
  bool allowOpaque;
  uint32_t numTris;
  int level;
  uint32_t microPerTri;
  uint32_t slotWords; // 32-bit words per triangle state slot
  uint32_t *states; // zero-initialized; transparent microtris stay unwritten
  uint32_t *triClass;
};

VISRTX_DEVICE vec4 bakeReadAttribute(const OmmBakeParams &p,
    MaterialAttribute attribute,
    uint32_t primID,
    const vec3 &b)
{
  const uint8_t id = static_cast<uint8_t>(attribute);
  const vec4 &uf = p.attrUniform[id];
  const auto &apFV = p.tri.vertexAttrFV[id];
  if (isPopulated(apFV)) {
    const uvec3 idx = uvec3(0, 1, 2) + primID * 3;
    return b.x * getAttributeValue(apFV, idx.x, uf)
        + b.y * getAttributeValue(apFV, idx.y, uf)
        + b.z * getAttributeValue(apFV, idx.z, uf);
  }
  const auto &ap = p.tri.vertexAttr[id];
  if (isPopulated(ap)) {
    const uvec3 idx =
        p.tri.indices ? p.tri.indices[primID] : 3 * primID + uvec3(0, 1, 2);
    return b.x * getAttributeValue(ap, idx.x, uf)
        + b.y * getAttributeValue(ap, idx.y, uf)
        + b.z * getAttributeValue(ap, idx.z, uf);
  }
  if (isPopulated(p.primAttr[id]))
    return getAttributeValue(p.primAttr[id], primID, uf);
  return uf;
}

// Conservative bounds of one factor over a microtriangle. Attribute factors
// are linear over the (micro)triangle, so corner values bound them exactly.
// Sampler factors scan the texel centers under the microtri's UV bbox dilated
// by the bilinear support; bilinear blends never exceed contributing texels,
// so the scan bounds the filtered field. Returns false when the bounds cannot
// be established (footprint too large) — caller must classify unknown.
VISRTX_DEVICE bool factorBounds(const OmmBakeParams &p,
    const OmmFactorView &f,
    uint32_t primID,
    const vec3 b[3],
    float &lo,
    float &hi)
{
  switch (f.type) {
  case MaterialParameterType::VALUE:
    lo = hi = f.value;
    return true;
  case MaterialParameterType::ATTRIBUTE: {
    lo = FLT_MAX;
    hi = -FLT_MAX;
    for (int i = 0; i < 3; i++) {
      const float v = bakeReadAttribute(p, f.attribute, primID, b[i])[f.channel];
      lo = fminf(lo, v);
      hi = fmaxf(hi, v);
    }
    return true;
  }
  case MaterialParameterType::SAMPLER: {
    const SamplerGPUData &s = f.sampler;
    vec2 uvLo(FLT_MAX), uvHi(-FLT_MAX);
    for (int i = 0; i < 3; i++) {
      const vec4 at = bakeReadAttribute(p, s.attribute, primID, b[i]);
      const vec4 tc = s.inTransform * at + s.inOffset;
      uvLo = glm::min(uvLo, vec2(tc));
      uvHi = glm::max(uvHi, vec2(tc));
    }
    const uvec2 size = s.image2D.size;
    const vec2 fsize(size);
    // Exact bilinear support of the bbox: a sample at u reads texels
    // floor(u*w - 0.5) and +1; nearest filtering reads a subset of these.
    const int x0 = int(floorf(uvLo.x * fsize.x - 0.5f));
    const int x1 = int(floorf(uvHi.x * fsize.x - 0.5f)) + 1;
    const int y0 = int(floorf(uvLo.y * fsize.y - 0.5f));
    const int y1 = int(floorf(uvHi.y * fsize.y - 0.5f)) + 1;
    const long long count =
        (long long)(x1 - x0 + 1) * (long long)(y1 - y0 + 1);
    if (count <= 0 || count > OMM_MAX_TEXELS_PER_SCAN)
      return false;
    lo = FLT_MAX;
    hi = -FLT_MAX;
    for (int ty = y0; ty <= y1; ty++) {
      for (int tx = x0; tx <= x1; tx++) {
        // Texel-center fetch through the shading texture object: exact texel
        // value, hardware wrap semantics identical to shading.
        vec4 t = make_vec4(tex2D<::float4>(s.image2D.texobj,
            (tx + 0.5f) / fsize.x,
            (ty + 0.5f) / fsize.y));
        if (s.numChannels < 4)
          t.w = 1.f;
        t = s.outTransform * t + s.outOffset;
        if (f.allChannels) {
          lo = fminf(lo, fminf(fminf(t.x, t.y), fminf(t.z, t.w)));
          hi = fmaxf(hi, fmaxf(fmaxf(t.x, t.y), fmaxf(t.z, t.w)));
        } else {
          const float v = t[f.channel];
          lo = fminf(lo, v);
          hi = fmaxf(hi, v);
        }
      }
    }
    return true;
  }
  default:
    return false;
  }
}

VISRTX_GLOBAL void bakeOmmStatesKernel(OmmBakeParams p)
{
  const uint32_t total = p.numTris * p.microPerTri;
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total;
       i += gridDim.x * blockDim.x) {
    const uint32_t primID = i / p.microPerTri;
    const uint32_t micro = i % p.microPerTri;

    float2 b0, b1, b2;
    optixMicromapIndexToBaseBarycentrics(micro, p.level, b0, b1, b2);
    const vec3 b[3] = {vec3(1.f - b0.x - b0.y, b0.x, b0.y),
        vec3(1.f - b1.x - b1.y, b1.x, b1.y),
        vec3(1.f - b2.x - b2.y, b2.x, b2.y)};

    float lo0, hi0, lo1, hi1;
    bool bounded = factorBounds(p, p.factors[0], primID, b, lo0, hi0)
        && factorBounds(p, p.factors[1], primID, b, lo1, hi1);

    bool transparent = false;
    bool opaque = false;
    if (bounded) {
      // alpha = colorAlpha * opacity; interval product over the 4 endpoint
      // combinations (factors may be negative through out-transforms).
      const float c[4] = {lo0 * lo1, lo0 * hi1, hi0 * lo1, hi0 * hi1};
      const float mn = fminf(fminf(c[0], c[1]), fminf(c[2], c[3]));
      const float mx = fmaxf(fmaxf(c[0], c[1]), fmaxf(c[2], c[3]));
      if (p.mode == AlphaMode::MASK) {
        transparent = mx < p.cutoff;
        // strict >: conservative under both the CUDA (>=) and MDL (>)
        // cutoff conventions
        opaque = p.allowOpaque && mn > p.cutoff;
      } else { // BLEND: exact endpoints only
        transparent = mx <= 0.f && mn >= 0.f;
        opaque = p.allowOpaque && mn >= 1.f;
      }
    }

    if (transparent) {
      // State TRANSPARENT == 0: the zero-initialized slot already encodes it.
      atomicOr(&p.triClass[primID], TRI_HAS_TRANSPARENT);
    } else if (opaque) {
      atomicOr(&p.states[primID * p.slotWords + (micro >> 4)],
          uint32_t(OPTIX_OPACITY_MICROMAP_STATE_OPAQUE) << (2 * (micro & 15)));
      atomicOr(&p.triClass[primID], TRI_HAS_OPAQUE);
    } else {
      atomicOr(&p.states[primID * p.slotWords + (micro >> 4)],
          uint32_t(OPTIX_OPACITY_MICROMAP_STATE_UNKNOWN_OPAQUE)
              << (2 * (micro & 15)));
      atomicOr(&p.triClass[primID], TRI_HAS_UNKNOWN);
    }
  }
}

// Mean UV footprint (in texels) of the sampler-driven factors, for the auto
// subdivision level.
VISRTX_GLOBAL void ommTexelCoverageKernel(OmmBakeParams p, float *sumCoverage)
{
  for (uint32_t primID = blockIdx.x * blockDim.x + threadIdx.x;
       primID < p.numTris;
       primID += gridDim.x * blockDim.x) {
    const vec3 b[3] = {
        vec3(1.f, 0.f, 0.f), vec3(0.f, 1.f, 0.f), vec3(0.f, 0.f, 1.f)};
    float coverage = 0.f;
    for (int fi = 0; fi < 2; fi++) {
      const OmmFactorView &f = p.factors[fi];
      if (f.type != MaterialParameterType::SAMPLER)
        continue;
      vec2 tc[3];
      for (int i = 0; i < 3; i++) {
        const vec4 at = bakeReadAttribute(p, f.sampler.attribute, primID, b[i]);
        tc[i] = vec2(f.sampler.inTransform * at + f.sampler.inOffset)
            * vec2(f.sampler.image2D.size);
      }
      const vec2 e0 = tc[1] - tc[0];
      const vec2 e1 = tc[2] - tc[0];
      coverage = fmaxf(coverage, 0.5f * fabsf(e0.x * e1.y - e0.y * e1.x));
    }
    atomicAdd(sumCoverage, coverage);
  }
}

// Monotonic total-order encoding of a float into uint32 bits: unsigned
// comparisons (and unsigned atomicMin/Max) then order exactly like the
// floats, including negatives. Signed-float textures with negative texels
// must not be clamped away — a signed out-transform coefficient can map a
// negative texel to a positive contribution downstream.
VISRTX_HOST_DEVICE uint32_t floatToOrderedBits(float f)
{
  uint32_t b;
  memcpy(&b, &f, sizeof(b));
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

VISRTX_HOST_DEVICE float orderedBitsToFloat(uint32_t u)
{
  uint32_t b = (u & 0x80000000u) ? (u & 0x7FFFFFFFu) : ~u;
  float f;
  memcpy(&f, &b, sizeof(f));
  return f;
}

// Per-channel min/max over a 2D texture's texels (texel-center fetches, same
// exactness argument as the bake scan). Feeds the whole-domain skip verdict.
// Results are ordered-bits encoded (see floatToOrderedBits).
VISRTX_GLOBAL void textureRangeKernel(
    SamplerGPUData s, uint32_t *mn4, uint32_t *mx4)
{
  const uvec2 size = s.image2D.size;
  // 64-bit texel count: 65536x65536 wraps a uint32 to 0 and the kernel would
  // silently scan nothing.
  const uint64_t total = uint64_t(size.x) * size.y;
  vec4 mn(FLT_MAX), mx(-FLT_MAX);
  for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total;
       i += uint64_t(gridDim.x) * blockDim.x) {
    vec4 t = make_vec4(tex2D<::float4>(s.image2D.texobj,
        (i % size.x + 0.5f) / float(size.x),
        (i / size.x + 0.5f) / float(size.y)));
    if (s.numChannels < 4)
      t.w = 1.f;
    mn = glm::min(mn, t);
    mx = glm::max(mx, t);
  }
  for (int c = 0; c < 4; c++) {
    atomicMin(&mn4[c], floatToOrderedBits(mn[c]));
    atomicMax(&mx4[c], floatToOrderedBits(mx[c]));
  }
}

// Counts non-transparent (bit0: OPAQUE=01 or UNKNOWN=11) and unknown (bit1)
// microtris; transparent = total - nonTransparent, opaque = the difference.
VISRTX_GLOBAL void countStatesKernel(const uint32_t *states,
    uint32_t numWords,
    uint32_t *nonTransparentCount,
    uint32_t *unknownCount)
{
  uint32_t nonTransparent = 0, unknown = 0;
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < numWords;
       i += gridDim.x * blockDim.x) {
    nonTransparent += __popc(states[i] & 0x55555555u);
    unknown += __popc(states[i] & 0xAAAAAAAAu);
  }
  atomicAdd(nonTransparentCount, nonTransparent);
  atomicAdd(unknownCount, unknown);
}

VISRTX_GLOBAL void gatherMixedSlotsKernel(const uint32_t *states,
    const uint32_t *mixedTriIndices,
    uint32_t numMixed,
    uint32_t slotWords,
    uint32_t *packed)
{
  const uint32_t total = numMixed * slotWords;
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total;
       i += gridDim.x * blockDim.x) {
    const uint32_t mixed = i / slotWords;
    const uint32_t word = i % slotWords;
    packed[i] = states[mixedTriIndices[mixed] * slotWords + word];
  }
}

// Host-side eligibility //////////////////////////////////////////////////////

bool geometryResolvesAttribute(
    const GeometryGPUData &ggd, MaterialAttribute attribute)
{
  // Bake reads must never fall through to instance-supplied values
  // (readAttributeValue consults them before the geometry uniform), so only
  // attributes the geometry itself populates are bakeable.
  if (attribute >= MaterialAttribute::OBJECT_POSITION)
    return false;
  const uint8_t id = static_cast<uint8_t>(attribute);
  return ggd.tri.vertexAttr[id].numChannels > 0
      || ggd.tri.vertexAttrFV[id].numChannels > 0
      || ggd.attr[id].numChannels > 0;
}

// Fills `out` from a MaterialParameter; false when the parameter cannot be
// conservatively evaluated by the bake kernel. channel < 0 bounds over all
// channels. `raw` (MDL) normalizes the sampler view to the wrapper's fixed
// uv0 / untransformed lookups.
bool resolveFactor(OmmFactorView &out,
    const MaterialParameter &mp,
    int channel,
    const GeometryGPUData &ggd,
    DeviceGlobalState *state,
    bool raw)
{
  out.allChannels = channel < 0;
  out.channel = std::max(channel, 0);
  out.type = mp.type;
  switch (mp.type) {
  case MaterialParameterType::VALUE:
    out.value = mp.value[out.channel];
    return true;
  case MaterialParameterType::ATTRIBUTE:
    out.attribute = mp.attribute;
    return !raw && geometryResolvesAttribute(ggd, mp.attribute);
  case MaterialParameterType::SAMPLER: {
    SamplerGPUData sd = state->registry.samplers.hostValue(mp.sampler);
    if (sd.type != SamplerType::TEXTURE2D)
      return false;
    if (raw) {
      // MDL lookups still apply the sampler's affine transforms
      // (tex_lookup_float*_2d → evaluateImageTextureSampler), but they wrap
      // uv0 BEFORE the in-transform — a non-identity in-transform therefore
      // has no faithful equivalent in this scan; refuse the bake rather than
      // bound the wrong function. The out-transform composes after the fetch
      // and is kept.
      if (sd.inTransform != mat4(1.f) || sd.inOffset != vec4(0.f))
        return false;
      sd.attribute = MaterialAttribute::ATTRIB_0;
    }
    if (!geometryResolvesAttribute(ggd, sd.attribute))
      return false;
    out.sampler = sd;
    return true;
  }
  default:
    return false;
  }
}

// Whole-domain range of one factor (any point on any surface using it);
// false when unbounded (attribute-driven). Sampler ranges are cached per
// sampler in DeviceGlobalState and refreshed when the sampler re-finalizes.
// Ranges are exact for signed texel values (ordered-bits atomics), so
// downstream proofs may rely on both endpoints.
bool factorDomainRange(const OmmFactorView &f,
    const MaterialParameter &mp,
    DeviceGlobalState *state,
    float &lo,
    float &hi)
{
  switch (f.type) {
  case MaterialParameterType::VALUE:
    lo = hi = f.value;
    return true;
  case MaterialParameterType::SAMPLER: {
    auto &cache = state->omm.samplerRanges[mp.sampler];
    auto *samplerObj =
        static_cast<Object *>(state->registry.samplers.hostObject(mp.sampler));
    const auto stamp = samplerObj->lastFinalized();
    if (cache.stamp != stamp) {
      const uint32_t mnInit[4] = {floatToOrderedBits(FLT_MAX),
          floatToOrderedBits(FLT_MAX),
          floatToOrderedBits(FLT_MAX),
          floatToOrderedBits(FLT_MAX)};
      const uint32_t mxInit[4] = {floatToOrderedBits(-FLT_MAX),
          floatToOrderedBits(-FLT_MAX),
          floatToOrderedBits(-FLT_MAX),
          floatToOrderedBits(-FLT_MAX)};
      DeviceBuffer mnBuf, mxBuf;
      mnBuf.upload(mnInit, 4);
      mxBuf.upload(mxInit, 4);
      const uvec2 size = f.sampler.image2D.size;
      const uint64_t total = uint64_t(size.x) * size.y;
      const uint32_t bs = 256;
      const uint32_t nb =
          uint32_t(std::min<uint64_t>(1024, (total + bs - 1) / bs));
      textureRangeKernel<<<nb, bs, 0, state->stream>>>(
          f.sampler, mnBuf.ptrAs<uint32_t>(), mxBuf.ptrAs<uint32_t>());
      uint32_t mn[4], mx[4];
      mnBuf.download(mn, 4);
      mxBuf.download(mx, 4);
      cudaStreamSynchronize(state->stream);
      cache.stamp = stamp;
      cache.mn = vec4(orderedBitsToFloat(mn[0]),
          orderedBitsToFloat(mn[1]),
          orderedBitsToFloat(mn[2]),
          orderedBitsToFloat(mn[3]));
      cache.mx = vec4(orderedBitsToFloat(mx[0]),
          orderedBitsToFloat(mx[1]),
          orderedBitsToFloat(mx[2]),
          orderedBitsToFloat(mx[3]));
    }
    // Interval through the sampler's affine out-transform; all-channel
    // factors reduce over every output channel.
    auto channelInterval = [&](int c, float &clo, float &chi) {
      double l = f.sampler.outOffset[c];
      double h = l;
      for (int j = 0; j < 4; j++) {
        const float coeff = f.sampler.outTransform[j][c];
        const float a = cache.mn[j], b = cache.mx[j];
        l += coeff * (coeff >= 0.f ? a : b);
        h += coeff * (coeff >= 0.f ? b : a);
      }
      clo = float(l);
      chi = float(h);
    };
    if (f.allChannels) {
      lo = FLT_MAX;
      hi = -FLT_MAX;
      for (int c = 0; c < 4; c++) {
        float clo, chi;
        channelInterval(c, clo, chi);
        lo = std::min(lo, clo);
        hi = std::max(hi, chi);
      }
    } else {
      channelInterval(f.channel, lo, hi);
    }
    return true;
  }
  default:
    return false;
  }
}

int autoSubdivisionLevel(const OmmBakeParams &p,
    bool anySamplerFactor,
    DeviceGlobalState *state)
{
  if (!anySamplerFactor)
    return OMM_ATTRIBUTE_ONLY_LEVEL;

  DeviceBuffer sumBuf;
  sumBuf.reserve(sizeof(float));
  cudaMemsetAsync(sumBuf.ptr(), 0, sizeof(float), state->stream);
  const uint32_t bs = 128;
  const uint32_t nb = std::min(4096u, (p.numTris + bs - 1) / bs);
  ommTexelCoverageKernel<<<nb, bs, 0, state->stream>>>(
      p, sumBuf.ptrAs<float>());
  float sum = 0.f;
  sumBuf.download(&sum); // syncs the stream copy
  cudaStreamSynchronize(state->stream);

  const float meanTexels = sum / float(std::max(p.numTris, 1u));
  const float microTarget = meanTexels / OMM_TARGET_TEXELS_PER_MICROTRI;
  // 4^level microtris per triangle
  const int level = int(std::ceil(std::log2(std::max(microTarget, 1.f)) / 2.f));
  return std::min(std::max(level, 0), OMM_MAX_LEVEL);
}

VISRTX_HOST_DEVICE uint64_t splitmix64(uint64_t x)
{
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}

// Position-mixed content hash of a device buffer, XOR-accumulated so lanes
// commute; `salt` keeps multi-buffer keys order-sensitive across buffers.
VISRTX_GLOBAL void hashBufferKernel(
    const uint8_t *data, size_t bytes, uint64_t salt, unsigned long long *acc)
{
  const size_t lanes = (bytes + 7) / 8;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < lanes;
       i += gridDim.x * blockDim.x) {
    uint64_t w = 0;
    const size_t o = i * 8;
    const int n = int(min(size_t(8), bytes - o));
    for (int b = 0; b < n; b++)
      w |= uint64_t(data[o + b]) << (8 * b);
    atomicXor(acc, (unsigned long long)splitmix64(w ^ splitmix64(i ^ salt)));
  }
}

} // namespace

// Everything the bake reads, resolved once by computeOpacityMicromapKey and
// handed to bakeOpacityMicromaps so cache keys and bakes can never disagree
// on eligibility — and a cache miss never resolves twice. Opaque outside
// this translation unit.
struct OmmBakeSetup
{
  MaterialAlphaSpec spec;
  GeometryGPUData ggd;
  uint32_t numTris{0};
  uint32_t numVertices{0};
  OmmBakeParams p{};
};

namespace {

// False when the (geometry, material) pair is OMM-ineligible or provably has
// no transparency anywhere (whole-domain verdict).
bool resolveBakeSetup(OmmBakeSetup &setup,
    Geometry *geometry,
    const Material *material,
    DeviceGlobalState *state)
{
  if (!state->omm.enabled)
    return false;

  setup.spec = material->alphaSpec();
  const auto &spec = setup.spec;
  if (!spec.bakeable || spec.mode == AlphaMode::OPAQUE)
    return false;
  if (!geometry->supportsOpacityMicromap())
    return false;

  setup.ggd = geometry->ommGeometryView();

  OptixBuildInput obi = {};
  geometry->populateBuildInput(obi);
  setup.numTris = obi.triangleArray.numIndexTriplets
      ? obi.triangleArray.numIndexTriplets
      : obi.triangleArray.numVertices / 3;
  setup.numVertices = obi.triangleArray.numVertices;
  if (setup.numTris == 0)
    return false;

  auto &p = setup.p;
  std::memcpy(&p.tri, &setup.ggd.tri, sizeof(p.tri));
  std::memcpy(&p.primAttr, &setup.ggd.attr, sizeof(p.primAttr));
  std::memcpy(&p.attrUniform, &setup.ggd.attrUniform, sizeof(p.attrUniform));
  p.mode = spec.mode;
  p.cutoff = spec.cutoff;
  p.numTris = setup.numTris;

  // MDL's mono opacity lookup has no defined channel — bound all of them.
  const int opacityChannel = spec.rawSamplerLookups ? -1 : 0;
  if (!resolveFactor(
          p.factors[0], spec.colorAlpha, 3, setup.ggd, state,
          spec.rawSamplerLookups)
      || !resolveFactor(p.factors[1],
          spec.opacity,
          opacityChannel,
          setup.ggd,
          state,
          spec.rawSamplerLookups))
    return false;

  // Hard OPAQUE states (experimental): only when transmission is provably
  // zero everywhere (an opaque-committed hit skips the transmission-aware
  // any-hit) and the geometry doesn't backface-cull (that cull also lives in
  // any-hit).
  p.allowOpaque = false;
  if (state->omm.opaqueStates) {
    const bool backfaceCulled = setup.ggd.type == GeometryType::TRIANGLE
        && setup.ggd.tri.cullBackfaces;
    OmmFactorView tf;
    float tl = 1.f, th = 1.f;
    // Same mono-lookup rule as the opacity factor above: raw MDL has no
    // defined channel, so the proof must bound ALL channels.
    const int transmissionChannel = spec.rawSamplerLookups ? -1 : 0;
    const bool transmissionZero = resolveFactor(tf,
                                      spec.transmission,
                                      transmissionChannel,
                                      setup.ggd,
                                      state,
                                      spec.rawSamplerLookups)
        && factorDomainRange(tf, spec.transmission, state, tl, th) && tl >= 0.f
        && th <= 0.f;
    p.allowOpaque = !backfaceCulled && transmissionZero;
  }

  // Whole-domain verdict: when both factors are boundable and their product
  // provably never reaches transparency (nor, with opaque states on,
  // provable opacity), no geometry using this material can gain anything
  // from an OMM — skip without touching the mesh.
  float dl0, dh0, dl1, dh1;
  if (factorDomainRange(p.factors[0], spec.colorAlpha, state, dl0, dh0)
      && factorDomainRange(p.factors[1], spec.opacity, state, dl1, dh1)) {
    const float c[4] = {dl0 * dl1, dl0 * dh1, dh0 * dl1, dh0 * dh1};
    const float mn = fminf(fminf(c[0], c[1]), fminf(c[2], c[3]));
    const float mx = fmaxf(fmaxf(c[0], c[1]), fmaxf(c[2], c[3]));
    const bool canBeTransparent =
        spec.mode == AlphaMode::MASK ? mn < spec.cutoff : mn <= 0.f;
    const bool canBeOpaque = p.allowOpaque
        && (spec.mode == AlphaMode::MASK ? mx > spec.cutoff : mx >= 1.f);
    if (!canBeTransparent && !canBeOpaque)
      return false;
  }

  return true;
}

} // namespace

// Public entry point /////////////////////////////////////////////////////////

bool bakeOpacityMicromaps(OpacityMicromapBuffers &out,
    OmmBakeSetup &setup,
    Object *reporter)
{
  // VISRTX_OMM_STATS=1 prints per-bake timing/coverage diagnostics.
  const bool stats = std::getenv("VISRTX_OMM_STATS") != nullptr;
  const auto t0 = std::chrono::steady_clock::now();
  auto elapsedMs = [&t0]() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0)
        .count();
  };

  out.reset();

  auto *state = reporter->deviceState();
  const uint32_t numTris = setup.numTris;
  OmmBakeParams &p = setup.p;

  const bool anySampler =
      p.factors[0].type == MaterialParameterType::SAMPLER
      || p.factors[1].type == MaterialParameterType::SAMPLER;

  int level = state->omm.subdivisionLevel;
  if (level < 0)
    level = autoSubdivisionLevel(p, anySampler, state);
  level = std::min(std::max(level, 0), OMM_MAX_LEVEL);
  // 2 bits per microtri; lower the level until scratch fits the budget.
  auto slotBytesFor = [](int l) {
    return std::max(size_t(1) << (2 * l) >> 2, sizeof(uint32_t));
  };
  while (level > 0 && slotBytesFor(level) * numTris > OMM_MAX_SCRATCH_BYTES)
    level--;

  p.level = level;
  p.microPerTri = 1u << (2 * level);
  const size_t slotBytes = slotBytesFor(level);
  p.slotWords = uint32_t(slotBytes / sizeof(uint32_t));

  DeviceBuffer stateScratch, classScratch;
  stateScratch.reserve(slotBytes * numTris);
  classScratch.reserve(sizeof(uint32_t) * numTris);
  cudaMemsetAsync(stateScratch.ptr(), 0, stateScratch.bytes(), state->stream);
  cudaMemsetAsync(classScratch.ptr(), 0, classScratch.bytes(), state->stream);
  p.states = stateScratch.ptrAs<uint32_t>();
  p.triClass = classScratch.ptrAs<uint32_t>();

  {
    const uint32_t bs = 128;
    const uint64_t total = uint64_t(numTris) * p.microPerTri;
    const uint32_t nb =
        uint32_t(std::min<uint64_t>(65536, (total + bs - 1) / bs));
    bakeOmmStatesKernel<<<nb, bs, 0, state->stream>>>(p);
  }

  DeviceBuffer countBuf;
  countBuf.reserve(2 * sizeof(uint32_t));
  cudaMemsetAsync(countBuf.ptr(), 0, 2 * sizeof(uint32_t), state->stream);
  {
    const uint32_t words = uint32_t(stateScratch.bytes() / sizeof(uint32_t));
    const uint32_t bs = 256;
    const uint32_t nb = std::min(2048u, (words + bs - 1) / bs);
    countStatesKernel<<<nb, bs, 0, state->stream>>>(
        stateScratch.ptrAs<uint32_t>(),
        words,
        countBuf.ptrAs<uint32_t>(),
        countBuf.ptrAs<uint32_t>() + 1);
  }

  std::vector<uint32_t> triClass(numTris);
  classScratch.download(triClass.data(), numTris);
  uint32_t counts[2] = {0, 0}; // nonTransparent, unknown
  countBuf.download(counts, 2);
  cudaStreamSynchronize(state->stream);

  const uint64_t totalMicroTris = uint64_t(numTris) * p.microPerTri;
  const double transparentFraction =
      1.0 - double(counts[0]) / double(totalMicroTris);
  const double opaqueFraction =
      double(counts[0] - counts[1]) / double(totalMicroTris);

  // Index buffer: uniform triangles use predefined indices (zero storage),
  // mixed triangles reference a real micromap.
  std::vector<int32_t> indices(numTris);
  std::vector<uint32_t> mixedTris;
  uint32_t numTransparentTris = 0;
  uint32_t numOpaqueTris = 0;
  for (uint32_t t = 0; t < numTris; t++) {
    const bool hasT = triClass[t] & TRI_HAS_TRANSPARENT;
    const bool hasU = triClass[t] & TRI_HAS_UNKNOWN;
    const bool hasO = triClass[t] & TRI_HAS_OPAQUE;
    if (hasT && !hasU && !hasO) {
      indices[t] = OPTIX_OPACITY_MICROMAP_PREDEFINED_INDEX_FULLY_TRANSPARENT;
      numTransparentTris++;
    } else if (hasO && !hasU && !hasT) {
      indices[t] = OPTIX_OPACITY_MICROMAP_PREDEFINED_INDEX_FULLY_OPAQUE;
      numOpaqueTris++;
    } else if (hasU && !hasT && !hasO) {
      indices[t] = OPTIX_OPACITY_MICROMAP_PREDEFINED_INDEX_FULLY_UNKNOWN_OPAQUE;
    } else {
      indices[t] = int32_t(mixedTris.size());
      mixedTris.push_back(t);
    }
  }

  if (numTransparentTris == 0 && numOpaqueTris == 0 && mixedTris.empty()) {
    reporter->reportMessage(ANARI_SEVERITY_DEBUG,
        "visrtx::OpacityMicromap skip: no provably transparent or opaque "
        "region");
    return false;
  }

  // OptiX requires a non-null micromap array in INDEXED mode even when every
  // triangle resolves to a predefined index (all-uniform bake, e.g. a sprite
  // atlas). Promote one triangle to a real micromap: its states in scratch are
  // uniform, so the encoding — and traversal behavior — is identical.
  if (mixedTris.empty()) {
    if (indices[0] == OPTIX_OPACITY_MICROMAP_PREDEFINED_INDEX_FULLY_TRANSPARENT)
      numTransparentTris--;
    else if (indices[0] == OPTIX_OPACITY_MICROMAP_PREDEFINED_INDEX_FULLY_OPAQUE)
      numOpaqueTris--;
    indices[0] = 0;
    mixedTris.push_back(0);
  }

  const uint32_t numMixed = uint32_t(mixedTris.size());

  if (numMixed > 0) {
    DeviceBuffer mixedIdxBuf, packedStates, descBuf;
    mixedIdxBuf.upload(mixedTris);
    packedStates.reserve(slotBytes * numMixed);
    {
      const uint32_t bs = 128;
      const uint32_t total = numMixed * p.slotWords;
      const uint32_t nb = std::min(65536u, (total + bs - 1) / bs);
      gatherMixedSlotsKernel<<<nb, bs, 0, state->stream>>>(
          stateScratch.ptrAs<uint32_t>(),
          mixedIdxBuf.ptrAs<uint32_t>(),
          numMixed,
          p.slotWords,
          packedStates.ptrAs<uint32_t>());
    }

    std::vector<OptixOpacityMicromapDesc> descs(numMixed);
    for (uint32_t i = 0; i < numMixed; i++) {
      descs[i].byteOffset = uint32_t(i * slotBytes);
      descs[i].subdivisionLevel = uint16_t(level);
      descs[i].format = OPTIX_OPACITY_MICROMAP_FORMAT_4_STATE;
    }
    descBuf.upload(descs);

    OptixOpacityMicromapHistogramEntry histogram = {};
    histogram.count = numMixed;
    histogram.subdivisionLevel = uint32_t(level);
    histogram.format = OPTIX_OPACITY_MICROMAP_FORMAT_4_STATE;

    OptixOpacityMicromapArrayBuildInput bi = {};
    bi.flags = OPTIX_OPACITY_MICROMAP_FLAG_NONE;
    bi.inputBuffer = (CUdeviceptr)packedStates.ptr();
    bi.perMicromapDescBuffer = (CUdeviceptr)descBuf.ptr();
    bi.perMicromapDescStrideInBytes = 0;
    bi.numMicromapHistogramEntries = 1;
    bi.micromapHistogramEntries = &histogram;

    OptixMicromapBufferSizes sizes = {};
    if (optixOpacityMicromapArrayComputeMemoryUsage(
            state->optixContext, &bi, &sizes)
        != OPTIX_SUCCESS) {
      reporter->reportMessage(ANARI_SEVERITY_WARNING,
          "visrtx::OpacityMicromap memory-usage query failed; rendering "
          "without OMM");
      out.reset();
      return false;
    }

    DeviceBuffer temp;
    temp.reserve(sizes.tempSizeInBytes);
    out.micromapArray.reserve(sizes.outputSizeInBytes);

    OptixMicromapBuffers buffers = {};
    buffers.output = (CUdeviceptr)out.micromapArray.ptr();
    buffers.outputSizeInBytes = out.micromapArray.bytes();
    buffers.temp = (CUdeviceptr)temp.ptr();
    buffers.tempSizeInBytes = temp.bytes();

    if (optixOpacityMicromapArrayBuild(
            state->optixContext, state->stream, &bi, &buffers)
        != OPTIX_SUCCESS) {
      reporter->reportMessage(ANARI_SEVERITY_WARNING,
          "visrtx::OpacityMicromap array build failed; rendering without OMM");
      out.reset();
      return false;
    }
    cudaStreamSynchronize(state->stream);

    out.usage[0].count = numMixed;
    out.usage[0].subdivisionLevel = uint32_t(level);
    out.usage[0].format = OPTIX_OPACITY_MICROMAP_FORMAT_4_STATE;
    out.numUsage = 1;
  }

  out.indexBuffer.upload(indices);
  out.attached = true;

  if (stats) {
    fprintf(stderr,
        "[omm] bake %.2f ms: %u tris level %d mixed %u transparentTris %u "
        "opaqueTris %u transparentFraction %.3f opaqueFraction %.3f\n",
        elapsedMs(),
        numTris,
        level,
        numMixed,
        numTransparentTris,
        numOpaqueTris,
        transparentFraction,
        opaqueFraction);
  }

  reporter->reportMessage(ANARI_SEVERITY_DEBUG,
      "visrtx::OpacityMicromap baked: %u tris (level %d, %u mixed, %u fully "
      "transparent, %zu KB)",
      numTris,
      level,
      numMixed,
      numTransparentTris,
      (out.micromapArray.bytes() + out.indexBuffer.bytes()) >> 10);

  return true;
}

bool computeOpacityMicromapKey(uint64_t &key,
    std::shared_ptr<OmmBakeSetup> &setup,
    Geometry *geometry,
    const Material *material,
    Object *reporter)
{
  auto *state = reporter->deviceState();
  setup = std::make_shared<OmmBakeSetup>();
  if (!resolveBakeSetup(*setup, geometry, material, state)) {
    setup.reset();
    return false;
  }
  const auto &spec = setup->spec;
  const auto &p = setup->p;

  // Scalar seed: everything non-buffer the bake result depends on.
  uint64_t seed = splitmix64(uint64_t(spec.mode));
  auto mixBits = [&seed](uint64_t v) { seed = splitmix64(seed ^ v); };
  auto mixFloat = [&](float v) {
    uint32_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    mixBits(bits);
  };
  mixFloat(spec.cutoff);
  mixBits(uint64_t(state->omm.subdivisionLevel) + 1);
  mixBits(uint64_t(setup->p.allowOpaque) + 3);
  mixBits(setup->numTris);

  // Buffers the bake reads: triangle indices plus, per factor, the populated
  // geometry attribute source (mirrors bakeReadAttribute's priority).
  struct BufferRef
  {
    const void *data;
    size_t bytes;
  };
  BufferRef buffers[3] = {};
  uint32_t numBuffers = 0;
  if (p.tri.indices)
    buffers[numBuffers++] = {p.tri.indices, sizeof(uvec3) * setup->numTris};

  auto addAttributeSource = [&](MaterialAttribute attribute) {
    if (attribute >= MaterialAttribute::OBJECT_POSITION)
      return; // not a geometry attribute (gated earlier; belt-and-braces)
    const uint8_t id = static_cast<uint8_t>(attribute);
    const AttributeData *src = nullptr;
    size_t count = 0;
    if (p.tri.vertexAttrFV[id].numChannels > 0) {
      src = &p.tri.vertexAttrFV[id];
      count = size_t(3) * setup->numTris;
    } else if (p.tri.vertexAttr[id].numChannels > 0) {
      src = &p.tri.vertexAttr[id];
      count = setup->numVertices;
    } else if (p.primAttr[id].numChannels > 0) {
      src = &p.primAttr[id];
      count = setup->numTris;
    }
    if (src && numBuffers < 3)
      buffers[numBuffers++] = {src->data, count * anari::sizeOf(src->type)};
  };

  for (const auto &f : p.factors) {
    // Channel selection changes the baked function even when every other
    // factor input matches (e.g. alpha-in-.w vs opacity-in-.x lookups of the
    // same texture must not share a bake). allChannels distinguishes MDL's
    // all-channel bound (channel -1, clamped to 0) from a plain channel-0
    // factor — a raw-MDL and a PBR material sharing sampler+geometry must
    // not share a bake either.
    mixBits(((uint64_t(uint32_t(f.channel)) << 1) | uint64_t(f.allChannels))
        + 0x30);
    switch (f.type) {
    case MaterialParameterType::VALUE:
      mixFloat(f.value);
      break;
    case MaterialParameterType::ATTRIBUTE:
      mixBits(uint64_t(f.attribute) + 0x10);
      addAttributeSource(f.attribute);
      break;
    case MaterialParameterType::SAMPLER: {
      // Texture content is covered by the sampler's finalize stamp (globally
      // monotonic — in-place image updates re-finalize the sampler).
      mixBits(uint64_t(f.sampler.attribute) + 0x20);
      mixBits(f.sampler.image2D.size.x);
      mixBits(f.sampler.image2D.size.y);
      const float *xf = &f.sampler.inTransform[0][0];
      for (int i = 0; i < 16; i++)
        mixFloat(xf[i]);
      for (int i = 0; i < 4; i++)
        mixFloat(f.sampler.inOffset[i]);
      const float *xo = &f.sampler.outTransform[0][0];
      for (int i = 0; i < 16; i++)
        mixFloat(xo[i]);
      for (int i = 0; i < 4; i++)
        mixFloat(f.sampler.outOffset[i]);
      addAttributeSource(f.sampler.attribute);
      break;
    }
    default:
      break;
    }
  }
  // Sampler content stamps (texture data identity).
  auto mixSamplerStamp = [&](const MaterialParameter &mp) {
    if (mp.type != MaterialParameterType::SAMPLER)
      return;
    auto *obj =
        static_cast<Object *>(state->registry.samplers.hostObject(mp.sampler));
    mixBits(uint64_t(obj->lastFinalized()));
  };
  mixSamplerStamp(spec.colorAlpha);
  mixSamplerStamp(spec.opacity);

  DeviceBuffer accBuf;
  accBuf.upload(&seed, 1);
  for (uint32_t i = 0; i < numBuffers; i++) {
    const size_t lanes = (buffers[i].bytes + 7) / 8;
    const uint32_t bs = 256;
    const uint32_t nb =
        uint32_t(std::min<size_t>(2048, (lanes + bs - 1) / bs));
    hashBufferKernel<<<nb, bs, 0, state->stream>>>(
        (const uint8_t *)buffers[i].data,
        buffers[i].bytes,
        seed ^ splitmix64(i + 1),
        accBuf.ptrAs<unsigned long long>());
  }
  uint64_t acc = 0;
  accBuf.download(&acc, 1);
  cudaStreamSynchronize(state->stream);

  key = acc;
  return true;
}

} // namespace visrtx
