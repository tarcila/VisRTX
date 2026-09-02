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

#include <vector_types.h> // ::float2, ::float4
#include <cstdint>

// Hash-based Owen-scrambled Sobol (Burley 2020 / pbrt-style scramble).
// Pattern: QMC sample *points*, PCG for binary decisions (RR, opacity).
//
// Dimension map — do not reuse 0–3 for lighting:
//   0..1  sub-pixel jitter
//   2..3  lens / DoF
//   4..5  first-bounce cosine hemisphere (env NEE / ambient)
//   6..9  first-bounce HDRI CDF (row, column, jitter u/v)
//   10..11 first-bounce light pick (ambient vs lights, instance)
//   12..15 first-bounce BSDF nextRay (then PCG)
//
// Pixel seed Owen-scrambles each dimension so neighboring pixels do not share
// a global Sobol sequence. sampleIndex is the existing frameID×iterations+i
// ordinal — progressive accumulation is unchanged.

namespace visrtx {

inline constexpr uint32_t kSobolDimPixelX = 0;
inline constexpr uint32_t kSobolDimPixelY = 1;
inline constexpr uint32_t kSobolDimLensU = 2;
inline constexpr uint32_t kSobolDimLensV = 3;
inline constexpr uint32_t kSobolDimHemiU = 4;
inline constexpr uint32_t kSobolDimHemiV = 5;
inline constexpr uint32_t kSobolDimCdfY = 6;
inline constexpr uint32_t kSobolDimCdfX = 7;
inline constexpr uint32_t kSobolDimCdfJitterU = 8;
inline constexpr uint32_t kSobolDimCdfJitterV = 9;
inline constexpr uint32_t kSobolDimLightPick0 = 10;
inline constexpr uint32_t kSobolDimLightPick1 = 11;
inline constexpr uint32_t kSobolDimNextRay0 = 12;
inline constexpr uint32_t kSobolNumDimensions = 16;

// clang-format off
#ifdef __CUDA_ARCH__
__device__ __constant__
#endif
static uint32_t kSobolDirection[16][32] = {
    {
        0x80000000u, 0x40000000u, 0x20000000u, 0x10000000u,
        0x08000000u, 0x04000000u, 0x02000000u, 0x01000000u,
        0x00800000u, 0x00400000u, 0x00200000u, 0x00100000u,
        0x00080000u, 0x00040000u, 0x00020000u, 0x00010000u,
        0x00008000u, 0x00004000u, 0x00002000u, 0x00001000u,
        0x00000800u, 0x00000400u, 0x00000200u, 0x00000100u,
        0x00000080u, 0x00000040u, 0x00000020u, 0x00000010u,
        0x00000008u, 0x00000004u, 0x00000002u, 0x00000001u,
    },
    {
        0x80000000u, 0xc0000000u, 0xa0000000u, 0xf0000000u,
        0x88000000u, 0xcc000000u, 0xaa000000u, 0xff000000u,
        0x80800000u, 0xc0c00000u, 0xa0a00000u, 0xf0f00000u,
        0x88880000u, 0xcccc0000u, 0xaaaa0000u, 0xffff0000u,
        0x80008000u, 0xc000c000u, 0xa000a000u, 0xf000f000u,
        0x88008800u, 0xcc00cc00u, 0xaa00aa00u, 0xff00ff00u,
        0x80808080u, 0xc0c0c0c0u, 0xa0a0a0a0u, 0xf0f0f0f0u,
        0x88888888u, 0xccccccccu, 0xaaaaaaaau, 0xffffffffu,
    },
    {
        0x80000000u, 0xc0000000u, 0x60000000u, 0x90000000u,
        0xe8000000u, 0x5c000000u, 0x8e000000u, 0xc5000000u,
        0x68800000u, 0x9cc00000u, 0xee600000u, 0x55900000u,
        0x80680000u, 0xc09c0000u, 0x60ee0000u, 0x90550000u,
        0xe8808000u, 0x5cc0c000u, 0x8e606000u, 0xc5909000u,
        0x6868e800u, 0x9c9c5c00u, 0xeeee8e00u, 0x5555c500u,
        0x8000e880u, 0xc0005cc0u, 0x60008e60u, 0x9000c590u,
        0xe8006868u, 0x5c009c9cu, 0x8e00eeeeu, 0xc5005555u,
    },
    {
        0x80000000u, 0xc0000000u, 0x20000000u, 0x50000000u,
        0xf8000000u, 0x74000000u, 0xa2000000u, 0x93000000u,
        0xd8800000u, 0x25400000u, 0x59e00000u, 0xe6d00000u,
        0x78080000u, 0xb40c0000u, 0x82020000u, 0xc3050000u,
        0x208f8000u, 0x51474000u, 0xfbea2000u, 0x75d93000u,
        0xa0858800u, 0x914e5400u, 0xdbe79e00u, 0x25db6d00u,
        0x58800080u, 0xe54000c0u, 0x79e00020u, 0xb6d00050u,
        0x800800f8u, 0xc00c0074u, 0x200200a2u, 0x50050093u,
    },
    {
        0x80000000u, 0x40000000u, 0x20000000u, 0xb0000000u,
        0xf8000000u, 0xdc000000u, 0x7a000000u, 0x9d000000u,
        0x5a800000u, 0x2fc00000u, 0xa1600000u, 0xf0b00000u,
        0xda880000u, 0x6fc40000u, 0x81620000u, 0x40bb0000u,
        0x22878000u, 0xb3c9c000u, 0xfb65a000u, 0xddb2d000u,
        0x78022800u, 0x9c0b3c00u, 0x5a0fb600u, 0x2d0ddb00u,
        0xa2878080u, 0xf3c9c040u, 0xdb65a020u, 0x6db2d0b0u,
        0x800228f8u, 0x400b3cdcu, 0x200fb67au, 0xb00ddb9du,
    },
    {
        0x80000000u, 0x40000000u, 0x60000000u, 0x30000000u,
        0xc8000000u, 0x24000000u, 0x56000000u, 0xfb000000u,
        0xe0800000u, 0x70400000u, 0xa8600000u, 0x14300000u,
        0x9ec80000u, 0xdf240000u, 0xb6d60000u, 0x8bbb0000u,
        0x48008000u, 0x64004000u, 0x36006000u, 0xcb003000u,
        0x2880c800u, 0x54402400u, 0xfe605600u, 0xef30fb00u,
        0x7e48e080u, 0xaf647040u, 0x1eb6a860u, 0x9f8b1430u,
        0xd6c81ec8u, 0xbb249f24u, 0x80d6d6d6u, 0x40bbbbbbu,
    },
    {
        0x80000000u, 0xc0000000u, 0xa0000000u, 0xd0000000u,
        0x58000000u, 0x94000000u, 0x3e000000u, 0xe3000000u,
        0xbe800000u, 0x23c00000u, 0x1e200000u, 0xf3100000u,
        0x46780000u, 0x67840000u, 0x78460000u, 0x84670000u,
        0xc6788000u, 0xa784c000u, 0xd846a000u, 0x5467d000u,
        0x9e78d800u, 0x33845400u, 0xe6469e00u, 0xb7673300u,
        0x20f86680u, 0x104477c0u, 0xf8668020u, 0x4477c010u,
        0x668020f8u, 0x77c01044u, 0x8020f866u, 0xc0104477u,
    },
    {
        0x80000000u, 0x40000000u, 0xa0000000u, 0x50000000u,
        0x28000000u, 0x14000000u, 0x82000000u, 0x41000000u,
        0xa8800000u, 0x54400000u, 0x22a00000u, 0x11500000u,
        0x80a80000u, 0x40540000u, 0xa0220000u, 0x50110000u,
        0x28808000u, 0x14404000u, 0x82a0a000u, 0x41505000u,
        0xa8a82800u, 0x54541400u, 0x22228200u, 0x11114100u,
        0x80002880u, 0x40001440u, 0xa00082a0u, 0x50004150u,
        0x2800a8a8u, 0x14005454u, 0x82002222u, 0x41001111u,
    },
    {
        0x80000000u, 0x40000000u, 0xa0000000u, 0x50000000u,
        0x38000000u, 0x8c000000u, 0x4e000000u, 0xaf000000u,
        0x56800000u, 0x33400000u, 0x80200000u, 0x40100000u,
        0xa0980000u, 0x50dc0000u, 0x38760000u, 0x8c230000u,
        0x4e188000u, 0xaf9c4000u, 0x5656a000u, 0x33335000u,
        0x8000b800u, 0x4000cc00u, 0xa000ee00u, 0x5000ff00u,
        0x38006e80u, 0x8c00bf40u, 0x4e00ce20u, 0xaf00ef10u,
        0x5680f618u, 0x3340639cu, 0x8020b856u, 0x4010cc33u,
    },
    {
        0x80000000u, 0x40000000u, 0xa0000000u, 0x50000000u,
        0x68000000u, 0xb4000000u, 0x92000000u, 0x89000000u,
        0x48800000u, 0xa4400000u, 0x5aa00000u, 0x6d500000u,
        0xb2e80000u, 0x99f40000u, 0x80b20000u, 0x40990000u,
        0xa0808000u, 0x50404000u, 0x68a0a000u, 0xb4505000u,
        0x92686800u, 0x89b4b400u, 0x48129200u, 0xa4c98900u,
        0x5ae8c880u, 0x6df4e440u, 0xb2b2faa0u, 0x99993d50u,
        0x80005ae8u, 0x40006df4u, 0xa000b2b2u, 0x50009999u,
    },
    {
        0x80000000u, 0xc0000000u, 0xa0000000u, 0x50000000u,
        0x18000000u, 0x74000000u, 0x8e000000u, 0xc3000000u,
        0xae800000u, 0x53c00000u, 0x16200000u, 0x77900000u,
        0x80380000u, 0xc0e40000u, 0xa0360000u, 0x50e70000u,
        0x18388000u, 0x74e4c000u, 0x8e36a000u, 0xc3e75000u,
        0xaeb89800u, 0x5324b400u, 0x16162e00u, 0x77779300u,
        0x80003680u, 0xc000e7c0u, 0xa0003820u, 0x5000e490u,
        0x180036b8u, 0x7400e724u, 0x8e003816u, 0xc300e477u,
    },
    {
        0x80000000u, 0x40000000u, 0xa0000000u, 0x50000000u,
        0x88000000u, 0xc4000000u, 0xe2000000u, 0xf5000000u,
        0xda800000u, 0x48400000u, 0x20200000u, 0x10100000u,
        0x28280000u, 0x94140000u, 0x6a220000u, 0x31310000u,
        0x38b88000u, 0xbd7d4000u, 0xfa96a000u, 0x58421000u,
        0x08000800u, 0x84000400u, 0x42000a00u, 0xa5000500u,
        0x52800880u, 0x8c400c40u, 0xc2200e20u, 0xe5100f50u,
        0xf2a80da8u, 0xdc540484u, 0x4a020202u, 0x21210101u,
    },
    {
        0x80000000u, 0x40000000u, 0xa0000000u, 0x50000000u,
        0x88000000u, 0xd4000000u, 0xca000000u, 0x71000000u,
        0x98800000u, 0xfd400000u, 0x4a200000u, 0x31100000u,
        0x38a80000u, 0xad540000u, 0xc2020000u, 0xe5250000u,
        0xf29a8000u, 0xdc484000u, 0x5aa42000u, 0x185a5000u,
        0xb8a80800u, 0xed540400u, 0x62020a00u, 0xb5250500u,
        0x7a9a8880u, 0x08484d40u, 0x90a42ca0u, 0x695a5710u,
        0x20280188u, 0x10140bd4u, 0x28220ea2u, 0x84350611u,
    },
    {
        0x80000000u, 0x40000000u, 0xe0000000u, 0xb0000000u,
        0x98000000u, 0x64000000u, 0xf2000000u, 0x7f000000u,
        0xd1800000u, 0x6ec00000u, 0x18200000u, 0x24100000u,
        0x12380000u, 0xcf2c0000u, 0x49a60000u, 0x0ad90000u,
        0xea1c8000u, 0x5b0fc000u, 0xc38c6000u, 0xa1f7b000u,
        0x51800800u, 0x2ec00400u, 0xf8200e00u, 0x94100b00u,
        0x8a380980u, 0xab2c0640u, 0xbba60f20u, 0x75d907f0u,
        0x3b9c8d18u, 0x35cfc6ecu, 0xdbac6182u, 0x85e7b241u,
    },
    {
        0x80000000u, 0x40000000u, 0xe0000000u, 0x30000000u,
        0x08000000u, 0x14000000u, 0x9a000000u, 0xcb000000u,
        0xb7800000u, 0x4d400000u, 0xf2200000u, 0xaf100000u,
        0xc5b80000u, 0xa24c0000u, 0xd7a20000u, 0x3d550000u,
        0x1a3e8000u, 0x8b2ec000u, 0x5797e000u, 0x7d5a5000u,
        0xfa380800u, 0xbb0c0400u, 0x5f820e00u, 0x69450300u,
        0x60068080u, 0x7022c140u, 0xe815e9a0u, 0x241f5cb0u,
        0x923e8378u, 0xdf2ec0d4u, 0x2d97e122u, 0x865a59f1u,
    },
    {
        0x80000000u, 0x40000000u, 0xa0000000u, 0x10000000u,
        0xe8000000u, 0x8c000000u, 0x7e000000u, 0x23000000u,
        0x57800000u, 0x4ac00000u, 0x9fa00000u, 0x96d00000u,
        0xa9880000u, 0x29d40000u, 0x68320000u, 0xcc370000u,
        0xde0d8000u, 0x332fc000u, 0xbfb06000u, 0xc6f97000u,
        0xe18d8800u, 0xb5efc400u, 0xfe106a00u, 0x63297100u,
        0xf7858680u, 0x5afbccc0u, 0x77826de0u, 0x1ace7330u,
        0xd78003f8u, 0x0ac0086cu, 0x3fa0041au, 0x86d00a5du,
    },
};
// clang-format on

VISRTX_HOST_DEVICE uint32_t reverseBits32(uint32_t x)
{
#ifdef __CUDA_ARCH__
  return __brev(x);
#else
  x = (x << 16) | (x >> 16);
  x = ((x & 0x00ff00ffu) << 8) | ((x & 0xff00ff00u) >> 8);
  x = ((x & 0x0f0f0f0fu) << 4) | ((x & 0xf0f0f0f0u) >> 4);
  x = ((x & 0x33333333u) << 2) | ((x & 0xccccccccu) >> 2);
  x = ((x & 0x55555555u) << 1) | ((x & 0xaaaaaaaau) >> 1);
  return x;
#endif
}

// pbrt-v4 OwenScramble: nested uniform scramble of a 32-bit Sobol value.
VISRTX_HOST_DEVICE uint32_t owenScramble(uint32_t v, uint32_t seed)
{
  v = reverseBits32(v);
  v ^= v * 0x3d20adeau;
  v += seed;
  v *= (seed >> 16) | 1u;
  v ^= v * 0x05526c56u;
  v ^= v * 0x53a22864u;
  return reverseBits32(v);
}

VISRTX_HOST_DEVICE uint32_t sobolUInt(uint32_t index, uint32_t dimension)
{
  if (dimension >= kSobolNumDimensions)
    dimension = 0;
  uint32_t x = 0;
  for (uint32_t bit = 0; index; ++bit, index >>= 1u) {
    if (index & 1u)
      x ^= kSobolDirection[dimension][bit];
  }
  return x;
}

VISRTX_HOST_DEVICE uint32_t owenPixelSeed(uint32_t x, uint32_t y)
{
  // Same mixer as the old Halton pixel hash; now seeds Owen, not the index.
  uint32_t v = x + y * 0x9E3779B9u + 0xB5297A4Du;
  v ^= v >> 16;
  v *= 0x7feb352du;
  v ^= v >> 15;
  v *= 0x846ca68bu;
  v ^= v >> 16;
  return v;
}

// Uniform in [0, 1). Mixes pixelSeed with dimension so dims stay independent.
VISRTX_HOST_DEVICE float owenSobol(
    uint32_t sampleIndex, uint32_t dimension, uint32_t pixelSeed)
{
  const uint32_t dimSeed = pixelSeed ^ (0x9e3779b9u * (dimension + 1u));
  const uint32_t scrambled =
      owenScramble(sobolUInt(sampleIndex, dimension), dimSeed);
  // 24-bit mantissa in [0, 1); never 1.0.
  return float(scrambled >> 8) * 0x1p-24f;
}

VISRTX_HOST_DEVICE ::float4 owenSobolCamera(
    uint32_t sampleIndex, uint32_t pixelSeed)
{
  return ::float4{owenSobol(sampleIndex, kSobolDimPixelX, pixelSeed),
      owenSobol(sampleIndex, kSobolDimPixelY, pixelSeed),
      owenSobol(sampleIndex, kSobolDimLensU, pixelSeed),
      owenSobol(sampleIndex, kSobolDimLensV, pixelSeed)};
}

VISRTX_HOST_DEVICE ::float2 owenSobolHemi(
    uint32_t sampleIndex, uint32_t pixelSeed)
{
  return ::float2{owenSobol(sampleIndex, kSobolDimHemiU, pixelSeed),
      owenSobol(sampleIndex, kSobolDimHemiV, pixelSeed)};
}

} // namespace visrtx
