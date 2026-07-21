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

namespace visrtx {

namespace {

constexpr int kThreadsPerBlock = 256;

__global__ void wavefrontRegenerateKernel(WavefrontPathSlot *slots,
    uint32_t waveBase,
    uint32_t numPixels,
    uint32_t totalSamples,
    uint32_t liveSlots)
{
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= liveSlots)
    return;

  const uint32_t sampleId = waveBase + i;
  if (sampleId >= totalSamples) {
    slots[i].alive = 0;
    return;
  }

  slots[i].pixel = sampleId % numPixels;
  slots[i].sampleIdx = sampleId / numPixels;
  slots[i].alive = 1;
}

} // namespace

void wavefrontRegenerate(cudaStream_t stream,
    WavefrontPathSlot *slots,
    uint32_t waveBase,
    uint32_t numPixels,
    uint32_t totalSamples,
    uint32_t liveSlots)
{
  if (liveSlots == 0)
    return;
  const uint32_t blocks = (liveSlots + kThreadsPerBlock - 1) / kThreadsPerBlock;
  wavefrontRegenerateKernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      slots, waveBase, numPixels, totalSamples, liveSlots);
}

} // namespace visrtx
