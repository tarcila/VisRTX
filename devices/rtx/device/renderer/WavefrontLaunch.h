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

#pragma once

#include "gpu/wavefrontPool.h"
// cuda
#include <cuda_runtime.h>
// std
#include <cstdint>

namespace visrtx {

// Fixed Path Pool capacity, in slots. Resolution-independent by design: the
// pool processes a frame in waves of at most this many samples regardless of
// frame size. At 12 bytes/slot this is ~12 MB.
constexpr uint32_t kWavefrontPoolCapacity = 1u << 20;

// Regenerate stage: assign each live slot the next pending (pixel, sampleIdx)
// for this wave. A slot whose global sample id has run past the frame's sample
// budget is marked dead and skipped by the trace launch. Sample id s maps to
// pixel s % numPixels and per-pixel ordinal s / numPixels, so consecutive slots
// in a wave target distinct pixels (as long as liveSlots <= numPixels) — which
// is what keeps this slice's non-atomic accumulation correct.
void wavefrontRegenerate(cudaStream_t stream,
    WavefrontPathSlot *slots,
    uint32_t waveBase,
    uint32_t numPixels,
    uint32_t totalSamples,
    uint32_t liveSlots);

} // namespace visrtx
