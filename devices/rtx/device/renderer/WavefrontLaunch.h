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

struct FrameGPUData;
struct WavefrontPathState;

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
    uint64_t waveBase,
    uint32_t numPixels,
    uint64_t totalSamples,
    uint32_t liveSlots);

// Shade-emit stage: read each slot's trace hit record, evaluate builtin shading
// statically (no optixDirectCall — the register-isolation point), pick a light
// for NEE, and write the deferred shade record (unshadowed term, shadowable
// direct term, shadow ray). Reads frameData from the device pointer (a CUDA
// kernel cannot see the OptiX __constant__ params).
void wavefrontShadeEmit(
    cudaStream_t stream, const FrameGPUData *frameData, uint32_t liveSlots);

// Resolve stage: deposit this bounce's throughput-weighted radiance (with the
// shadow-ray visibility) and, unless the path terminates (miss / max depth /
// throughput collapse), sample a diffuse continuation ray for the next bounce.
void wavefrontResolve(cudaStream_t stream,
    const FrameGPUData *frameData,
    uint32_t liveSlots,
    uint32_t bounce,
    uint32_t maxDepth);

// Alive-path compaction: gather the slots whose path survived this bounce into
// a dense prefix of the destination buffers, so the next bounce launches over
// only the survivors. Copies the cross-bounce state (slot = pixel/sample, path
// = throughput/rng/continuation ray); the per-bounce hit/shade scratch is not
// moved (it is rewritten next bounce over the dense front). `outCount` (1
// device word, zeroed here) receives the survivor count — the host reads it
// back to size the next bounce's launches.
void wavefrontCompactAlive(cudaStream_t stream,
    const WavefrontPathSlot *srcSlots,
    const WavefrontPathState *srcPaths,
    WavefrontPathSlot *dstSlots,
    WavefrontPathState *dstPaths,
    uint32_t inCount,
    uint32_t *outCount);

// MDL material-sorted compaction: partition the live pool's MDL hits into a
// packed slot-index array grouped by compiled material. `baseIndices[0..M)` is
// each material's callableBaseIndex (the partition key). Single pass: each
// material owns the fixed-stride region `packed[bucket * stride ..]`, and the
// per-material atomic `cursor[bucket]` gives the append position AND is that
// material's final slot count. All outputs live on the device — the
// per-material shade launch reads its count from `cursor[bucket]`, so no host
// readback / stream sync is needed. `cursor` is M words; `stride` >= liveSlots.
void wavefrontMdlCompact(cudaStream_t stream,
    const FrameGPUData *frameData,
    const uint32_t *baseIndices,
    uint32_t numMaterials,
    uint32_t liveSlots,
    uint32_t stride,
    uint32_t *cursor,
    uint32_t *packed);

} // namespace visrtx
