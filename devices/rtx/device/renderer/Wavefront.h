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

#include "Renderer.h"
#include "utility/DeviceBuffer.h"
#ifdef USE_MDL
#include "WavefrontMdlKernelCache.h"
// std
#include <utility>
#include <vector>
#endif

namespace visrtx {

// Wavefront renderer subtype (ticket 05). Execution model: a host-driven cycle
// of launches over a fixed-size Path Pool. This slice (05c) drives the pool: a
// regenerate stage assigns pool slots the frame's samples wave by wave, and a
// trace launch shades each slot's direct-visibility sample. Trace and shade are
// still one OptiX launch with a placeholder shade; slice 05d splits shading
// into a dedicated CUDA stage with the full atomic accumulate / AOV protocol.
struct Wavefront : public Renderer
{
  Wavefront(DeviceGlobalState *s);
  void commitParameters() override;
  void populateFrameData(FrameGPUData &fd) const override;
  void launchFrame(cudaStream_t stream,
      CUdeviceptr frameData,
      size_t frameDataSize,
      uvec2 launchSize) override;
  OptixModule optixModule() const override;
  Span<HitgroupFunctionNames> hitgroupSbtNames() const override;
  Span<std::string> missSbtNames() const override;

  static ptx_blob ptx();

 private:
  void ensurePool() const;

#ifdef USE_MDL
  // Build/refresh the per-compiled-material MDL shade kernels from the material
  // registry's PTX blobs, keyed off its update timestamp (same signal the OptiX
  // pipeline rebuild uses). Full invalidate-and-rebuild on any change, so a
  // freed registry slot reused by a new material never resolves a stale kernel.
  void refreshMdlKernels(cudaStream_t stream) const;

  mutable WavefrontMdlKernelCache m_mdlKernels;
  mutable helium::TimeStamp m_lastMdlKernelUpdate{};
  mutable bool m_mdlKernelsBuilt{false};
  // (callableBaseIndex, kernel) for each built MDL material; the shade dispatch
  // launches one per entry over that material's compacted slot list.
  mutable std::vector<std::pair<uint32_t, WavefrontMdlKernel>> m_mdlShaders;

  // Single-pass material-sorted compaction buffers. m_mdlPacked: one pool-
  // capacity stride per built material, holding that material's packed slot
  // indices. m_mdlBaseIndices: per-material partition key (callableBaseIndex),
  // uploaded on refresh. m_mdlCounts: per-material atomic cursor == final slot
  // count.
  mutable DeviceBuffer m_mdlPacked;
  mutable DeviceBuffer m_mdlBaseIndices;
  mutable DeviceBuffer m_mdlCounts;
#endif

  int m_maxDepth{4}; // path-tracing bounce depth

  // Fixed-capacity Path Pool (resolution-independent). Mutable so the const
  // populateFrameData() can lazily allocate before publishing the pointers.
  // m_poolSlots: per-slot (pixel, sampleIdx). m_poolHits: per-slot trace
  // result. m_poolShade: per-slot deferred shading state around the shadow
  // trace. m_poolPaths: per-slot path state (throughput, continuation ray)
  // across bounces. m_launch: 1-element (stage, bounce) selector for the shared
  // raygen, patched per launch on the stream.
  mutable DeviceBuffer m_poolSlots;
  mutable DeviceBuffer m_poolHits;
  mutable DeviceBuffer m_poolShade;
  mutable DeviceBuffer m_poolPaths;
  mutable DeviceBuffer m_launch;

  // Alive-path compaction: after each bounce the surviving slots/paths are
  // gathered into these alternate buffers (ping-pong with m_poolSlots/m_pool
  // Paths), so later bounces launch over only the survivors. m_aliveCount is
  // the 1-word device survivor counter, read back each bounce to size launches.
  mutable DeviceBuffer m_poolSlotsAlt;
  mutable DeviceBuffer m_poolPathsAlt;
  mutable DeviceBuffer m_aliveCount;
};

} // namespace visrtx
