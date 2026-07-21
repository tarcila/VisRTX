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

  // Fixed-capacity Path Pool (resolution-independent). Mutable so the const
  // populateFrameData() can lazily allocate before publishing the pointers.
  // m_poolSlots: per-slot (pixel, sampleIdx). m_poolHits: per-slot trace result.
  // m_poolShade: per-slot deferred shading state around the shadow trace.
  // m_stage: 1-element selector switching the shared raygen between the primary
  // and shadow traces (patched per launch on the stream).
  mutable DeviceBuffer m_poolSlots;
  mutable DeviceBuffer m_poolHits;
  mutable DeviceBuffer m_poolShade;
  mutable DeviceBuffer m_stage;
};

} // namespace visrtx
