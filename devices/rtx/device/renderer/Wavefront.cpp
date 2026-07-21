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

#include "Wavefront.h"
#include "WavefrontLaunch.h"
#include "optix_visrtx.h"
// ptx
#include "Wavefront_ptx.h"
// std
#include <algorithm>

namespace visrtx {

static const std::vector<HitgroupFunctionNames> g_wavefrontHitNames = {
    {"__closesthit__primary", "__anyhit__primary"},
    {"__closesthit__shadow", "__anyhit__shadow"}};

static const std::vector<std::string> g_wavefrontMissNames = {
    "__miss__", "__miss__"};

Wavefront::Wavefront(DeviceGlobalState *s) : Renderer(s, 1.f) {}

void Wavefront::commitParameters()
{
  Renderer::commitParameters();
  // Checkerboarding halves the launch dimensions; the pool's slot->pixel
  // mapping assumes a full-frame pixel grid, so disable it for now. The base
  // clamps m_spp to 1 whenever checkerboarding was requested, so restore the
  // requested pixelSamples here — otherwise checkerboarding=true silently
  // renders at 1 spp instead of just being ignored.
  m_checkerboard = false;
  m_spp = std::max(1, getParam<int>("pixelSamples", 1));
}

void Wavefront::ensurePool() const
{
  // reserve() only grows, so this is a no-op after the first frame.
  m_poolSlots.reserve(
      size_t(kWavefrontPoolCapacity) * sizeof(WavefrontPathSlot));
  m_poolHits.reserve(
      size_t(kWavefrontPoolCapacity) * sizeof(WavefrontHitRecord));
}

void Wavefront::populateFrameData(FrameGPUData &fd) const
{
  Renderer::populateFrameData(fd);
  ensurePool();
  fd.wavefrontSlots = m_poolSlots.ptrAs<WavefrontPathSlot>();
  fd.wavefrontHits = m_poolHits.ptrAs<WavefrontHitRecord>();
}

void Wavefront::launchFrame(cudaStream_t stream,
    CUdeviceptr frameData,
    size_t frameDataSize,
    uvec2 launchSize)
{
  ensurePool();

  const uint32_t numPixels = launchSize.x * launchSize.y;
  if (numPixels == 0)
    return;

  const uint32_t samplesPerPixel = uint32_t(std::max(spp(), 1));
  // 64-bit: numPixels * spp overflows uint32 for large frames (8K x high spp),
  // and a uint32 waveBase would wrap past 2^32 mid-loop and re-enter at 0 —
  // an infinite host loop, not merely truncated sampling.
  const uint64_t totalSamples = uint64_t(numPixels) * samplesPerPixel;

  // The atomic shade path lets a wave run several samples of one pixel
  // concurrently, so the pool can use its full capacity. CLAMP/TRIM keep
  // per-pixel running statistics that scatter-add would corrupt, so they stay
  // capped to one slot per pixel per wave (distinct pixels, no concurrency).
  const bool atomicSafe = m_fireflyFilterMode == FireflyFilterMode::NONE
      || m_fireflyFilterMode == FireflyFilterMode::TONEMAP;
  const uint32_t cap = atomicSafe ? kWavefrontPoolCapacity
                                  : std::min(kWavefrontPoolCapacity, numPixels);
  const uint32_t liveSlots =
      uint32_t(std::min<uint64_t>(cap, totalSamples));
  auto *slots = m_poolSlots.ptrAs<WavefrontPathSlot>();
  auto *frameDataPtr = reinterpret_cast<const FrameGPUData *>(frameData);

  // Host-driven cycle loop. Each wave: (1) regenerate assigns up to liveSlots
  // samples to the pool; (2) a trace-only OptiX launch fills each slot's hit
  // record — no shading in the pipeline; (3) a CUDA shade stage reads the hit
  // records and accumulates. Waves repeat until the sample budget is spent.
  for (uint64_t waveBase = 0; waveBase < totalSamples; waveBase += liveSlots) {
    wavefrontRegenerate(
        stream, slots, waveBase, numPixels, totalSamples, liveSlots);
    OPTIX_CHECK(optixLaunch(pipeline(),
        stream,
        frameData,
        frameDataSize,
        sbt(),
        liveSlots,
        1,
        1));
    wavefrontShade(stream, frameDataPtr, liveSlots);
  }
}

OptixModule Wavefront::optixModule() const
{
  return deviceState()->rendererModules.wavefront;
}

Span<HitgroupFunctionNames> Wavefront::hitgroupSbtNames() const
{
  return make_Span(g_wavefrontHitNames.data(), g_wavefrontHitNames.size());
}

Span<std::string> Wavefront::missSbtNames() const
{
  return make_Span(g_wavefrontMissNames.data(), g_wavefrontMissNames.size());
}

ptx_blob Wavefront::ptx()
{
  return {Wavefront_ptx, sizeof(Wavefront_ptx)};
}

} // namespace visrtx
