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

// Wavefront renderer PTX (ticket 05c). The trace launch runs 1D over Path Pool
// slots: each thread reads its slot's assigned (pixel, sampleIdx), casts the
// camera ray, traces, and accumulates one direct-visibility sample. The hit
// programs populate a SurfaceHit and never invoke shading callables — the
// pipeline does traversal and cut-plane culling only. Shading is still a
// placeholder matte+ambient term (moves to a dedicated CUDA shade stage with
// the full accumulate/AOV protocol in slice 05d).

#include <limits>
#include "gpu/cameraCreateRay.h"
#include "gpu/evalShading.h"
#include "gpu/gpu_decl.h"
#include "gpu/renderer/common.h"
#include "gpu/renderer/raygen_helpers.h"
#include "gpu/shadingState.h"
#include "gpu/shading_api.h"

namespace visrtx {

DECLARE_FRAME_DATA(frameData)

// Build a ScreenSample for an explicit pixel (the pool decouples a slot from
// its pixel, so we cannot use createScreenSample's optixGetLaunchIndex path).
// Same pixel-stream / frame-seed structure as createScreenSample, but keyed on
// an explicit pixel and the sample's accumulated ordinal rather than the launch
// index and bare frameID.
VISRTX_DEVICE ScreenSample poolScreenSample(
    const FrameGPUData &fd, uint32_t linearPixel, uint32_t accumSampleIdx)
{
  ScreenSample ss;
  const uint32_t w = fd.fb.size.x;
  const int x = int(linearPixel % w);
  const int y = int(linearPixel / w);
  const uint64_t pixelLinear = uint64_t(linearPixel);
  const uint64_t streamId = detail::pcg_mix64(pixelLinear);
  // Key the seed on the sample's accumulated ordinal so every sample of a pixel
  // (across waves and progressive frames) gets a distinct RNG stream — the
  // decoupled budget re-seeds here per sample instead of advancing one stream.
  const uint64_t frameSeed = detail::pcg_mix64(uint64_t(accumSampleIdx)
      ^ (pixelLinear << 1u) ^ 0xD1B54A32D192ED03ULL);
  pcg_init(&ss.rs, frameSeed, streamId);
  ss.pixel.x = x;
  ss.pixel.y = y;
  ss.frameData = &fd;
  ss.shadowContribWeight = 1.0f;
  return ss;
}

// OptiX programs /////////////////////////////////////////////////////////////

VISRTX_GLOBAL void __closesthit__shadow()
{
  // no-op
}

VISRTX_GLOBAL void __anyhit__shadow()
{
  ray::cullCutPlane();
  SurfaceHit hit;
  ray::populateSurfaceHit(hit);

  auto &o = ray::rayData<float>();

  if (hit.material->isFullyOpaque) {
    o = 1.0f;
    optixTerminateRay();
    return;
  }

  const auto &fd = frameData;
  const auto &md = *hit.material;
  MaterialShadingState shadingState;
  materialInitShading(&shadingState, fd, md, hit);

  accumulateValue(o, materialEvaluateOpacity(shadingState), o);
  if (o >= OPACITY_THRESHOLD)
    optixTerminateRay();
  else
    optixIgnoreIntersection();
}

VISRTX_GLOBAL void __anyhit__primary()
{
  ray::cullbackFaces();
  ray::cullCutPlane();
}

VISRTX_GLOBAL void __closesthit__primary()
{
  ray::populateHit();
}

VISRTX_GLOBAL void __miss__()
{
  // no-op
}

// Trace-only raygen: cast the slot's camera ray, intersect, and write the hit
// record for the CUDA shade stage. No shading callables run in this pipeline —
// that is the whole point of the trace/shade split.
VISRTX_GLOBAL void __raygen__()
{
  const uint32_t slotIdx = optixGetLaunchIndex().x;
  const WavefrontPathSlot slot = frameData.wavefrontSlots[slotIdx];
  if (!slot.alive)
    return;

  // The camera QMC (Halton) index must be the sample's ordinal across the whole
  // accumulation, not just within this launchFrame: frameID advances by spp per
  // launch (Frame.cu), so the accumulated index is frameID + the per-pixel
  // ordinal. Feeding only slot.sampleIdx (0..spp-1) would replay the identical
  // sub-pixel/lens samples every progressive frame and never converge AA/DoF.
  const uint32_t cameraSampleIdx = uint32_t(frameData.fb.frameID) + slot.sampleIdx;
  ScreenSample ss = poolScreenSample(frameData, slot.pixel, cameraSampleIdx);
  const bool isVeryFirstRay =
      slot.sampleIdx == 0 && frameData.fb.frameID == 0;
  Ray ray = makePrimaryRay(ss, cameraSampleIdx, isVeryFirstRay);
  applyCuttingPlane(frameData.renderer.cutPlane, ray);

  WavefrontHitRecord &rec = frameData.wavefrontHits[slotIdx];
  rec.hit.foundHit = false;
  intersectSurface(ss,
      ray,
      RayType::PRIMARY,
      &rec.hit,
      primaryRayOptiXFlags(frameData.renderer));
  rec.rayDir = ray.dir;
}

} // namespace visrtx
