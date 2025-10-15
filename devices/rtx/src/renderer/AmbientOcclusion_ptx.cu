/*
 * Copyright (c) 2019-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <limits>
#include "gpu/evalShading.h"
#include "gpu/shadingState.h"
#include "gpu/shading_api.h"

namespace visrtx {

enum class RayType
{
  PRIMARY = 0,
  AO = 1
};

DECLARE_FRAME_DATA(frameData)

// Helper functions ///////////////////////////////////////////////////////////

VISRTX_DEVICE vec4 shadeSurface(
    ScreenSample &ss, const Ray &ray, const SurfaceHit &hit)
{
  const auto &rendererParams = frameData.renderer;
  const auto &aoParams = rendererParams.params.ao;

  // Compute ambient light contribution //
  const float aoFactor = aoParams.aoSamples > 0
      ? computeAO(ss,
            ray,
            RayType::AO,
            hit,
            rendererParams.occlusionDistance,
            aoParams.aoSamples)
      : 1.f;

  MaterialShadingState shadingState;
  materialInitShading(&shadingState, frameData, *hit.material, hit);
  auto materialBaseColor = materialEvaluateTint(shadingState);
  auto materialOpacity = materialEvaluateOpacity(shadingState);

  const auto lighting =
      aoFactor * rendererParams.ambientIntensity * rendererParams.ambientColor;

  return vec4(materialBaseColor * lighting, materialOpacity);
}

// OptiX programs /////////////////////////////////////////////////////////////

VISRTX_GLOBAL void __closesthit__ao()
{
  // no-op
}

VISRTX_GLOBAL void __anyhit__ao()
{
  SurfaceHit hit;
  ray::populateSurfaceHit(hit);

  const auto &fd = frameData;
  const auto &md = *hit.material;
  MaterialShadingState shadingState;
  materialInitShading(&shadingState, fd, md, hit);

  auto &o = ray::rayData<float>();
  accumulateValue(o, materialEvaluateOpacity(shadingState), o);
  if (o >= 0.99f)
    optixTerminateRay();
  else
    optixIgnoreIntersection();
}

VISRTX_GLOBAL void __anyhit__primary()
{
  ray::cullbackFaces();
}

VISRTX_GLOBAL void __closesthit__primary()
{
  ray::populateHit();
}

VISRTX_GLOBAL void __miss__()
{
  // no-op
}

VISRTX_GLOBAL void __raygen__()
{
  auto &rendererParams = frameData.renderer;

  auto ss = createScreenSample(frameData);
  if (pixelOutOfFrame(ss.pixel, frameData.fb))
    return;

  for (int i = 0; i < frameData.renderer.numIterations; i++) {
    auto ray = makePrimaryRay(ss);

    vec3 outputColor(0.f);
    float outputOpacity = 0.f;
    vec3 outputNormal = -ray.dir;
    float depth = 1e30f;
    uint32_t primID = ~0u;
    uint32_t objID = ~0u;
    uint32_t instID = ~0u;
    bool firstHit = true;

    SurfaceHit surfaceHit;

    do {
      // Our final sample color and opacity
      vec3 sampleColor{};
      float sampleOpacity{};

      // Volume sampling information when applicable
      uint32_t volumeObjID;
      uint32_t volumeInstID;
      vec3 volumeSampleColor;
      float volumeSampleOpacity;

      // First try and get a surface hit
      surfaceHit.foundHit = false;
      intersectSurface(ss,
          ray,
          RayType::PRIMARY,
          &surfaceHit,
          primaryRayOptiXFlags(rendererParams));

      if (surfaceHit.foundHit) {
        // We have a hit, check whether we did traverse a volume before hitting
        // the surface

        const float volumeDepth = rayMarchAllVolumes(ss,
            ray,
            RayType::PRIMARY,
            surfaceHit.t,
            rendererParams.inverseVolumeSamplingRate,
            volumeSampleColor,
            volumeSampleOpacity,
            volumeObjID,
            volumeInstID);

        const bool hasVolume = volumeDepth < surfaceHit.t;
        if (firstHit) {
          if (hasVolume) {
            outputNormal = -ray.dir;
            depth = volumeDepth;
            primID = 0;
            objID = volumeObjID;
            instID = volumeInstID;

            firstHit = false;
          } else {
            outputNormal = surfaceHit.Ns;
            depth = surfaceHit.t;
            primID = computeGeometryPrimId(surfaceHit);
            objID = surfaceHit.objID;
            instID = surfaceHit.instID;

            firstHit = false;
          }
        }

        if (hasVolume) {
          accumulateValue(sampleColor,
              volumeSampleColor * volumeSampleOpacity,
              sampleOpacity);
          accumulateValue(sampleOpacity, volumeSampleOpacity, sampleOpacity);
        }

        // Finally shade the actual surface we hit
        const auto surfaceSample = shadeSurface(ss, ray, surfaceHit);
        accumulateValue(
            sampleColor, vec3(surfaceSample) * surfaceSample.a, sampleOpacity);
        accumulateValue(sampleOpacity, surfaceSample.a, sampleOpacity);

        accumulateValue(
            outputColor, sampleColor * sampleOpacity, outputOpacity);
        accumulateValue(outputOpacity, sampleOpacity, outputOpacity);

        // Prepare for next ray
        ray.t.lower = surfaceHit.t + surfaceHit.epsilon;
      } else {
        // No hit, let's try and see if we do traverse any volumes
        const float volumeDepth = rayMarchAllVolumes(ss,
            ray,
            RayType::PRIMARY,
            std::numeric_limits<float>::max(),
            rendererParams.inverseVolumeSamplingRate,
            volumeSampleColor,
            volumeSampleOpacity,
            volumeObjID,
            volumeInstID);

        if (volumeDepth != std::numeric_limits<float>::max()) {
          if (firstHit) {
            depth = volumeDepth;
            primID = 0;
            objID = volumeObjID;
            instID = volumeInstID;
          }

          accumulateValue(sampleColor,
              volumeSampleColor * volumeSampleOpacity,
              sampleOpacity);
          accumulateValue(sampleOpacity, volumeSampleOpacity, sampleOpacity);
        }

        const auto background = getBackground(frameData, ss.screen, ray.dir);
        accumulateValue(
            sampleColor, vec3(background) * background.a, sampleOpacity);
        accumulateValue(sampleOpacity, background.a, sampleOpacity);

        accumulateValue(
            outputColor, sampleColor * sampleOpacity, outputOpacity);
        accumulateValue(outputOpacity, sampleOpacity, outputOpacity);

        break;
      }
    } while (outputOpacity < 0.99f);

    accumResults(frameData.fb,
        ss.pixel,
        vec4(outputColor, outputOpacity),
        depth,
        outputColor,
        outputNormal,
        primID,
        objID,
        instID,
        i);
  }
}

} // namespace visrtx
