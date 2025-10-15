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

#include <curand.h>
#include <cmath>
#include <glm/common.hpp>
#include <glm/ext/vector_float3.hpp>
#include <glm/ext/vector_float4.hpp>
#include <glm/geometric.hpp>
#include <glm/vector_relational.hpp>
#include <limits>
#include "gpu/evalShading.h"
#include "gpu/gpu_math.h"
#include "gpu/gpu_objects.h"
#include "gpu/intersectRay.h"
#include "gpu/sampleLight.h"
#include "gpu/shadingState.h"
#include "gpu/shading_api.h"

namespace visrtx {

enum class RayType
{
  PRIMARY = 0,
  SHADOW = 1,
  BOUNCE = 2
};

struct RayAttenuation
{
  const Ray *ray{nullptr};
  float attenuation{0.f};
};

DECLARE_FRAME_DATA(frameData)

// Helper functions ///////////////////////////////////////////////////////////

VISRTX_DEVICE float volumeAttenuation(ScreenSample &ss, Ray r)
{
  RayAttenuation ra;
  ra.ray = &r;
  intersectVolume(
      ss, r, RayType::SHADOW, &ra, OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT);
  return ra.attenuation;
}

VISRTX_DEVICE vec4 shadeSurface(
    ScreenSample &ss, const Ray &ray, const SurfaceHit &hit)
{
  const auto &rendererParams = frameData.renderer;
  const auto &directLightParams = rendererParams.params.directLight;

  auto &world = frameData.world;

  // Compute ambient light contribution //
  const float aoFactor = directLightParams.aoSamples > 0
      ? computeAO(ss,
            ray,
            RayType::SHADOW,
            hit,
            rendererParams.occlusionDistance,
            directLightParams.aoSamples)
      : 1.f;

  MaterialShadingState shadingState;
  materialInitShading(&shadingState, frameData, *hit.material, hit);

  vec3 contrib = materialEvaluateEmission(shadingState, -ray.dir);
  float opacity = materialEvaluateOpacity(shadingState);

  // Handle ambient light contribution
  if (rendererParams.ambientIntensity > 0.0f) {
#define USE_SAMPLED_AMBIENT_LIGHT 0
#if USE_SAMPLED_AMBIENT_LIGHT
    const LightSample ls = {
        .radiance =
            rendererParams.ambientColor * rendererParams.ambientIntensity,
        .dir = sampleHemisphere(ss.rs, hit.Ns),
        .dist = 1e20f,
        .pdf = 1.0f,
    };
    contrib = materialShadeSurface(shadingState, hit, ls, -ray.dir);
#else
    contrib += rendererParams.ambientColor * rendererParams.ambientIntensity
        * materialEvaluateTint(shadingState);
#endif
  }

  // Handle all lights contributions
  for (size_t i = 0; i < world.numLightInstances; i++) {
    auto *inst = world.lightInstances + i;
    if (!inst)
      continue;

    for (size_t l = 0; l < inst->numLights; l++) {
      const auto lightSample =
          sampleLight(ss, hit, inst->indices[l], inst->xfm);
      if (lightSample.pdf == 0.0f)
        continue;

      const Ray shadowRay = {
          hit.hitpoint + hit.Ng * hit.epsilon,
          lightSample.dir,
          {hit.epsilon, lightSample.dist},
      };

      const float surface_o =
        1.f - surfaceAttenuation(ss, shadowRay, RayType::SHADOW);
      const float volume_o = 1.f - volumeAttenuation(ss, shadowRay);
      const float attenuation = surface_o * volume_o;

      if (attenuation <= 1.0e-12f)
        continue;

      const vec3 thisLightContrib =
          materialShadeSurface(shadingState, hit, lightSample, -ray.dir);

      if (glm::any(glm::isnan(thisLightContrib)))
        continue;

      contrib += thisLightContrib * attenuation;
    }
  }

  // Take AO in account
  contrib *= aoFactor;

  // Then proceed with the bounce rays
  vec3 nextRayContrib(0.0f);
  vec3 nextRayContribWeight = vec3(1.f);

  Ray bounceRay = ray;
  SurfaceHit bounceHit = hit;

  for (int depth = 0; depth < frameData.renderer.maxRayDepth; ++depth) {
    NextRay nextRay = materialNextRay(shadingState, bounceRay, ss.rs);

    if (glm::all(glm::lessThan(
            glm::abs(vec3(nextRay.direction)), glm::vec3(1.0e-8f))))
      break;

    nextRayContribWeight *= vec3(nextRay.contributionWeight);
    if (glm::all(glm::lessThan(nextRayContribWeight, glm::vec3(1.0e-8f))))
      break;

    bounceRay = {
        bounceHit.hitpoint
            + bounceHit.Ng
                * std::copysignf(
                    bounceHit.epsilon, dot(bounceHit.Ns, nextRay.direction)),
        normalize(vec3(nextRay.direction)),
    };

    bounceHit.foundHit = false;
    bounceHit.isFrontFace = true;
    intersectSurface(ss, bounceRay, RayType::BOUNCE, &bounceHit);

    // We hit something. Gather its contribution.
    if (bounceHit.foundHit) {
      // This HDRI search is not ideal. It does not account for light instance
      // transformations and should be reworked later on.
      auto hdri = (frameData.world.hdri != -1)
          ? &frameData.registry.lights[frameData.world.hdri]
          : nullptr;

      LightSample lightSample;
      // If we have an active HDRI, sample it.
      if (hdri && hdri->hdri.visible) {
        lightSample = detail::sampleHDRILight(
            *hdri, glm::identity<mat4>(), bounceHit, ss.rs);
      } else {
        // Otherwise fallback to some simple background probing.
        lightSample = {
            rendererParams.ambientColor * rendererParams.ambientIntensity,
            bounceHit.Ns,
            std::numeric_limits<float>::max(),
            1.0f / (4.0f * float(M_PI)),
        };
      }
      materialInitShading(
          &shadingState, frameData, *bounceHit.material, bounceHit);
      nextRayContrib = materialShadeSurface(
          shadingState, bounceHit, lightSample, bounceRay.dir);

      if (glm::any(glm::isnan(nextRayContrib))) {
        break;
      }
      contrib += nextRayContrib * nextRayContribWeight;
    } else {
      // This HDRI search is not ideal. It does not account for light instance
      // transformations and should be reworked later on.
      auto hdri = (frameData.world.hdri != -1)
          ? &frameData.registry.lights[frameData.world.hdri]
          : nullptr;
      // No hit, get background contribution.
      vec3 radiance;
      // If we have an active HDRI, sample it.
      if (hdri && hdri->hdri.visible) {
        radiance =
            detail::sampleHDRILight(*hdri, glm::identity<mat4>(), bounceRay.dir)
                .radiance;
      } else {
        radiance =
            rendererParams.ambientColor * rendererParams.ambientIntensity;
      }
      nextRayContrib = radiance;
      contrib += nextRayContrib * nextRayContribWeight;
      break;
    }
  }

  return vec4(contrib, opacity);
}

// OptiX programs /////////////////////////////////////////////////////////////

VISRTX_GLOBAL void __closesthit__shadow()
{
  // no-op
}

VISRTX_GLOBAL void __anyhit__shadow()
{
  auto &rendererParams = frameData.renderer;

  if (ray::isIntersectingSurfaces()) {
    SurfaceHit hit;
    ray::populateSurfaceHit(hit);

    MaterialShadingState shadingState;
    materialInitShading(&shadingState, frameData, *hit.material, hit);
    auto opacity = materialEvaluateOpacity(shadingState);

    auto &o = ray::rayData<float>();

    accumulateValue(o, opacity, o);

    if (o >= 0.99f)
      optixTerminateRay();
    else
      optixIgnoreIntersection();
  } else {
    auto &ra = ray::rayData<RayAttenuation>();
    VolumeHit hit;
    ray::populateVolumeHit(hit);
    rayMarchVolume(ray::screenSample(),
        hit,
        ra.attenuation,
        rendererParams.inverseVolumeSamplingRate);
    if (ra.attenuation < 0.99f)
      optixIgnoreIntersection();
  }
}

VISRTX_GLOBAL void __anyhit__primary()
{
  ray::cullbackFaces();
}

VISRTX_GLOBAL void __closesthit__primary()
{
  ray::populateHit();
}

VISRTX_GLOBAL void __anyhit__bounce()
{
  ray::cullbackFaces();
}

VISRTX_GLOBAL void __closesthit__bounce()
{
  ray::populateHit();
}

VISRTX_GLOBAL void __miss__()
{
  // TODO
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
