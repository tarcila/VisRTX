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

#include <optix_device.h>
#define VISRTX_DEBUGGING 1
#include "gpu/gpu_debug.h"

#include "gpu/createScreenSample.h"
#include "gpu/evalShading.h"
#include "gpu/gpu_debug.h"
#include "gpu/gpu_decl.h"
#include "gpu/gpu_math.h"
#include "gpu/gpu_util.h"
#include "gpu/intersectRay.h"
#include "gpu/populateHit.h"
#include "gpu/sampleLight.h"
#include "gpu/shadingState.h"
#include "gpu/volumeIntegration.h"

#include <limits>

namespace visrtx {

enum class RayType
{
  SHADING,
  SHADOW,
};

DECLARE_FRAME_DATA(frameData)

struct VolumeSample
{
  bool foundHit;
  vec3 color;
  float opacity;
  float depth;
  float transmittance;
  uint32_t objID;
  uint32_t instID;
};

struct SampleDetails
{
  vec3 color;
  float opacity;
  float depth;
  vec3 albedo;
  vec3 normal;
};

// Helper functions

VISRTX_DEVICE void setPixelIds(const FramebufferGPUData &fb,
    const uvec2 &pixel,
    uint32_t primID,
    uint32_t objID,
    uint32_t instID)
{
  const uint32_t idx = detail::pixelIndex(fb, pixel);

  if (fb.buffers.primID)
    fb.buffers.primID[idx] = primID;
  if (fb.buffers.objID)
    fb.buffers.objID[idx] = objID;
  if (fb.buffers.instID)
    fb.buffers.instID[idx] = instID;
}

VISRTX_DEVICE void accumPixelSample(const FrameGPUData &frame,
    const uvec2 &pixel,
    const SampleDetails &sample,
    const int frameIDOffset = 0)
{
  const auto &fb = frame.fb;
  const uint32_t idx = detail::pixelIndex(fb, pixel);

  const auto frameID = fb.frameID + frameIDOffset;

  // Conditionally apply tonemapping during accumulation
  if (frame.renderer.tonemap)
    detail::accumValue(fb.buffers.colorAccumulation,
        idx,
        vec4(detail::tonemap(sample.color), sample.opacity));
  else
    detail::accumValue(
        fb.buffers.colorAccumulation, idx, vec4(sample.color, sample.opacity));
  detail::accumValue(fb.buffers.albedo, idx, sample.albedo);
  detail::accumValue(fb.buffers.normal, idx, sample.normal);

  const auto accumColor = fb.buffers.colorAccumulation[idx];
  // Conditionally apply inverse tonemapping on output
  const float frameDivisor = float(fb.frameID + frameIDOffset + 1);
  const auto normalizedColor = accumColor / frameDivisor;
  const auto outputColor = frame.renderer.tonemap
      ? detail::inverseTonemap(normalizedColor)
      : normalizedColor;
  detail::writeOutputColor(fb, outputColor, idx);

  if (fb.checkerboardID == 0 && frameID == 0) {
    auto adjPix = uvec2(pixel.x + 1, pixel.y + 0);
    if (!pixelOutOfFrame(adjPix, fb)) {
      detail::writeOutputColor(fb, accumColor, detail::pixelIndex(fb, adjPix));
    }

    adjPix = uvec2(pixel.x + 0, pixel.y + 1);
    if (!pixelOutOfFrame(adjPix, fb)) {
      detail::writeOutputColor(fb, accumColor, detail::pixelIndex(fb, adjPix));
    }

    adjPix = uvec2(pixel.x + 1, pixel.y + 1);
    if (!pixelOutOfFrame(adjPix, fb)) {
      detail::writeOutputColor(fb, accumColor, detail::pixelIndex(fb, adjPix));
    }
  }
}

// OptiX programs /////////////////////////////////////////////////////////////

VISRTX_GLOBAL void __closesthit__shading()
{
  ray::populateHit();
}

VISRTX_GLOBAL void __anyhit__shading()
{
  // no-op
}

VISRTX_GLOBAL void __miss__shading()
{
  if (ray::isIntersectingSurfaces()) {
    auto &hit = ray::rayData<SurfaceHit>();
    hit.foundHit = false;
  } else {
    auto &hit = ray::rayData<VolumeHit>();
    hit.foundHit = false;
  }
}

VISRTX_GLOBAL void __closesthit__shadow()
{
  // no-op
}

VISRTX_GLOBAL void __anyhit__shadow()
{
  if (ray::isIntersectingSurfaces()) {
    SurfaceHit hit;
    ray::populateSurfaceHit(hit);

    MaterialShadingState shadingState;
    materialInitShading(&shadingState, frameData, *hit.material, hit);
    auto tint = materialEvaluateTint(shadingState);
    auto opacity = materialEvaluateOpacity(shadingState);

    auto &o = ray::rayData<vec3>();

    // accumulateValue(o, tint * opacity, o);
    o *= tint * opacity;

    if (glm::all(
            glm::lessThanEqual(o, vec3(std::numeric_limits<float>::epsilon()))))
      optixTerminateRay();
    else
      optixIgnoreIntersection();
  } else {
    auto &o = ray::rayData<vec3>();
    VolumeHit hit;
    ray::populateVolumeHit(hit);

    // For Woodcock tracking in shadow rays:
    // Sample one scatter event per volume intersection
    // transmittance = 0.0 means ray scattered (blocked)
    // transmittance = 1.0 means ray passed through (no interaction)

    vec3 albedo = vec3(0.0f);
    float extinction = 0.0f;
    float transmittance = 0.0f;

    sampleDistanceVolume(ray::screenSample(),
        hit,
        albedo,
        extinction,
        transmittance);

    // If transmittance is 0 (scattered), the ray is blocked
    // If transmittance is 1 (no interaction), ray passes through
    // Get actual attenuation color from albedo and extinction
    if (transmittance <= 0.5f) {
      o *= albedo * (1.0f - extinction);
      optixTerminateRay();
    } else {
      optixIgnoreIntersection();
    }
  }
}

VISRTX_GLOBAL void __miss__shadow()
{
}

VISRTX_DEVICE vec3 surfaceAttenuation(ScreenSample &ss, Ray r)
{
  vec3 attenuation = vec3(1.0f);
  intersectSurface(
      ss, r, RayType::SHADOW, &attenuation, OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT);
  return attenuation;
}

VISRTX_DEVICE vec3 volumeAttenuation(ScreenSample &ss, Ray r)
{
  vec3 attenuation = vec3(1.0f);
  intersectVolume(
      ss, r, RayType::SHADOW, &attenuation, OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT);
  return attenuation;
}

VISRTX_DEVICE LightSample sampleLights(ScreenSample &ss,
    const FrameGPUData &frameData,
    const vec3 &position,
    const SurfaceHit &hit)
{
  const auto &world = frameData.world;

  // Sampling is not uniform, depends on how the light are gathered under the
  // instancers. To be fixed somehow.
  const auto instanceIndex = curand(&ss.rs) % world.numLightInstances;
  const auto inst = world.lightInstances + instanceIndex;
  if (!inst)
    return {};

  const auto lightIndex = curand(&ss.rs) % inst->numLights;
  return sampleLight(ss, hit, inst->indices[lightIndex], inst->xfm);
}

VISRTX_DEVICE
void sampleVolumes(ScreenSample &ss,
    const Ray &ray,
    RayType rayType,
    VolumeSample *volumeSample)
{
  float transmittance = 0.0f;
  volumeSample->depth = sampleDistanceAllVolumes(ss,
      ray,
      RayType::SHADING,
      ray.t.upper,
      volumeSample->color,
      volumeSample->opacity,
      transmittance,
      volumeSample->objID,
      volumeSample->instID);
  // transmittance: 0.0 = scattered, 1.0 = no interaction
  volumeSample->foundHit =
      transmittance < 0.5f && volumeSample->depth < ray.t.upper;
  volumeSample->transmittance = transmittance;
}

VISRTX_GLOBAL void __raygen__()
{
  if (frameData.world.numLightInstances == 0)
    return;

  auto ss = createScreenSample(frameData);
  if (pixelOutOfFrame(ss.pixel, frameData.fb))
    return;

  const auto &rendererParams = frameData.renderer;

  SurfaceHit surfaceHit = {};
  VolumeSample volumeSample = {};

  for (int i = 0; i < rendererParams.numIterations; ++i) {
    SampleDetails sample;
    vec3 nextContribution;

    // First go with the main ray, pixel centered if first frame.
    // Jittered samples are produced by next iterations.
    auto ray = makePrimaryRay(ss, i == 0 && ss.frameData->fb.frameID == 0);
    intersectSurface(ss,
        ray,
        RayType::SHADING,
        &surfaceHit,
        primaryRayOptiXFlags(rendererParams));

    auto volumeRay = Ray{
        ray.org,
        ray.dir,
        {ray.t.lower, surfaceHit.foundHit ? surfaceHit.t : ray.t.upper},
    };
    volumeSample = {};
    sampleVolumes(ss, volumeRay, RayType::SHADING, &volumeSample);

    if (!volumeSample.foundHit && !surfaceHit.foundHit) {
      // No hit, retrun the background contribution.
      const auto color = getBackground(frameData, ss.screen, ray.dir);
      sample.color = vec3(color) * color.a;
      sample.opacity = color.a;
      sample.depth = std::numeric_limits<float>::max();
      sample.albedo = vec3(color) * color.a;
      sample.normal = -ray.dir;

      accumPixelSample(frameData, ss.pixel, sample, i);
      continue;
    }

    if (volumeSample.foundHit) {
      // Volume scattering event occurred
      setPixelIds(
          frameData.fb, ss.pixel, 0, volumeSample.objID, volumeSample.instID);

      // volumeSample.color contains albedo (scattering color)
      // volumeSample.opacity contains the actual opacity value from transfer
      // function

      sample.color = volumeSample.color;
      sample.opacity = 1.0f; // Scattering event is opaque
      sample.depth = volumeSample.depth;
      sample.albedo = volumeSample.color;
      sample.normal = -ray.dir;

      nextContribution = volumeSample.color;
          //* volumeSample.opacity; // Weight by albedo for next bounce
    }

    if (surfaceHit.foundHit) {
      LightSample lightSample =
          sampleLights(ss, frameData, surfaceHit.hitpoint, surfaceHit);

      //const Ray shadowRay = {
      //    surfaceHit.hitpoint + surfaceHit.Ng * surfaceHit.epsilon,
      //    lightSample.dir,
      //    {surfaceHit.epsilon, lightSample.dist },
      //};

      const Ray shadowRay = {
    surfaceHit.hitpoint,
    lightSample.dir,
    {surfaceHit.epsilon, lightSample.dist },
      };

      // Shadows
      const auto surface_o = surfaceAttenuation(ss, shadowRay);
      const auto volume_o = volumeAttenuation(ss, shadowRay);
      const auto attenuation = surface_o * volume_o;

      // Shading
      MaterialShadingState shadingState;
      materialInitShading(
          &shadingState, frameData, *surfaceHit.material, surfaceHit);

      const vec3 materialEmission =
          materialEvaluateEmission(shadingState, -ray.dir);
      const vec3 materialTint = materialEvaluateTint(shadingState);
      const float materialOpacity = materialEvaluateOpacity(shadingState);

      const vec3 materialShadedTint =
          materialShadeSurface(shadingState, surfaceHit, lightSample, -ray.dir);

      const vec3 materialColor =
          materialEmission + (materialShadedTint * attenuation);

      if (volumeSample.foundHit) {
        // Volume was hit - add surface contribution to existing volume
        // contribution
        accumulateValue(sample.color, materialColor, sample.opacity);
        accumulateValue(sample.albedo, materialTint, sample.opacity);
        accumulateValue(sample.opacity, materialOpacity, sample.opacity);
      } else {
        // No volume - just surface
        setPixelIds(frameData.fb,
            ss.pixel,
            surfaceHit.primID,
            surfaceHit.objID,
            surfaceHit.instID);

        sample.color = materialColor;
        sample.opacity = materialOpacity;
        sample.depth = surfaceHit.t;
        sample.albedo = materialTint;
        sample.normal = surfaceHit.Ns;
      }

      if (rendererParams.maxRayDepth > 1) {
        auto nextRay = materialNextRay(shadingState, ray, ss.rs);
        if (glm::any(glm::isnan(nextRay.contributionWeight))
            || glm::all(glm::lessThan(
                nextRay.contributionWeight, glm::vec3(1.0e-8f)))) {
          // Nothing more to accumulate, store and proceed to next iteration
          accumPixelSample(frameData, ss.pixel, sample, i);
          continue;
        }

        nextContribution = nextRay.contributionWeight;

        ray = Ray{
            surfaceHit.hitpoint
                + surfaceHit.Ng
                    * std::copysignf(surfaceHit.epsilon,
                        dot(surfaceHit.Ns, nextRay.direction)),
            normalize(vec3(nextRay.direction)),
        };
      } else {
        accumPixelSample(frameData, ss.pixel, sample, i);
        continue;
      }
    }

#if 1
    // Go on with additional bounces
    for (int d = 1; d < rendererParams.maxRayDepth; ++d) {
      intersectSurface(ss,
          ray,
          RayType::SHADING,
          &surfaceHit,
          primaryRayOptiXFlags(rendererParams));

      auto volumeRay = Ray{
          ray.org,
          ray.dir,
          {ray.t.lower, surfaceHit.t},
      };
      sampleVolumes(ss, volumeRay, RayType::SHADING, &volumeSample);

      if (!volumeSample.foundHit && !surfaceHit.foundHit) {
        const auto background = getBackground(frameData, ss.screen, ray.dir);
        const auto color = vec3(background) * background.a;
        const auto opacity = background.a;

        // For reflection, we can add background directly
        sample.color += color * nextContribution;
        sample.albedo += color * nextContribution;
        sample.opacity += opacity * glm::luminosity(nextContribution);
        break;
      }

      if (volumeSample.foundHit) {
        // Volume scattering: weight albedo by path contribution
        const auto color = volumeSample.color; // Albedo color
        float opacity = volumeSample.opacity; // Opacity from transfer function

        sample.color += color * nextContribution;
        sample.albedo += color * nextContribution;
        sample.opacity += opacity * glm::luminosity(nextContribution);

        // For volume scattering, we need to continue the path from the scatter
        // point Sample new direction from phase function
        vec3 newDir = randomDir(ss.rs);
        vec3 scatterPoint = ray.org + ray.dir * volumeSample.depth;

        // ray = Ray{scatterPoint, normalize(newDir), {1e-6f, 1e30f}};
        nextContribution *=
            volumeSample.color; // Weight by albedo for next bounce
      }

      if (surfaceHit.foundHit) {
        LightSample lightSample =
            sampleLights(ss, frameData, surfaceHit.hitpoint, surfaceHit);

        const Ray shadowRay = {
            surfaceHit.hitpoint,
            lightSample.dir,
            {surfaceHit.epsilon, lightSample.dist},
        };

        // Shadows
        const auto surface_o = surfaceAttenuation(ss, shadowRay);
        const auto volume_o = volumeAttenuation(ss, shadowRay);
        const auto attenuation = surface_o * volume_o;

        // Shading
        MaterialShadingState shadingState;
        materialInitShading(
            &shadingState, frameData, *surfaceHit.material, surfaceHit);

        const vec3 materialEmission =
            materialEvaluateEmission(shadingState, -ray.dir);
        const vec3 materialTint = materialEvaluateTint(shadingState);
        const float materialOpacity = materialEvaluateOpacity(shadingState);

        const vec3 materialShadedTint = materialShadeSurface(
            shadingState, surfaceHit, lightSample, -ray.dir);

        const vec3 materialColor = materialEmission
            + (materialShadedTint * materialOpacity) * attenuation;

        // Direct addition, already weighted by nextContribution
        sample.color += materialColor * nextContribution;
        sample.albedo += materialTint * nextContribution;
        sample.opacity += materialOpacity * glm::luminosity(nextContribution);

        NextRay nextRay = materialNextRay(shadingState, ray, ss.rs);
        if (glm::all(glm::lessThan(
                glm::abs(vec3(nextRay.direction)), glm::vec3(1.0e-8f)))) {
          break;
        }

        nextContribution *= nextRay.contributionWeight;
        if (glm::all(glm::lessThan(nextContribution, glm::vec3(1.0e-8f)))) {
          break;
        }

        ray = Ray{surfaceHit.hitpoint
                + surfaceHit.Ng
                    * std::copysignf(surfaceHit.epsilon,
                        dot(surfaceHit.Ns, nextRay.direction)),
            normalize(vec3(nextRay.direction)),
            {1e-6f, 1e30f}};
      }
    }
#endif
    accumPixelSample(frameData, ss.pixel, sample, i);
  }
}

} // namespace visrtx
