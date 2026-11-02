// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "gpu/lightProxyRadiance.h"
#include "gpu/renderer/fog.h"
#include "gpu/renderer/volumePreview.h"
#include "gpu/renderer/volumePreviewFloat.h"
#include "gpu/sampleOrdinal.h"
#include "gpu/volumeIntegration.h"

namespace visrtx {

template <typename ShadingPolicy>
VISRTX_DEVICE void renderInteractivePixel(FrameGPUData &fd, ScreenSample ss)
{
  namespace vp = practical_preview;
  for (int iteration = 0; iteration < fd.renderer.numIterations; ++iteration) {
    const bool first = iteration == 0 && fd.fb.frameID == 0;
    const uint32_t ordinal = sampleOrdinal(fd.fb.frameID, iteration);
    Ray ray = makePrimaryRay(ss, ordinal, first);
    const vec3 cameraOrigin = ray.org;
    const vec3 cameraFog = fogColor(fd, ss.pixel, ray.dir);
    applyCuttingPlane(fd.renderer.cutPlane, ray);
    const float end = ray.t.upper;
    const auto opticalRay = aggregate_volume::worldDistanceRay(ray);
    const double speed = sqrt(double(ray.dir.x) * ray.dir.x
        + double(ray.dir.y) * ray.dir.y + double(ray.dir.z) * ray.dir.z);
    double opticalCursor = opticalRay.lower;
    vp::CameraIntegral integral;
    float normal[3]{};
    vec3 guideAlbedo(0), guideNormal(0);
    float depth = std::numeric_limits<float>::max();
    uint32_t primitive = ~0u, object = ~0u, instance = ~0u;
    ScreenSample medium = ss;
    const uint64_t pixel =
        uint64_t(ss.pixel.x) + uint64_t(ss.pixel.y) * fd.fb.size.x;
    pcg_init(&medium.rs,
        detail::pcg_mix64(
            pixel ^ (uint64_t(ordinal) << 32) ^ 0x3273d4c981afULL),
        detail::pcg_mix64(pixel ^ 0xa97f524bULL));
    while (true) {
      SurfaceHit hit{};
      if (ray.t.lower < end)
        intersectSurface(ss,
            ray,
            RayType::PRIMARY,
            &hit,
            primaryRayOptiXFlags(fd.renderer),
            ShadingPolicy::primaryVisibilityMask());
      const float far = hit.foundHit ? hit.t : end;
      const double opticalEnd = double(far) * speed;
      if (opticalCursor < opticalEnd)
        vp::cameraSegment(
            medium, opticalRay, opticalCursor, opticalEnd, integral);
      opticalCursor = opticalEnd;

      // Preserve the legacy approximate TF picking path, isolated from optical
      // color/AOV sampling. Principled picking remains deferred.
      bool pickedVolume = false;
      if (first && fd.fb.buffers.depth) {
        ScreenSample picking = ss;
        vec3 unusedColor(0);
        float unusedOpacity = 0;
        uint32_t v = ~0u, i = ~0u;
        const float picked = rayMarchAllVolumes(picking,
            ray,
            RayType::PRIMARY,
            far,
            fd.renderer.inverseVolumeSamplingRate,
            unusedColor,
            unusedOpacity,
            v,
            i);
        if (picked < far) {
          depth = picked;
          primitive = object = v;
          instance = i;
          pickedVolume = true;
        }
      }
      if (!hit.foundHit)
        break;
      const auto appearance = fogAppearance(fd, hit.hitpoint, cameraOrigin);
      const float coverage = vp::luminance(integral.uncovered);
      if (hit.isLightProxy()) {
        const vec3 radiance = lightProxyRadiance(fd, hit, ray.org);
        for (int c = 0; c < 3; ++c) {
          integral.color[c] +=
              (integral.fogOwned[c]
                  + integral.fogFree[c] * appearance.visibility)
                  * float(radiance[c])
              + integral.fogFree[c] * appearance.fogFraction
                  * float(cameraFog[c]);
          integral.fogFree[c] = integral.fogOwned[c] = integral.uncovered[c] =
              0.f;
        }
        guideNormal += float(coverage) * (-ray.dir);
        if (!pickedVolume)
          depth = hit.t;
        if (first)
          setPixelIds(fd.fb, ss.pixel, depth, primitive, object, instance);
        break;
      }
      MaterialShadingState shading;
      materialInitShading(&shading, fd, *hit.material, hit);
      const float alpha = materialEvaluateOpacity(shading);
      const vec3 tau = materialEvaluateTransmission(shading);
      const vec3 radiance = ShadingPolicy::shadeSurface(shading, ss, ray, hit);
      const vec3 tint = materialEvaluateTint(shading);
      vec3 n = materialEvaluateNormal(shading);
      const float length = glm::length(n);
      n = length > 0.f ? n / length : vec3(0);
      for (int c = 0; c < 3; ++c) {
        const float free = integral.fogFree[c], owned = integral.fogOwned[c];
        integral.color[c] += alpha
            * ((owned + free * appearance.visibility) * float(radiance[c])
                + free * appearance.fogFraction * float(cameraFog[c]));
        integral.albedo[c] += alpha * coverage * float(tint[c]);
        normal[c] += alpha * coverage * float(n[c]);
        const float transmission = (1.f - alpha) + alpha * float(tau[c]);
        integral.fogOwned[c] = owned * transmission
            + free * alpha * float(tau[c]) * appearance.visibility;
        integral.fogFree[c] = free * (1.f - alpha);
        integral.uncovered[c] *= 1.f - alpha;
      }
      guideAlbedo += float(alpha * coverage) * tint;
      guideNormal += float(coverage)
          * n; // retained surface-only heuristic, not public normal
      if (!pickedVolume) {
        depth = hit.t;
        primitive = hit.primID;
        object = hit.objID;
        instance = hit.instID;
      }
      // Preserve legacy per-hit depth arbitration (nearest when depth is
      // requested), rather than publishing only the final transparent layer.
      if (first)
        setPixelIds(fd.fb, ss.pixel, depth, primitive, object, instance);
      if (alpha == 1.f && tau == vec3(0))
        break; // physically opaque surface, not an opacity threshold
      ray.t.lower = hit.t + hit.epsilon;
    }
    if (first)
      setPixelIds(fd.fb, ss.pixel, depth, primitive, object, instance);
    vec3 backdrop;
    for (int c = 0; c < 3; ++c)
      backdrop[c] = float(integral.fogFree[c] + integral.fogOwned[c]);
    vec3 hdri;
    if (getBackgroundLight(fd, ray.dir, hdri)) {
      for (int c = 0; c < 3; ++c) {
        integral.color[c] +=
            (integral.fogFree[c] + integral.fogOwned[c]) * float(hdri[c]);
        integral.uncovered[c] = 0.f;
      }
      backdrop = vec3(0);
    }
    detail::accumValue(fd.fb.buffers.backdropWeight,
        detail::pixelIndex(fd.fb, ss.pixel),
        backdrop);
    const vec3 color(float(integral.color[0]),
        float(integral.color[1]),
        float(integral.color[2]));
    const vec3 albedo(float(integral.albedo[0]),
        float(integral.albedo[1]),
        float(integral.albedo[2]));
    const vec3 publicNormal{
        float(normal[0]), float(normal[1]), float(normal[2])};
    accumPixelSample(fd,
        ss.pixel,
        vec4(color, float(1.f - vp::luminance(integral.uncovered))),
        albedo,
        publicNormal,
        &guideAlbedo,
        &guideNormal);
  }
}
} // namespace visrtx
