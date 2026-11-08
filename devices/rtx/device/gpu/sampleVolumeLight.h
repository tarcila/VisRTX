// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "gpu/sampleLight.h"

namespace visrtx {
// Full-sphere NEE shared by medium transport algorithms. Unlike surface NEE,
// this never folds directions about a shading normal. Each returned density
// includes the discrete light selection probability.
VISRTX_DEVICE LightSample sampleVolumeLight(ScreenSample &ss,
    const FrameGPUData &fd,
    const vec3 &origin,
    bool shadowEndpoint = false)
{
  const auto &world = fd.world;
  const auto &rp = fd.renderer;
  const float radius = world.sceneRadius;
  const float ambientPower = rp.ambientIntensity > 0.f
      ? luminance(rp.ambientColor) * rp.ambientIntensity * kPi * kPi * radius
          * radius
      : 0.f;
  const bool ambient = ambientPower > 0.f;
  const size_t n = world.numLightInstances;
  if (!n && !ambient)
    return {};
  const float total = world.totalLightPower + ambientPower;
  size_t selected;
  float pickPdf;
  bool pickAmbient = false;
  if (!(total > 0.f)) {
    const size_t count = n + size_t(ambient);
    selected = glm::min(size_t(pcg_uniform(&ss.rs) * float(count)), count - 1);
    pickAmbient = selected == n;
    pickPdf = 1.f / float(count);
  } else {
    pickAmbient = ambient && pcg_uniform(&ss.rs) * total < ambientPower;
    if (pickAmbient) {
      selected = 0;
      pickPdf = ambientPower / total;
    } else {
      selected = glm::min(size_t(detail::inverseSampleCDF(
                              world.lightPickCdf, int(n), pcg_uniform(&ss.rs))),
          n - 1);
      pickPdf = world.lightPickDelta[selected] * world.totalLightPower / total;
    }
  }
  if (pickAmbient)
    return {rp.ambientColor * rp.ambientIntensity,
        randomDir(ss.rs),
        FLT_MAX,
        pickPdf / (4.f * kPi)};
  const auto &li = world.lightInstances[selected];
  auto ls =
      sampleLight(ss, origin, li.lightIndex, li.xfm, li.surfaceInstanceIndex);
  ls.pdf *= pickPdf;
  // Sampled geometry is the endpoint, not an intervening opaque blocker.
  if (shadowEndpoint
      && fd.registry.lights[li.lightIndex].type == LightType::GEOMETRY)
    ls.dist *= 1.f - GEOMETRY_LIGHT_SHADOW_EPSILON;
  return ls;
}
} // namespace visrtx
