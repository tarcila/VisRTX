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

// Wavefront builtin physicallyBased cutout opacity. A BLEND physicallyBased
// material with opacity 0.5 is stochastically transparent in the wavefront
// path: each sample either fully shades the surface or passes the ray straight
// through. Two consequences, both asserted:
//  - accumulated coverage (framebuffer alpha) converges to the authored 0.5;
//  - the ray passes THROUGH on transparent samples, so the blue environment
//    behind the sphere shows through — the sphere carries significant blue.
// The pre-slice behaviour (opacity*shading dimming) yields alpha ~0.5 too but
// NEVER reveals the background: the red sphere stays red. So the blue-through
// check is what pins the stochastic cutout. The same path serves `matte`.

// anari_cpp
#define ANARI_EXTENSION_UTILITY_IMPL
#include <anari/anari_cpp/ext/std.h>
#include <anari/anari_cpp.hpp>
// VisRTX
#include <anari/ext/visrtx/makeVisRTXDevice.h>
// std
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using uvec2 = std::array<unsigned int, 2>;
using vec3 = std::array<float, 3>;
using vec4 = std::array<float, 4>;

static void statusFunc(const void *,
    ANARIDevice,
    ANARIObject source,
    ANARIDataType,
    ANARIStatusSeverity severity,
    ANARIStatusCode,
    const char *message)
{
  if (severity == ANARI_SEVERITY_FATAL_ERROR
      || severity == ANARI_SEVERITY_ERROR) {
    fprintf(stderr, "[ERROR][%p] %s\n", source, message);
    std::exit(1);
  }
}

int main()
{
  auto device = makeVisRTXDevice(statusFunc);

  auto positionsArray = anari::newArray1D(device, ANARI_FLOAT32_VEC3, 1);
  {
    auto *positions = anari::map<vec3>(device, positionsArray);
    positions[0] = vec3{0.f, 0.f, 0.f};
    anari::unmap(device, positionsArray);
  }
  auto geometry = anari::newObject<anari::Geometry>(device, "sphere");
  anari::setAndReleaseParameter(
      device, geometry, "vertex.position", positionsArray);
  anari::setParameter(device, geometry, "radius", 0.9f);
  anari::commitParameters(device, geometry);

  auto material = anari::newObject<anari::Material>(device, "physicallyBased");
  anari::setParameter(device, material, "baseColor", vec3{0.8f, 0.1f, 0.1f});
  anari::setParameter(device, material, "opacity", 0.5f);
  anari::setParameter(device, material, "alphaMode", "blend");
  anari::commitParameters(device, material);

  auto surface = anari::newObject<anari::Surface>(device);
  anari::setAndReleaseParameter(device, surface, "geometry", geometry);
  anari::setAndReleaseParameter(device, surface, "material", material);
  anari::commitParameters(device, surface);

  // White directional light for the opaque-sample surface shading.
  auto light = anari::newObject<anari::Light>(device, "directional");
  anari::setParameter(device, light, "direction", vec3{0.f, 0.f, 1.f});
  anari::setParameter(device, light, "irradiance", 2.f);
  anari::commitParameters(device, light);

  // Blue environment behind the sphere — only reachable by rays that pass
  // through the cutout holes (HDRI is what secondary/continuation rays see).
  constexpr uint32_t W = 8, H = 4;
  std::vector<vec3> texels(W * H, vec3{0.f, 0.f, 1.f});
  auto radiance = anari::newArray2D(device, ANARI_FLOAT32_VEC3, W, H);
  std::memcpy(anari::map<vec3>(device, radiance),
      texels.data(),
      texels.size() * sizeof(vec3));
  anari::unmap(device, radiance);
  auto env = anari::newObject<anari::Light>(device, "hdri");
  anari::setParameter(device, env, "direction", vec3{0.f, 0.f, 1.f});
  anari::setParameter(device, env, "up", vec3{0.f, 1.f, 0.f});
  anari::setParameter(device, env, "scale", 1.f);
  anari::setAndReleaseParameter(device, env, "radiance", radiance);
  anari::commitParameters(device, env);

  std::array<anari::Light, 2> lights = {light, env};
  auto world = anari::newObject<anari::World>(device);
  anari::setParameterArray1D(device, world, "surface", &surface, 1);
  anari::setParameterArray1D(device, world, "light", lights.data(), 2);
  anari::release(device, surface);
  anari::release(device, light);
  anari::release(device, env);
  anari::commitParameters(device, world);

  auto camera = anari::newObject<anari::Camera>(device, "perspective");
  const uvec2 imageSize = {256, 256};
  anari::setParameter(device, camera, "position", vec3{0.f, 0.f, -2.f});
  anari::setParameter(device, camera, "direction", vec3{0.f, 0.f, 1.f});
  anari::setParameter(device, camera, "up", vec3{0.f, 1.f, 0.f});
  anari::setParameter(
      device, camera, "aspect", imageSize[0] / float(imageSize[1]));
  anari::commitParameters(device, camera);

  auto renderer = anari::newObject<anari::Renderer>(device, "wavefront");
  anari::setParameter(device, renderer, "background", vec4{0.f, 0.f, 0.f, 0.f});
  anari::setParameter(device, renderer, "ambientRadiance", 0.f);
  // Many samples so the stochastic coverage converges to the authored opacity.
  anari::setParameter(device, renderer, "pixelSamples", 256);
  anari::commitParameters(device, renderer);

  auto frame = anari::newObject<anari::Frame>(device);
  anari::setParameter(device, frame, "size", imageSize);
  anari::setParameter(device, frame, "channel.color", ANARI_FLOAT32_VEC4);
  anari::setParameter(device, frame, "channel.depth", ANARI_FLOAT32);
  anari::setParameter(device, frame, "world", world);
  anari::setParameter(device, frame, "camera", camera);
  anari::setParameter(device, frame, "renderer", renderer);
  anari::commitParameters(device, frame);

  anari::render(device, frame);
  anari::wait(device, frame);

  auto color = anari::map<vec4>(device, frame, "channel.color");
  auto depth = anari::map<float>(device, frame, "channel.depth");
  const size_t numPixels = size_t(color.width) * color.height;
  size_t sphere = 0;
  double alphaSum = 0.0, blueSum = 0.0;
  for (size_t i = 0; i < numPixels; ++i) {
    if (!(depth.data[i] < 100.f))
      continue; // background (primary miss)
    ++sphere;
    alphaSum += color.data[i][3];
    blueSum += color.data[i][2];
  }
  const double meanAlpha = sphere ? alphaSum / double(sphere) : 0.0;
  const double meanBlue = sphere ? blueSum / double(sphere) : 0.0;
  anari::unmap(device, frame, "channel.color");
  anari::unmap(device, frame, "channel.depth");

  anari::release(device, camera);
  anari::release(device, renderer);
  anari::release(device, world);
  anari::release(device, frame);
  anari::release(device, device);

  printf(
      "wavefront PBR cutout: %zu sphere px, mean alpha %.3f, mean blue %.3f\n",
      sphere,
      meanAlpha,
      meanBlue);

  if (sphere < 1000) {
    fprintf(stderr, "FAIL: sphere not resolved in the depth channel\n");
    return 1;
  }
  // Coverage converges to the authored cutout_opacity 0.5.
  if (meanAlpha < 0.4 || meanAlpha > 0.6) {
    fprintf(stderr,
        "FAIL: mean coverage %.3f is not ~0.5 — MDL cutout opacity is not "
        "stochastically applied\n",
        meanAlpha);
    return 1;
  }
  // The blue environment shows through the holes. A red diffuse surface barely
  // reflects the blue env, so substantial blue can only come from pass-through
  // — which the deterministic dimming (ray stops at the surface) never
  // produces.
  if (meanBlue < 0.2) {
    fprintf(stderr,
        "FAIL: mean blue %.3f too low — the cutout is not passing rays through "
        "to reveal the background\n",
        meanBlue);
    return 1;
  }

  printf("PASS\n");
  return 0;
}
