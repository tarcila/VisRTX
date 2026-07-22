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

// Furnace test for environment MIS. A white (albedo 1) diffuse sphere immersed
// in a uniform environment of radiance L must reflect exactly L (energy
// conservation: rho * L = L), so it is INVISIBLE against the background. Any
// deviation is a bug. In particular, if the renderer both next-event-samples
// the environment AND deposits it again on the BSDF-escape miss without
// multiple- importance-sampling weights, the surface receives the environment
// twice and reads ~2x the background. This measures the sphere/background
// ratio.

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

  // White Lambertian: albedo 1, so outgoing radiance must equal the
  // environment.
  auto material = anari::newObject<anari::Material>(device, "matte");
  anari::setParameter(device, material, "color", vec3{1.f, 1.f, 1.f});
  anari::commitParameters(device, material);

  auto surface = anari::newObject<anari::Surface>(device);
  anari::setAndReleaseParameter(device, surface, "geometry", geometry);
  anari::setAndReleaseParameter(device, surface, "material", material);
  anari::commitParameters(device, surface);

  // Uniform grey environment, radiance L = 0.5 on every channel.
  constexpr float L = 0.5f;
  constexpr uint32_t W = 8, H = 4;
  std::vector<vec3> texels(W * H, vec3{L, L, L});
  auto radiance = anari::newArray2D(device, ANARI_FLOAT32_VEC3, W, H);
  std::memcpy(anari::map<vec3>(device, radiance),
      texels.data(),
      texels.size() * sizeof(vec3));
  anari::unmap(device, radiance);
  auto light = anari::newObject<anari::Light>(device, "hdri");
  anari::setParameter(device, light, "direction", vec3{0.f, 0.f, 1.f});
  anari::setParameter(device, light, "up", vec3{0.f, 1.f, 0.f});
  anari::setParameter(device, light, "scale", 1.f);
  anari::setAndReleaseParameter(device, light, "radiance", radiance);
  anari::commitParameters(device, light);

  auto world = anari::newObject<anari::World>(device);
  anari::setParameterArray1D(device, world, "surface", &surface, 1);
  anari::setParameterArray1D(device, world, "light", &light, 1);
  anari::release(device, surface);
  anari::release(device, light);
  anari::commitParameters(device, world);

  auto camera = anari::newObject<anari::Camera>(device, "perspective");
  const uvec2 imageSize = {256, 256};
  anari::setParameter(device, camera, "position", vec3{0.f, 0.f, -2.5f});
  anari::setParameter(device, camera, "direction", vec3{0.f, 0.f, 1.f});
  anari::setParameter(device, camera, "up", vec3{0.f, 1.f, 0.f});
  anari::setParameter(
      device, camera, "aspect", imageSize[0] / float(imageSize[1]));
  anari::commitParameters(device, camera);

  auto renderer = anari::newObject<anari::Renderer>(device, "wavefront");
  anari::setParameter(device, renderer, "ambientRadiance", 0.f);
  anari::setParameter(device, renderer, "pixelSamples", 512);
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
  double sphereSum = 0.0, bgSum = 0.0;
  size_t sphereN = 0, bgN = 0;
  for (size_t i = 0; i < numPixels; ++i) {
    const double lum =
        (double(color.data[i][0]) + color.data[i][1] + color.data[i][2]) / 3.0;
    if (depth.data[i] < 100.f) {
      sphereSum += lum;
      ++sphereN;
    } else {
      bgSum += lum;
      ++bgN;
    }
  }
  anari::unmap(device, frame, "channel.color");
  anari::unmap(device, frame, "channel.depth");

  anari::release(device, camera);
  anari::release(device, renderer);
  anari::release(device, world);
  anari::release(device, frame);
  anari::release(device, device);

  const double sphere = sphereN ? sphereSum / double(sphereN) : 0.0;
  const double bg = bgN ? bgSum / double(bgN) : 0.0;
  const double ratio = bg > 0.0 ? sphere / bg : 0.0;

  printf("wavefront furnace: sphere=%.4f, env=%.4f, ratio=%.3f (want ~1.0)\n",
      sphere,
      bg,
      ratio);

  if (sphereN < 1000 || bgN < 1000) {
    fprintf(stderr, "FAIL: sphere/background not resolved\n");
    return 1;
  }
  // Energy conservation: a white diffuse sphere in a uniform environment must
  // match it. Allow 10% for residual noise + the sphere's own darkening at
  // grazing silhouette pixels; a double-count reads ~2.0.
  if (ratio < 0.85 || ratio > 1.15) {
    fprintf(stderr,
        "FAIL: furnace ratio %.3f is not ~1.0 — the environment is not "
        "energy-conserving (double-counted NEE + BSDF escape without MIS?)\n",
        ratio);
    return 1;
  }

  printf("PASS\n");
  return 0;
}
