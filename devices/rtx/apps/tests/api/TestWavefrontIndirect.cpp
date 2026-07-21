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

// The wavefront renderer path-traces indirect bounces. A receiver sphere sits
// beside a large reflector sphere; a directional light lights the reflector's
// receiver-facing side, so the receiver's reflector-facing side is lit only by
// the bounce. Rendering at maxDepth 1 (direct only) vs 4 must brighten the
// receiver — indirect illumination. Isolating the receiver by objectId excludes
// the reflector. With bounces disabled the receiver brightness is unchanged.

// anari_cpp
#define ANARI_EXTENSION_UTILITY_IMPL
#include <anari/anari_cpp.hpp>
#include <anari/anari_cpp/ext/std.h>
// VisRTX
#include <anari/ext/visrtx/makeVisRTXDevice.h>
// std
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

static uint32_t luminance(uint32_t px)
{
  const uint32_t r = px & 0xff, g = (px >> 8) & 0xff, b = (px >> 16) & 0xff;
  return (r * 54 + g * 183 + b * 19) >> 8;
}

static anari::Surface sphere(
    anari::Device device, vec3 center, float radius, uint32_t id)
{
  auto positions = anari::newArray1D(device, ANARI_FLOAT32_VEC3, 1);
  *anari::map<vec3>(device, positions) = center;
  anari::unmap(device, positions);
  auto geometry = anari::newObject<anari::Geometry>(device, "sphere");
  anari::setAndReleaseParameter(device, geometry, "vertex.position", positions);
  anari::setParameter(device, geometry, "radius", radius);
  anari::commitParameters(device, geometry);
  auto material = anari::newObject<anari::Material>(device, "matte");
  anari::setParameter(device, material, "color", vec3{0.9f, 0.9f, 0.9f});
  anari::commitParameters(device, material);
  auto surface = anari::newObject<anari::Surface>(device);
  anari::setAndReleaseParameter(device, surface, "geometry", geometry);
  anari::setAndReleaseParameter(device, surface, "material", material);
  anari::setParameter(device, surface, "id", id);
  anari::commitParameters(device, surface);
  return surface;
}

// Mean luminance over the receiver (objectId 1).
static double renderReceiverMean(anari::Device device, int maxDepth)
{
  std::vector<anari::Surface> surfaces;
  surfaces.push_back(sphere(device, vec3{-0.7f, 0.f, 0.f}, 0.6f, 1u)); // receiver
  surfaces.push_back(sphere(device, vec3{0.9f, 0.f, 0.f}, 1.0f, 2u)); // reflector

  auto world = anari::newObject<anari::World>(device);
  anari::setParameterArray1D(
      device, world, "surface", surfaces.data(), surfaces.size());
  for (auto s : surfaces)
    anari::release(device, s);
  auto light = anari::newObject<anari::Light>(device, "directional");
  // Travels +X: lights the reflector's -X face (toward the receiver); the
  // receiver's +X face (toward the reflector) is not directly lit.
  anari::setParameter(device, light, "direction", vec3{1.f, 0.f, 0.2f});
  anari::setParameter(device, light, "irradiance", 3.f);
  anari::commitParameters(device, light);
  anari::setParameterArray1D(device, world, "light", &light, 1);
  anari::release(device, light);
  anari::commitParameters(device, world);

  auto camera = anari::newObject<anari::Camera>(device, "perspective");
  const uvec2 imageSize = {256, 256};
  anari::setParameter(device, camera, "position", vec3{0.f, 0.f, -4.f});
  anari::setParameter(device, camera, "direction", vec3{0.f, 0.f, 1.f});
  anari::setParameter(device, camera, "up", vec3{0.f, 1.f, 0.f});
  anari::setParameter(
      device, camera, "aspect", imageSize[0] / float(imageSize[1]));
  anari::commitParameters(device, camera);

  auto renderer = anari::newObject<anari::Renderer>(device, "wavefront");
  anari::setParameter(device, renderer, "background", vec4{0.f, 0.f, 0.f, 1.f});
  anari::setParameter(device, renderer, "ambientRadiance", 0.f); // NEE + GI only
  anari::setParameter(device, renderer, "maxDepth", maxDepth);
  anari::setParameter(device, renderer, "pixelSamples", 4);
  anari::commitParameters(device, renderer);

  auto frame = anari::newObject<anari::Frame>(device);
  anari::setParameter(device, frame, "size", imageSize);
  anari::setParameter(device, frame, "channel.color", ANARI_UFIXED8_RGBA_SRGB);
  anari::setParameter(device, frame, "channel.objectId", ANARI_UINT32);
  anari::setParameter(device, frame, "world", world);
  anari::setParameter(device, frame, "camera", camera);
  anari::setParameter(device, frame, "renderer", renderer);
  anari::commitParameters(device, frame);

  anari::render(device, frame);
  anari::wait(device, frame);

  auto color = anari::map<uint32_t>(device, frame, "channel.color");
  auto obj = anari::map<uint32_t>(device, frame, "channel.objectId");
  double sum = 0.0;
  size_t n = 0;
  for (size_t i = 0; i < size_t(color.width) * color.height; ++i) {
    if (obj.data[i] == 1u) {
      sum += luminance(color.data[i]);
      ++n;
    }
  }
  anari::unmap(device, frame, "channel.color");
  anari::unmap(device, frame, "channel.objectId");

  anari::release(device, camera);
  anari::release(device, renderer);
  anari::release(device, world);
  anari::release(device, frame);
  return n ? sum / double(n) : 0.0;
}

int main()
{
  auto device = makeVisRTXDevice(statusFunc);
  const double direct = renderReceiverMean(device, 1);
  const double withGI = renderReceiverMean(device, 4);
  anari::release(device, device);

  const double gain = direct > 0.0 ? withGI / direct - 1.0 : 0.0;
  printf("wavefront indirect: receiver mean %.2f (direct) -> %.2f (maxDepth 4), "
         "+%.1f%%\n",
      direct,
      withGI,
      100.0 * gain);

  // The reflector's bounce measurably brightens the receiver. Without indirect
  // bounces the two renders match on the receiver.
  if (gain < 0.05) {
    fprintf(stderr,
        "FAIL: indirect bounces added only %.1f%% to the receiver — the path "
        "is not carrying light past the first bounce\n",
        100.0 * gain);
    return 1;
  }

  printf("PASS\n");
  return 0;
}
