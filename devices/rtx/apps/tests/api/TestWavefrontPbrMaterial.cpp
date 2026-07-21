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

// The wavefront CUDA shade stage evaluates the physicallyBased material's base
// color statically (no optixDirectCall). This renders a green PBR sphere and
// asserts the lit surface is green-dominant — which the pre-port neutral-gray
// fallback (equal RGB) would fail, so it guards the static PBR path specifically.

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

static anari::World generateScene(anari::Device device)
{
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
  anari::setParameter(device, material, "baseColor", vec3{0.1f, 0.8f, 0.1f});
  anari::commitParameters(device, material);

  auto surface = anari::newObject<anari::Surface>(device);
  anari::setAndReleaseParameter(device, surface, "geometry", geometry);
  anari::setAndReleaseParameter(device, surface, "material", material);
  anari::commitParameters(device, surface);

  auto world = anari::newObject<anari::World>(device);
  anari::setParameterArray1D(device, world, "surface", &surface, 1);
  anari::release(device, surface);
  anari::commitParameters(device, world);

  return world;
}

int main()
{
  auto device = makeVisRTXDevice(statusFunc);
  auto world = generateScene(device);

  auto camera = anari::newObject<anari::Camera>(device, "perspective");
  const vec3 eye = {0.f, 0.f, -2.f};
  const vec3 dir = {0.f, 0.f, 1.f};
  const vec3 up = {0.f, 1.f, 0.f};
  const uvec2 imageSize = {256, 256};
  anari::setParameter(device, camera, "position", eye);
  anari::setParameter(device, camera, "direction", dir);
  anari::setParameter(device, camera, "up", up);
  anari::setParameter(
      device, camera, "aspect", imageSize[0] / float(imageSize[1]));
  anari::commitParameters(device, camera);

  auto renderer = anari::newObject<anari::Renderer>(device, "wavefront");
  anari::setParameter(device, renderer, "background", vec4{0.f, 0.f, 0.f, 1.f});
  anari::commitParameters(device, renderer);

  auto frame = anari::newObject<anari::Frame>(device);
  anari::setParameter(device, frame, "size", imageSize);
  anari::setParameter(device, frame, "channel.color", ANARI_UFIXED8_RGBA_SRGB);
  anari::setParameter(device, frame, "world", world);
  anari::setParameter(device, frame, "camera", camera);
  anari::setParameter(device, frame, "renderer", renderer);
  anari::commitParameters(device, frame);

  anari::render(device, frame);
  anari::wait(device, frame);

  auto fb = anari::map<uint32_t>(device, frame, "channel.color");
  const size_t numPixels = size_t(fb.width) * fb.height;
  size_t greenDominant = 0;
  size_t lit = 0;
  for (size_t i = 0; i < numPixels; ++i) {
    const uint32_t px = fb.data[i];
    const uint32_t r = px & 0xff;
    const uint32_t g = (px >> 8) & 0xff;
    const uint32_t b = (px >> 16) & 0xff;
    if ((px & 0x00ffffffu) == 0)
      continue; // background
    ++lit;
    // Green base color -> G clearly the largest channel. The old neutral-gray
    // fallback has r==g==b and would never satisfy this.
    if (g > r + 16 && g > b + 16)
      ++greenDominant;
  }
  anari::unmap(device, frame, "channel.color");

  anari::release(device, camera);
  anari::release(device, renderer);
  anari::release(device, world);
  anari::release(device, frame);
  anari::release(device, device);

  printf("wavefront PBR: %zu lit, %zu green-dominant (%.1f%% of lit)\n",
      lit,
      greenDominant,
      lit ? 100.0 * double(greenDominant) / double(lit) : 0.0);

  if (lit == 0 || double(greenDominant) / double(lit) < 0.9) {
    fprintf(stderr,
        "FAIL: PBR sphere is not green-dominant — base color not evaluated "
        "statically (neutral-gray fallback?)\n");
    return 1;
  }

  printf("PASS\n");
  return 0;
}
