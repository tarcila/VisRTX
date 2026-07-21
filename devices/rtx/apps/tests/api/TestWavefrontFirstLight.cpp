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

// First-light smoke for the wavefront renderer subtype (ticket 05b). Renders a
// single matte sphere and asserts the wavefront pipeline shades the geometry —
// i.e. a meaningful fraction of the frame is lit foreground, distinct from the
// background. This exercises the subtype registration, its own OptiX module and
// pipeline, and end-to-end render through the ANARI API. It is NOT yet a parity
// test against the quality renderer (that arrives with slice 05d).

// anari_cpp
#define ANARI_EXTENSION_UTILITY_IMPL
#include <anari/anari_cpp.hpp>
#include <anari/anari_cpp/ext/std.h>
// VisRTX
#include <anari/ext/visrtx/makeVisRTXDevice.h>
// std
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
  if (severity == ANARI_SEVERITY_FATAL_ERROR) {
    fprintf(stderr, "[FATAL][%p] %s\n", source, message);
    std::exit(1);
  } else if (severity == ANARI_SEVERITY_ERROR) {
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

  auto material = anari::newObject<anari::Material>(device, "matte");
  anari::setParameter(device, material, "color", vec3{0.8f, 0.2f, 0.2f});
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
  const vec4 backgroundColor = {0.f, 0.f, 0.f, 1.f};
  anari::setParameter(device, renderer, "background", backgroundColor);
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
  size_t litPixels = 0;
  for (size_t i = 0; i < numPixels; ++i) {
    // Non-background = any RGB channel nonzero (background is pure black).
    if ((fb.data[i] & 0x00ffffffu) != 0)
      ++litPixels;
  }
  anari::unmap(device, frame, "channel.color");

  anari::release(device, camera);
  anari::release(device, renderer);
  anari::release(device, world);
  anari::release(device, frame);
  anari::release(device, device);

  // A radius-0.9 sphere at 256x256 from this camera covers a large central
  // disc; require at least 10% lit to prove the wavefront pipeline shaded the
  // geometry (not merely rendered background) while staying robust to exact
  // projection.
  const double litFraction = double(litPixels) / double(numPixels);
  printf("wavefront first light: %zu/%zu lit (%.1f%%)\n",
      litPixels,
      numPixels,
      100.0 * litFraction);
  if (litFraction < 0.10) {
    fprintf(stderr,
        "FAIL: wavefront renderer produced too few lit pixels (%.1f%%)\n",
        100.0 * litFraction);
    return 1;
  }

  printf("PASS\n");
  return 0;
}
