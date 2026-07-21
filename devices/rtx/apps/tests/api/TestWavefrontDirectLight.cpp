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

// The wavefront CUDA shade stage does next-event estimation against the scene's
// analytic lights. With ambient disabled, the only way a surface can be lit is
// the directional light — so a matte sphere renders illuminated with NEE and
// black without it. Rendering both isolates the direct-lighting contribution:
// a broken/ignored light path would leave the sphere dark in both cases.

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

// A matte sphere, optionally lit by a single directional light travelling +Z
// (toward the scene from the camera side), so it illuminates the camera-facing
// hemisphere. Ambient is disabled so the light is the only illumination.
static size_t renderAndCountLit(anari::Device device, bool withLight)
{
  auto positionsArray = anari::newArray1D(device, ANARI_FLOAT32_VEC3, 1);
  {
    auto *p = anari::map<vec3>(device, positionsArray);
    p[0] = vec3{0.f, 0.f, 0.f};
    anari::unmap(device, positionsArray);
  }
  auto geometry = anari::newObject<anari::Geometry>(device, "sphere");
  anari::setAndReleaseParameter(
      device, geometry, "vertex.position", positionsArray);
  anari::setParameter(device, geometry, "radius", 0.9f);
  anari::commitParameters(device, geometry);

  auto material = anari::newObject<anari::Material>(device, "matte");
  anari::setParameter(device, material, "color", vec3{0.8f, 0.8f, 0.8f});
  anari::commitParameters(device, material);

  auto surface = anari::newObject<anari::Surface>(device);
  anari::setAndReleaseParameter(device, surface, "geometry", geometry);
  anari::setAndReleaseParameter(device, surface, "material", material);
  anari::commitParameters(device, surface);

  auto world = anari::newObject<anari::World>(device);
  anari::setParameterArray1D(device, world, "surface", &surface, 1);
  anari::release(device, surface);
  if (withLight) {
    auto light = anari::newObject<anari::Light>(device, "directional");
    anari::setParameter(device, light, "direction", vec3{0.f, 0.f, 1.f});
    anari::setParameter(device, light, "irradiance", 2.f);
    anari::commitParameters(device, light);
    anari::setParameterArray1D(device, world, "light", &light, 1);
    anari::release(device, light);
  }
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
  anari::setParameter(device, renderer, "background", vec4{0.f, 0.f, 0.f, 1.f});
  anari::setParameter(device, renderer, "ambientRadiance", 0.f); // NEE only
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
  size_t lit = 0;
  for (size_t i = 0; i < size_t(fb.width) * fb.height; ++i)
    if ((fb.data[i] & 0x00ffffffu) != 0)
      ++lit;
  anari::unmap(device, frame, "channel.color");

  anari::release(device, camera);
  anari::release(device, renderer);
  anari::release(device, world);
  anari::release(device, frame);
  return lit;
}

int main()
{
  auto device = makeVisRTXDevice(statusFunc);

  const size_t litNoLight = renderAndCountLit(device, false);
  const size_t litWithLight = renderAndCountLit(device, true);

  anari::release(device, device);

  printf("wavefront direct light: lit pixels — ambient-only-off=%zu, "
         "directional=%zu\n",
      litNoLight,
      litWithLight);

  // Ambient off and no light: the sphere is unlit -> essentially black.
  if (litNoLight > 4000) {
    fprintf(stderr,
        "FAIL: sphere is lit (%zu px) with no light and ambient off — shading "
        "is not honoring ambientRadiance / leaking illumination\n",
        litNoLight);
    return 1;
  }
  // The directional light must illuminate a large part of the visible sphere.
  if (litWithLight < 15000) {
    fprintf(stderr,
        "FAIL: directional light illuminated only %zu px — NEE is not lighting "
        "the surface\n",
        litWithLight);
    return 1;
  }

  printf("PASS\n");
  return 0;
}
