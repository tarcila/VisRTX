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

// The wavefront shade stage does next-event estimation against Geometry Lights
// (emissive surfaces). A matte receiver (objectId 1) is lit only by an emissive
// sphere (objectId 2) — ambient off, no analytic lights. The receiver is
// illuminated only if Geometry-Light NEE works; when those lights are skipped
// (or the emissive sphere is made non-emitting) the receiver stays dark.

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

static anari::Surface matteSphere(
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

static anari::Surface emissiveSphere(
    anari::Device device, vec3 center, float radius, float emissive, uint32_t id)
{
  auto positions = anari::newArray1D(device, ANARI_FLOAT32_VEC3, 1);
  *anari::map<vec3>(device, positions) = center;
  anari::unmap(device, positions);
  auto geometry = anari::newObject<anari::Geometry>(device, "sphere");
  anari::setAndReleaseParameter(device, geometry, "vertex.position", positions);
  anari::setParameter(device, geometry, "radius", radius);
  anari::commitParameters(device, geometry);
  auto material = anari::newObject<anari::Material>(device, "physicallyBased");
  anari::setParameter(device, material, "baseColor", vec3{0.f, 0.f, 0.f});
  anari::setParameter(
      device, material, "emissive", vec3{emissive, emissive, emissive});
  anari::commitParameters(device, material);
  auto surface = anari::newObject<anari::Surface>(device);
  anari::setAndReleaseParameter(device, surface, "geometry", geometry);
  anari::setAndReleaseParameter(device, surface, "material", material);
  anari::setParameter(device, surface, "id", id);
  anari::commitParameters(device, surface);
  return surface;
}

// Count lit receiver (objectId 1) pixels for a given emitter radiance.
static size_t receiverLit(anari::Device device, float emissive)
{
  std::vector<anari::Surface> surfaces;
  surfaces.push_back(matteSphere(device, vec3{-0.2f, 0.f, 0.f}, 0.7f, 1u));
  surfaces.push_back(
      emissiveSphere(device, vec3{1.3f, 0.f, 0.f}, 0.5f, emissive, 2u));

  auto world = anari::newObject<anari::World>(device);
  anari::setParameterArray1D(
      device, world, "surface", surfaces.data(), surfaces.size());
  for (auto s : surfaces)
    anari::release(device, s);
  anari::commitParameters(device, world); // no analytic lights

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
  anari::setParameter(device, renderer, "ambientRadiance", 0.f);
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
  size_t lit = 0;
  for (size_t i = 0; i < size_t(color.width) * color.height; ++i)
    if (obj.data[i] == 1u && (color.data[i] & 0x00ffffffu) != 0)
      ++lit;
  anari::unmap(device, frame, "channel.color");
  anari::unmap(device, frame, "channel.objectId");

  anari::release(device, camera);
  anari::release(device, renderer);
  anari::release(device, world);
  anari::release(device, frame);
  return lit;
}

int main()
{
  auto device = makeVisRTXDevice(statusFunc);
  const size_t litDark = receiverLit(device, 0.f); // emitter off -> no light
  const size_t litEmit = receiverLit(device, 8.f); // emitter on -> geometry light
  anari::release(device, device);

  printf("wavefront geometry light: receiver lit pixels — emitter off=%zu, "
         "emitter on=%zu\n",
      litDark,
      litEmit);

  // With no light of any kind the receiver is dark.
  if (litDark > 2000) {
    fprintf(stderr,
        "FAIL: receiver is lit (%zu px) with a non-emitting sphere and no other "
        "light — illumination is leaking\n",
        litDark);
    return 1;
  }
  // The emissive sphere's Geometry Light must illuminate the receiver.
  if (litEmit < 400) {
    fprintf(stderr,
        "FAIL: emissive sphere illuminated only %zu receiver px — Geometry-Light "
        "NEE is not lighting the surface\n",
        litEmit);
    return 1;
  }

  printf("PASS\n");
  return 0;
}
