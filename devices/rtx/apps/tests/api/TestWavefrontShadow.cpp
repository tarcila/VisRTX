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

// The wavefront shadow trace stage occludes next-event estimation. A receiver
// sphere is lit by a directional light; adding an occluder sphere between the
// light and the receiver casts a shadow on the receiver. Using the objectId AOV
// to look only at receiver pixels (id 1), the total light landing on the
// receiver must drop once the occluder is present. Restricting to the receiver
// excludes the occluder's own silhouette; without a working shadow trace the
// receiver's total brightness is unchanged.

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
  return (r * 54 + g * 183 + b * 19) >> 8; // ~Rec.709
}

static anari::Surface makeSphere(
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
  anari::setParameter(device, material, "color", vec3{0.8f, 0.8f, 0.8f});
  anari::commitParameters(device, material);

  auto surface = anari::newObject<anari::Surface>(device);
  anari::setAndReleaseParameter(device, surface, "geometry", geometry);
  anari::setAndReleaseParameter(device, surface, "material", material);
  anari::setParameter(device, surface, "id", id); // objectId AOV
  anari::commitParameters(device, surface);
  return surface;
}

// Receiver sphere (id 1) lit by a directional light; optionally an occluder
// sphere (id 2) between the light and the receiver. The angled light lands the
// shadow on the receiver's visible face.
static void render(anari::Device device,
    bool withOccluder,
    std::vector<uint32_t> &colorOut,
    std::vector<uint32_t> &objIdOut)
{
  std::vector<anari::Surface> surfaces;
  surfaces.push_back(makeSphere(device, vec3{0.f, 0.f, 0.f}, 0.9f, 1u));
  if (withOccluder)
    surfaces.push_back(
        makeSphere(device, vec3{0.6f, 0.6f, -1.2f}, 0.4f, 2u));

  auto world = anari::newObject<anari::World>(device);
  anari::setParameterArray1D(
      device, world, "surface", surfaces.data(), surfaces.size());
  for (auto s : surfaces)
    anari::release(device, s);
  auto light = anari::newObject<anari::Light>(device, "directional");
  // Angled so the occluder's shadow lands on the receiver offset from the
  // occluder's own screen silhouette.
  anari::setParameter(device, light, "direction", vec3{-0.4f, -0.4f, 1.f});
  anari::setParameter(device, light, "irradiance", 2.f);
  anari::commitParameters(device, light);
  anari::setParameterArray1D(device, world, "light", &light, 1);
  anari::release(device, light);
  anari::commitParameters(device, world);

  auto camera = anari::newObject<anari::Camera>(device, "perspective");
  const uvec2 imageSize = {256, 256};
  anari::setParameter(device, camera, "position", vec3{0.f, 0.f, -3.f});
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
  anari::setParameter(device, frame, "channel.objectId", ANARI_UINT32);
  anari::setParameter(device, frame, "world", world);
  anari::setParameter(device, frame, "camera", camera);
  anari::setParameter(device, frame, "renderer", renderer);
  anari::commitParameters(device, frame);

  anari::render(device, frame);
  anari::wait(device, frame);

  auto fb = anari::map<uint32_t>(device, frame, "channel.color");
  colorOut.assign(fb.data, fb.data + size_t(fb.width) * fb.height);
  anari::unmap(device, frame, "channel.color");
  auto ob = anari::map<uint32_t>(device, frame, "channel.objectId");
  objIdOut.assign(ob.data, ob.data + size_t(ob.width) * ob.height);
  anari::unmap(device, frame, "channel.objectId");

  anari::release(device, camera);
  anari::release(device, renderer);
  anari::release(device, world);
  anari::release(device, frame);
}

int main()
{
  auto device = makeVisRTXDevice(statusFunc);
  std::vector<uint32_t> colorA, objA, colorB, objB;
  render(device, false, colorA, objA);
  render(device, true, colorB, objB);
  anari::release(device, device);

  // Total light landing on the receiver (objectId 1) with and without the
  // occluder. The occluder's own pixels (id 2) are excluded, so any drop is the
  // cast shadow removing direct light from the receiver.
  constexpr uint32_t kReceiverId = 1;
  uint64_t sumA = 0, sumB = 0;
  size_t pixA = 0, pixB = 0;
  for (size_t i = 0; i < colorA.size(); ++i) {
    if (objA[i] == kReceiverId) {
      sumA += luminance(colorA[i]);
      ++pixA;
    }
    if (objB[i] == kReceiverId) {
      sumB += luminance(colorB[i]);
      ++pixB;
    }
  }

  const double drop = sumA > 0 ? 1.0 - double(sumB) / double(sumA) : 0.0;
  printf("wavefront shadow: receiver light sum %llu (%zu px) -> %llu (%zu px), "
         "drop %.1f%%\n",
      (unsigned long long)sumA,
      pixA,
      (unsigned long long)sumB,
      pixB,
      100.0 * drop);

  // The occluder shadows a meaningful part of the receiver, so its total
  // received light must drop clearly. With no shadow trace the receiver is lit
  // identically in both renders (drop ~ 0).
  if (drop < 0.1) {
    fprintf(stderr,
        "FAIL: receiver light dropped only %.1f%% with an occluder present — "
        "the shadow trace stage is not occluding direct lighting\n",
        100.0 * drop);
    return 1;
  }

  printf("PASS\n");
  return 0;
}
