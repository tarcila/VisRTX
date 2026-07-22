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

// Wavefront MDL material-sorted compaction (ticket 10). Two spheres with two
// DISTINCT compiled MDL materials (red + green diffuse) are lit and rendered
// with the wavefront renderer. Each material is shaded by its own per-compiled-
// material CUDA kernel, dispatched over a compacted per-material slot list. A
// correct compaction shades each sphere with its own tint; a partition bug
// (slots leaking between buckets) would cross-contaminate or blank a sphere. So
// the pass condition is BOTH a red-dominant AND a green-dominant cluster.

// anari_cpp
#define ANARI_EXTENSION_UTILITY_IMPL
#include <anari/anari_cpp/ext/std.h>
#include <anari/anari_cpp.hpp>
// VisRTX
#include <anari/ext/visrtx/makeVisRTXDevice.h>
// std
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

using uvec2 = std::array<unsigned int, 2>;
using vec3 = std::array<float, 3>;
using vec4 = std::array<float, 4>;

static const char *MDL_RED = R"mdl(mdl 1.6;
import ::df::*;
export material red_diffuse() = material(
    surface: material_surface(
        scattering: df::diffuse_reflection_bsdf(
            tint: color(0.8, 0.1, 0.1))));
)mdl";

static const char *MDL_GREEN = R"mdl(mdl 1.6;
import ::df::*;
export material green_diffuse() = material(
    surface: material_surface(
        scattering: df::diffuse_reflection_bsdf(
            tint: color(0.1, 0.8, 0.1))));
)mdl";

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

static anari::Surface makeSphere(anari::Device device,
    vec3 center,
    const char *source,
    const char *materialName)
{
  auto positionsArray = anari::newArray1D(device, ANARI_FLOAT32_VEC3, 1);
  {
    auto *positions = anari::map<vec3>(device, positionsArray);
    positions[0] = center;
    anari::unmap(device, positionsArray);
  }
  auto geometry = anari::newObject<anari::Geometry>(device, "sphere");
  anari::setAndReleaseParameter(
      device, geometry, "vertex.position", positionsArray);
  anari::setParameter(device, geometry, "radius", 0.7f);
  anari::commitParameters(device, geometry);

  auto material = anari::newObject<anari::Material>(device, "mdl");
  anari::setParameter(device, material, "sourceType", "code");
  anari::setParameter(device, material, "source", source);
  anari::setParameter(device, material, "materialName", materialName);
  anari::commitParameters(device, material);

  auto surface = anari::newObject<anari::Surface>(device);
  anari::setAndReleaseParameter(device, surface, "geometry", geometry);
  anari::setAndReleaseParameter(device, surface, "material", material);
  anari::commitParameters(device, surface);
  return surface;
}

int main()
{
  auto device = makeVisRTXDevice(statusFunc);

  auto redSphere =
      makeSphere(device, vec3{-1.f, 0.f, 0.f}, MDL_RED, "red_diffuse");
  auto greenSphere =
      makeSphere(device, vec3{1.f, 0.f, 0.f}, MDL_GREEN, "green_diffuse");
  std::array<anari::Surface, 2> surfaces = {redSphere, greenSphere};

  auto light = anari::newObject<anari::Light>(device, "directional");
  anari::setParameter(device, light, "direction", vec3{0.f, 0.f, 1.f});
  anari::setParameter(device, light, "irradiance", 2.f);
  anari::commitParameters(device, light);

  auto world = anari::newObject<anari::World>(device);
  anari::setParameterArray1D(device, world, "surface", surfaces.data(), 2);
  anari::setParameterArray1D(device, world, "light", &light, 1);
  anari::release(device, redSphere);
  anari::release(device, greenSphere);
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
  anari::setParameter(device, renderer, "ambientRadiance", 0.f);
  anari::setParameter(device, renderer, "pixelSamples", 16);
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
  size_t lit = 0, redDominant = 0, greenDominant = 0;
  for (size_t i = 0; i < numPixels; ++i) {
    const uint32_t px = fb.data[i];
    if ((px & 0x00ffffffu) == 0)
      continue;
    ++lit;
    const uint32_t r = px & 0xff;
    const uint32_t g = (px >> 8) & 0xff;
    const uint32_t b = (px >> 16) & 0xff;
    if (r > g + 16 && r > b + 16)
      ++redDominant;
    else if (g > r + 16 && g > b + 16)
      ++greenDominant;
  }
  anari::unmap(device, frame, "channel.color");

  anari::release(device, camera);
  anari::release(device, renderer);
  anari::release(device, world);
  anari::release(device, frame);
  anari::release(device, device);

  printf("wavefront MDL multi: %zu lit, %zu red, %zu green\n",
      lit,
      redDominant,
      greenDominant);

  // Each material must own a substantial cluster. A compaction partition bug
  // would leave one sphere black (0 of its color) or paint it the other tint.
  const size_t minCluster = 1000;
  if (redDominant < minCluster || greenDominant < minCluster) {
    fprintf(stderr,
        "FAIL: expected both a red and a green cluster (>= %zu px each) — "
        "material-sorted compaction mis-partitioned the pool\n",
        minCluster);
    return 1;
  }

  printf("PASS\n");
  return 0;
}
