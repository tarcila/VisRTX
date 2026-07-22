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

// Wavefront MDL importance-sampled continuation bounce (ticket 10). A specular
// (mirror) MDL material has NO diffuse response, so next-event estimation
// contributes ~nothing: the sphere's colour comes ENTIRELY from the sampled
// continuation ray reflecting the environment. Against a blue background a
// correct MDL BSDF sample makes the sphere reflect blue; the pre-slice diffuse
// fallback (throughput *= albedo, cosine bounce) would render it near-black.
// Sphere pixels are isolated from the (equally blue) background via the depth
// channel: the sphere has finite depth, a primary miss is at infinity.

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
#include <cstring>
#include <vector>

using uvec2 = std::array<unsigned int, 2>;
using vec3 = std::array<float, 3>;
using vec4 = std::array<float, 4>;

static const char *MDL_MIRROR = R"mdl(mdl 1.6;
import ::df::*;
export material mirror() = material(
    surface: material_surface(
        scattering: df::specular_bsdf(tint: color(1.0))));
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

  auto material = anari::newObject<anari::Material>(device, "mdl");
  anari::setParameter(device, material, "sourceType", "code");
  anari::setParameter(device, material, "source", MDL_MIRROR);
  anari::setParameter(device, material, "materialName", "mirror");
  anari::commitParameters(device, material);

  auto surface = anari::newObject<anari::Surface>(device);
  anari::setAndReleaseParameter(device, surface, "geometry", geometry);
  anari::setAndReleaseParameter(device, surface, "material", material);
  anari::commitParameters(device, surface);

  // Uniform blue environment — the only thing the mirror can reflect. Read on a
  // secondary-ray miss via getBackgroundLight (HDRI lights only; the composite
  // `background` param is not visible to reflection rays).
  constexpr uint32_t W = 8, H = 4;
  std::vector<vec3> texels(W * H, vec3{0.f, 0.f, 1.f});
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
  anari::setParameter(device, camera, "position", vec3{0.f, 0.f, -2.f});
  anari::setParameter(device, camera, "direction", vec3{0.f, 0.f, 1.f});
  anari::setParameter(device, camera, "up", vec3{0.f, 1.f, 0.f});
  anari::setParameter(
      device, camera, "aspect", imageSize[0] / float(imageSize[1]));
  anari::commitParameters(device, camera);

  auto renderer = anari::newObject<anari::Renderer>(device, "wavefront");
  anari::setParameter(device, renderer, "ambientRadiance", 0.f);
  anari::setParameter(device, renderer, "pixelSamples", 16);
  anari::commitParameters(device, renderer);

  auto frame = anari::newObject<anari::Frame>(device);
  anari::setParameter(device, frame, "size", imageSize);
  anari::setParameter(device, frame, "channel.color", ANARI_UFIXED8_RGBA_SRGB);
  anari::setParameter(device, frame, "channel.depth", ANARI_FLOAT32);
  anari::setParameter(device, frame, "world", world);
  anari::setParameter(device, frame, "camera", camera);
  anari::setParameter(device, frame, "renderer", renderer);
  anari::commitParameters(device, frame);

  anari::render(device, frame);
  anari::wait(device, frame);

  auto color = anari::map<uint32_t>(device, frame, "channel.color");
  auto depth = anari::map<float>(device, frame, "channel.depth");
  const size_t numPixels = size_t(color.width) * color.height;
  size_t sphere = 0, sphereBlue = 0;
  for (size_t i = 0; i < numPixels; ++i) {
    // Finite depth => primary ray hit the sphere; a miss is at ~1e30.
    if (!(depth.data[i] < 100.f))
      continue;
    ++sphere;
    const uint32_t px = color.data[i];
    const uint32_t r = px & 0xff;
    const uint32_t g = (px >> 8) & 0xff;
    const uint32_t b = (px >> 16) & 0xff;
    if (b > r + 16 && b > g + 16)
      ++sphereBlue;
  }
  anari::unmap(device, frame, "channel.color");
  anari::unmap(device, frame, "channel.depth");

  anari::release(device, camera);
  anari::release(device, renderer);
  anari::release(device, world);
  anari::release(device, frame);
  anari::release(device, device);

  printf("wavefront MDL specular: %zu sphere px, %zu blue (%.1f%%)\n",
      sphere,
      sphereBlue,
      sphere ? 100.0 * double(sphereBlue) / double(sphere) : 0.0);

  if (sphere < 1000) {
    fprintf(stderr, "FAIL: sphere not resolved in the depth channel\n");
    return 1;
  }
  // Most of the mirror reflects the blue environment. The diffuse fallback
  // would leave the specular sphere near-black (no diffuse NEE response),
  // failing this.
  if (double(sphereBlue) / double(sphere) < 0.5) {
    fprintf(stderr,
        "FAIL: specular sphere does not reflect the blue background — the MDL "
        "BSDF sample is not driving the continuation bounce\n");
    return 1;
  }

  printf("PASS\n");
  return 0;
}
