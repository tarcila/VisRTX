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

// Numeric parity: the wavefront and interactive renderers share the exact same
// PhysicallyBased BRDF (gpu/physicallyBasedBsdf.h). This validates that the
// shared BRDF *evaluation* (pbrInitState + pbrEvalNEE / NEE direct lighting)
// produces matching results by rendering a LONE convex PBR sphere under a
// directional light with each renderer and comparing the pixel-averaged image.
//
// A lone convex sphere is deliberate: its indirect bounces escape immediately
// (GI ~ 0), so the renderers' different path-termination policies (the
// wavefront caps at maxDepth; quality uses Russian roulette; interactive
// marches direct only) do not confound the comparison — both reduce to the same
// direct-lit image. The bounce/continuation machinery is covered separately by
// the metallic and transmission tests (which require a bounce to reveal the
// environment).

// anari_cpp
#define ANARI_EXTENSION_UTILITY_IMPL
#include <anari/anari_cpp/ext/std.h>
#include <anari/anari_cpp.hpp>
// VisRTX
#include <anari/ext/visrtx/makeVisRTXDevice.h>
// std
#include <array>
#include <cmath>
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

static anari::Surface pbrSurface(
    anari::Device device, anari::Geometry geom, vec3 baseColor)
{
  auto material = anari::newObject<anari::Material>(device, "physicallyBased");
  anari::setParameter(device, material, "baseColor", baseColor);
  anari::setParameter(device, material, "metallic", 0.f);
  anari::setParameter(device, material, "roughness", 0.8f);
  anari::commitParameters(device, material);
  auto surface = anari::newObject<anari::Surface>(device);
  anari::setAndReleaseParameter(device, surface, "geometry", geom);
  anari::setAndReleaseParameter(device, surface, "material", material);
  anari::commitParameters(device, surface);
  return surface;
}

static anari::World generateScene(anari::Device device)
{
  auto sphereGeom = anari::newObject<anari::Geometry>(device, "sphere");
  {
    auto arr = anari::newArray1D(device, ANARI_FLOAT32_VEC3, 1);
    *anari::map<vec3>(device, arr) = vec3{0.f, 0.f, 0.f};
    anari::unmap(device, arr);
    anari::setAndReleaseParameter(device, sphereGeom, "vertex.position", arr);
    anari::setParameter(device, sphereGeom, "radius", 0.9f);
    anari::commitParameters(device, sphereGeom);
  }

  auto surface = pbrSurface(device, sphereGeom, vec3{0.75f, 0.4f, 0.25f});

  auto light = anari::newObject<anari::Light>(device, "directional");
  anari::setParameter(device, light, "direction", vec3{0.3f, -0.4f, 1.f});
  anari::setParameter(device, light, "irradiance", 3.f);
  anari::commitParameters(device, light);

  auto world = anari::newObject<anari::World>(device);
  anari::setParameterArray1D(device, world, "surface", &surface, 1);
  anari::setParameterArray1D(device, world, "light", &light, 1);
  anari::release(device, surface);
  anari::release(device, light);
  anari::commitParameters(device, world);
  return world;
}

static std::vector<vec4> renderWith(anari::Device device,
    anari::World world,
    anari::Camera camera,
    const char *subtype,
    uvec2 size)
{
  auto renderer = anari::newObject<anari::Renderer>(device, subtype);
  anari::setParameter(device, renderer, "background", vec4{0.f, 0.f, 0.f, 1.f});
  anari::setParameter(device, renderer, "ambientRadiance", 0.f);
  anari::setParameter(device, renderer, "pixelSamples", 512);
  anari::commitParameters(device, renderer);

  auto frame = anari::newObject<anari::Frame>(device);
  anari::setParameter(device, frame, "size", size);
  anari::setParameter(device, frame, "channel.color", ANARI_FLOAT32_VEC4);
  anari::setParameter(device, frame, "world", world);
  anari::setParameter(device, frame, "camera", camera);
  anari::setParameter(device, frame, "renderer", renderer);
  anari::commitParameters(device, frame);

  anari::render(device, frame);
  anari::wait(device, frame);

  auto fb = anari::map<vec4>(device, frame, "channel.color");
  std::vector<vec4> out(fb.data, fb.data + size_t(fb.width) * fb.height);
  anari::unmap(device, frame, "channel.color");
  anari::release(device, renderer);
  anari::release(device, frame);
  return out;
}

int main()
{
  auto device = makeVisRTXDevice(statusFunc);
  auto world = generateScene(device);

  const uvec2 imageSize = {200, 200};
  auto camera = anari::newObject<anari::Camera>(device, "perspective");
  anari::setParameter(device, camera, "position", vec3{0.f, 0.f, -2.5f});
  anari::setParameter(device, camera, "direction", vec3{0.f, 0.f, 1.f});
  anari::setParameter(device, camera, "up", vec3{0.f, 1.f, 0.f});
  anari::setParameter(
      device, camera, "aspect", imageSize[0] / float(imageSize[1]));
  anari::commitParameters(device, camera);

  const auto ref = renderWith(device, world, camera, "interactive", imageSize);
  const auto wav = renderWith(device, world, camera, "wavefront", imageSize);

  anari::release(device, camera);
  anari::release(device, world);
  anari::release(device, device);

  // Parity = agreement of the pixel-AVERAGED image (the estimator mean), which
  // cancels the independent per-pixel Monte-Carlo noise of both renderers over
  // ~1e4 pixels and leaves only systematic bias (a missing 1/pi, a double-
  // counted term, a wrong lobe weight). The per-pixel |Δ| is reported too as a
  // noise indicator, but the assertion is on the bias.
  double refC[3] = {0, 0, 0}, wavC[3] = {0, 0, 0}, diffSum = 0.0, refSum = 0.0;
  size_t lit = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    if (ref[i][0] + ref[i][1] + ref[i][2] < 1e-3f)
      continue; // background
    ++lit;
    for (int c = 0; c < 3; ++c) {
      refC[c] += ref[i][c];
      wavC[c] += wav[i][c];
      diffSum += std::fabs(double(ref[i][c]) - double(wav[i][c]));
      refSum += double(ref[i][c]);
    }
  }

  double biasNum = 0.0, biasDen = 0.0;
  for (int c = 0; c < 3; ++c) {
    biasNum += std::fabs(refC[c] - wavC[c]);
    biasDen += refC[c];
  }
  const double bias = biasDen > 0.0 ? biasNum / biasDen : 1.0;
  const double perPixelNoise = refSum > 0.0 ? diffSum / refSum : 1.0;

  printf(
      "wavefront/interactive parity: %zu lit px, mean-image bias=%.2f%%, "
      "per-pixel noise=%.1f%%\n",
      lit,
      100.0 * bias,
      100.0 * perPixelNoise);

  if (lit < 1000) {
    fprintf(stderr, "FAIL: scene did not render (too few lit pixels)\n");
    return 1;
  }
  // Same BRDF, both unbiased -> the averaged images must agree tightly. 3%
  // absorbs residual noise-of-the-mean and minor sampling-strategy differences
  // while still catching a real systematic divergence.
  if (bias > 0.03) {
    fprintf(stderr,
        "FAIL: wavefront and interactive PBR means disagree by %.2f%% — the "
        "shared BRDF is not producing matching results\n",
        100.0 * bias);
    return 1;
  }

  printf("PASS\n");
  return 0;
}
