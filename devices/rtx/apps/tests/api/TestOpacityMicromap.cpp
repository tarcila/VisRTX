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

// Opacity Micromaps (ADR 0009) are a pure traversal accelerator: the
// conservative bake emits only TRANSPARENT (alpha provably 0) and
// UNKNOWN_OPAQUE states, so every surviving hit runs today's exact shading
// paths. Pins:
//   1. omm=true renders bit-identically to omm=false for mask AND
//      binary-alpha blend cutouts (Quality + Interactive).
//   2. The bake actually engages (debug message observed) — guards against
//      the accelerator silently disabling itself.
//   3. Cutout holes really show the backdrop (the micromap doesn't
//      over-classify to transparent).

// anari_cpp
#include <anari/anari_cpp/ext/std.h>
#include <anari/anari_cpp.hpp>
// VisRTX
#include <anari/ext/visrtx/makeVisRTXDevice.h>
// std
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using uvec2 = std::array<unsigned int, 2>;
using vec2 = std::array<float, 2>;
using vec3 = std::array<float, 3>;
using vec4 = std::array<float, 4>;

static std::atomic<int> g_bakeMessages{0};

static void statusFunc(const void *,
    ANARIDevice,
    ANARIObject source,
    ANARIDataType,
    ANARIStatusSeverity severity,
    ANARIStatusCode,
    const char *message)
{
  if (message && std::strstr(message, "OpacityMicromap baked"))
    g_bakeMessages++;
  if (severity == ANARI_SEVERITY_FATAL_ERROR) {
    fprintf(stderr, "[FATAL][%p] %s\n", source, message);
    std::exit(1);
  } else if (severity == ANARI_SEVERITY_ERROR)
    fprintf(stderr, "[ERROR][%p] %s\n", source, message);
}

static constexpr uvec2 IMAGE_SIZE = {128, 128};
static constexpr int PIXEL_SAMPLES = 16;
static constexpr uint32_t TEX_SIZE = 32; // alpha texture resolution

// Camera-filling quad at z=0 with a nearest-filtered leaf-card-style alpha
// texture: left half transparent, right half opaque (alpha strictly 0 or 1 →
// the with/without-OMM shading factor sets are identical, so frames must
// match bitwise). Contiguous transparent regions are what the conservative
// bake is designed to carve — per-texel noise legitimately stays unknown.
static anari::Surface makeCutoutQuad(ANARIDevice device, const char *alphaMode)
{
  const std::array<vec3, 4> pos = {vec3{-2.f, -2.f, 0.f},
      vec3{2.f, -2.f, 0.f},
      vec3{2.f, 2.f, 0.f},
      vec3{-2.f, 2.f, 0.f}};
  const std::array<vec2, 4> uv = {
      vec2{0.f, 0.f}, vec2{1.f, 0.f}, vec2{1.f, 1.f}, vec2{0.f, 1.f}};
  const std::array<std::array<unsigned, 3>, 2> idx = {
      std::array<unsigned, 3>{0, 1, 2}, std::array<unsigned, 3>{0, 2, 3}};

  auto geom = anari::newObject<anari::Geometry>(device, "triangle");
  anari::setParameterArray1D(device, geom, "vertex.position", pos.data(), 4);
  anari::setParameterArray1D(device, geom, "vertex.attribute0", uv.data(), 4);
  anari::setParameterArray1D(device, geom, "primitive.index", idx.data(), 2);
  anari::commitParameters(device, geom);

  std::vector<vec4> img(TEX_SIZE * TEX_SIZE);
  for (uint32_t y = 0; y < TEX_SIZE; y++) {
    for (uint32_t x = 0; x < TEX_SIZE; x++)
      img[y * TEX_SIZE + x] = {0.1f, 0.8f, 0.2f, x < TEX_SIZE / 2 ? 0.f : 1.f};
  }
  auto sampler = anari::newObject<anari::Sampler>(device, "image2D");
  anari::setParameter(device, sampler, "inAttribute", "attribute0");
  anari::setParameter(device, sampler, "filter", "nearest");
  anari::setParameterArray2D(
      device, sampler, "image", img.data(), TEX_SIZE, TEX_SIZE);
  anari::commitParameters(device, sampler);

  auto mat = anari::newObject<anari::Material>(device, "physicallyBased");
  anari::setAndReleaseParameter(device, mat, "baseColor", sampler);
  anari::setParameter(device, mat, "roughness", 1.f);
  anari::setParameter(device, mat, "metallic", 0.f);
  anari::setParameter(device, mat, "alphaMode", alphaMode);
  anari::setParameter(device, mat, "alphaCutoff", 0.5f);
  anari::commitParameters(device, mat);

  auto surface = anari::newObject<anari::Surface>(device);
  anari::setAndReleaseParameter(device, surface, "geometry", geom);
  anari::setAndReleaseParameter(device, surface, "material", mat);
  anari::commitParameters(device, surface);
  return surface;
}

// Opaque backdrop behind the cutout so holes are visibly red.
static anari::Surface makeBackdrop(ANARIDevice device)
{
  const std::array<vec3, 4> pos = {vec3{-4.f, -4.f, 1.f},
      vec3{4.f, -4.f, 1.f},
      vec3{4.f, 4.f, 1.f},
      vec3{-4.f, 4.f, 1.f}};
  const std::array<std::array<unsigned, 3>, 2> idx = {
      std::array<unsigned, 3>{0, 1, 2}, std::array<unsigned, 3>{0, 2, 3}};

  auto geom = anari::newObject<anari::Geometry>(device, "triangle");
  anari::setParameterArray1D(device, geom, "vertex.position", pos.data(), 4);
  anari::setParameterArray1D(device, geom, "primitive.index", idx.data(), 2);
  anari::commitParameters(device, geom);

  auto mat = anari::newObject<anari::Material>(device, "matte");
  anari::setParameter(device, mat, "color", vec3{0.8f, 0.1f, 0.1f});
  anari::commitParameters(device, mat);

  auto surface = anari::newObject<anari::Surface>(device);
  anari::setAndReleaseParameter(device, surface, "geometry", geom);
  anari::setAndReleaseParameter(device, surface, "material", mat);
  anari::commitParameters(device, surface);
  return surface;
}

static std::vector<vec4> renderFrame(
    ANARIDevice device, anari::World world, const char *rendererType)
{
  auto camera = anari::newObject<anari::Camera>(device, "perspective");
  anari::setParameter(device, camera, "position", vec3{0.f, 0.f, -3.f});
  anari::setParameter(device, camera, "direction", vec3{0.f, 0.f, 1.f});
  anari::setParameter(device, camera, "up", vec3{0.f, 1.f, 0.f});
  anari::setParameter(
      device, camera, "aspect", IMAGE_SIZE[0] / float(IMAGE_SIZE[1]));
  anari::commitParameters(device, camera);

  auto renderer = anari::newObject<anari::Renderer>(device, rendererType);
  anari::setParameter(device, renderer, "background", vec4{0.f, 0.f, 0.f, 1.f});
  // Ambient stays off: ambient-bounce rays trace with DISABLE_ANYHIT, where
  // OMM-culled cutouts legitimately diverge from the non-OMM render (ADR 0009
  // determinism scope). This test pins the guaranteed domain — primary
  // visibility and shadows.
  anari::setParameter(device, renderer, "ambientRadiance", 0.f);
  anari::setParameter(device, renderer, "pixelSamples", PIXEL_SAMPLES);
  anari::setParameter(device, renderer, "fireflyFilterMode", "none");
  anari::setParameter(device, renderer, "denoise", false);
  anari::commitParameters(device, renderer);

  auto makeFrame = [&]() {
    auto frame = anari::newObject<anari::Frame>(device);
    anari::setParameter(device, frame, "size", IMAGE_SIZE);
    anari::setParameter(device, frame, "channel.color", ANARI_FLOAT32_VEC4);
    anari::setParameter(device, frame, "world", world);
    anari::setParameter(device, frame, "camera", camera);
    anari::setParameter(device, frame, "renderer", renderer);
    anari::commitParameters(device, frame);
    return frame;
  };

  // OMM bakes are bake-on-stable (deferred one rebuild pass); a warm-up
  // render lets them settle and attach before the measured frame.
  {
    auto warmup = makeFrame();
    anari::render(device, warmup);
    anari::wait(device, warmup);
    anari::render(device, warmup);
    anari::wait(device, warmup);
    anari::release(device, warmup);
  }

  auto frame = makeFrame();
  anari::release(device, camera);
  anari::release(device, renderer);
  anari::render(device, frame);
  anari::wait(device, frame);

  auto fb = anari::map<vec4>(device, frame, "channel.color");
  std::vector<vec4> pixels(fb.data, fb.data + IMAGE_SIZE[0] * IMAGE_SIZE[1]);
  anari::unmap(device, frame, "channel.color");
  anari::release(device, frame);
  return pixels;
}

static void setOmmEnabled(ANARIDevice device, bool enabled)
{
  anari::setParameter(device, device, "omm", enabled);
  anari::commitParameters(device, device);
}

int main()
{
  auto device = makeVisRTXDevice(statusFunc);

  bool ok = true;
  for (const char *alphaMode : {"mask", "blend"}) {
    auto cutout = makeCutoutQuad(device, alphaMode);
    auto backdrop = makeBackdrop(device);
    const std::array<anari::Surface, 2> surfaces = {cutout, backdrop};

    auto world = anari::newObject<anari::World>(device);
    anari::setParameterArray1D(device, world, "surface", surfaces.data(), 2);
    anari::release(device, cutout);
    anari::release(device, backdrop);

    auto light = anari::newObject<anari::Light>(device, "directional");
    anari::setParameter(device, light, "direction", vec3{0.f, 0.f, 1.f});
    anari::setParameter(device, light, "irradiance", 1.f);
    anari::commitParameters(device, light);
    anari::setParameterArray1D(device, world, "light", &light, 1);
    anari::release(device, light);
    anari::commitParameters(device, world);

    // Content-dedup means one bake per world: the second renderer adopts the
    // cached micromap without re-baking, so the counter spans both renderers.
    g_bakeMessages = 0;
    for (const char *renderer : {"quality", "interactive"}) {
      setOmmEnabled(device, false);
      const auto reference = renderFrame(device, world, renderer);

      setOmmEnabled(device, true);
      const auto withOmm = renderFrame(device, world, renderer);

      double maxDiff = 0.0;
      double redMean = 0.0, greenMean = 0.0;
      for (size_t i = 0; i < reference.size(); i++) {
        for (int c = 0; c < 4; c++) {
          maxDiff = std::max(maxDiff,
              double(std::fabs(reference[i][c] - withOmm[i][c])));
        }
        redMean += withOmm[i][0];
        greenMean += withOmm[i][1];
      }
      redMean /= double(reference.size());
      greenMean /= double(reference.size());

      printf("%s/%s: maxDiff=%g bakes=%d redMean=%f greenMean=%f\n",
          alphaMode,
          renderer,
          maxDiff,
          g_bakeMessages.load(),
          redMean,
          greenMean);

      if (g_bakeMessages.load() == 0) {
        fprintf(stderr,
            "FAIL[%s/%s]: no OpacityMicromap bake happened yet — accelerator "
            "silently off\n",
            alphaMode,
            renderer);
        ok = false;
      }
      if (maxDiff != 0.0) {
        fprintf(stderr,
            "FAIL[%s/%s]: OMM changed the image (maxDiff=%g)\n",
            alphaMode,
            renderer,
            maxDiff);
        ok = false;
      }
      // Both the cutout (green) and the backdrop-through-holes (red) must be
      // present: catches both over-transparent and over-opaque bakes.
      if (redMean < 0.01 || greenMean < 0.01) {
        fprintf(stderr,
            "FAIL[%s/%s]: cutout pattern wrong (redMean=%f greenMean=%f)\n",
            alphaMode,
            renderer,
            redMean,
            greenMean);
        ok = false;
      }
    }

    anari::release(device, world);
  }

  anari::release(device, device);

  if (!ok)
    return 1;
  printf("opacity micromap semantics passed\n");
  return 0;
}
