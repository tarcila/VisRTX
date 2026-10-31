// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

// Public ANARI regression. A full-coverage quad has a linear vertex-color
// gradient, so its albedo directly observes the camera's subpixel coordinates.
// This checks actual raygen ordinals, mapping and normalization, not an image
// comparison between different PCG-dependent batching schedules.
#include <anari/anari_cpp/ext/std.h>
#include <anari/ext/visrtx/makeVisRTXDevice.h>
#include <anari/anari_cpp.hpp>
#include "gpu/owenSobol.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using Vec3 = std::array<float, 3>;
using Vec4 = std::array<float, 4>;
using UVec2 = std::array<unsigned, 2>;
constexpr unsigned SIZE = 8;

static void status(const void *,
    ANARIDevice,
    ANARIObject,
    ANARIDataType,
    ANARIStatusSeverity severity,
    ANARIStatusCode,
    const char *message)
{
  if (severity <= ANARI_SEVERITY_ERROR) {
    std::fprintf(stderr, "%s\n", message);
    std::exit(1);
  }
}

struct Image
{
  std::vector<Vec3> albedo;
  std::vector<Vec4> color;
  bool operator==(const Image &other) const
  {
    return albedo == other.albedo && color == other.color;
  }
};

struct Scene
{
  anari::Device d;
  anari::World world;
  anari::Camera camera;
  anari::Renderer renderer;

  Scene(anari::Device device, const char *subtype, int spp, bool checkerboard)
      : d(device)
  {
    const std::array<Vec3, 4> positions = {Vec3{-1.f, -1.f, 0.f},
        Vec3{1.f, -1.f, 0.f},
        Vec3{1.f, 1.f, 0.f},
        Vec3{-1.f, 1.f, 0.f}};
    const std::array<Vec4, 4> colors = {Vec4{0.f, 0.f, .25f, 1.f},
        Vec4{1.f, 0.f, .25f, 1.f},
        Vec4{1.f, 1.f, .25f, 1.f},
        Vec4{0.f, 1.f, .25f, 1.f}};
    auto geometry = anari::newObject<anari::Geometry>(d, "quad");
    anari::setParameterArray1D(
        d, geometry, "vertex.position", positions.data(), 4);
    anari::setParameterArray1D(d, geometry, "vertex.color", colors.data(), 4);
    anari::commitParameters(d, geometry);
    auto material = anari::newObject<anari::Material>(d, "matte");
    anari::setParameter(d, material, "color", "color");
    anari::commitParameters(d, material);
    auto surface = anari::newObject<anari::Surface>(d);
    anari::setAndReleaseParameter(d, surface, "geometry", geometry);
    anari::setAndReleaseParameter(d, surface, "material", material);
    anari::commitParameters(d, surface);
    world = anari::newObject<anari::World>(d);
    anari::setParameterArray1D(d, world, "surface", &surface, 1);
    anari::release(d, surface);
    anari::commitParameters(d, world);
    camera = anari::newObject<anari::Camera>(d, "orthographic");
    anari::setParameter(d, camera, "position", Vec3{0.f, 0.f, 1.f});
    anari::setParameter(d, camera, "direction", Vec3{0.f, 0.f, -1.f});
    anari::setParameter(d, camera, "up", Vec3{0.f, 1.f, 0.f});
    anari::setParameter(d, camera, "height", 2.f);
    anari::setParameter(d, camera, "aspect", 1.f);
    anari::commitParameters(d, camera);
    renderer = anari::newObject<anari::Renderer>(d, subtype);
    anari::setParameter(d, renderer, "pixelSamples", spp);
    anari::setParameter(d, renderer, "checkerboarding", checkerboard);
    anari::setParameter(d, renderer, "denoise", false);
    anari::setParameter(d, renderer, "fireflyFilterMode", "none");
    anari::setParameter(d, renderer, "ambientRadiance", 1.f);
    anari::setParameter(d, renderer, "ambientSamples", 0);
    anari::setParameter(d, renderer, "sampleLimit", 0);
    anari::commitParameters(d, renderer);
  }
  ~Scene()
  {
    anari::release(d, renderer);
    anari::release(d, camera);
    anari::release(d, world);
  }
  anari::Frame newFrame()
  {
    auto f = anari::newObject<anari::Frame>(d);
    anari::setParameter(d, f, "size", UVec2{SIZE, SIZE});
    anari::setParameter(d, f, "channel.color", ANARI_FLOAT32_VEC4);
    anari::setParameter(d, f, "channel.albedo", ANARI_FLOAT32_VEC3);
    anari::setParameter(d, f, "world", world);
    anari::setParameter(d, f, "camera", camera);
    anari::setParameter(d, f, "renderer", renderer);
    // Explicit public restart avoids unrelated deferred scene finalizations.
    anari::setParameter(d, f, "accumulationVersion", uint64_t(1));
    anari::commitParameters(d, f);
    return f;
  }
  Image render(anari::Frame f)
  {
    anari::render(d, f);
    anari::wait(d, f);
    Image image;
    auto a = anari::map<Vec3>(d, f, "channel.albedo");
    auto c = anari::map<Vec4>(d, f, "channel.color");
    if (!a.data || !c.data || a.width != SIZE || a.height != SIZE
        || c.width != SIZE || c.height != SIZE) {
      std::fprintf(stderr, "missing or incorrectly sized image channel\n");
      std::exit(1);
    }
    image.albedo.assign(a.data, a.data + SIZE * SIZE);
    image.color.assign(c.data, c.data + SIZE * SIZE);
    anari::unmap(d, f, "channel.albedo");
    anari::unmap(d, f, "channel.color");
    return image;
  }
};

// The expected *prefix* enumerates indices from zero directly. It deliberately
// does not call sampleOrdinal or reproduce the launch's addressing expression.
static Vec3 expected(unsigned x, unsigned y, int count)
{
  Vec3 result{0.f, 0.f, .25f};
  const auto seed = visrtx::owenPixelSeed(x, y);
  for (int ordinal = 0; ordinal < count; ++ordinal) {
    const auto jitter = visrtx::owenSobolCamera(uint32_t(ordinal), seed);
    result[0] += (x + (ordinal == 0 ? .5f : jitter.x)) / SIZE;
    result[1] += (y + (ordinal == 0 ? .5f : jitter.y)) / SIZE;
  }
  result[0] /= count;
  result[1] /= count;
  return result;
}

static bool run(
    anari::Device d, const char *subtype, int spp, bool checkerboard)
{
  Scene scene(d, subtype, spp, checkerboard);
  auto frame = scene.newFrame();
  std::vector<Image> reference;
  // Frame's initial reset precedes its first version latch. Observe the fresh
  // image, then let the public version mechanism latch before recording the
  // progressive trace. Do not change Frame's inherited reset lifecycle here.
  Image fresh = scene.render(frame);
  bool pending = false;
  anari::getProperty(d, frame, "nextFrameReset", pending, ANARI_WAIT);
  bool ok = pending;
  const int launches = checkerboard ? 12 : 3;
  for (int run = 0; run < 3; ++run) {
    if (run == 1) {
      anari::setParameter(d, frame, "accumulationVersion", uint64_t(2));
      anari::commitParameters(d, frame);
    } else if (run == 2) {
      anari::release(d, frame);
      frame = scene.newFrame();
      fresh = scene.render(frame);
      anari::getProperty(d, frame, "nextFrameReset", pending, ANARI_WAIT);
      ok &= pending;
    }
    for (int launch = 0; launch < launches; ++launch) {
      const Image actual = scene.render(frame);
      int samples = -1;
      if (!anari::getProperty(d, frame, "numSamples", samples, ANARI_WAIT)
          || samples
              != (checkerboard ? (launch + 1) / 4 : (launch + 1) * spp)) {
        std::fprintf(stderr,
            "%s spp=%d checker=%d launch=%d: wrong sample count %d\n",
            subtype,
            spp,
            checkerboard,
            launch,
            samples);
        ok = false;
      }
      for (unsigned y = 0; y < SIZE; ++y) {
        for (unsigned x = 0; x < SIZE; ++x) {
          int count = (launch + 1) * spp;
          const int divisor = checkerboard ? launch / 4 + 1 : launch * spp + 1;
          if (checkerboard) {
            const int tile = (x & 1) | ((y & 1) << 1);
            count = launch / 4 + int(tile <= launch % 4);
          }
          // The mapped albedo channel retains its existing pre-launch
          // (completedSamples+1) divisor, including sparse checkerboard tiles.
          // Color compositing has its own normalization/initial tile redirect.
          // Preserve both; addressing must not repair counter normalization.
          Vec3 want = count ? expected(x, y, count) : Vec3{0.f, 0.f, 0.f};
          for (float &v : want)
            v *= float(count) / divisor;
          for (int c = 0; c < 3; ++c) {
            const float got = actual.albedo[y * SIZE + x][c];
            if (!std::isfinite(got) || std::abs(got - want[c]) > 2e-6f) {
              std::fprintf(stderr,
                  "%s spp=%d checker=%d run=%d launch=%d pixel=(%u,%u) channel=%d: %.9g != %.9g\n",
                  subtype,
                  spp,
                  checkerboard,
                  run,
                  launch,
                  x,
                  y,
                  c,
                  got,
                  want[c]);
              anari::release(d, frame);
              return false;
            }
          }
        }
      }
      if (launch == 0 && run != 1 && !(fresh == actual)) {
        std::fprintf(stderr,
            "%s fresh first launch differs from reset ordinal zero\n",
            subtype);
        ok = false;
      }
      if (run == 0)
        reference.push_back(actual);
      else if (!(actual == reference[launch])) {
        std::fprintf(stderr,
            "%s spp=%d checker=%d run=%d launch=%d not repeatable\n",
            subtype,
            spp,
            checkerboard,
            run,
            launch);
        ok = false;
      }
    }
  }
  anari::release(d, frame);
  std::printf("%s spp=%d checker=%d: %s\n",
      subtype,
      spp,
      checkerboard,
      ok ? "ok" : "FAIL");
  return ok;
}

int main()
{
  auto device = makeVisRTXDevice(status);
  bool ok = true;
  for (const char *subtype : {"quality", "interactive"}) {
    for (int spp : {1, 2, 3, 4, 8})
      ok &= run(device, subtype, spp, false);
    // Requested spp=8 must be forced to effective spp=1 by the real renderer.
    ok &= run(device, subtype, 8, true);
  }
  anari::release(device, device);
  return ok ? 0 : 1;
}
