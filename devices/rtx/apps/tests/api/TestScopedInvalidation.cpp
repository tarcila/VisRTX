/*
 * Copyright (c) 2019-2026 NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

// Scoped invalidation regressions (ADR 0009): detect per-Frame accumulation
// resets through "numSamples", and BVH/light-set work through debug counters.
// Shared and independent cameras exercise per-consumer dependency stamps.

#include <anari/anari_cpp/ext/std.h>
#include <anari/ext/visrtx/makeVisRTXDevice.h>
#include <algorithm>
#include <anari/anari_cpp.hpp>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

using float3 = std::array<float, 3>;
using uint2 = std::array<uint32_t, 2>;

static int g_failures = 0;

#define CHECK(cond, ...)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);                \
      std::fprintf(stderr, __VA_ARGS__);                                       \
      std::fprintf(stderr, "\n");                                              \
      g_failures++;                                                            \
    }                                                                          \
  } while (0)

static void statusFunc(const void *,
    ANARIDevice,
    ANARIObject,
    ANARIDataType,
    ANARIStatusSeverity severity,
    ANARIStatusCode,
    const char *message)
{
  if (severity <= ANARI_SEVERITY_ERROR) {
    std::fprintf(stderr, "[ANARI] %s\n", message);
    g_failures++;
  }
}

namespace {

// Rebuild counters read back from the device.
struct BuildCounts
{
  uint32_t groupBLAS{0};
  uint32_t groupLightSet{0};
  uint32_t worldTLAS{0};
  uint32_t worldLightSet{0};
};

// Two-Frame scene derived from _tmp/repro.cpp.
struct Harness
{
  anari::Device d{nullptr};
  anari::Geometry geom{nullptr};
  anari::Array1D vertices{nullptr};
  anari::Material mat{nullptr};
  anari::Surface surf{nullptr};
  anari::Light light{nullptr};
  anari::Group group{nullptr};
  anari::Instance inst{nullptr};
  anari::World world{nullptr};
  anari::Camera cam{nullptr};
  anari::Renderer ren{nullptr};
  std::array<anari::Frame, 2> frames{};
  std::array<int32_t, 2> lastSamples{-1, -1};

  explicit Harness(anari::Device dev) : d(dev)
  {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    std::vector<float3> pos(256);
    for (auto &p : pos)
      p = {u(rng), u(rng), u(rng)};
    geom = anari::newObject<anari::Geometry>(d, "sphere");
    vertices = anariNewArray1D(
        d, nullptr, nullptr, nullptr, ANARI_FLOAT32_VEC3, pos.size());
    auto *mapped = static_cast<float3 *>(anariMapArray(d, vertices));
    std::copy(pos.begin(), pos.end(), mapped);
    anariUnmapArray(d, vertices);
    anari::setParameter(d, geom, "vertex.position", vertices);
    anari::setParameter(d, geom, "radius", 0.05f);
    anari::commitParameters(d, geom);
    mat = anari::newObject<anari::Material>(d, "matte");
    anari::setParameter(d, mat, "color", float3{0.8f, 0.4f, 0.1f});
    anari::commitParameters(d, mat);
    surf = anari::newObject<anari::Surface>(d);
    anari::setParameter(d, surf, "geometry", geom);
    anari::setParameter(d, surf, "material", mat);
    anari::commitParameters(d, surf);
    light = anari::newObject<anari::Light>(d, "directional");
    anari::setParameter(d, light, "direction", float3{-1.f, -1.f, -1.f});
    anari::commitParameters(d, light);

    // Explicit Group/Instance so the Group's counters are reachable.
    group = anari::newObject<anari::Group>(d);
    anari::setParameterArray1D(d, group, "surface", &surf, 1);
    anari::setParameterArray1D(d, group, "light", &light, 1);
    anari::commitParameters(d, group);
    inst = anari::newObject<anari::Instance>(d, "transform");
    anari::setParameter(d, inst, "group", group);
    anari::commitParameters(d, inst);
    world = anari::newObject<anari::World>(d);
    anari::setParameterArray1D(d, world, "instance", &inst, 1);
    anari::commitParameters(d, world);

    const uint2 size = {64, 48};
    cam = anari::newObject<anari::Camera>(d, "perspective");
    anari::setParameter(d, cam, "position", float3{0.f, 0.f, 3.f});
    anari::setParameter(d, cam, "direction", float3{0.f, 0.f, -1.f});
    anari::setParameter(d, cam, "up", float3{0.f, 1.f, 0.f});
    anari::setParameter(d, cam, "aspect", size[0] / float(size[1]));
    anari::commitParameters(d, cam);
    ren = anari::newObject<anari::Renderer>(d, "default");
    anari::setParameter(d, ren, "ambientRadiance", 0.3f);
    anari::commitParameters(d, ren);

    for (auto &f : frames) {
      f = anari::newObject<anari::Frame>(d);
      anari::setParameter(d, f, "size", size);
      anari::setParameter(d, f, "channel.color", ANARI_UFIXED8_RGBA_SRGB);
      anari::setParameter(d, f, "camera", cam);
      anari::setParameter(d, f, "renderer", ren);
      anari::setParameter(d, f, "world", world);
      anari::setParameter(d, f, "accumulation", true);
      anari::commitParameters(d, f);
    }
  }

  ~Harness()
  {
    for (auto f : frames)
      anari::release(d, f);
    anari::release(d, ren);
    anari::release(d, cam);
    anari::release(d, world);
    anari::release(d, inst);
    anari::release(d, group);
    anari::release(d, light);
    anari::release(d, surf);
    anari::release(d, mat);
    anari::release(d, geom);
    anari::release(d, vertices);
  }

  // Renders one sample on frame 'i'; returns true if it accumulated (its
  // numSamples rose since the previous render), false on an Accumulation
  // Reset.
  bool renderAndCheck(int i)
  {
    anari::render(d, frames[i]);
    anari::wait(d, frames[i]);
    int32_t n = -1;
    anari::getProperty(d, frames[i], "numSamples", n, ANARI_WAIT);
    const bool accumulated = n > lastSamples[i];
    lastSamples[i] = n;
    return accumulated;
  }

  BuildCounts counts() const
  {
    BuildCounts c;
    anari::getProperty(d, group, "numBLASBuilds", c.groupBLAS, ANARI_NO_WAIT);
    anari::getProperty(
        d, group, "numLightSetBuilds", c.groupLightSet, ANARI_NO_WAIT);
    anari::getProperty(d, world, "numTLASBuilds", c.worldTLAS, ANARI_NO_WAIT);
    anari::getProperty(
        d, world, "numLightSetBuilds", c.worldLightSet, ANARI_NO_WAIT);
    return c;
  }
};

void testBaselineTwoFramesAccumulate(anari::Device d)
{
  Harness h(d);

  // One warm-up round must suffice even with shared dependencies.
  h.renderAndCheck(0);
  h.renderAndCheck(1);
  const auto built = h.counts();
  CHECK(built.groupBLAS >= 1, "initial BLAS build not counted");
  CHECK(built.groupLightSet >= 1, "initial light-set build not counted");
  CHECK(built.worldTLAS >= 1, "initial TLAS build not counted");

  for (int s = 0; s < 6; s++) {
    for (int i = 0; i < 2; i++) {
      CHECK(h.renderAndCheck(i),
          "frame %d reset at step %d (numSamples=%d)",
          i,
          s,
          h.lastSamples[i]);
    }
  }

  const auto after = h.counts();
  CHECK(after.groupBLAS == built.groupBLAS,
      "BLAS rebuilt with no scene change (%u -> %u)",
      built.groupBLAS,
      after.groupBLAS);
  CHECK(after.groupLightSet == built.groupLightSet,
      "light set rebuilt with no scene change (%u -> %u)",
      built.groupLightSet,
      after.groupLightSet);
  CHECK(after.worldTLAS == built.worldTLAS,
      "TLAS rebuilt with no scene change (%u -> %u)",
      built.worldTLAS,
      after.worldTLAS);
}

void testResetDetected(anari::Device d)
{
  // Harness self-check: a camera re-commit is a documented Accumulation
  // Reset; the harness must see it on both frames.
  Harness h(d);
  for (int s = 0; s < 3; s++) {
    h.renderAndCheck(0);
    h.renderAndCheck(1);
  }
  anari::setParameter(d, h.cam, "position", float3{0.f, 0.f, 3.5f});
  anari::commitParameters(d, h.cam);
  CHECK(!h.renderAndCheck(0), "camera commit did not reset frame 0");
  CHECK(!h.renderAndCheck(1), "camera commit did not reset frame 1");
}

void completionCallback(const void *userData, ANARIDevice, ANARIFrame)
{
  ++*static_cast<int *>(const_cast<void *>(userData));
}

void testFrameDependencies(anari::Device d)
{
  Harness h(d);
  auto otherCam = anari::newObject<anari::Camera>(d, "perspective");
  anari::setParameter(d, otherCam, "position", float3{0.f, 0.f, 3.f});
  anari::setParameter(d, otherCam, "direction", float3{0.f, 0.f, -1.f});
  anari::commitParameters(d, otherCam);
  anari::setParameter(d, h.frames[1], "camera", otherCam);
  anari::commitParameters(d, h.frames[1]);
  for (int s = 0; s < 3; ++s) {
    h.renderAndCheck(0);
    h.renderAndCheck(1);
  }
  const auto before = h.counts();

  auto unrelated = anari::newObject<anari::Camera>(d, "orthographic");
  anari::commitParameters(d, unrelated);
  CHECK(h.renderAndCheck(0), "unrelated camera reset A");
  CHECK(h.renderAndCheck(1), "unrelated camera reset B");
  anari::release(d, unrelated);

  auto c = anari::newObject<anari::Frame>(d);
  anari::setParameter(d, c, "size", uint2{64, 48});
  anari::setParameter(d, c, "camera", h.cam);
  anari::setParameter(d, c, "renderer", h.ren);
  anari::setParameter(d, c, "world", h.world);
  anari::commitParameters(d, c);
  anari::render(d, c);
  anari::wait(d, c);
  CHECK(h.renderAndCheck(0), "new shared frame C reset A");
  CHECK(h.renderAndCheck(1), "new shared frame C reset B");
  anari::release(d, c);

  anari::setParameter(d, h.cam, "position", float3{0.f, 0.f, 3.5f});
  anari::commitParameters(d, h.cam);
  CHECK(!h.renderAndCheck(0), "changed camera did not reset A");
  CHECK(h.renderAndCheck(1), "A's camera reset B");
  CHECK(h.renderAndCheck(0), "A failed to resume accumulation");
  anari::setParameter(d, h.cam, "position", float3{0.f, 0.f, 3.5f});
  anari::commitParameters(d, h.cam);
  CHECK(h.renderAndCheck(0), "unchanged camera reset A");
  CHECK(h.renderAndCheck(1), "unchanged camera reset B");

  int callbacks = 0;
  ANARIFrameCompletionCallback callback = completionCallback;
  anariSetParameter(d,
      h.frames[0],
      "frameCompletionCallback",
      ANARI_FRAME_COMPLETION_CALLBACK,
      &callback);
  anariSetParameter(d,
      h.frames[0],
      "frameCompletionCallbackUserData",
      ANARI_VOID_POINTER,
      &callbacks);
  anari::commitParameters(d, h.frames[0]);
  CHECK(h.renderAndCheck(0), "callback-only commit reset A");
  CHECK(callbacks == 1, "completion callback did not run");
  CHECK(h.renderAndCheck(1), "callback-only commit reset B");

  anari::setParameter(d, h.frames[0], "size", uint2{80, 60});
  anari::commitParameters(d, h.frames[0]);
  CHECK(!h.renderAndCheck(0), "size change did not reset A");
  CHECK(h.renderAndCheck(1), "size change reset B");
  CHECK(h.renderAndCheck(0), "A did not accumulate after resize");
  const auto after = h.counts();
  CHECK(after.groupBLAS == before.groupBLAS, "frame/camera edit rebuilt BLAS");
  CHECK(after.worldTLAS == before.worldTLAS, "frame/camera edit rebuilt TLAS");
  CHECK(after.groupLightSet == before.groupLightSet,
      "frame/camera edit rebuilt light set");

  anari::setParameter(d, h.mat, "color", float3{0.1f, 0.2f, 0.3f});
  anari::commitParameters(d, h.mat);
  CHECK(!h.renderAndCheck(0), "world-content edit did not reset A");
  CHECK(!h.renderAndCheck(1), "world-content edit did not reset B");

  anari::setParameter(d, h.ren, "ambientRadiance", 0.4f);
  anari::commitParameters(d, h.ren);
  h.renderAndCheck(0);
  h.renderAndCheck(1);
  CHECK(h.lastSamples[0] == 1, "shared renderer did not reset A");
  CHECK(h.lastSamples[1] == 1, "shared renderer did not reset B");
  anari::release(d, otherCam);
}

void testManualResize(anari::Device d)
{
  Harness h(d);
  const uint64_t version = 1;
  anari::setParameter(d, h.frames[0], "accumulationVersion", version);
  anari::commitParameters(d, h.frames[0]);
  for (int s = 0; s < 3; ++s) {
    h.renderAndCheck(0);
    h.renderAndCheck(1);
  }
  anari::setParameter(d, h.cam, "position", float3{0.f, 0.f, 4.f});
  anari::commitParameters(d, h.cam);
  CHECK(h.renderAndCheck(0), "camera reset A in manual mode");
  CHECK(!h.renderAndCheck(1), "shared camera did not reset automatic B");
  anari::setParameter(d, h.frames[0], "size", uint2{96, 72});
  anari::commitParameters(d, h.frames[0]);
  CHECK(!h.renderAndCheck(0), "resize did not reset A in manual mode");
  CHECK(h.renderAndCheck(0), "manual A did not accumulate after resize");
  CHECK(h.renderAndCheck(1), "manual resize reset B");
  for (const auto size : {uint2{32, 24}, uint2{64, 48}}) {
    anari::setParameter(d, h.frames[0], "size", size);
    anari::commitParameters(d, h.frames[0]);
    CHECK(!h.renderAndCheck(0), "manual shrink/growth did not reset A");
    CHECK(h.renderAndCheck(0), "manual A did not resume after resize");
    CHECK(h.renderAndCheck(1), "manual shrink/growth reset B");
  }
  for (const auto *channel : {"channel.color",
           "channel.albedo",
           "channel.normal",
           "channel.depth",
           "channel.primitiveId",
           "channel.objectId",
           "channel.instanceId"}) {
    const auto type = std::string(channel) == "channel.color"
        ? ANARI_FLOAT32_VEC4
        : std::string(channel) == "channel.depth" ? ANARI_FLOAT32
        : std::string(channel).find("Id") != std::string::npos
        ? ANARI_UINT32
        : ANARI_FLOAT32_VEC3;
    anari::setParameter(d, h.frames[0], channel, type);
    anari::commitParameters(d, h.frames[0]);
    CHECK(!h.renderAndCheck(0), "manual %s change did not reset A", channel);
    CHECK(h.renderAndCheck(0), "manual A did not resume after %s", channel);
    CHECK(h.renderAndCheck(1), "manual %s change reset B", channel);
    anari::unsetParameter(d, h.frames[0], channel);
    anari::commitParameters(d, h.frames[0]);
    CHECK(!h.renderAndCheck(0), "manual %s removal did not reset A", channel);
    CHECK(h.renderAndCheck(0), "manual A did not resume after removal");
    CHECK(h.renderAndCheck(1), "manual channel removal reset B");
  }
  anari::setParameter(d, h.mat, "color", float3{0.2f, 0.3f, 0.4f});
  anari::setParameter(d, h.ren, "ambientRadiance", 0.6f);
  anari::commitParameters(d, h.mat);
  anari::commitParameters(d, h.ren);
  CHECK(h.renderAndCheck(0), "dependency changes reset manual A");
  CHECK(!h.renderAndCheck(1), "dependency changes did not reset automatic B");
  for (const bool denoise : {true, false}) {
    anari::setParameter(d, h.ren, "denoise", denoise);
    anari::setParameter(d, h.ren, "denoiseMode", "colorAlbedoNormal");
    anari::commitParameters(d, h.ren);
    CHECK(h.renderAndCheck(0), "denoiser scratch change reset manual A");
    CHECK(h.renderAndCheck(1), "denoiser scratch change reset automatic B");
  }
}

void testRendererPostOnly(anari::Device d)
{
  Harness h(d);
  anari::setParameter(d, h.frames[0], "channel.color", ANARI_FLOAT32_VEC4);
  anari::commitParameters(d, h.frames[0]);
  for (int s = 0; s < 4; ++s) {
    h.renderAndCheck(0);
    h.renderAndCheck(1);
  }
  const auto built = h.counts();
  auto checkBoth = [&]() {
    CHECK(h.renderAndCheck(1), "post-only change reset packed B");
    CHECK(h.renderAndCheck(0), "post-only change reset float A");
    for (int i = 0; i < 2; ++i) {
      uint32_t w = 0, height = 0;
      ANARIDataType type = ANARI_UNKNOWN;
      const auto *pixels =
          anariMapFrame(d, h.frames[i], "channel.color", &w, &height, &type);
      CHECK(pixels && w == 64 && height == 48, "invalid post-only output");
      if (pixels && type == ANARI_FLOAT32_VEC4) {
        const auto *values = static_cast<const float *>(pixels);
        float energy = 0.f;
        for (size_t p = 0; p < size_t(w) * height * 4; ++p) {
          CHECK(std::isfinite(values[p]), "nonfinite post-only float output");
          if (p % 4 != 3)
            energy += std::abs(values[p]);
        }
        CHECK(energy > 1.f, "post-only float output lost color history");
      } else if (pixels) {
        const auto *values = static_cast<const uint32_t *>(pixels);
        uint64_t energy = 0;
        for (size_t p = 0; p < size_t(w) * height; ++p) {
          CHECK((values[p] >> 24) == 255, "invalid packed output alpha");
          energy += values[p] & 0x00ffffffu;
        }
        CHECK(energy > 0, "post-only packed output lost color history");
      }
      anariUnmapFrame(d, h.frames[i], "channel.color");
    }
  };
  for (const bool enabled : {true, false, true}) {
    anari::setParameter(d, h.ren, "denoise", enabled);
    anari::commitParameters(d, h.ren);
    checkBoth();
    for (const auto *mode : {"colorAlbedo", "colorAlbedoNormal", "color"}) {
      anari::setParameter(d, h.ren, "denoiseMode", mode);
      anari::commitParameters(d, h.ren);
      checkBoth();
    }
  }
  for (const int start : {100, 0, -1}) {
    anari::setParameter(d, h.ren, "denoiseStart", start);
    anari::commitParameters(d, h.ren);
    checkBoth();
  }
  anari::setParameter(d, h.ren, "sampleLimit", 2);
  anari::commitParameters(d, h.ren);
  const auto stopped = h.lastSamples;
  h.renderAndCheck(0);
  h.renderAndCheck(1);
  CHECK(h.lastSamples == stopped,
      "lowering sampleLimit lost history or rendered");
  for (const bool enabled : {false, true}) {
    anari::setParameter(d, h.ren, "denoise", enabled);
    anari::setParameter(d, h.ren, "denoiseMode", "colorAlbedoNormal");
    anari::commitParameters(d, h.ren);
    h.renderAndCheck(0);
    h.renderAndCheck(1);
    CHECK(h.lastSamples == stopped, "capped post-only change lost history");
    for (int i = 0; i < 2; ++i) {
      uint32_t w, height;
      ANARIDataType type;
      const auto *pixels =
          anariMapFrame(d, h.frames[i], "channel.color", &w, &height, &type);
      CHECK(pixels, "missing capped output");
      if (pixels && type == ANARI_FLOAT32_VEC4) {
        const auto *values = static_cast<const float *>(pixels);
        for (size_t p = 0; p < size_t(w) * height; ++p) {
          CHECK(std::isfinite(values[4 * p]), "nonfinite capped color");
          CHECK(std::abs(values[4 * p + 3] - 1.f) < 1e-5f,
              "invalid capped alpha");
        }
      } else if (pixels) {
        const auto *values = static_cast<const uint32_t *>(pixels);
        for (size_t p = 0; p < size_t(w) * height; ++p)
          CHECK((values[p] >> 24) == 255, "invalid capped packed alpha");
      }
      anariUnmapFrame(d, h.frames[i], "channel.color");
    }
  }
  anari::setParameter(d, h.ren, "sampleLimit", 100);
  anari::commitParameters(d, h.ren);
  checkBoth();
  CHECK(
      h.lastSamples[0] == stopped[0] + 1 && h.lastSamples[1] == stopped[1] + 1,
      "sampleLimit increase did not resume history");
  CHECK(h.counts().groupBLAS == built.groupBLAS
          && h.counts().worldTLAS == built.worldTLAS,
      "post-only change rebuilt BVHs");
  for (const auto *parameter : {"ambientRadiance", "lightFalloff"}) {
    anari::setParameter(d, h.ren, "denoise", false);
    anari::setParameter(d, h.ren, parameter, 0.7f);
    anari::commitParameters(d, h.ren);
    CHECK(!h.renderAndCheck(0), "mixed %s commit did not reset A", parameter);
    CHECK(!h.renderAndCheck(1), "mixed %s commit did not reset B", parameter);
    CHECK(h.renderAndCheck(0), "A did not resume after mixed commit");
    CHECK(h.renderAndCheck(1), "B did not resume after mixed commit");
  }
}

void testRendererSubtypeAndBackground(anari::Device d)
{
  for (const auto *subtype : {"fast", "quality"}) {
    Harness h(d);
    auto renderer = anari::newObject<anari::Renderer>(d, subtype);
    // fast overrides this authored ambient color when fixedAmbientLighting
    // is on. The comparison must use its final effective values.
    anari::setParameter(d, renderer, "ambientColor", float3{0.2f, 0.3f, 0.4f});
    anari::commitParameters(d, renderer);
    for (auto f : h.frames) {
      anari::setParameter(d, f, "renderer", renderer);
      anari::commitParameters(d, f);
    }
    for (int s = 0; s < 3; ++s) {
      h.renderAndCheck(0);
      h.renderAndCheck(1);
    }
    anari::setParameter(d, renderer, "denoise", true);
    anari::commitParameters(d, renderer);
    CHECK(h.renderAndCheck(0), "%s denoise reset A", subtype);
    CHECK(h.renderAndCheck(1), "%s denoise reset B", subtype);
    const auto *parameter =
        std::string(subtype) == "fast" ? "ambientSamples" : "maxRayDepth";
    anari::setParameter(d, renderer, parameter, 3);
    anari::setParameter(d, renderer, "denoiseStart", 100);
    anari::commitParameters(d, renderer);
    CHECK(!h.renderAndCheck(0),
        "%s mixed subtype commit did not reset A",
        subtype);
    CHECK(!h.renderAndCheck(1),
        "%s mixed subtype commit did not reset B",
        subtype);
    anari::release(d, renderer);
  }

  Harness h(d);
  auto background =
      anariNewArray2D(d, nullptr, nullptr, nullptr, ANARI_FLOAT32_VEC4, 2, 2);
  auto *pixels =
      static_cast<std::array<float, 4> *>(anariMapArray(d, background));
  std::fill_n(pixels, 4, std::array<float, 4>{0.1f, 0.2f, 0.3f, 1.f});
  anariUnmapArray(d, background);
  anari::setParameter(d, h.ren, "background", background);
  anari::commitParameters(d, h.ren);
  for (int s = 0; s < 3; ++s) {
    h.renderAndCheck(0);
    h.renderAndCheck(1);
  }
  // A post-only commit must not mask a concurrent data notification.
  anari::setParameter(d, h.ren, "denoise", true);
  anari::commitParameters(d, h.ren);
  pixels = static_cast<std::array<float, 4> *>(anariMapArray(d, background));
  pixels[0][0] = 0.6f;
  anariUnmapArray(d, background);
  CHECK(!h.renderAndCheck(0), "background notification did not reset A");
  CHECK(!h.renderAndCheck(1), "background notification did not reset B");
  CHECK(h.renderAndCheck(0), "A did not resume after background edit");
  CHECK(h.renderAndCheck(1), "B did not resume after background edit");
  anariMapArray(d, background);
  anariUnmapArray(d, background);
  CHECK(!h.renderAndCheck(0), "identical background unmap did not reset A");
  CHECK(!h.renderAndCheck(1), "identical background unmap did not reset B");
  anari::release(d, background);
}

void testGuideHistory(anari::Device d)
{
  Harness h(d);
  auto *positions = static_cast<float3 *>(anariMapArray(d, h.vertices));
  std::fill_n(positions, 256, float3{0.f, 0.f, 0.f});
  anariUnmapArray(d, h.vertices);
  anari::setParameter(d, h.geom, "radius", 2.5f);
  anari::commitParameters(d, h.geom);
  for (auto f : h.frames) {
    anari::setParameter(d, f, "channel.color", ANARI_FLOAT32_VEC4);
    anari::commitParameters(d, f);
  }
  // B has always collected guides; A must retain the same history even with
  // neither output channels nor denoising requested initially.
  anari::setParameter(d, h.frames[1], "channel.albedo", ANARI_FLOAT32_VEC3);
  anari::setParameter(d, h.frames[1], "channel.normal", ANARI_FLOAT32_VEC3);
  anari::commitParameters(d, h.frames[1]);
  for (int s = 0; s < 8; ++s) {
    h.renderAndCheck(0);
    h.renderAndCheck(1);
  }
  for (const auto *mode :
      {"colorAlbedoNormal", "color", "colorAlbedo", "colorAlbedoNormal"}) {
    anari::setParameter(d, h.ren, "denoise", true);
    anari::setParameter(d, h.ren, "denoiseMode", mode);
    anari::commitParameters(d, h.ren);
    CHECK(h.renderAndCheck(0), "guide change reset A");
    CHECK(h.renderAndCheck(1), "guide change reset B");
    uint32_t w, height;
    ANARIDataType type;
    const auto *a = static_cast<const float *>(
        anariMapFrame(d, h.frames[0], "channel.color", &w, &height, &type));
    const auto *b = static_cast<const float *>(
        anariMapFrame(d, h.frames[1], "channel.color", &w, &height, &type));
    CHECK(a && b, "missing guide-history output");
    if (a && b) {
      float energy = 0.f;
      for (size_t p = 0; p < size_t(w) * height * 4; ++p) {
        CHECK(std::isfinite(a[p]) && std::isfinite(b[p]),
            "nonfinite guide output");
        CHECK(
            std::abs(a[p] - b[p]) < 1e-5f, "guide history mismatch at %zu", p);
        if (p % 4 != 3)
          energy += std::abs(a[p]);
      }
      CHECK(energy > 1.f, "guide history produced empty output");
    }
    anariUnmapFrame(d, h.frames[0], "channel.color");
    anariUnmapFrame(d, h.frames[1], "channel.color");
    const auto *albedo = static_cast<const float3 *>(
        anariMapFrame(d, h.frames[1], "channel.albedo", &w, &height, &type));
    const auto *normal = static_cast<const float3 *>(
        anariMapFrame(d, h.frames[1], "channel.normal", &w, &height, &type));
    CHECK(albedo && normal, "missing guide outputs");
    if (albedo && normal) {
      for (size_t p = 0; p < size_t(w) * height; ++p) {
        float norm2 = 0.f;
        for (size_t c = 0; c < 3; ++c) {
          CHECK(std::isfinite(albedo[p][c]) && std::isfinite(normal[p][c]),
              "nonfinite guide history");
          CHECK(std::abs(albedo[p][c] - float3{0.8f, 0.4f, 0.1f}[c]) < 1e-5f,
              "albedo history lost at %zu",
              p);
          norm2 += normal[p][c] * normal[p][c];
        }
        CHECK(std::abs(norm2 - 1.f) < 1e-5f, "normal history lost at %zu", p);
      }
    }
    anariUnmapFrame(d, h.frames[1], "channel.albedo");
    anariUnmapFrame(d, h.frames[1], "channel.normal");
  }
}

void testCameraSubtypeParameters(anari::Device d)
{
  Harness h(d);
  for (int s = 0; s < 3; ++s) {
    h.renderAndCheck(0);
    h.renderAndCheck(1);
  }
  for (const auto *name : {"fovy", "focusDistance", "apertureRadius"}) {
    anari::setParameter(d, h.cam, name, 0.5f);
    anari::commitParameters(d, h.cam);
    CHECK(!h.renderAndCheck(0), "%s did not reset A", name);
    CHECK(!h.renderAndCheck(1), "%s did not reset B", name);
    CHECK(h.renderAndCheck(0), "A did not resume after %s", name);
    CHECK(h.renderAndCheck(1), "B did not resume after %s", name);
    anari::setParameter(d, h.cam, name, 0.5f);
    anari::commitParameters(d, h.cam);
    CHECK(h.renderAndCheck(0), "unchanged %s reset A", name);
    CHECK(h.renderAndCheck(1), "unchanged %s reset B", name);
  }
  auto ortho = anari::newObject<anari::Camera>(d, "orthographic");
  anari::setParameter(d, ortho, "position", float3{0.f, 0.f, 3.f});
  anari::setParameter(d, ortho, "direction", float3{0.f, 0.f, -1.f});
  anari::commitParameters(d, ortho);
  anari::setParameter(d, h.frames[0], "camera", ortho);
  anari::commitParameters(d, h.frames[0]);
  CHECK(!h.renderAndCheck(0), "camera handle change did not reset A");
  CHECK(h.renderAndCheck(1), "camera handle change reset B");
  CHECK(h.renderAndCheck(0), "A did not resume with orthographic camera");
  anari::setParameter(d, ortho, "height", 2.f);
  anari::commitParameters(d, ortho);
  CHECK(!h.renderAndCheck(0), "orthographic height did not reset A");
  CHECK(h.renderAndCheck(1), "orthographic height reset B");
  CHECK(h.renderAndCheck(0), "A did not resume after height change");
  anari::setParameter(d, ortho, "height", 2.f);
  anari::commitParameters(d, ortho);
  CHECK(h.renderAndCheck(0), "unchanged orthographic height reset A");
  anari::release(d, ortho);
}

void testWorldDependencies(anari::Device d, bool zeroInstance)
{
  Harness a(d);
  Harness b(d);
  if (zeroInstance) {
    anari::unsetParameter(d, a.world, "instance");
    anari::setParameterArray1D(d, a.world, "surface", &a.surf, 1);
    anari::setParameterArray1D(d, a.world, "light", &a.light, 1);
    anari::commitParameters(d, a.world);
  }
  // Share the renderer and camera: only the world graph distinguishes Frames.
  anari::setParameter(d, a.frames[1], "world", b.world);
  anari::commitParameters(d, a.frames[1]);
  for (int s = 0; s < 3; ++s) {
    a.renderAndCheck(0);
    a.renderAndCheck(1);
  }
  auto checkAOnly = [&]() {
    CHECK(!a.renderAndCheck(0),
        "world A change did not reset A (zero=%d)",
        zeroInstance);
    CHECK(
        a.renderAndCheck(1), "world A change reset B (zero=%d)", zeroInstance);
    CHECK(a.renderAndCheck(0), "A failed to resume accumulation");
  };

  const auto initialA = a.counts();
  const auto initialB = b.counts();
  anari::setParameter(d, a.mat, "color", float3{0.2f, 0.5f, 0.7f});
  anari::commitParameters(d, a.mat);
  checkAOnly();
  CHECK(a.counts().worldTLAS == initialA.worldTLAS,
      "material edit rebuilt A TLAS (zero=%d)",
      zeroInstance);
  CHECK(b.counts().worldTLAS == initialB.worldTLAS,
      "material edit rebuilt B TLAS (zero=%d)",
      zeroInstance);

  auto *mapped = static_cast<float3 *>(anariMapArray(d, a.vertices));
  (*mapped)[0] += 0.1f;
  anariUnmapArray(d, a.vertices);
  checkAOnly();
  CHECK(b.counts().groupBLAS == initialB.groupBLAS,
      "vertex edit rebuilt unrelated BLAS (zero=%d)",
      zeroInstance);
  CHECK(b.counts().worldTLAS == initialB.worldTLAS,
      "vertex edit rebuilt unrelated TLAS (zero=%d)",
      zeroInstance);

  anari::unsetParameter(d, a.geom, "vertex.position");
  anari::commitParameters(d, a.geom);
  checkAOnly();
  CHECK(b.counts().worldTLAS == initialB.worldTLAS,
      "invalid geometry rebuilt unrelated TLAS (zero=%d)",
      zeroInstance);
  anari::setParameter(d, a.geom, "vertex.position", a.vertices);
  anari::commitParameters(d, a.geom);
  checkAOnly();
  // Identical bytes still constitute a change; uploads alone do not.
  anariMapArray(d, a.vertices);
  anariUnmapArray(d, a.vertices);
  checkAOnly();

  const auto beforeLightA = a.counts();
  const auto beforeLightB = b.counts();
  anari::setParameter(d, a.light, "irradiance", 2.f);
  anari::commitParameters(d, a.light);
  checkAOnly();
  CHECK(a.counts().worldTLAS == beforeLightA.worldTLAS,
      "light-only edit rebuilt A TLAS (zero=%d)",
      zeroInstance);
  CHECK(b.counts().worldTLAS == beforeLightB.worldTLAS,
      "light-only edit rebuilt B TLAS (zero=%d)",
      zeroInstance);

  if (!zeroInstance) {
    std::array<float, 16> transform{
        1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0.1f, 0, 0, 1};
    anariSetParameter(
        d, a.inst, "transform", ANARI_FLOAT32_MAT4, transform.data());
    anari::commitParameters(d, a.inst);
    checkAOnly();
  }

  auto orphan = anari::newObject<anari::Surface>(d);
  anari::setParameter(d, orphan, "geometry", a.geom);
  anari::setParameter(d, orphan, "material", a.mat);
  anari::commitParameters(d, orphan);
  CHECK(a.renderAndCheck(0), "unreachable surface reset A");
  CHECK(a.renderAndCheck(1), "unreachable surface reset B");
  anari::release(d, orphan);

  anari::setParameter(d, b.surf, "geometry", a.geom);
  anari::commitParameters(d, b.surf);
  CHECK(a.renderAndCheck(0), "B's surface handle change reset A");
  CHECK(!a.renderAndCheck(1), "B's surface handle change did not reset B");
  a.renderAndCheck(1);
  anari::setParameter(d, a.geom, "radius", 0.07f);
  anari::commitParameters(d, a.geom);
  CHECK(!a.renderAndCheck(0), "shared geometry did not reset A");
  CHECK(!a.renderAndCheck(1), "shared geometry did not reset B");
  a.renderAndCheck(0);
  a.renderAndCheck(1);

  // Replacing a dependency must remove the old upward edge.
  anari::setParameter(d, b.surf, "geometry", b.geom);
  anari::commitParameters(d, b.surf);
  CHECK(a.renderAndCheck(0), "B's geometry replacement reset A");
  CHECK(!a.renderAndCheck(1), "B's geometry replacement did not reset B");
  a.renderAndCheck(1);
  anari::setParameter(d, a.geom, "radius", 0.08f);
  anari::commitParameters(d, a.geom);
  checkAOnly();
}

void testScopedBLAS(anari::Device d)
{
  Harness a(d);
  Harness b(d);
  anari::setParameter(d, a.frames[1], "world", b.world);
  anari::commitParameters(d, a.frames[1]);

  // Leave a valid primitive in A when its original geometry becomes invalid,
  // so the build counter also detects removal from the partition.
  auto secondSurface = anari::newObject<anari::Surface>(d);
  anari::setParameter(d, secondSurface, "geometry", b.geom);
  anari::setParameter(d, secondSurface, "material", b.mat);
  anari::commitParameters(d, secondSurface);
  const std::array<anari::Surface, 2> surfaces{a.surf, secondSurface};
  anari::setParameterArray1D(
      d, a.group, "surface", surfaces.data(), surfaces.size());
  anari::commitParameters(d, a.group);

  auto secondInstance = anari::newObject<anari::Instance>(d, "transform");
  anari::setParameter(d, secondInstance, "group", a.group);
  anari::commitParameters(d, secondInstance);
  const std::array<anari::Instance, 2> instances{a.inst, secondInstance};
  anari::setParameterArray1D(
      d, a.world, "instance", instances.data(), instances.size());
  anari::commitParameters(d, a.world);
  for (int s = 0; s < 3; ++s) {
    a.renderAndCheck(0);
    a.renderAndCheck(1);
  }

  auto beforeA = a.counts();
  auto beforeB = b.counts();
  anari::setParameter(d, a.mat, "color", float3{0.2f, 0.3f, 0.4f});
  anari::commitParameters(d, a.mat);
  CHECK(!a.renderAndCheck(0), "material edit did not reset A");
  CHECK(a.renderAndCheck(1), "material edit reset B");
  CHECK(a.counts().groupBLAS == beforeA.groupBLAS,
      "material edit rebuilt A BLAS");
  CHECK(b.counts().groupBLAS == beforeB.groupBLAS,
      "material edit rebuilt B BLAS");
  CHECK(a.counts().worldTLAS == beforeA.worldTLAS,
      "material edit rebuilt A TLAS");
  CHECK(b.counts().worldTLAS == beforeB.worldTLAS,
      "material edit rebuilt B TLAS");
  a.renderAndCheck(0);

  auto checkStructuralChange = [&]() {
    CHECK(!a.renderAndCheck(0), "structural edit did not reset A");
    CHECK(a.renderAndCheck(1), "structural edit reset B");
    CHECK(a.counts().groupBLAS == beforeA.groupBLAS + 1,
        "shared group did not rebuild exactly once");
    CHECK(b.counts().groupBLAS == beforeB.groupBLAS,
        "structural edit rebuilt unrelated B BLAS");
    CHECK(b.counts().worldTLAS == beforeB.worldTLAS,
        "structural edit rebuilt unrelated B TLAS");
    CHECK(a.counts().worldTLAS == beforeA.worldTLAS + 1,
        "structural edit did not rebuild A TLAS once");
    CHECK(a.renderAndCheck(0), "A failed to resume accumulation");
    beforeA = a.counts();
    beforeB = b.counts();
  };
  auto *mapped = static_cast<float3 *>(anariMapArray(d, a.vertices));
  (*mapped)[0] += 0.2f;
  anariUnmapArray(d, a.vertices);
  checkStructuralChange();

  anari::unsetParameter(d, a.geom, "vertex.position");
  anari::commitParameters(d, a.geom);
  checkStructuralChange();
  anari::setParameter(d, a.geom, "vertex.position", a.vertices);
  anari::commitParameters(d, a.geom);
  checkStructuralChange();

  anari::setParameter(d, a.surf, "visible", false);
  anari::commitParameters(d, a.surf);
  checkStructuralChange();
  anari::setParameter(d, a.surf, "visible", true);
  anari::commitParameters(d, a.surf);
  checkStructuralChange();

  // A Group shared across Worlds has one build consumer but two TLAS
  // consumers. Reading bounds must not consume either World's change.
  anari::setParameter(d, b.inst, "group", a.group);
  anari::commitParameters(d, b.inst);
  a.renderAndCheck(0);
  a.renderAndCheck(1);
  a.renderAndCheck(0);
  a.renderAndCheck(1);
  beforeA = a.counts();
  beforeB = b.counts();
  anari::setParameter(d, a.geom, "radius", 0.06f);
  anari::commitParameters(d, a.geom);
  std::array<float, 6> bounds{};
  anariGetProperty(d,
      a.group,
      "bounds",
      ANARI_FLOAT32_BOX3,
      bounds.data(),
      sizeof(bounds),
      ANARI_WAIT);
  CHECK(!a.renderAndCheck(1), "shared group did not reset B");
  CHECK(!a.renderAndCheck(0), "shared group did not reset A");
  CHECK(a.counts().groupBLAS == beforeA.groupBLAS + 1,
      "shared group rebuilt more than once across Worlds");
  CHECK(a.counts().worldTLAS == beforeA.worldTLAS + 1,
      "shared group did not rebuild A TLAS");
  CHECK(b.counts().worldTLAS == beforeB.worldTLAS + 1,
      "shared group did not rebuild B TLAS");
  anari::release(d, secondInstance);
  anari::release(d, secondSurface);
}

void testScopedTLAS(anari::Device d)
{
  Harness a(d);
  Harness b(d);
  anari::setParameter(d, a.frames[1], "world", b.world);
  anari::commitParameters(d, a.frames[1]);
  for (int s = 0; s < 3; ++s) {
    a.renderAndCheck(0);
    a.renderAndCheck(1);
  }
  auto beforeA = a.counts();
  const auto beforeB = b.counts();
  auto checkAOnly = [&]() {
    // Render B first: flushing A's changes must not rebuild B's TLAS.
    CHECK(a.renderAndCheck(1), "instance edit reset unrelated B");
    CHECK(!a.renderAndCheck(0), "instance edit did not reset A");
    CHECK(a.counts().worldTLAS == beforeA.worldTLAS + 1,
        "instance edit did not rebuild A TLAS exactly once");
    CHECK(b.counts().worldTLAS == beforeB.worldTLAS,
        "instance edit rebuilt unrelated B TLAS");
    CHECK(a.counts().groupBLAS == beforeA.groupBLAS,
        "instance edit rebuilt A BLAS");
    CHECK(b.counts().groupBLAS == beforeB.groupBLAS,
        "instance edit rebuilt B BLAS");
    CHECK(a.renderAndCheck(0), "A failed to resume after instance edit");
    beforeA = a.counts();
  };

  std::array<float, 16> transform{
      1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0.1f, 0, 0, 1};
  anariSetParameter(
      d, a.inst, "transform", ANARI_FLOAT32_MAT4, transform.data());
  anari::commitParameters(d, a.inst);
  checkAOnly();

  // Array data changes must still re-finalize Instance's packed parameters.
  auto transforms =
      anariNewArray1D(d, nullptr, nullptr, nullptr, ANARI_FLOAT32_MAT4, 1);
  auto *mapped = static_cast<float *>(anariMapArray(d, transforms));
  std::copy(transform.begin(), transform.end(), mapped);
  anariUnmapArray(d, transforms);
  anari::setParameter(d, a.inst, "transform", transforms);
  anari::commitParameters(d, a.inst);
  checkAOnly();
  mapped = static_cast<float *>(anariMapArray(d, transforms));
  mapped[12] += 0.1f;
  anariUnmapArray(d, transforms);
  checkAOnly();

  auto second = anari::newObject<anari::Instance>(d, "transform");
  anari::setParameter(d, second, "group", a.group);
  anari::commitParameters(d, second);
  CHECK(a.renderAndCheck(1), "unreachable instance reset B");
  CHECK(a.renderAndCheck(0), "unreachable instance reset A");
  CHECK(a.counts().worldTLAS == beforeA.worldTLAS,
      "unreachable instance rebuilt A TLAS");
  CHECK(b.counts().worldTLAS == beforeB.worldTLAS,
      "unreachable instance rebuilt B TLAS");

  auto instances =
      anariNewArray1D(d, nullptr, nullptr, nullptr, ANARI_INSTANCE, 2);
  auto *handles = static_cast<anari::Instance *>(anariMapArray(d, instances));
  handles[0] = a.inst;
  handles[1] = second;
  anariUnmapArray(d, instances);
  anari::setParameter(d, a.world, "instance", instances);
  anari::commitParameters(d, a.world);
  checkAOnly();
  handles = static_cast<anari::Instance *>(anariMapArray(d, instances));
  handles[1] = nullptr;
  anariUnmapArray(d, instances);
  checkAOnly();

  // Invalid Instances remain observed, so restoring a group re-adds them.
  anari::unsetParameter(d, a.inst, "group");
  anari::commitParameters(d, a.inst);
  checkAOnly();
  anari::setParameter(d, a.inst, "group", a.group);
  anari::commitParameters(d, a.inst);
  checkAOnly();
  anari::setParameter(d, a.inst, "group", b.group);
  anari::commitParameters(d, a.inst);
  checkAOnly();

  // The old Group edge is gone after the handle replacement.
  anari::setParameter(d, a.geom, "radius", 0.09f);
  anari::commitParameters(d, a.geom);
  CHECK(a.renderAndCheck(0), "replaced group reset A");
  CHECK(a.renderAndCheck(1), "unreachable group reset B");
  CHECK(a.counts().worldTLAS == beforeA.worldTLAS,
      "replaced group rebuilt A TLAS");
  CHECK(b.counts().worldTLAS == beforeB.worldTLAS,
      "unreachable group rebuilt B TLAS");
  anari::release(d, instances);
  anari::release(d, second);
  anari::release(d, transforms);
}

void testScopedLightSets(anari::Device d, bool zeroInstance)
{
  Harness a(d);
  Harness b(d);
  auto material = anari::newObject<anari::Material>(d, "pbr");
  anari::commitParameters(d, material);
  anari::setParameter(d, a.surf, "material", material);
  anari::commitParameters(d, a.surf);
  if (zeroInstance) {
    anari::unsetParameter(d, a.world, "instance");
    anari::setParameterArray1D(d, a.world, "surface", &a.surf, 1);
    anari::setParameterArray1D(d, a.world, "light", &a.light, 1);
    anari::commitParameters(d, a.world);
  }
  anari::setParameter(d, a.frames[1], "world", b.world);
  anari::commitParameters(d, a.frames[1]);
  for (int s = 0; s < 3; ++s) {
    a.renderAndCheck(0);
    a.renderAndCheck(1);
  }
  auto beforeA = a.counts();
  auto beforeB = b.counts();
  auto checkNoBVH = [&]() {
    CHECK(a.counts().groupBLAS == beforeA.groupBLAS,
        "light-set edit rebuilt A BLAS (zero=%d)",
        zeroInstance);
    CHECK(a.counts().worldTLAS == beforeA.worldTLAS,
        "light-set edit rebuilt A TLAS (zero=%d)",
        zeroInstance);
    CHECK(b.counts().groupBLAS == beforeB.groupBLAS,
        "light-set edit rebuilt B BLAS");
    CHECK(b.counts().worldTLAS == beforeB.worldTLAS,
        "light-set edit rebuilt B TLAS");
  };
  auto checkAOnly = [&](uint32_t expectedLights) {
    CHECK(a.renderAndCheck(1), "light-set edit reset B");
    CHECK(!a.renderAndCheck(0), "light-set edit did not reset A");
    checkNoBVH();
    CHECK(a.counts().worldLightSet == beforeA.worldLightSet + 1,
        "A light set did not rebuild exactly once (zero=%d)",
        zeroInstance);
    CHECK(b.counts().worldLightSet == beforeB.worldLightSet,
        "A light-set edit rebuilt B world light set");
    CHECK(b.counts().groupLightSet == beforeB.groupLightSet,
        "A light-set edit rebuilt B group light set");
    if (!zeroInstance) {
      CHECK(a.counts().groupLightSet == beforeA.groupLightSet + 1,
          "A group light set did not rebuild exactly once");
    }
    uint32_t lights = 0;
    anari::getProperty(d, a.world, "numLightInstances", lights, ANARI_WAIT);
    CHECK(lights == expectedLights,
        "expected %u lights, got %u",
        expectedLights,
        lights);
    CHECK(a.renderAndCheck(0), "A did not resume after light-set edit");
    CHECK(a.counts().worldLightSet == beforeA.worldLightSet + 1,
        "A light set stayed dirty after rebuild");
    beforeA = a.counts();
  };

  anari::setParameter(d, a.light, "irradiance", 2.f);
  anari::commitParameters(d, a.light);
  checkAOnly(1);
  for (const auto emission :
      {float3{2.f, 1.f, 1.f}, float3{3.f, 2.f, 1.f}, float3{0.f, 0.f, 0.f}}) {
    anari::setParameter(d, material, "emissive", emission);
    anari::commitParameters(d, material);
    checkAOnly(emission[0] > 0.f ? 2 : 1);
  }
  // A non-emission edit resets the image, not either light-set consumer.
  anari::setParameter(d, material, "baseColor", float3{0.2f, 0.3f, 0.4f});
  anari::commitParameters(d, material);
  CHECK(!a.renderAndCheck(0), "baseColor edit did not reset A");
  CHECK(a.renderAndCheck(1), "baseColor edit reset B");
  checkNoBVH();
  CHECK(a.counts().worldLightSet == beforeA.worldLightSet,
      "baseColor edit rebuilt A world light set");
  CHECK(a.counts().groupLightSet == beforeA.groupLightSet,
      "baseColor edit rebuilt A group light set");
  a.renderAndCheck(0);

  auto orphan = anari::newObject<anari::Light>(d, "directional");
  anari::commitParameters(d, orphan);
  CHECK(a.renderAndCheck(0), "unreachable light reset A");
  CHECK(a.renderAndCheck(1), "unreachable light reset B");
  CHECK(a.counts().worldLightSet == beforeA.worldLightSet,
      "unreachable light rebuilt A light set");
  CHECK(b.counts().worldLightSet == beforeB.worldLightSet,
      "unreachable light rebuilt B light set");
  anari::release(d, orphan);

  if (!zeroInstance) {
    // The Group consumes once; both Worlds must still consume independently.
    anari::setParameter(d, b.inst, "group", a.group);
    anari::commitParameters(d, b.inst);
    a.renderAndCheck(1);
    a.renderAndCheck(1);
    beforeA = a.counts();
    beforeB = b.counts();
    anari::setParameter(d, material, "emissive", float3{1.f, 1.f, 1.f});
    anari::commitParameters(d, material);
    CHECK(!a.renderAndCheck(1), "shared emission did not reset B");
    CHECK(!a.renderAndCheck(0), "shared emission did not reset A");
    checkNoBVH();
    CHECK(a.counts().groupLightSet == beforeA.groupLightSet + 1,
        "shared Group rebuilt light set more than once");
    CHECK(a.counts().worldLightSet == beforeA.worldLightSet + 1,
        "shared emission did not rebuild A light set");
    CHECK(b.counts().worldLightSet == beforeB.worldLightSet + 1,
        "shared emission did not rebuild B light set");
    uint32_t lights = 0;
    anari::getProperty(d, b.world, "numLightInstances", lights, ANARI_WAIT);
    CHECK(lights == 2, "shared emission missing from B light list");
  }
  anari::release(d, material);
}

void testScopedVolumeBLAS(anari::Device d)
{
  Harness a(d);
  Harness b(d);
  anari::setParameter(d, a.frames[1], "world", b.world);
  anari::commitParameters(d, a.frames[1]);
  std::array<anari::Array3D, 2> data{};
  for (auto &array : data) {
    array =
        anariNewArray3D(d, nullptr, nullptr, nullptr, ANARI_FLOAT32, 2, 2, 2);
    auto *values = static_cast<float *>(anariMapArray(d, array));
    std::fill(values, values + 8, 0.5f);
    anariUnmapArray(d, array);
  }
  std::array<anari::SpatialField, 2> fields{};
  std::array<anari::Volume, 2> volumes{};
  const std::array<anari::Group, 2> groups{a.group, b.group};
  for (size_t i = 0; i < fields.size(); ++i) {
    fields[i] = anari::newObject<anari::SpatialField>(d, "structuredRegular");
    anari::setParameter(d, fields[i], "data", data[i]);
    anari::commitParameters(d, fields[i]);
    volumes[i] = anari::newObject<anari::Volume>(d, "transferFunction1D");
    anari::setParameter(d, volumes[i], "value", fields[i]);
    anari::setParameter(d, volumes[i], "color", float3{0.4f, 0.5f, 0.6f});
    anari::setParameter(d, volumes[i], "opacity", 0.1f);
    anari::commitParameters(d, volumes[i]);
    anari::setParameterArray1D(d, groups[i], "volume", &volumes[i], 1);
    anari::commitParameters(d, groups[i]);
  }
  for (int s = 0; s < 3; ++s) {
    a.renderAndCheck(0);
    a.renderAndCheck(1);
  }
  const auto beforeA = a.counts();
  const auto beforeB = b.counts();
  anari::setParameter(d, volumes[0], "color", float3{0.6f, 0.5f, 0.4f});
  anari::commitParameters(d, volumes[0]);
  CHECK(!a.renderAndCheck(0), "volume color did not reset A");
  CHECK(a.renderAndCheck(1), "volume color reset B");
  CHECK(
      a.counts().groupBLAS == beforeA.groupBLAS, "volume color rebuilt A BLAS");
  CHECK(
      a.counts().worldTLAS == beforeA.worldTLAS, "volume color rebuilt A TLAS");
  a.renderAndCheck(0);

  anari::setParameter(d, fields[0], "origin", float3{0.1f, 0.f, 0.f});
  anari::commitParameters(d, fields[0]);
  CHECK(!a.renderAndCheck(0), "field bounds did not reset A");
  CHECK(a.renderAndCheck(1), "field bounds reset B");
  CHECK(a.counts().groupBLAS == beforeA.groupBLAS + 1,
      "field bounds did not rebuild only A's volume BLAS");
  CHECK(a.counts().worldTLAS == beforeA.worldTLAS + 1,
      "field bounds did not rebuild A TLAS");
  CHECK(b.counts().groupBLAS == beforeB.groupBLAS,
      "field bounds rebuilt unrelated B BLAS");
  CHECK(b.counts().worldTLAS == beforeB.worldTLAS,
      "field bounds rebuilt unrelated B TLAS");
  auto checkEdit = [&](const char *name, bool structural, bool empty = false) {
    const auto previousA = a.counts();
    const auto previousB = b.counts();
    CHECK(!a.renderAndCheck(0), "%s did not reset A", name);
    CHECK(a.renderAndCheck(1), "%s reset B", name);
    CHECK(a.counts().groupBLAS == previousA.groupBLAS + (structural && !empty),
        "%s: wrong A BLAS count",
        name);
    CHECK(a.counts().worldTLAS == previousA.worldTLAS + structural,
        "%s: wrong A TLAS count",
        name);
    CHECK(
        b.counts().groupBLAS == previousB.groupBLAS, "%s rebuilt B BLAS", name);
    CHECK(
        b.counts().worldTLAS == previousB.worldTLAS, "%s rebuilt B TLAS", name);
    CHECK(a.renderAndCheck(0), "%s caused a second reset", name);
  };
  a.renderAndCheck(0);

  auto colors =
      anariNewArray1D(d, nullptr, nullptr, nullptr, ANARI_FLOAT32_VEC3, 2);
  auto *colorValues = static_cast<float3 *>(anariMapArray(d, colors));
  colorValues[0] = {0.1f, 0.2f, 0.3f};
  colorValues[1] = {0.5f, 0.6f, 0.7f};
  anariUnmapArray(d, colors);
  anari::setParameter(d, volumes[0], "color", colors);
  anari::commitParameters(d, volumes[0]);
  checkEdit("color map", false);
  colorValues = static_cast<float3 *>(anariMapArray(d, colors));
  colorValues[0][0] = 0.4f;
  anariUnmapArray(d, colors);
  checkEdit("color map unmap", false);

  auto opacity =
      anariNewArray1D(d, nullptr, nullptr, nullptr, ANARI_FLOAT32, 2);
  auto *opacityValues = static_cast<float *>(anariMapArray(d, opacity));
  opacityValues[0] = 0.1f;
  opacityValues[1] = 0.2f;
  anariUnmapArray(d, opacity);
  anari::setParameter(d, volumes[0], "opacity", opacity);
  anari::commitParameters(d, volumes[0]);
  checkEdit("opacity map", false);
  opacityValues = static_cast<float *>(anariMapArray(d, opacity));
  opacityValues[1] = 0.3f;
  anariUnmapArray(d, opacity);
  checkEdit("opacity map unmap", false);

  auto *values = static_cast<float *>(anariMapArray(d, data[0]));
  values[0] = 0.7f;
  anariUnmapArray(d, data[0]);
  checkEdit("field data unmap", false);
  anariMapArray(d, data[0]);
  anariUnmapArray(d, data[0]);
  checkEdit("identical field data unmap", false);
  anari::setParameter(d, fields[0], "filter", "nearest");
  anari::commitParameters(d, fields[0]);
  checkEdit("field filter", false);
  anari::setParameter(d,
      fields[0],
      "roi",
      std::array<float3, 2>{float3{0.2f, 0.f, 0.f}, float3{0.8f, 1.f, 1.f}});
  anari::commitParameters(d, fields[0]);
  checkEdit("field ROI", false);
  anari::setParameter(d, fields[0], "spacing", float3{0.8f, 1.f, 1.f});
  anari::commitParameters(d, fields[0]);
  checkEdit("field spacing", true);
  anari::setParameter(d, fields[0], "dataCentering", "cell");
  anari::commitParameters(d, fields[0]);
  checkEdit("field centering", true);

  auto largerData =
      anariNewArray3D(d, nullptr, nullptr, nullptr, ANARI_FLOAT32, 3, 2, 2);
  values = static_cast<float *>(anariMapArray(d, largerData));
  std::fill(values, values + 12, 0.4f);
  anariUnmapArray(d, largerData);
  anari::setParameter(d, fields[0], "data", largerData);
  anari::commitParameters(d, fields[0]);
  checkEdit("field dimensions", true);
  anari::unsetParameter(d, fields[0], "data");
  anari::commitParameters(d, fields[0]);
  checkEdit("invalid field", true, true);
  anari::setParameter(d, fields[0], "data", largerData);
  anari::commitParameters(d, fields[0]);
  checkEdit("valid field", true);

  // A distinct field handle with identical bounds is not structural.
  auto replacement =
      anari::newObject<anari::SpatialField>(d, "structuredRegular");
  anari::setParameter(d, replacement, "data", largerData);
  anari::setParameter(d, replacement, "origin", float3{0.1f, 0.f, 0.f});
  anari::setParameter(d, replacement, "spacing", float3{0.8f, 1.f, 1.f});
  anari::setParameter(d, replacement, "dataCentering", "cell");
  anari::commitParameters(d, replacement);
  anari::setParameter(d, volumes[0], "value", replacement);
  anari::commitParameters(d, volumes[0]);
  checkEdit("equal-bounds field swap", false);

  // Sharing the field resets both consumers even if B renders first.
  anari::setParameter(d, volumes[1], "value", replacement);
  anari::commitParameters(d, volumes[1]);
  a.renderAndCheck(0);
  a.renderAndCheck(1);
  a.renderAndCheck(1);
  const auto sharedA = a.counts();
  const auto sharedB = b.counts();
  anariMapArray(d, largerData);
  anariUnmapArray(d, largerData);
  CHECK(!a.renderAndCheck(1), "shared field data did not reset B");
  CHECK(!a.renderAndCheck(0), "shared field data did not reset A");
  CHECK(a.counts().groupBLAS == sharedA.groupBLAS
          && b.counts().groupBLAS == sharedB.groupBLAS,
      "shared field data rebuilt BLAS");
  CHECK(a.counts().worldTLAS == sharedA.worldTLAS
          && b.counts().worldTLAS == sharedB.worldTLAS,
      "shared field data rebuilt TLAS");

  auto rectilinear =
      anari::newObject<anari::SpatialField>(d, "structuredRectilinear");
  std::array<anari::Array1D, 3> coords{};
  const std::array<const char *, 3> coordNames{"coordsX", "coordsY", "coordsZ"};
  for (size_t axis = 0; axis < coords.size(); ++axis) {
    const size_t count = axis == 0 ? 3 : 2;
    coords[axis] =
        anariNewArray1D(d, nullptr, nullptr, nullptr, ANARI_FLOAT32, count);
    auto *c = static_cast<float *>(anariMapArray(d, coords[axis]));
    for (size_t i = 0; i < count; ++i)
      c[i] = float(i);
    anariUnmapArray(d, coords[axis]);
    anari::setParameter(d, rectilinear, coordNames[axis], coords[axis]);
  }
  anari::setParameter(d, rectilinear, "data", largerData);
  anari::commitParameters(d, rectilinear);
  anari::setParameter(d, volumes[0], "value", rectilinear);
  anari::commitParameters(d, volumes[0]);
  a.renderAndCheck(0);
  a.renderAndCheck(0);
  a.renderAndCheck(1);
  auto *xCoords = static_cast<float *>(anariMapArray(d, coords[0]));
  xCoords[1] = 0.8f;
  anariUnmapArray(d, coords[0]);
  checkEdit("rectilinear interior coordinate", false);
  xCoords = static_cast<float *>(anariMapArray(d, coords[0]));
  xCoords[2] = 2.2f;
  anariUnmapArray(d, coords[0]);
  checkEdit("rectilinear endpoint", true);
  for (auto array : coords)
    anari::release(d, array);
  anari::release(d, rectilinear);
  anari::release(d, replacement);
  anari::release(d, largerData);
  anari::release(d, colors);
  anari::release(d, opacity);
  for (size_t i = 0; i < fields.size(); ++i) {
    anari::release(d, volumes[i]);
    anari::release(d, fields[i]);
    anari::release(d, data[i]);
  }
}

// Managed arrays ensure unmaps, rather than object commits, drive the edits.
anari::Array1D makeOwnerArray(anari::Device d, ANARIDataType type, size_t count)
{
  auto array = anariNewArray1D(d, nullptr, nullptr, nullptr, type, count);
  auto *mapped = anariMapArray(d, array);
  if (type == ANARI_UINT32) {
    std::fill_n(static_cast<uint32_t *>(mapped), count, 0u);
  } else {
    const size_t channels = type == ANARI_FLOAT32_VEC3 ? 3 : 1;
    std::fill_n(static_cast<float *>(mapped), count * channels, 0.1f);
  }
  anariUnmapArray(d, array);
  return array;
}

void testGeometryArrayOwners(anari::Device d)
{
  Harness a(d), b(d);
  for (auto renderer : {a.ren, b.ren}) {
    anari::setParameter(d, renderer, "sampleLimit", 0);
    anari::commitParameters(d, renderer);
  }
  auto warm = [&]() {
    for (int i = 0; i < 3; ++i) {
      a.renderAndCheck(0);
      b.renderAndCheck(0);
    }
  };
  auto check = [&](const char *subtype, const char *param, bool reset) {
    const auto before = a.counts();
    CHECK(b.renderAndCheck(0), "%s %s reset unrelated B", subtype, param);
    CHECK(a.renderAndCheck(0) != reset,
        "%s %s: expected A reset=%d",
        subtype,
        param,
        reset);
    CHECK((a.counts().groupBLAS > before.groupBLAS) == reset,
        "%s %s: owner did not re-finalize conservatively",
        subtype,
        param);
    CHECK(a.renderAndCheck(0), "%s %s reset A twice", subtype, param);
  };

  for (const char *subtype :
      {"sphere", "triangle", "quad", "cylinder", "cone", "curve"}) {
    auto geom = anari::newObject<anari::Geometry>(d, subtype);
    const size_t count = std::string(subtype) == "triangle" ? 3 : 4;
    auto positions = makeOwnerArray(d, ANARI_FLOAT32_VEC3, count);
    auto *p = static_cast<float3 *>(anariMapArray(d, positions));
    p[0] = {-0.5f, -0.5f, 0.f};
    p[1] = {0.5f, -0.5f, 0.f};
    p[2] = {0.5f, 0.5f, 0.f};
    if (count == 4)
      p[3] = {-0.5f, 0.5f, 0.f};
    anariUnmapArray(d, positions);
    anari::setParameter(d, geom, "vertex.position", positions);
    anari::setParameter(d, geom, "radius", 0.1f);
    auto coneRadius = makeOwnerArray(d, ANARI_FLOAT32, count);
    if (std::string(subtype) == "cone")
      anari::setParameter(d, geom, "vertex.radius", coneRadius);
    anari::commitParameters(d, geom);
    anari::setParameter(d, a.surf, "geometry", geom);
    anari::commitParameters(d, a.surf);
    warm();

    std::vector<std::string> params{"primitive.id"};
    for (const char *prefix : {"primitive.", "vertex."}) {
      for (const char *attr :
          {"attribute0", "attribute1", "attribute2", "attribute3", "color"})
        params.push_back(std::string(prefix) + attr);
    }
    const std::string type(subtype);
    if (type == "triangle") {
      for (const char *attr : {"attribute0",
               "attribute1",
               "attribute2",
               "attribute3",
               "color",
               "normal",
               "tangent"})
        params.push_back(std::string("faceVarying.") + attr);
      params.push_back("vertex.tangent");
    }
    if (type == "triangle" || type == "quad")
      params.push_back("vertex.normal");
    if (type == "cylinder")
      params.push_back("primitive.radius");
    if (type == "cone" || type == "curve" || type == "sphere")
      params.push_back("vertex.radius");

    for (const auto &param : params) {
      const auto elementType = param == "primitive.id" ? ANARI_UINT32
          : param.find("radius") != std::string::npos  ? ANARI_FLOAT32
                                                       : ANARI_FLOAT32_VEC3;
      auto array = makeOwnerArray(d, elementType, count);
      auto replacement = makeOwnerArray(d, elementType, count);
      anari::setParameter(d, geom, param.c_str(), array);
      anari::commitParameters(d, geom);
      warm();
      auto *data = anariMapArray(d, array);
      if (elementType == ANARI_UINT32)
        static_cast<uint32_t *>(data)[0] = 7;
      else
        static_cast<float *>(data)[0] = 0.2f;
      anariUnmapArray(d, array);
      check(subtype, param.c_str(), true);
      anariMapArray(d, array);
      anariUnmapArray(d, array);
      check(subtype, param.c_str(), true);

      anari::setParameter(d, geom, param.c_str(), replacement);
      anari::commitParameters(d, geom);
      warm();
      anariMapArray(d, array);
      anariUnmapArray(d, array);
      check(subtype, param.c_str(), false);
      anariMapArray(d, replacement);
      anariUnmapArray(d, replacement);
      check(subtype, param.c_str(), true);
      anari::unsetParameter(d, geom, param.c_str());
      if (type == "cone" && param == "vertex.radius")
        anari::setParameter(d, geom, param.c_str(), coneRadius);
      anari::commitParameters(d, geom);
      warm();
      anariMapArray(d, replacement);
      anariUnmapArray(d, replacement);
      check(subtype, param.c_str(), false);
      anari::release(d, array);
      anari::release(d, replacement);
    }
    // Aliased bindings must retain the edge until the last binding is removed.
    auto alias = makeOwnerArray(d, ANARI_FLOAT32_VEC3, count);
    anari::setParameter(d, geom, "primitive.attribute0", alias);
    anari::setParameter(d, geom, "vertex.attribute0", alias);
    anari::commitParameters(d, geom);
    warm();
    anari::unsetParameter(d, geom, "vertex.attribute0");
    anari::commitParameters(d, geom);
    warm();
    anariMapArray(d, alias);
    anariUnmapArray(d, alias);
    check(subtype, "one alias removed", true);
    anari::unsetParameter(d, geom, "primitive.attribute0");
    anari::commitParameters(d, geom);
    warm();
    anariMapArray(d, alias);
    anariUnmapArray(d, alias);
    check(subtype, "last alias removed", false);
    anari::release(d, alias);

    // Two worlds observing the same owner must consume its change separately.
    anari::setParameter(d, b.surf, "geometry", geom);
    anari::commitParameters(d, b.surf);
    warm();
    anariMapArray(d, positions);
    anariUnmapArray(d, positions);
    CHECK(!b.renderAndCheck(0), "%s shared array did not reset B", subtype);
    CHECK(!a.renderAndCheck(0), "%s shared array did not reset A", subtype);
    anari::setParameter(d, b.surf, "geometry", b.geom);
    anari::commitParameters(d, b.surf);
    anari::release(d, geom);
    anari::release(d, positions);
    anari::release(d, coneRadius);
  }
}

void testFieldAndLightArrayOwners(anari::Device d)
{
  Harness a(d), b(d);
  auto warm = [&]() {
    for (int i = 0; i < 3; ++i) {
      a.renderAndCheck(0);
      b.renderAndCheck(0);
    }
  };
  auto check = [&](const char *edit, bool reset, bool geometry) {
    const auto before = a.counts();
    CHECK(b.renderAndCheck(0), "%s reset unrelated B", edit);
    CHECK(a.renderAndCheck(0) != reset, "%s expected A reset=%d", edit, reset);
    const auto after = a.counts();
    CHECK((after.groupBLAS > before.groupBLAS) == (reset && geometry),
        "%s incorrect BLAS work",
        edit);
    CHECK(a.renderAndCheck(0), "%s reset twice", edit);
  };
  auto data =
      anariNewArray3D(d, nullptr, nullptr, nullptr, ANARI_FLOAT32, 4, 4, 4);
  auto fillField = [&](anari::Array3D array, float scale) {
    auto *v = static_cast<float *>(anariMapArray(d, array));
    for (int i = 0; i < 64; ++i)
      v[i] = scale * float(i / 16);
    anariUnmapArray(d, array);
  };
  fillField(data, 1.f);
  auto field = anari::newObject<anari::SpatialField>(d, "structuredRegular");
  anari::setParameter(d, field, "data", data);
  anari::commitParameters(d, field);
  auto iso = anari::newObject<anari::Geometry>(d, "isosurface");
  auto values = makeOwnerArray(d, ANARI_FLOAT32, 1);
  auto *v = static_cast<float *>(anariMapArray(d, values));
  v[0] = 1.5f;
  anariUnmapArray(d, values);
  anari::setParameter(d, iso, "field", field);
  anari::setParameter(d, iso, "isovalue", values);
  anari::commitParameters(d, iso);
  anari::setParameter(d, a.surf, "geometry", iso);
  anari::commitParameters(d, a.surf);
  warm();
  fillField(data, 2.f);
  check("field to isosurface repacking", true, true);
  anariMapArray(d, data);
  anariUnmapArray(d, data);
  check("identical field to isosurface", true, true);
  v = static_cast<float *>(anariMapArray(d, values));
  v[0] = 2.f;
  anariUnmapArray(d, values);
  check("isovalue unmap", true, true);
  anari::setParameter(d, iso, "isovalue", 1.f);
  anari::commitParameters(d, iso);
  warm();
  anariMapArray(d, values);
  anariUnmapArray(d, values);
  check("removed isovalue array", false, true);
  auto newData =
      anariNewArray3D(d, nullptr, nullptr, nullptr, ANARI_FLOAT32, 4, 4, 4);
  fillField(newData, 1.f);
  anari::setParameter(d, field, "data", newData);
  anari::commitParameters(d, field);
  warm();
  fillField(data, 3.f);
  check("replaced field data", false, true);

  auto hdri = anari::newObject<anari::Light>(d, "hdri");
  auto makeRadiance = [&]() {
    auto r =
        anariNewArray2D(d, nullptr, nullptr, nullptr, ANARI_FLOAT32_VEC3, 4, 2);
    auto *p = static_cast<float3 *>(anariMapArray(d, r));
    std::fill_n(p, 8, float3{1.f, 1.f, 1.f});
    anariUnmapArray(d, r);
    return r;
  };
  auto radiance = makeRadiance();
  anari::setParameter(d, hdri, "radiance", radiance);
  anari::commitParameters(d, hdri);
  anari::setParameterArray1D(d, a.group, "light", &hdri, 1);
  anari::commitParameters(d, a.group);
  warm();
  auto *r = static_cast<float3 *>(anariMapArray(d, radiance));
  r[0] = {2.f, 1.f, 1.f};
  anariUnmapArray(d, radiance);
  check("HDRI radiance", true, false);
  anariMapArray(d, radiance);
  anariUnmapArray(d, radiance);
  check("identical HDRI radiance", true, false);
  auto newRadiance = makeRadiance();
  anari::setParameter(d, hdri, "radiance", newRadiance);
  anari::commitParameters(d, hdri);
  warm();
  anariMapArray(d, radiance);
  anariUnmapArray(d, radiance);
  check("replaced HDRI radiance", false, false);
  anari::unsetParameter(d, hdri, "radiance");
  anari::commitParameters(d, hdri);
  warm();
  anariMapArray(d, newRadiance);
  anariUnmapArray(d, newRadiance);
  check("removed HDRI radiance", false, false);
  anari::release(d, newRadiance);
  anari::release(d, radiance);
  anari::release(d, hdri);
  anari::release(d, newData);
  anari::release(d, values);
  anari::release(d, iso);
  anari::release(d, field);
  anari::release(d, data);
}

void testRebuildCounted(anari::Device d)
{
  // Harness self-check: a geometry change rebuilds the Group's BLAS and the
  // World's TLAS, and the counters see it.
  Harness h(d);
  h.renderAndCheck(0);
  h.renderAndCheck(1);
  const auto before = h.counts();
  anari::setParameter(d, h.geom, "radius", 0.06f);
  anari::commitParameters(d, h.geom);
  h.renderAndCheck(0);
  const auto after = h.counts();
  CHECK(after.groupBLAS > before.groupBLAS,
      "geometry change did not rebuild BLAS (%u -> %u)",
      before.groupBLAS,
      after.groupBLAS);
  CHECK(after.worldTLAS > before.worldTLAS,
      "geometry change did not rebuild TLAS (%u -> %u)",
      before.worldTLAS,
      after.worldTLAS);
}

} // namespace

int main()
{
  auto d = makeVisRTXDevice(statusFunc);
  if (!d) {
    std::fprintf(stderr, "failed to create VisRTX device\n");
    return EXIT_FAILURE;
  }

  testBaselineTwoFramesAccumulate(d);
  testResetDetected(d);
  testFrameDependencies(d);
  testManualResize(d);
  testRendererPostOnly(d);
  testRendererSubtypeAndBackground(d);
  testGuideHistory(d);
  testCameraSubtypeParameters(d);
  testWorldDependencies(d, false);
  testWorldDependencies(d, true);
  testRebuildCounted(d);
  testScopedBLAS(d);
  testScopedTLAS(d);
  testScopedLightSets(d, false);
  testScopedLightSets(d, true);
  testScopedVolumeBLAS(d);
  testGeometryArrayOwners(d);
  testFieldAndLightArrayOwners(d);

  anari::release(d, d);

  if (g_failures) {
    std::fprintf(stderr, "%d failure(s)\n", g_failures);
    return EXIT_FAILURE;
  }
  std::printf("PASS\n");
  return EXIT_SUCCESS;
}
