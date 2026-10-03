/*
 * Copyright (c) 2019-2026 NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

// Scoped invalidation harness (ADR 0009): two Frames sharing camera, renderer
// and world. Detects an Accumulation Reset per Frame through "numSamples" and
// counts BVH (BLAS/TLAS) and light-set rebuilds through the Group/World debug
// counters. The baseline case: two undisturbed Frames keep accumulating.

#include <anari/anari_cpp/ext/std.h>
#include <anari/ext/visrtx/makeVisRTXDevice.h>
#include <anari/anari_cpp.hpp>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
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
};

// Two-Frame scene derived from _tmp/repro.cpp.
struct Harness
{
  anari::Device d{nullptr};
  anari::Geometry geom{nullptr};
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
    anari::setParameterArray1D(
        d, geom, "vertex.position", pos.data(), pos.size());
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
    return c;
  }
};

void testBaselineTwoFramesAccumulate(anari::Device d)
{
  Harness h(d);

  // Warm-up: the first render of each frame builds everything. Today a
  // second Frame's first render still resets the first one once (the
  // cross-Frame reset tickets 02-03 remove), so warm up for two rounds.
  for (int s = 0; s < 2; s++) {
    h.renderAndCheck(0);
    h.renderAndCheck(1);
  }
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
  testRebuildCounted(d);

  anari::release(d, d);

  if (g_failures) {
    std::fprintf(stderr, "%d failure(s)\n", g_failures);
    return EXIT_FAILURE;
  }
  std::printf("PASS\n");
  return EXIT_SUCCESS;
}
