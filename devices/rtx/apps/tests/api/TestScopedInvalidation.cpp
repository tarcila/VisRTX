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

void testScopedVolumeBLAS(anari::Device d)
{
  Harness a(d);
  Harness b(d);
  anari::setParameter(d, a.frames[1], "world", b.world);
  anari::commitParameters(d, a.frames[1]);
  const std::array<float, 8> values{0, 1, 0, 1, 0, 1, 0, 1};
  auto data = anariNewArray3D(
      d, values.data(), nullptr, nullptr, ANARI_FLOAT32, 2, 2, 2);
  std::array<anari::SpatialField, 2> fields{};
  std::array<anari::Volume, 2> volumes{};
  const std::array<anari::Group, 2> groups{a.group, b.group};
  for (size_t i = 0; i < fields.size(); ++i) {
    fields[i] = anari::newObject<anari::SpatialField>(d, "structuredRegular");
    anari::setParameter(d, fields[i], "data", data);
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
  for (size_t i = 0; i < fields.size(); ++i) {
    anari::release(d, volumes[i]);
    anari::release(d, fields[i]);
  }
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
  testCameraSubtypeParameters(d);
  testWorldDependencies(d, false);
  testWorldDependencies(d, true);
  testRebuildCounted(d);
  testScopedBLAS(d);
  testScopedTLAS(d);
  testScopedVolumeBLAS(d);

  anari::release(d, d);

  if (g_failures) {
    std::fprintf(stderr, "%d failure(s)\n", g_failures);
    return EXIT_FAILURE;
  }
  std::printf("PASS\n");
  return EXIT_SUCCESS;
}
