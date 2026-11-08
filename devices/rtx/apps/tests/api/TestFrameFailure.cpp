// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

// Public ANARI scene/render/map/status boundary. The controlled GPU fault is
// compiled only with VISRTX_BUILD_TESTS, not exposed as an ANARI parameter.
#include <anari/anari_cpp/ext/std.h>
#include <anari/ext/visrtx/makeVisRTXDevice.h>
#include <anari/anari_cpp.hpp>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <thread>

using Vec4 = std::array<float, 4>;
static std::atomic<int> failures{0};
static void check(bool ok, const char *message)
{
  if (!ok) {
    fprintf(stderr, "FAIL: %s\n", message);
    ++failures;
  }
}

struct State
{
  anari::Frame frame{};
  std::atomic<int> errors{0};
  std::atomic<int> completions{0};
  bool expectFailure{false};
  bool gateCompletion{false};
  std::atomic<bool> entered{false};
  std::atomic<bool> release{false};
};
static int samples(anari::Device d, anari::Frame f, uint32_t flags = ANARI_WAIT)
{
  int n = -1;
  check(anariGetProperty(d, f, "numSamples", ANARI_INT32, &n, sizeof(n), flags),
      "numSamples exists as INT32");
  return n;
}
static void readFailed(anari::Device d, anari::Frame f)
{
  check(samples(d, f) == 0, "failed result has zero samples (WAIT query)");
  check(samples(d, f, ANARI_NO_WAIT) == 0,
      "failed result has zero samples (NO_WAIT query)");
  for (const char *channel : {"color",
           "depth",
           "primitiveId",
           "objectId",
           "instanceId",
           "albedo",
           "normal"}) {
    for (const char *suffix : {"", "CUDA"}) {
      const std::string name = std::string("channel.") + channel + suffix;
      unsigned width = 99, height = 99;
      ANARIDataType type = ANARI_FLOAT32;
      const void *p = anariMapFrame(d, f, name.c_str(), &width, &height, &type);
      check(!p && !width && !height && type == ANARI_UNKNOWN,
          "every failed CPU/CUDA channel maps null/zero/UNKNOWN");
      anariUnmapFrame(d, f, name.c_str());
    }
  }
}
static Vec4 color(anari::Device d, anari::Frame f)
{
  auto p = anari::map<Vec4>(d, f, "channel.color");
  check(p.data && p.width == 8 && p.height == 8
          && p.pixelType == ANARI_FLOAT32_VEC4,
      "successful color maps");
  Vec4 result{};
  if (p.data)
    result = p.data[0];
  anari::unmap(d, f, "channel.color");
  return result;
}
static void status(const void *user,
    ANARIDevice d,
    ANARIObject source,
    ANARIDataType sourceType,
    ANARIStatusSeverity severity,
    ANARIStatusCode,
    const char *message)
{
  if (severity > ANARI_SEVERITY_ERROR)
    return;
  auto &s = *(State *)user;
  fprintf(stderr, "status: %s\n", message);
  check(s.expectFailure && source == s.frame && sourceType == ANARI_FRAME,
      "error is sourced from the failed Frame");
  const int previous = s.errors.fetch_add(1);
  check(previous < 3, "no recursively emitted failure diagnostics");
  if (source == s.frame && previous < 3)
    readFailed(d, s.frame);
  check(
      std::strstr(message, "majorant") || std::strstr(message, "missing ANARI"),
      "error identifies cause");
  if (std::strstr(message, "majorant"))
    check(std::strstr(message, "volume 17, instance 23"),
        "GPU record preserves failing volume and instance identity");
}
static void complete(const void *user, ANARIDevice d, ANARIFrame f)
{
  auto &s = *(State *)user;
  ++s.completions;
  if (s.expectFailure) {
    check(s.errors > 0, "error is published before completion callback");
    readFailed(d, f);
  } else {
    check(samples(d, f) > 0, "successful callback sees published count");
    color(d, f); // successful callbacks can use CUDA-backed accessors too
  }
  if (s.gateCompletion) {
    s.entered = true;
    while (!s.release.load())
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}
static bool await(const std::function<bool()> &predicate)
{
  const auto limit =
      std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (!predicate()) {
    if (std::chrono::steady_clock::now() > limit)
      return false;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return true;
}
static anari::Frame makeFrame(anari::Device d,
    anari::World w,
    anari::Camera c,
    anari::Renderer r,
    bool manual)
{
  auto f = anari::newObject<anari::Frame>(d);
  anari::setParameter(d, f, "size", std::array<unsigned, 2>{8, 8});
  anari::setParameter(d, f, "channel.color", ANARI_FLOAT32_VEC4);
  anari::setParameter(d, f, "channel.depth", ANARI_FLOAT32);
  for (const char *name :
      {"channel.primitiveId", "channel.objectId", "channel.instanceId"})
    anari::setParameter(d, f, name, ANARI_UINT32);
  for (const char *name : {"channel.albedo", "channel.normal"})
    anari::setParameter(d, f, name, ANARI_FLOAT32_VEC3);
  anari::setParameter(d, f, "world", w);
  anari::setParameter(d, f, "camera", c);
  anari::setParameter(d, f, "renderer", r);
  if (manual)
    anari::setParameter(d, f, "accumulationVersion", uint64_t(7));
  anari::commitParameters(d, f);
  return f;
}
static void render(anari::Device d, anari::Frame f)
{
  anari::render(d, f);
  check(anariFrameReady(d, f, ANARI_WAIT), "WAIT returns completion");
}
static void sameColor(const Vec4 &a, const Vec4 &b, const char *message)
{
  for (int c = 0; c < 4; ++c)
    check(std::abs(a[c] - b[c]) < 1e-5f, message);
}
static void runCase(const char *renderer, bool manual, bool denoise)
{
  fprintf(
      stderr, "case: %s, manual=%d, denoise=%d\n", renderer, manual, denoise);
  State s;
  auto d = makeVisRTXDevice(status, &s);
  check(d != nullptr, "device exists");
  if (!d)
    return;
  // Nonzero surface radiance exercises accumulation, not only the background
  // compositor. The quad fills every pixel with constant emission.
  using Vec3 = std::array<float, 3>;
  const Vec3 positions[] = {
      {-2.f, -2.f, 0.f}, {2.f, -2.f, 0.f}, {2.f, 2.f, 0.f}, {-2.f, 2.f, 0.f}};
  auto geometry = anari::newObject<anari::Geometry>(d, "quad");
  anari::setParameterArray1D(d, geometry, "vertex.position", positions, 4);
  anari::commitParameters(d, geometry);
  auto material = anari::newObject<anari::Material>(d, "physicallyBased");
  anari::setParameter(d, material, "baseColor", Vec3{0.f, 0.f, 0.f});
  anari::setParameter(d, material, "metallic", 0.f);
  anari::setParameter(d, material, "specular", 0.f);
  anari::setParameter(d, material, "emissive", Vec3{0.8f, 0.1f, 0.2f});
  anari::commitParameters(d, material);
  auto surface = anari::newObject<anari::Surface>(d);
  anari::setAndReleaseParameter(d, surface, "geometry", geometry);
  anari::setParameter(d, surface, "material", material);
  anari::commitParameters(d, surface);
  auto w = anari::newObject<anari::World>(d);
  anari::setParameterArray1D(d, w, "surface", &surface, 1);
  anari::release(d, surface);
  anari::commitParameters(d, w);
  auto c = anari::newObject<anari::Camera>(d, "orthographic");
  anari::setParameter(d, c, "position", Vec3{0.f, 0.f, 2.f});
  anari::setParameter(d, c, "direction", Vec3{0.f, 0.f, -1.f});
  anari::setParameter(d, c, "up", Vec3{0.f, 1.f, 0.f});
  anari::setParameter(d, c, "height", 1.f);
  anari::commitParameters(d, c);
  auto r = anari::newObject<anari::Renderer>(d, renderer);
  anari::setParameter(d, r, "denoise", denoise);
  anari::setParameter(d, r, "fireflyFilterMode", "none");
  anari::setParameter(d, r, "pixelSamples", 1);
  anari::setParameter(d, r, "background", Vec4{0.f, 0.f, 0.f, 1.f});
  anari::setParameter(d, r, "ambientRadiance", 0.f);
  anari::commitParameters(d, r);
  s.frame = makeFrame(d, w, c, r, manual);
  auto other = makeFrame(d, w, c, r, manual);
  ANARIFrameCompletionCallback cb = complete;
  anariSetParameter(d,
      s.frame,
      "frameCompletionCallback",
      ANARI_FRAME_COMPLETION_CALLBACK,
      &cb);
  anari::setParameter(
      d, s.frame, "frameCompletionCallbackUserData", (void *)&s);
  anari::commitParameters(d, s.frame);
  // Flush all scene setup before accumulating either Frame.
  render(d, other);
  render(d, other);
  const int otherCount = samples(d, other);
  const Vec4 otherColor = color(d, other);
  check(otherColor[0] > 0.1f, "fixture accumulates nonzero surface radiance");
  // Existing accumulation bookkeeping consumes the initial dependency/version
  // reset on the second Launch. Warm it up before the three-sample assertion.
  render(d, s.frame);
  for (int i = 0; i < 3; ++i)
    render(d, s.frame);
  check(samples(d, s.frame) == 3, "valid samples accumulate before failure");
  check(s.completions == 4 && s.errors == 0,
      "successful callbacks run once each");

  s.expectFailure = true;
  s.gateCompletion = true;
  setenv("VISRTX_TEST_FRAME_FAILURE", "1", 1);
  anari::render(d, s.frame);
  unsetenv("VISRTX_TEST_FRAME_FAILURE");
  check(await([&] { return s.entered.load(); }),
      "completion reaches published outcome");
  check(!anariFrameReady(d, s.frame, ANARI_NO_WAIT),
      "publication does not imply completion of callback");
  std::atomic<bool> waitReturned{false};
  std::thread waiter([&] {
    check(anariFrameReady(d, s.frame, ANARI_WAIT), "failed WAIT returns true");
    waitReturned = true;
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  check(!waitReturned, "ordinary WAIT includes callback completion");
  s.release = true;
  waiter.join();
  s.gateCompletion = false;
  check(s.errors == 1 && s.completions == 5,
      "one error and one completion for failure");
  readFailed(d, s.frame);
  check(samples(d, other) == otherCount,
      "failure does not clear other Frame samples");
  sameColor(color(d, other),
      otherColor,
      "failure does not clear other Frame channels");

  // A retry with no correction revalidates; NO_WAIT must eventually complete.
  setenv("VISRTX_TEST_FRAME_FAILURE", "1", 1);
  anari::render(d, s.frame);
  unsetenv("VISRTX_TEST_FRAME_FAILURE");
  check(await([&] { return anariFrameReady(d, s.frame, ANARI_NO_WAIT); }),
      "NO_WAIT eventually completes on repeated failure");
  check(s.errors == 2 && s.completions == 6,
      "retry delivers exactly one completion");

  // Correction commits cannot resurrect the old output. Manual version stays 7.
  anari::setParameter(d, material, "emissive", Vec3{0.1f, 0.6f, 0.3f});
  anari::commitParameters(d, material);
  readFailed(d, s.frame);
  s.expectFailure = false;
  render(d, s.frame);
  check(samples(d, s.frame) == 1,
      "recovery forces fresh accumulation in either mode");
  auto fresh = makeFrame(d, w, c, r, manual);
  render(d, fresh);
  sameColor(
      color(d, s.frame), color(d, fresh), "recovery matches a fresh Frame");
  check(color(d, s.frame)[0] < otherColor[0] * 0.5f,
      "recovery does not retain old accumulated emission");
  check(
      samples(d, other) == otherCount, "recovery preserves other Frame count");
  sameColor(
      color(d, other), otherColor, "recovery preserves other Frame output");
  check(s.errors == 2 && s.completions == 7,
      "successful recovery clears failure");

  // Preflight uses the identical publication/callback policy.
  s.expectFailure = true;
  anari::unsetParameter(d, s.frame, "camera");
  anari::commitParameters(d, s.frame);
  render(d, s.frame);
  check(
      s.errors == 3 && s.completions == 8, "preflight failure completes once");
  readFailed(d, s.frame);
  for (auto o : {ANARIObject(fresh),
           ANARIObject(other),
           ANARIObject(s.frame),
           ANARIObject(r),
           ANARIObject(c),
           ANARIObject(w),
           ANARIObject(material)})
    anariRelease(d, o);
  anariRelease(d, d);
}
int main()
{
  unsetenv("VISRTX_TEST_FRAME_FAILURE");
  for (const char *r : {"quality", "interactive"})
    for (bool manual : {false, true})
      for (bool denoise : {false, true})
        runCase(r, manual, denoise);
  return failures ? 1 : 0;
}
