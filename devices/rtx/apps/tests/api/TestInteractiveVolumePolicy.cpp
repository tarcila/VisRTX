// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <anari/anari.h>
#include <anari/ext/visrtx/makeVisRTXDevice.h>

#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {
void require(bool condition, const char *message)
{
  if (!condition)
    throw std::runtime_error(message);
}

struct Context
{
  ANARIDevice device{};
  std::vector<ANARIObject> objects;
  std::atomic<int> warnings{0}, errors{0};
  Context();
  ~Context();
  template <typename T>
  T keep(T object);
  static void status(const void *,
      ANARIDevice,
      ANARIObject,
      ANARIDataType,
      ANARIStatusSeverity,
      ANARIStatusCode,
      const char *);
};
void Context::status(const void *user,
    ANARIDevice,
    ANARIObject,
    ANARIDataType,
    ANARIStatusSeverity severity,
    ANARIStatusCode,
    const char *message)
{
  auto &c = *static_cast<Context *>(const_cast<void *>(user));
  if (std::strstr(message, "effective denoiseMode is color"))
    ++c.warnings;
  if (severity <= ANARI_SEVERITY_ERROR) {
    ++c.errors;
    std::fprintf(stderr, "%s\n", message);
  }
}
Context::Context()
{
  device = makeVisRTXDevice(status, this);
  require(device != nullptr, "device creation");
  anariCommitParameters(device, device);
}
Context::~Context()
{
  for (auto i = objects.rbegin(); i != objects.rend(); ++i)
    anariRelease(device, *i);
  anariRelease(device, device);
}
template <typename T>
T Context::keep(T object)
{
  require(object != nullptr, "object creation");
  objects.push_back(reinterpret_cast<ANARIObject>(object));
  return object;
}
void discovery(ANARIDevice d)
{
  for (const char *subtype : {"interactive", "default"}) {
    for (const char *name :
        {"volumeSamplingRate", "volumeSamplingRateShadows"}) {
      auto info = [&](const char *key) {
        return static_cast<const float *>(anariGetParameterInfo(d,
            ANARI_RENDERER,
            subtype,
            name,
            ANARI_FLOAT32,
            key,
            ANARI_FLOAT32));
      };
      const auto *value = info("default"), *minimum = info("minimum");
      require(value && *value == 1.f, "experimental rate default");
      require(minimum
              && *minimum
                  == (std::strcmp(name, "volumeSamplingRate") == 0 ? 1e-3f
                                                                   : 1e-4f),
          "rate minimum");
      require(!info("maximum"), "obsolete rate ceiling");
      auto desc = static_cast<const char *>(anariGetParameterInfo(d,
          ANARI_RENDERER,
          subtype,
          name,
          ANARI_FLOAT32,
          "description",
          ANARI_STRING));
      require(desc && std::strstr(desc, "experimental")
              && std::strstr(desc, "finite"),
          "rate qualification disclosure");
    }
    auto mode = static_cast<const char *>(anariGetParameterInfo(d,
        ANARI_RENDERER,
        subtype,
        "denoiseMode",
        ANARI_STRING,
        "default",
        ANARI_STRING));
    require(mode && std::strcmp(mode, "color") == 0, "denoise default");
    auto enabled = static_cast<const int *>(anariGetParameterInfo(d,
        ANARI_RENDERER,
        subtype,
        "denoise",
        ANARI_BOOL,
        "default",
        ANARI_BOOL));
    require(enabled && !*enabled, "denoise disabled default");
    auto start = static_cast<const int *>(anariGetParameterInfo(d,
        ANARI_RENDERER,
        subtype,
        "denoiseStart",
        ANARI_INT32,
        "default",
        ANARI_INT32));
    require(start && *start == 0, "denoise start default");
  }
}
} // namespace

int main()
try {
  Context c;
  auto d = c.device;
  discovery(d);
  auto commit = [&](ANARIObject o) { anariCommitParameters(d, o); };
  auto set =
      [&](ANARIObject o, const char *name, ANARIDataType type, const auto &v) {
        anariSetParameter(d, o, name, type, &v);
      };
  auto renderer = c.keep(anariNewRenderer(d, "interactive"));
  int enabled = 1, disabled = 0;
  set(renderer, "denoise", ANARI_BOOL, enabled);
  anariSetParameter(d, renderer, "fireflyFilterMode", ANARI_STRING, "none");
  commit(renderer);
  auto camera = c.keep(anariNewCamera(d, "orthographic"));
  set(camera,
      "position",
      ANARI_FLOAT32_VEC3,
      std::array<float, 3>{.5f, .5f, 2});
  set(camera, "direction", ANARI_FLOAT32_VEC3, std::array<float, 3>{0, 0, -1});
  set(camera, "height", ANARI_FLOAT32, .5f);
  commit(camera);
  auto volume = c.keep(anariNewVolume(d, "principledVolume"));
  set(volume,
      "bounds",
      ANARI_FLOAT32_BOX3,
      std::array<float, 6>{0, 0, 0, 1, 1, 1});
  set(volume, "density", ANARI_FLOAT32, 0.f);
  set(volume, "emission", ANARI_FLOAT32_VEC3, std::array<float, 3>{1, 1, 1});
  commit(volume);
  auto array =
      c.keep(anariNewArray1D(d, nullptr, nullptr, nullptr, ANARI_VOLUME, 1));
  *static_cast<ANARIVolume *>(anariMapArray(d, array)) = volume;
  anariUnmapArray(d, array);
  auto world = c.keep(anariNewWorld(d));
  set(world, "volume", ANARI_ARRAY1D, array);
  commit(world);
  auto frame = c.keep(anariNewFrame(d));
  set(frame, "size", ANARI_UINT32_VEC2, std::array<unsigned, 2>{4, 4});
  set(frame, "channel.color", ANARI_DATA_TYPE, ANARI_FLOAT32_VEC4);
  set(frame, "channel.albedo", ANARI_DATA_TYPE, ANARI_FLOAT32_VEC3);
  set(frame, "channel.normal", ANARI_DATA_TYPE, ANARI_FLOAT32_VEC3);
  set(frame, "world", ANARI_WORLD, world);
  set(frame, "camera", ANARI_CAMERA, camera);
  set(frame, "renderer", ANARI_RENDERER, renderer);
  commit(frame);
  auto render = [&](int warnings) {
    anariRenderFrame(d, frame);
    anariFrameReady(d, frame, ANARI_WAIT);
    require(c.errors == 0, "unexpected API error");
    require(c.warnings == warnings, "warning revision/deduplication");
    uint32_t w, h;
    ANARIDataType type;
    const auto *pixels = static_cast<const float *>(
        anariMapFrame(d, frame, "channel.color", &w, &h, &type));
    require(
        pixels && w == 4 && h == 4 && type == ANARI_FLOAT32_VEC4, "color map");
    for (int i = 0; i < 64; ++i)
      require(std::isfinite(pixels[i]), "nonfinite output");
    anariUnmapFrame(d, frame, "channel.color");
  };
  auto mode = [&](const char *value) {
    anariSetParameter(d, renderer, "denoiseMode", ANARI_STRING, value);
    commit(renderer);
  };
  render(0); // Unset mode means color, matching discovery.
  mode("colorAlbedoNormal");
  render(1);
  render(1);
  mode("colorAlbedo");
  render(2);
  render(2);
  mode("color");
  render(2);
  mode("colorAlbedoNormal");
  render(3);
  set(renderer, "denoise", ANARI_BOOL, disabled);
  commit(renderer);
  render(3);
  set(renderer, "denoise", ANARI_BOOL, enabled);
  commit(renderer);
  render(4);
  renderer = c.keep(anariNewRenderer(d, "default"));
  set(renderer, "denoise", ANARI_BOOL, enabled);
  mode("colorAlbedoNormal");
  set(frame, "renderer", ANARI_RENDERER, renderer);
  commit(frame);
  render(5);
  render(5);
  set(volume, "emission", ANARI_FLOAT32_VEC3, std::array<float, 3>{2, 2, 2});
  commit(volume);
  render(6);
  render(6);
  anariUnsetParameter(d, world, "volume");
  commit(world);
  render(6);
  set(world, "volume", ANARI_ARRAY1D, array);
  commit(world);
  render(7);
  render(7);
  std::puts("Interactive/default metadata and denoiser warning revisions pass");
  return 0;
} catch (const std::exception &e) {
  std::fprintf(stderr, "FAIL: %s\n", e.what());
  return 1;
}
