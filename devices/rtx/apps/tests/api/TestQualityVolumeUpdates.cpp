// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#include <anari/anari.h>
#include <anari/ext/visrtx/makeVisRTXDevice.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {
using V3 = std::array<float, 3>;
using V4 = std::array<float, 4>;
constexpr unsigned SIZE = 24;
constexpr int SPP = 16;
struct State
{
  float cameraX = 0.f, lightX = 0.f, irradiance = .15f;
  float translate = 0.f, scale = 1.f, density = 1.f, temperature = 1800.f;
  float reflectance = .7f, intensity = .02f, emission = 1.f;
  bool reverse = false;
  uint32_t randomSeed = 0;
};
struct Image
{
  std::vector<V4> pixels;
  int samples = 0;
};
static void require(bool condition, const char *message)
{
  if (!condition)
    throw std::runtime_error(message);
}
struct Scene
{
  explicit Scene(const State &state, ANARIDevice sharedDevice = nullptr);
  ~Scene();
  Scene(const Scene &) = delete;
  Scene &operator=(const Scene &) = delete;
  template <typename T>
  T keep(T object);
  template <typename T>
  void set(
      ANARIObject object, const char *name, ANARIDataType type, const T &value);
  void commit(ANARIObject object);
  ANARIArray1D array(
      ANARIDataType type, const void *values, size_t count, size_t bytes);
  ANARISpatialField field(ANARIArray3D &data, ANARIDataType type);
  void apply(const State &state, bool force = false);
  Image render();
  static void status(const void *,
      ANARIDevice,
      ANARIObject,
      ANARIDataType,
      ANARIStatusSeverity,
      ANARIStatusCode,
      const char *);
  ANARIDevice d{};
  ANARIWorld world{};
  ANARICamera camera{};
  ANARILight light{};
  ANARIMaterial material{};
  ANARIVolume volume{};
  ANARIInstance instance{};
  ANARIFrame frame{};
  ANARIRenderer renderer{};
  ANARIArray3D density{}, temperature{}, emission{};
  std::vector<ANARIObject> objects;
  State current;
  int errors = 0;
};
void Scene::status(const void *user,
    ANARIDevice,
    ANARIObject,
    ANARIDataType,
    ANARIStatusSeverity severity,
    ANARIStatusCode,
    const char *message)
{
  if (severity <= ANARI_SEVERITY_ERROR) {
    ++const_cast<Scene *>(static_cast<const Scene *>(user))->errors;
    std::fprintf(stderr, "ANARI: %s\n", message);
  }
}
template <typename T>
T Scene::keep(T object)
{
  objects.push_back((ANARIObject)object);
  return object;
}
template <typename T>
void Scene::set(ANARIObject o, const char *n, ANARIDataType t, const T &v)
{
  anariSetParameter(d, o, n, t, &v);
}
void Scene::commit(ANARIObject o)
{
  anariCommitParameters(d, o);
}
ANARIArray1D Scene::array(
    ANARIDataType type, const void *values, size_t count, size_t bytes)
{
  auto a = keep(anariNewArray1D(d, nullptr, nullptr, nullptr, type, count));
  std::memcpy(anariMapArray(d, a), values, bytes);
  anariUnmapArray(d, a);
  return a;
}
ANARISpatialField Scene::field(ANARIArray3D &data, ANARIDataType type)
{
  data = keep(anariNewArray3D(d, nullptr, nullptr, nullptr, type, 2, 2, 2));
  auto f = keep(anariNewSpatialField(d, "structuredRegular"));
  set(f, "data", ANARI_ARRAY3D, data);
  set(f, "origin", ANARI_FLOAT32_VEC3, V3{.75f, -.25f, .75f});
  set(f, "spacing", ANARI_FLOAT32_VEC3, V3{.5f, .5f, .5f});
  anariSetParameter(d, f, "filter", ANARI_STRING, "linear");
  commit(f);
  return f;
}
Scene::Scene(const State &state, ANARIDevice sharedDevice)
{
  if (sharedDevice) {
    d = sharedDevice;
    anariRetain(d, d);
  } else {
    d = makeVisRTXDevice(status, this);
    commit(d);
  }
  volume = keep(anariNewVolume(d, "principledVolume"));
  set(volume,
      "bounds",
      ANARI_FLOAT32_BOX3,
      std::array<float, 6>{.75f, -.25f, .75f, 1.25f, .25f, 1.25f});
  set(volume, "transmissionColor", ANARI_FLOAT32_VEC3, V3{.6f, .7f, .8f});
  set(volume, "scatteringColor", ANARI_FLOAT32_VEC3, V3{.2f, .3f, .4f});
  auto f = field(density, ANARI_FLOAT32);
  set(volume, "density.field", ANARI_SPATIAL_FIELD, f);
  f = field(temperature, ANARI_FLOAT32);
  set(volume, "temperature.field", ANARI_SPATIAL_FIELD, f);
  f = field(emission, ANARI_FLOAT32_VEC3);
  set(volume, "emission.field", ANARI_SPATIAL_FIELD, f);
  auto group = keep(anariNewGroup(d));
  auto a = array(ANARI_VOLUME, &volume, 1, sizeof(volume));
  set(group, "volume", ANARI_ARRAY1D, a);
  commit(group);
  instance = keep(anariNewInstance(d, "transform"));
  set(instance, "group", ANARI_GROUP, group);
  auto geometry = keep(anariNewGeometry(d, "quad"));
  const std::array<V3, 4> points{
      {{-2, -2, 0}, {2, -2, 0}, {2, 2, 0}, {-2, 2, 0}}};
  a = array(ANARI_FLOAT32_VEC3, points.data(), 4, sizeof(points));
  set(geometry, "vertex.position", ANARI_ARRAY1D, a);
  commit(geometry);
  material = keep(anariNewMaterial(d, "matte"));
  auto surface = keep(anariNewSurface(d));
  set(surface, "geometry", ANARI_GEOMETRY, geometry);
  set(surface, "material", ANARI_MATERIAL, material);
  commit(surface);
  light = keep(anariNewLight(d, "directional"));
  world = keep(anariNewWorld(d));
  a = array(ANARI_INSTANCE, &instance, 1, sizeof(instance));
  set(world, "instance", ANARI_ARRAY1D, a);
  a = array(ANARI_SURFACE, &surface, 1, sizeof(surface));
  set(world, "surface", ANARI_ARRAY1D, a);
  a = array(ANARI_LIGHT, &light, 1, sizeof(light));
  set(world, "light", ANARI_ARRAY1D, a);
  commit(world);
  camera = keep(anariNewCamera(d, "orthographic"));
  set(camera, "direction", ANARI_FLOAT32_VEC3, V3{0, 0, -1});
  set(camera, "up", ANARI_FLOAT32_VEC3, V3{0, 1, 0});
  set(camera, "height", ANARI_FLOAT32, 1.f);
  renderer = keep(anariNewRenderer(d, "quality"));
  set(renderer, "pixelSamples", ANARI_INT32, SPP);
  set(renderer, "sampleLimit", ANARI_INT32, 0);
  set(renderer, "maxRayDepth", ANARI_INT32, 32);
  set(renderer, "background", ANARI_FLOAT32_VEC4, V4{0, 0, 0, 0});
  anariSetParameter(d, renderer, "fireflyFilterMode", ANARI_STRING, "none");
  commit(renderer);
  apply(state, true);
  frame = keep(anariNewFrame(d));
  set(frame, "size", ANARI_UINT32_VEC2, std::array<unsigned, 2>{SIZE, SIZE});
  set(frame, "channel.color", ANARI_DATA_TYPE, ANARI_FLOAT32_VEC4);
  set(frame, "renderer", ANARI_RENDERER, renderer);
  set(frame, "camera", ANARI_CAMERA, camera);
  set(frame, "world", ANARI_WORLD, world);
  commit(frame);
}
Scene::~Scene()
{
  for (auto i = objects.rbegin(); i != objects.rend(); ++i)
    anariRelease(d, *i);
  anariRelease(d, d);
}
void Scene::apply(const State &s, bool force)
{
  if (force || s.cameraX != current.cameraX) {
    set(camera, "position", ANARI_FLOAT32_VEC3, V3{s.cameraX, 0, 3});
    commit(camera);
  }
  if (force || s.lightX != current.lightX
      || s.irradiance != current.irradiance) {
    set(light, "direction", ANARI_FLOAT32_VEC3, V3{s.lightX, 0, -1});
    set(light, "irradiance", ANARI_FLOAT32, s.irradiance);
    commit(light);
  }
  if (force || s.translate != current.translate || s.scale != current.scale) {
    std::array<float, 16> transform{
        s.scale, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, s.translate, 0, 0, 1};
    set(instance, "transform", ANARI_FLOAT32_MAT4, transform);
    commit(instance);
  }
  if (force || s.reflectance != current.reflectance) {
    set(material,
        "color",
        ANARI_FLOAT32_VEC3,
        V3{s.reflectance, s.reflectance, s.reflectance});
    commit(material);
  }
  if (force || s.density != current.density) {
    auto p = static_cast<float *>(anariMapArray(d, density));
    for (int i = 0; i < 8; ++i)
      p[i] = s.density * (i % 2 ? .5f : 1.f);
    anariUnmapArray(d, density);
  }
  if (force || s.temperature != current.temperature) {
    auto p = static_cast<float *>(anariMapArray(d, temperature));
    for (int i = 0; i < 8; ++i)
      p[i] = s.temperature + (i % 2 ? 200.f : 0.f);
    anariUnmapArray(d, temperature);
  }
  if (force || s.reverse != current.reverse) {
    auto p = static_cast<V3 *>(anariMapArray(d, emission));
    for (int i = 0; i < 8; ++i) {
      float w = ((i % 2) != s.reverse) ? 2.f : 0.f;
      p[i] = {w, .2f * w, .1f * w};
    }
    anariUnmapArray(d, emission);
  }
  if (force || s.intensity != current.intensity
      || s.emission != current.emission) {
    set(volume, "blackbodyIntensity", ANARI_FLOAT32, s.intensity);
    set(volume,
        "emission",
        ANARI_FLOAT32_VEC3,
        V3{s.emission, s.emission, s.emission});
    commit(volume);
  }
  if (force || s.randomSeed != current.randomSeed) {
    set(renderer, "randomSeed", ANARI_UINT32, s.randomSeed);
    commit(renderer);
  }
  current = s;
}
Image Scene::render()
{
  anariRenderFrame(d, frame);
  anariFrameReady(d, frame, ANARI_WAIT);
  Image result;
  require(anariGetProperty(d,
              frame,
              "numSamples",
              ANARI_INT32,
              &result.samples,
              sizeof(result.samples),
              ANARI_WAIT),
      "sample property");
  uint32_t w, h;
  ANARIDataType type;
  const auto *p = static_cast<const V4 *>(
      anariMapFrame(d, frame, "channel.color", &w, &h, &type));
  require(
      p && w == SIZE && h == SIZE && type == ANARI_FLOAT32_VEC4, "frame map");
  result.pixels.assign(p, p + w * h);
  anariUnmapFrame(d, frame, "channel.color");
  require(errors == 0, "device errors");
  return result;
}
void compare(const Image &a, const Image &b, const char *label)
{
  require(a.samples == SPP && b.samples == SPP,
      "edited/fresh frame did not start a new accumulation");
  double maxError = 0.;
  for (size_t i = 0; i < a.pixels.size(); ++i)
    for (int c = 0; c < 4; ++c) {
      const double x = a.pixels[i][c], y = b.pixels[i][c];
      require(std::isfinite(x) && std::isfinite(y), "nonfinite updated image");
      maxError = std::max(maxError, std::abs(x - y));
      if (std::abs(x - y) > 2e-6 + 2e-5 * std::abs(y)) {
        std::fprintf(stderr,
            "%s pixel %zu channel %d: %.9g vs fresh %.9g\n",
            label,
            i,
            c,
            x,
            y);
        throw std::runtime_error("updated image differs from fresh scene");
      }
    }
  std::printf(
      "%s: samples=%d max paired error=%.9g\n", label, a.samples, maxError);
}
} // namespace
int main(int argc, char **argv)
{
  const bool separateDevices =
      argc == 2 && std::strcmp(argv[1], "--separate-devices") == 0;
  require(argc == 1 || separateDevices, "unknown option");
  State state;
  Scene live(state);
  (void)live.render();
  (void)live.render();
  require(live.render().samples > SPP, "static frame never accumulates");
  Image seedZero;
  const std::array<const char *, 14> names{"camera",
      "light direction",
      "light intensity",
      "instance translation",
      "instance scale",
      "density array",
      "temperature array",
      "emission array",
      "material",
      "source off",
      "source recovery",
      "restore all",
      "random seed",
      "seed zero recovery"};
  for (int cycle = 0; cycle < 2; ++cycle)
    for (size_t step = 0; step < names.size(); ++step) {
      switch (step) {
      case 0:
        state.cameraX = .15f;
        break;
      case 1:
        state.lightX = .6f;
        break;
      case 2:
        state.irradiance = .4f;
        break;
      case 3:
        state.translate = .2f;
        break;
      case 4:
        state.scale = 1.2f;
        break;
      case 5:
        state.density = 1.7f;
        break;
      case 6:
        state.temperature = 2200.f;
        break;
      case 7:
        state.reverse = true;
        break;
      case 8:
        state.reflectance = .3f;
        break;
      case 9:
        state.intensity = 0.f;
        state.emission = 0.f;
        break;
      case 10:
        state.intensity = .02f;
        state.emission = 1.f;
        break;
      case 11:
        state = State{};
        break;
      case 12:
        state.randomSeed = 0x91c375adu;
        break;
      case 13:
        state.randomSeed = 0;
        break;
      }
      live.apply(state);
      const auto updated = live.render();
      if (step == 11)
        seedZero = updated;
      if (step == 12)
        require(updated.pixels != seedZero.pixels, "randomSeed has no effect");
      if (step == 13)
        compare(updated, seedZero, "seed zero reproducibility");
      {
        Scene fresh(state, separateDevices ? nullptr : live.d);
        compare(updated, fresh.render(), names[step]);
      }
      (void)live.render();
      require(live.render().samples > SPP,
          "post-edit frame never resumes accumulation");
    }
  std::puts("Quality dynamic updates: 28 edited-vs-fresh scenes passed");
}
