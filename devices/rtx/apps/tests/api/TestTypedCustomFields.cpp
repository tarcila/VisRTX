// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#include <anari/anari.h>
#include <anari/ext/visrtx/makeVisRTXDevice.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
namespace {
using V3 = std::array<float, 3>;
using V4 = std::array<float, 4>;
using Box = std::array<float, 6>;
constexpr unsigned SIZE = 24;
void require(bool ok, const char *message)
{
  if (!ok)
    throw std::runtime_error(message);
}
struct Scene
{
  Scene(
      const char *rgb = "testTypedRGBOptix", const char *renderer = "quality");
  ~Scene();
  template <typename T>
  T keep(T);
  template <typename T>
  void set(ANARIObject, const char *, ANARIDataType, const T &);
  void commit(ANARIObject);
  ANARIArray1D array(ANARIDataType, const void *, size_t, size_t);
  bool property(ANARIObject, const char *);
  std::vector<V4> read();
  std::vector<V4> render();
  void receiver();
  void release(ANARIObject);
  static void status(const void *,
      ANARIDevice,
      ANARIObject,
      ANARIDataType,
      ANARIStatusSeverity,
      ANARIStatusCode,
      const char *);
  ANARIDevice d{};
  ANARIWorld world{};
  ANARIVolume volume{};
  ANARISpatialField density{}, emission{};
  ANARICamera camera{};
  ANARIFrame frame{};
  std::vector<ANARIObject> objects;
  int errors = 0, warnings = 0;
};
void Scene::status(const void *p,
    ANARIDevice,
    ANARIObject,
    ANARIDataType,
    ANARIStatusSeverity severity,
    ANARIStatusCode,
    const char *message)
{
  auto &s = *const_cast<Scene *>(static_cast<const Scene *>(p));
  if (severity <= ANARI_SEVERITY_ERROR) {
    ++s.errors;
    std::fprintf(stderr, "ANARI: %s\n", message);
  }
  if (severity == ANARI_SEVERITY_WARNING) {
    ++s.warnings;
    std::fprintf(stderr, "ANARI warning: %s\n", message);
  }
}
template <typename T>
T Scene::keep(T value)
{
  objects.push_back((ANARIObject)value);
  return value;
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
    ANARIDataType type, const void *data, size_t count, size_t bytes)
{
  auto a = keep(anariNewArray1D(d, nullptr, nullptr, nullptr, type, count));
  std::memcpy(anariMapArray(d, a), data, bytes);
  anariUnmapArray(d, a);
  return a;
}
Scene::Scene(const char *rgb, const char *renderType)
{
  d = makeVisRTXDevice(status, this);
  commit(d);
  density = keep(anariNewSpatialField(d, "testTypedScalar"));
  commit(density);
  emission = keep(anariNewSpatialField(d, rgb));
  commit(emission);
  volume = keep(anariNewVolume(d, "principledVolume"));
  set(volume, "bounds", ANARI_FLOAT32_BOX3, Box{0, 0, 0, 1, 1, 1});
  set(volume, "transmissionColor", ANARI_FLOAT32_VEC3, V3{.5f, .5f, .5f});
  set(volume, "scatteringColor", ANARI_FLOAT32_VEC3, V3{0, 0, 0});
  set(volume, "density.field", ANARI_SPATIAL_FIELD, density);
  set(volume, "emission", ANARI_FLOAT32_VEC3, V3{1, 1, 1});
  set(volume, "emission.field", ANARI_SPATIAL_FIELD, emission);
  commit(volume);
  world = keep(anariNewWorld(d));
  auto a = array(ANARI_VOLUME, &volume, 1, sizeof(volume));
  set(world, "volume", ANARI_ARRAY1D, a);
  commit(world);
  camera = keep(anariNewCamera(d, "orthographic"));
  set(camera, "position", ANARI_FLOAT32_VEC3, V3{.5f, .5f, 2});
  set(camera, "direction", ANARI_FLOAT32_VEC3, V3{0, 0, -1});
  set(camera, "up", ANARI_FLOAT32_VEC3, V3{0, 1, 0});
  set(camera, "height", ANARI_FLOAT32, 1.f);
  commit(camera);
  auto r = keep(anariNewRenderer(d, renderType));
  set(r, "pixelSamples", ANARI_INT32, 256);
  set(r, "sampleLimit", ANARI_INT32, 0);
  set(r, "maxRayDepth", ANARI_INT32, 32);
  set(r, "background", ANARI_FLOAT32_VEC4, V4{0, 0, 0, 0});
  anariSetParameter(d, r, "fireflyFilterMode", ANARI_STRING, "none");
  commit(r);
  frame = keep(anariNewFrame(d));
  set(frame, "size", ANARI_UINT32_VEC2, std::array<unsigned, 2>{SIZE, SIZE});
  set(frame, "channel.color", ANARI_DATA_TYPE, ANARI_FLOAT32_VEC4);
  set(frame, "world", ANARI_WORLD, world);
  set(frame, "camera", ANARI_CAMERA, camera);
  set(frame, "renderer", ANARI_RENDERER, r);
  commit(frame);
}
Scene::~Scene()
{
  for (auto i = objects.rbegin(); i != objects.rend(); ++i)
    anariRelease(d, *i);
  anariRelease(d, d);
}
void Scene::release(ANARIObject o)
{
  objects.erase(std::find(objects.begin(), objects.end(), o));
  anariRelease(d, o);
}
bool Scene::property(ANARIObject o, const char *name)
{
  bool b = false;
  require(anariGetProperty(d, o, name, ANARI_BOOL, &b, sizeof(b), ANARI_WAIT),
      "missing boolean property");
  return b;
}
std::vector<V4> Scene::read()
{
  anariFrameReady(d, frame, ANARI_WAIT);
  uint32_t w, h;
  ANARIDataType type;
  const auto *p = static_cast<const V4 *>(
      anariMapFrame(d, frame, "channel.color", &w, &h, &type));
  require(
      p && w == SIZE && h == SIZE && type == ANARI_FLOAT32_VEC4, "frame map");
  std::vector<V4> out(p, p + w * h);
  anariUnmapFrame(d, frame, "channel.color");
  require(errors == 0, "device errors");
  return out;
}
std::vector<V4> Scene::render()
{
  anariRenderFrame(d, frame);
  return read();
}
std::vector<V4> homogeneous(float density, V3 source = {.2f, .4f, .6f})
{
  const double sigma = double(density) * std::log(2.);
  const double t = std::exp(-sigma);
  const double weight = sigma > 0. ? -std::expm1(-sigma) / sigma : 1.;
  return std::vector<V4>(SIZE * SIZE,
      V4{float(source[0] * weight),
          float(source[1] * weight),
          float(source[2] * weight),
          float(1. - t)});
}
void check(const std::vector<V4> &image,
    const std::vector<V4> &expected,
    const char *name,
    bool deterministic = false)
{
  for (int c = 0; c < 4; ++c) {
    double sum = 0., squares = 0.;
    for (size_t i = 0; i < image.size(); ++i) {
      const double delta = double(image[i][c]) - expected[i][c];
      require(std::isfinite(delta), "nonfinite custom sample");
      sum += delta;
      squares += delta * delta;
      if (deterministic) {
        if (std::abs(delta) > 2e-5 + 2e-4 * std::abs(expected[i][c]))
          std::fprintf(stderr,
              "%s pixel %zu channel %d: %.9g expected %.9g\n",
              name,
              i,
              c,
              image[i][c],
              expected[i][c]);
        require(std::abs(delta) <= 2e-5 + 2e-4 * std::abs(expected[i][c]),
            "custom preview analytic segment");
      }
    }
    const double n = double(image.size()), mean = sum / n,
                 variance = std::max(0., (squares - sum * sum / n) / (n - 1.));
    if (std::abs(mean) > 6. * std::sqrt(variance / n) + 5e-5) {
      std::fprintf(stderr, "%s channel %d mean error %.9g\n", name, c, mean);
      throw std::runtime_error("custom analytic mean");
    }
  }
  std::printf("%s passed\n", name);
}
void Scene::receiver()
{
  set(volume, "transmissionColor", ANARI_FLOAT32_VEC3, V3{1, 1, 1});
  commit(volume);
  set(emission, "value", ANARI_FLOAT32_VEC3, V3{1, 1, 1});
  commit(emission);
  auto group = keep(anariNewGroup(d));
  auto a = array(ANARI_VOLUME, &volume, 1, sizeof(volume));
  set(group, "volume", ANARI_ARRAY1D, a);
  commit(group);
  auto instance = keep(anariNewInstance(d, "transform"));
  set(instance, "group", ANARI_GROUP, group);
  set(instance,
      "transform",
      ANARI_FLOAT32_MAT4,
      std::array<float, 16>{
          .5f, 0, 0, 0, 0, .5f, 0, 0, 0, 0, .5f, 0, .75f, -.25f, .75f, 1});
  commit(instance);
  anariUnsetParameter(d, world, "volume");
  a = array(ANARI_INSTANCE, &instance, 1, sizeof(instance));
  set(world, "instance", ANARI_ARRAY1D, a);
  auto g = keep(anariNewGeometry(d, "quad"));
  const V3 vertices[4]{{-2, -2, 0}, {2, -2, 0}, {2, 2, 0}, {-2, 2, 0}};
  a = array(ANARI_FLOAT32_VEC3, vertices, 4, sizeof(vertices));
  set(g, "vertex.position", ANARI_ARRAY1D, a);
  commit(g);
  auto m = keep(anariNewMaterial(d, "matte"));
  set(m, "color", ANARI_FLOAT32_VEC3, V3{.7f, .7f, .7f});
  commit(m);
  auto surface = keep(anariNewSurface(d));
  set(surface, "geometry", ANARI_GEOMETRY, g);
  set(surface, "material", ANARI_MATERIAL, m);
  commit(surface);
  a = array(ANARI_SURFACE, &surface, 1, sizeof(surface));
  set(world, "surface", ANARI_ARRAY1D, a);
  commit(world);
  set(camera, "position", ANARI_FLOAT32_VEC3, V3{0, 0, 3});
  commit(camera);
}
} // namespace
int main()
{
  {
    Scene s;
    require(s.property(s.density, "visrtx.typedSampling"),
        "scalar opt-in unavailable");
    require(s.property(s.emission, "visrtx.typedSampling"),
        "RGB opt-in unavailable");
    require(!s.property(s.emission, "visrtx.typedCudaSampling"),
        "OptiX-only provider gained CUDA capability");
    check(s.render(), homogeneous(1), "typed scalar/RGB camera");
    s.set(s.density, "slope", ANARI_FLOAT32, 1.f);
    s.commit(s.density);
    auto varying = homogeneous(1);
    for (unsigned x = 0; x < SIZE; ++x) {
      V4 e{};
      for (double offset : {-.5 / std::sqrt(3.), .5 / std::sqrt(3.)}) {
        const double rho = 1. + (double(x) + .5 + offset) / SIZE;
        const double sigma = rho * std::log(2.), t = std::exp(-sigma),
                     w = -std::expm1(-sigma) / sigma;
        for (int c = 0; c < 3; ++c)
          e[c] += float(.5 * .2 * double(c + 1) * w);
        e[3] += float(.5 * (1. - t));
      }
      for (unsigned y = 0; y < SIZE; ++y)
        varying[y * SIZE + x] = e;
    }
    check(s.render(), varying, "procedural scalar / conservative global range");
    s.set(s.density, "slope", ANARI_FLOAT32, 0.f);
    s.commit(s.density);
    s.set(s.density, "roi", ANARI_FLOAT32_BOX3, Box{.5f, 0, 0, 1, 1, 1});
    s.commit(s.density);
    auto expected = homogeneous(1), vacuum = homogeneous(0);
    for (unsigned y = 0; y < SIZE; ++y)
      for (unsigned x = 0; x < SIZE / 2; ++x)
        expected[y * SIZE + x] = vacuum[y * SIZE + x];
    check(s.render(),
        expected,
        "ROI before poisoned sampler / emission outside density");
    s.set(s.volume, "density.outside", ANARI_FLOAT32, 2.f);
    s.commit(s.volume);
    const auto outside = homogeneous(2);
    for (unsigned y = 0; y < SIZE; ++y)
      for (unsigned x = 0; x < SIZE / 2; ++x)
        expected[y * SIZE + x] = outside[y * SIZE + x];
    check(s.render(), expected, "nonzero binding-owned outside value");
    s.set(s.volume, "density.outside", ANARI_FLOAT32, 0.f);
    s.commit(s.volume);
    s.set(s.density, "roi", ANARI_FLOAT32_BOX3, Box{.8f, 0, 0, .2f, 1, 1});
    s.commit(s.density);
    check(s.render(), vacuum, "empty ROI keeps independent source");
    s.set(s.density, "roi", ANARI_FLOAT32_BOX3, Box{0, 0, 0, 1, 1, 1});
    s.commit(s.density);
    s.set(
        s.volume, "transmissionDistance.field", ANARI_SPATIAL_FIELD, s.density);
    s.commit(s.volume);
    check(s.render(), homogeneous(1), "shared custom binding");
    anariUnsetParameter(s.d, s.volume, "transmissionDistance.field");
    s.commit(s.volume);
    s.set(s.density, "value", ANARI_FLOAT32, 2.f);
    s.commit(s.density);
    check(s.render(), homogeneous(2), "remaining observer after unbind");
    s.set(s.density, "hasRange", ANARI_BOOL, false);
    s.commit(s.density);
    require(!s.property(s.density, "visrtx.typedSampling"),
        "missing range accepted");
    require(
        !s.property(s.volume, "valid"), "volume accepted missing custom range");
    check(s.render(),
        std::vector<V4>(SIZE * SIZE, V4{}),
        "missing range omits whole volume");
    s.set(s.density, "hasRange", ANARI_BOOL, true);
    s.commit(s.density);
    check(s.render(), homogeneous(2), "range recovery without volume recommit");
    for (auto key : {"unknownProvider", "badArity"}) {
      s.set(s.density, key, ANARI_BOOL, true);
      s.commit(s.density);
      require(!s.property(s.density, "visrtx.typedSampling"),
          "bad custom identity/arity accepted");
      s.set(s.density, key, ANARI_BOOL, false);
      s.commit(s.density);
    }
    s.set(s.density, "contractVersion", ANARI_UINT32, uint32_t(2));
    s.commit(s.density);
    require(!s.property(s.density, "visrtx.typedSampling"),
        "bad custom version accepted");
    s.set(s.density, "contractVersion", ANARI_UINT32, uint32_t(1));
    s.commit(s.density);
    check(s.render(), homogeneous(2), "version recovery");
    s.set(s.volume, "density.field", ANARI_SPATIAL_FIELD, s.emission);
    s.commit(s.volume);
    require(!s.property(s.volume, "valid"), "RGB-to-scalar binding accepted");
    s.set(s.volume, "density.field", ANARI_SPATIAL_FIELD, s.density);
    s.commit(s.volume);
    auto legacy = s.keep(anariNewSpatialField(s.d, "testLegacyScalar"));
    s.commit(legacy);
    require(s.property(legacy, "valid")
            && !s.property(legacy, "visrtx.typedSampling"),
        "legacy implicit opt-in");
    s.set(s.volume, "density.field", ANARI_SPATIAL_FIELD, legacy);
    s.commit(s.volume);
    require(!s.property(s.volume, "valid"),
        "legacy-only principled binding accepted");
    s.set(s.volume, "density.field", ANARI_SPATIAL_FIELD, s.density);
    s.commit(s.volume);
    s.release(s.density);
    s.release(s.emission);
    check(s.render(), homogeneous(2), "application-handle release");
  }
  {
    Scene s;
    s.set(s.density, "value", ANARI_FLOAT32, -1.f);
    s.commit(s.density);
    s.set(s.volume, "density", ANARI_FLOAT32, -1.f);
    s.set(s.emission, "value", ANARI_FLOAT32_VEC3, V3{-.2f, -.4f, -.6f});
    s.commit(s.emission);
    s.set(s.volume, "emission", ANARI_FLOAT32_VEC3, V3{-1, -1, -1});
    s.commit(s.volume);
    check(
        s.render(), homogeneous(1), "signed raw custom ranges and multipliers");
  }
  {
    Scene s("testTypedRGBOptix", "interactive");
    check(s.render(), homogeneous(1), "custom Interactive wide path", true);
  }
  for (const char *type :
      {"testTypedRGBOptix", "testTypedRGBCuda", "testTypedRGBOwnedCuda"}) {
    Scene s(type);
    require(s.property(s.emission, "visrtx.typedCudaSampling")
            == bool(std::strcmp(type, "testTypedRGBOptix") != 0),
        "CUDA capability mismatch");
    s.receiver();
    check(s.render(),
        std::vector<V4>(
            SIZE * SIZE, V4{.010027149f, .010027149f, .010027149f, 1.f}),
        type);
  }
  for (const char *type : {"testTypedRGBOwned", "testTypedRGBOwnedCuda"}) {
    Scene s(type);
    check(s.render(), homogeneous(1), "leased source generation");
    anariRenderFrame(s.d, s.frame);
    s.set(s.emission, "value", ANARI_FLOAT32_VEC3, V3{.4f, .2f, .1f});
    s.commit(s.emission);
    require(s.property(s.emission, "visrtx.typedSampling"),
        "leased replacement unavailable");
    check(s.read(),
        homogeneous(1),
        "in-flight old generation survives WAIT flush");
    check(s.render(),
        homogeneous(1, {.4f, .2f, .1f}),
        "replacement resource generation");
    std::vector<ANARISpatialField> churn;
    for (int i = 0; i < 128; ++i) {
      auto f = s.keep(anariNewSpatialField(s.d, "testTypedScalar"));
      s.commit(f);
      churn.push_back(f);
    }
    check(s.render(),
        homogeneous(1, {.4f, .2f, .1f}),
        "registry growth preserves snapshot payload");
    for (auto f : churn)
      s.release(f);
    s.set(s.emission, "omitResources", ANARI_BOOL, true);
    s.commit(s.emission);
    require(!s.property(s.emission, "visrtx.typedSampling"),
        "missing lease accepted");
    s.set(s.emission, "omitResources", ANARI_BOOL, false);
    s.commit(s.emission);
    check(
        s.render(), homogeneous(1, {.4f, .2f, .1f}), "resource lease recovery");
    V3 value{.1f, .3f, .5f};
    auto data = s.array(ANARI_FLOAT32_VEC3, &value, 1, sizeof(value));
    s.set(s.emission, "valueArray", ANARI_ARRAY1D, data);
    s.commit(s.emission);
    check(s.render(), homogeneous(1, value), "array-backed immutable source");
    value = {.5f, .3f, .1f};
    std::memcpy(anariMapArray(s.d, data), &value, sizeof(value));
    anariUnmapArray(s.d, data);
    check(s.render(),
        homogeneous(1, value),
        "custom array edit without field/volume recommit");
    s.release(data);
    check(s.render(), homogeneous(1, value), "custom array handle release");
  }
  {
    Scene s("testTypedRGBOptix", "fast");
    auto legacy = s.keep(anariNewSpatialField(s.d, "testLegacyScalar"));
    s.commit(legacy);
    auto tf = s.keep(anariNewVolume(s.d, "transferFunction1D"));
    s.set(tf, "value", ANARI_SPATIAL_FIELD, legacy);
    s.set(tf, "color", ANARI_FLOAT32_VEC3, V3{1, 0, 0});
    s.set(tf, "opacity", ANARI_FLOAT32, .5f);
    s.commit(tf);
    auto a = s.array(ANARI_VOLUME, &tf, 1, sizeof(tf));
    s.set(s.world, "volume", ANARI_ARRAY1D, a);
    s.commit(s.world);
    require(s.property(tf, "valid"), "legacy TF rejected");
    const auto image = s.render();
    double red = 0.;
    for (const auto &p : image)
      red += p[0];
    require(red / double(image.size()) > .1,
        "legacy scalar callable no longer renders");
  }
  {
    Scene s;
    auto scalar = s.keep(anariNewSpatialField(s.d, "testTypedScalarOptix"));
    s.commit(scalar);
    require(s.property(scalar, "visrtx.typedSampling")
            && !s.property(scalar, "visrtx.typedCudaSampling"),
        "OptiX-only scalar capability");
    s.set(s.volume, "density.field", ANARI_SPATIAL_FIELD, scalar);
    s.commit(s.volume);
    auto tf = s.keep(anariNewVolume(s.d, "transferFunction1D"));
    s.set(tf, "value", ANARI_SPATIAL_FIELD, scalar);
    s.set(tf, "color", ANARI_FLOAT32_VEC3, V3{1, 0, 0});
    s.set(tf, "opacity", ANARI_FLOAT32, .5f);
    s.commit(tf);
    const ANARIVolume mixed[]{s.volume, tf};
    auto a = s.array(ANARI_VOLUME, mixed, 2, sizeof(mixed));
    s.set(s.world, "volume", ANARI_ARRAY1D, a);
    s.commit(s.world);
    auto expected = homogeneous(2, {float(.2 + std::log(2.)), .4f, .6f});
    check(
        s.render(), expected, "shared custom field / mixed TF and principled");
  }
  {
    Scene s;
    auto legacy = s.keep(anariNewSpatialField(s.d, "testLegacyScalar"));
    s.set(legacy, "slope", ANARI_FLOAT32, 1.f);
    s.commit(legacy);
    auto geometry = s.keep(anariNewGeometry(s.d, "isosurface"));
    s.set(geometry, "field", ANARI_SPATIAL_FIELD, legacy);
    s.set(geometry, "isovalue", ANARI_FLOAT32, 1.5f);
    s.commit(geometry);
    auto material = s.keep(anariNewMaterial(s.d, "matte"));
    s.set(material, "color", ANARI_FLOAT32_VEC3, V3{.7f, .7f, .7f});
    s.commit(material);
    auto surface = s.keep(anariNewSurface(s.d));
    s.set(surface, "geometry", ANARI_GEOMETRY, geometry);
    s.set(surface, "material", ANARI_MATERIAL, material);
    s.commit(surface);
    anariUnsetParameter(s.d, s.world, "volume");
    auto a = s.array(ANARI_SURFACE, &surface, 1, sizeof(surface));
    s.set(s.world, "surface", ANARI_ARRAY1D, a);
    auto light = s.keep(anariNewLight(s.d, "directional"));
    s.set(light, "direction", ANARI_FLOAT32_VEC3, V3{-1, 0, 0});
    s.set(light, "irradiance", ANARI_FLOAT32, 1.f);
    s.commit(light);
    a = s.array(ANARI_LIGHT, &light, 1, sizeof(light));
    s.set(s.world, "light", ANARI_ARRAY1D, a);
    s.commit(s.world);
    s.set(s.camera, "position", ANARI_FLOAT32_VEC3, V3{2, .5f, .5f});
    s.set(s.camera, "direction", ANARI_FLOAT32_VEC3, V3{-1, 0, 0});
    s.set(s.camera, "up", ANARI_FLOAT32_VEC3, V3{0, 0, 1});
    s.commit(s.camera);
    const float reflected = .7f / float(std::acos(-1.));
    check(s.render(),
        std::vector<V4>(SIZE * SIZE, V4{reflected, reflected, reflected, 1}),
        "legacy scalar init/value/normal hooks");
  }
  std::puts("Typed custom public controls passed");
}
