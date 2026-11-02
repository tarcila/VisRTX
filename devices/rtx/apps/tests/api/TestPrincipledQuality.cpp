// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#include <anari/ext/visrtx/makeVisRTXDevice.h>
#include <algorithm>
#include <anari/anari_cpp.hpp>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

using RGB = std::array<float, 3>;
using RGBA = std::array<float, 4>;
struct Scene
{
  ANARIDevice d{};
  ANARIWorld world{};
  ANARIRenderer renderer{};
  ANARICamera camera{};
  std::vector<ANARIObject> objects;
  int errors = 0;
  Scene();
  ~Scene();
  template <typename T>
  T keep(T object);
  template <typename T>
  void set(
      ANARIObject object, const char *name, ANARIDataType type, const T &value);
  void commit(ANARIObject object);
  ANARIVolume volume(RGB color, RGB albedo, RGB source, float density = 1.f);
  void volumes(std::vector<ANARIVolume> list);
  void environment(bool visible);
  void directional();
  ANARIVolume transfer();
  ANARIMaterial surface(bool glass, float opacity);
  std::vector<std::array<double, 10>> render();
};
static void require(bool b, const char *m)
{
  if (!b)
    throw std::runtime_error(m);
}
static void status(const void *p,
    ANARIDevice,
    ANARIObject,
    ANARIDataType,
    ANARIStatusSeverity severity,
    ANARIStatusCode,
    const char *message)
{
  if (severity <= ANARI_SEVERITY_ERROR) {
    ++const_cast<Scene *>(static_cast<const Scene *>(p))->errors;
    std::fprintf(stderr, "ANARI: %s\n", message);
  }
}
template <typename T>
T Scene::keep(T object)
{
  objects.push_back(reinterpret_cast<ANARIObject>(object));
  return object;
}
template <typename T>
void Scene::set(
    ANARIObject object, const char *name, ANARIDataType type, const T &value)
{
  anariSetParameter(d, object, name, type, &value);
}
void Scene::commit(ANARIObject o)
{
  anariCommitParameters(d, o);
}
Scene::Scene()
{
  d = makeVisRTXDevice(status, this);
  commit(d);
  world = keep(anariNewWorld(d));
  commit(world);
  renderer = keep(anariNewRenderer(d, "quality"));
  set(renderer, "pixelSamples", ANARI_INT32, 128);
  set(renderer, "maxRayDepth", ANARI_INT32, 64);
  set(renderer, "background", ANARI_FLOAT32_VEC4, RGBA{0, 0, 0, 0});
  anariSetParameter(d, renderer, "fireflyFilterMode", ANARI_STRING, "none");
  commit(renderer);
  camera = keep(anariNewCamera(d, "orthographic"));
  set(camera, "position", ANARI_FLOAT32_VEC3, RGB{.5, .5, -1});
  set(camera, "direction", ANARI_FLOAT32_VEC3, RGB{0, 0, 1});
  set(camera, "up", ANARI_FLOAT32_VEC3, RGB{0, 1, 0});
  set(camera, "height", ANARI_FLOAT32, .5f);
  commit(camera);
}
Scene::~Scene()
{
  for (auto it = objects.rbegin(); it != objects.rend(); ++it)
    anariRelease(d, *it);
  anariRelease(d, d);
}
ANARIVolume Scene::volume(RGB color, RGB albedo, RGB source, float density)
{
  auto v = keep(anariNewVolume(d, "principledVolume"));
  set(v, "bounds", ANARI_FLOAT32_BOX3, std::array<float, 6>{0, 0, 0, 1, 1, 1});
  set(v, "transmissionColor", ANARI_FLOAT32_VEC3, color);
  set(v, "scatteringColor", ANARI_FLOAT32_VEC3, albedo);
  set(v, "emission", ANARI_FLOAT32_VEC3, source);
  set(v, "density", ANARI_FLOAT32, density);
  commit(v);
  return v;
}
void Scene::volumes(std::vector<ANARIVolume> list)
{
  if (list.empty())
    anariUnsetParameter(d, world, "volume");
  else {
    auto a = keep(anariNewArray1D(
        d, nullptr, nullptr, nullptr, ANARI_VOLUME, list.size()));
    std::memcpy(
        anariMapArray(d, a), list.data(), list.size() * sizeof(ANARIVolume));
    anariUnmapArray(d, a);
    set(world, "volume", ANARI_ARRAY1D, a);
  }
  commit(world);
}
void Scene::environment(bool visible)
{
  auto data = keep(
      anariNewArray2D(d, nullptr, nullptr, nullptr, ANARI_FLOAT32_VEC3, 8, 4));
  auto *p = static_cast<RGB *>(anariMapArray(d, data));
  std::fill(p, p + 32, RGB{1, 1, 1});
  anariUnmapArray(d, data);
  auto light = keep(anariNewLight(d, "hdri"));
  set(light, "radiance", ANARI_ARRAY2D, data);
  set(light, "visible", ANARI_BOOL, visible);
  commit(light);
  auto a = keep(anariNewArray1D(d, nullptr, nullptr, nullptr, ANARI_LIGHT, 1));
  *static_cast<ANARILight *>(anariMapArray(d, a)) = light;
  anariUnmapArray(d, a);
  set(world, "light", ANARI_ARRAY1D, a);
  commit(world);
}
void Scene::directional()
{
  auto light = keep(anariNewLight(d, "directional"));
  set(light, "direction", ANARI_FLOAT32_VEC3, RGB{0, 0, 1});
  set(light, "irradiance", ANARI_FLOAT32, float(std::acos(-1.)));
  commit(light);
  auto a = keep(anariNewArray1D(d, nullptr, nullptr, nullptr, ANARI_LIGHT, 1));
  *static_cast<ANARILight *>(anariMapArray(d, a)) = light;
  anariUnmapArray(d, a);
  set(world, "light", ANARI_ARRAY1D, a);
  commit(world);
}
ANARIVolume Scene::transfer()
{
  auto a = keep(
      anariNewArray3D(d, nullptr, nullptr, nullptr, ANARI_FLOAT32, 2, 2, 2));
  auto *p = static_cast<float *>(anariMapArray(d, a));
  std::fill(p, p + 8, .5f);
  anariUnmapArray(d, a);
  auto field = keep(anariNewSpatialField(d, "structuredRegular"));
  set(field, "data", ANARI_ARRAY3D, a);
  commit(field);
  auto tf = keep(anariNewVolume(d, "transferFunction1D"));
  set(tf, "value", ANARI_SPATIAL_FIELD, field);
  set(tf, "color", ANARI_FLOAT32_VEC3, RGB{0, 2, 0});
  set(tf, "opacity", ANARI_FLOAT32, .5f);
  set(tf, "unitDistance", ANARI_FLOAT32, 2.f);
  commit(tf);
  return tf;
}
ANARIMaterial Scene::surface(bool glass, float opacity)
{
  auto a = keep(
      anariNewArray1D(d, nullptr, nullptr, nullptr, ANARI_FLOAT32_VEC3, 4));
  const RGB p[4]{{-1, -1, 2}, {2, -1, 2}, {2, 2, 2}, {-1, 2, 2}};
  std::memcpy(anariMapArray(d, a), p, sizeof(p));
  anariUnmapArray(d, a);
  auto g = keep(anariNewGeometry(d, "quad"));
  set(g, "vertex.position", ANARI_ARRAY1D, a);
  commit(g);
  auto m = keep(anariNewMaterial(d, glass ? "physicallyBased" : "matte"));
  set(m, glass ? "baseColor" : "color", ANARI_FLOAT32_VEC3, RGB{1, 1, 1});
  set(m, "opacity", ANARI_FLOAT32, opacity);
  if (glass) {
    set(m, "transmission", ANARI_FLOAT32, 1.f);
    set(m, "ior", ANARI_FLOAT32, 1.f);
    set(m, "specular", ANARI_FLOAT32, 0.f);
  }
  commit(m);
  auto s = keep(anariNewSurface(d));
  set(s, "geometry", ANARI_GEOMETRY, g);
  set(s, "material", ANARI_MATERIAL, m);
  commit(s);
  auto sa =
      keep(anariNewArray1D(d, nullptr, nullptr, nullptr, ANARI_SURFACE, 1));
  *static_cast<ANARISurface *>(anariMapArray(d, sa)) = s;
  anariUnmapArray(d, sa);
  set(world, "surface", ANARI_ARRAY1D, sa);
  commit(world);
  return m;
}
std::vector<std::array<double, 10>> Scene::render()
{
  auto f = keep(anariNewFrame(d));
  set(f, "size", ANARI_UINT32_VEC2, std::array<uint32_t, 2>{32, 32});
  set(f, "channel.color", ANARI_DATA_TYPE, ANARI_FLOAT32_VEC4);
  set(f, "channel.albedo", ANARI_DATA_TYPE, ANARI_FLOAT32_VEC3);
  set(f, "channel.normal", ANARI_DATA_TYPE, ANARI_FLOAT32_VEC3);
  set(f, "renderer", ANARI_RENDERER, renderer);
  set(f, "camera", ANARI_CAMERA, camera);
  set(f, "world", ANARI_WORLD, world);
  commit(f);
  anariRenderFrame(d, f);
  anariFrameReady(d, f, ANARI_WAIT);
  require(errors == 0, "render errors");
  std::vector<std::array<double, 10>> out(1024);
  uint32_t width, height;
  ANARIDataType type;
  auto *color = static_cast<const RGBA *>(
      anariMapFrame(d, f, "channel.color", &width, &height, &type));
  require(color && width == 32 && height == 32, "color map");
  for (size_t i = 0; i < out.size(); ++i)
    for (int c = 0; c < 4; ++c)
      out[i][c] = color[i][c];
  anariUnmapFrame(d, f, "channel.color");
  int offset = 4;
  for (const char *name : {"channel.albedo", "channel.normal"}) {
    auto *data = static_cast<const RGB *>(
        anariMapFrame(d, f, name, &width, &height, &type));
    require(data != nullptr, "AOV map");
    for (size_t i = 0; i < out.size(); ++i)
      for (int c = 0; c < 3; ++c)
        out[i][offset + c] = data[i][c];
    offset += 3;
    anariUnmapFrame(d, f, name);
  }
  return out;
}
static void expect(const char *label,
    const std::vector<std::array<double, 10>> &image,
    std::array<double, 10> target)
{
  for (int c = 0; c < 10; ++c) {
    long double sum = 0, squares = 0;
    for (const auto &p : image) {
      require(std::isfinite(p[c]), "nonfinite output");
      sum += p[c];
      squares += p[c] * p[c];
    }
    const double mean = double(sum / image.size());
    const double variance = std::max(
        0., double((squares - sum * sum / image.size()) / (image.size() - 1)));
    // Fixed before rendering: six standard errors across independent pixels,
    // plus 2e-4 absolute for the float frame resolve. No image-dependent
    // tuning.
    const double tolerance = 6. * sqrt(variance / image.size()) + 2e-4;
    if (std::abs(mean - target[c]) > tolerance) {
      std::fprintf(stderr,
          "%s channel %d: %.9g expected %.9g tolerance %.9g\n",
          label,
          c,
          mean,
          target[c],
          tolerance);
      throw std::runtime_error("Quality analytic image control");
    }
  }
}
int main()
try {
  Scene scene;
  auto v = scene.volume({1, 1, 1}, {0, 0, 0}, {2, 3, 5}, 0);
  scene.volumes({v});
  expect("vacuum source", scene.render(), {2, 3, 5, 0});
  const RGB color{.8f, .5f, .2f}, source{.4f, .3f, .2f};
  auto absorber = scene.volume(color, {0, 0, 0}, source);
  scene.set(scene.renderer, "background", ANARI_FLOAT32_VEC4, RGBA{2, 3, 5, 0});
  scene.commit(scene.renderer);
  for (int copies : {1, 2}) {
    scene.volumes(std::vector<ANARIVolume>(size_t(copies), absorber));
    std::array<double, 10> target{};
    const double background[3]{2, 3, 5}, luma[3]{.2126, .7152, .0722};
    target[3] = 1.;
    for (int c = 0; c < 3; ++c) {
      const double t = std::pow(double(color[c]), copies);
      target[c] = source[c] * (1 - t) / (-std::log(double(color[c])))
          + background[c] * t;
      target[3] -= luma[c] * t;
    }
    expect(copies == 1 ? "colored source/backdrop" : "duplicate overlap",
        scene.render(),
        target);
  }
  const auto tf = scene.transfer();
  scene.volumes({absorber, tf});
  std::array<double, 10> mixed{};
  mixed[3] = 1.;
  const double tfa = std::log(2.) / 2., bg[3]{2, 3, 5},
               luma[3]{.2126, .7152, .0722};
  for (int channel = 0; channel < 3; ++channel) {
    const double a = -std::log(double(color[channel])) + tfa, t = std::exp(-a);
    mixed[channel] =
        (source[channel] + (channel == 1 ? 2 * tfa : 0.)) * (1 - t) / a
        + bg[channel] * t;
    mixed[3] -= luma[channel] * t;
  }
  expect("TF emission-absorption overlap", scene.render(), mixed);
  auto shadow = scene.volume(color, {0, 0, 0}, {0, 0, 0});
  scene.volumes({shadow});
  scene.surface(false, 1.f);
  scene.directional();
  const double coverage =
      luma[0] * color[0] + luma[1] * color[1] + luma[2] * color[2];
  expect("surface light connection through RGB medium",
      scene.render(),
      {double(color[0]) * color[0],
          double(color[1]) * color[1],
          double(color[2]) * color[2],
          1,
          coverage,
          coverage,
          coverage,
          0,
          0,
          -coverage});
  anariUnsetParameter(scene.d, scene.world, "light");
  scene.commit(scene.world);
  scene.volumes({});
  scene.surface(false, .25f);
  expect("coverage holes",
      scene.render(),
      {1.5, 2.25, 3.75, .25, .25, .25, .25, 0, 0, -.25});
  scene.surface(true, 1.f);
  expect("glass is not screen-plate transmission",
      scene.render(),
      {0, 0, 0, 1, 1, 1, 1, 0, 0, -1});
  anariUnsetParameter(scene.d, scene.world, "surface");
  scene.commit(scene.world);
  scene.set(scene.renderer, "background", ANARI_FLOAT32_VEC4, RGBA{0, 0, 0, 0});
  scene.commit(scene.renderer);
  const float c = std::exp(-1.f);
  const double sigma = -std::log(double(c));
  auto cloud = scene.volume({c, c, c},
      {.5f, .5f, .5f},
      {float(.5 * sigma), float(.5 * sigma), float(.5 * sigma)});
  scene.set(cloud, "anisotropy", ANARI_FLOAT32, .4f);
  scene.set(cloud, "secondaryAnisotropy", ANARI_FLOAT32, -.2f);
  scene.set(cloud, "secondaryWeight", ANARI_FLOAT32, .3f);
  scene.commit(cloud);
  scene.volumes({cloud});
  scene.environment(true);
  const double first = .5 * (1 - double(c));
  expect("multiple-scattering HDRI equilibrium",
      scene.render(),
      {1, 1, 1, 1, first, first, first});
  scene.environment(false);
  expect("hidden environment illumination",
      scene.render(),
      {1 - double(c),
          1 - double(c),
          1 - double(c),
          1 - double(c),
          first,
          first,
          first});
  anariUnsetParameter(scene.d, scene.world, "light");
  scene.commit(scene.world);
  auto thermal = scene.volume({c, c, c}, {0, 0, 0}, {0, 0, 0});
  scene.set(thermal, "temperature", ANARI_FLOAT32, 6500.f);
  scene.set(thermal, "blackbodyIntensity", ANARI_FLOAT32, 2.f);
  scene.commit(thermal);
  scene.volumes({thermal});
  std::ifstream oracle(BLACKBODY_ORACLE);
  require(bool(oracle), "thermal oracle missing");
  std::string line;
  std::getline(oracle, line);
  std::getline(oracle, line);
  double temperature, r, g, b, y;
  std::array<double, 10> thermalTarget{};
  bool found = false;
  while (oracle >> temperature >> r >> g >> b >> y)
    if (temperature == 6500.) {
      thermalTarget = {
          2 * r * (1 - c), 2 * g * (1 - c), 2 * b * (1 - c), 1 - double(c)};
      found = true;
      break;
    }
  require(found, "thermal reference missing");
  expect("calibrated thermal source", scene.render(), thermalTarget);
  scene.set(thermal, "scatteringColor", ANARI_FLOAT32_VEC3, RGB{1, 1, 1});
  scene.commit(thermal);
  expect("pure scattering has no thermal source",
      scene.render(),
      {0, 0, 0, 1 - double(c), 1 - double(c), 1 - double(c), 1 - double(c)});
  scene.set(scene.renderer, "maxRayDepth", ANARI_INT32, 1);
  scene.commit(scene.renderer);
  expect("coverage independent of radiance depth",
      scene.render(),
      {0, 0, 0, 1 - double(c), 1 - double(c), 1 - double(c), 1 - double(c)});
#ifdef VISRTX_NATIVE_PBR
  scene.volumes({});
  scene.environment(true);
  scene.set(scene.renderer, "maxRayDepth", ANARI_INT32, 8);
  scene.commit(scene.renderer);
  const auto sheen = scene.surface(true, 1.f);
  scene.set(sheen, "baseColor", ANARI_FLOAT32_VEC3, RGB{0, 0, 0});
  scene.set(sheen, "transmission", ANARI_FLOAT32, 0.f);
  scene.set(sheen, "sheenColor", ANARI_FLOAT32_VEC3, RGB{.8f, .4f, .2f});
  scene.set(sheen, "sheenRoughness", ANARI_FLOAT32, .5f);
  scene.commit(sheen);
  // At normal view and alphaS=1/4, integrate the accepted Charlie kernel:
  // 2*pi * integral_0^1 [3/(4*pi)*(1-mu)^2*mu/(4+1e-6)] dmu.
  const double charlie = 1. / (8. * (4. + 1e-6));
  expect("native sheen full-kernel environment",
      scene.render(),
      {.8 * charlie, .4 * charlie, .2 * charlie, 1, 0, 0, 0, 0, 0, -1});
#endif
  std::puts(
      "Principled Quality: public source, coverage, overlaps and scattering controls passed");
  return 0;
} catch (const std::exception &e) {
  std::fprintf(stderr, "FAIL: %s\n", e.what());
  return 1;
}
