// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#include <anari/ext/visrtx/makeVisRTXDevice.h>
#include <anari/anari_cpp.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

using RGB = std::array<float, 3>;
using RGBA = std::array<float, 4>;
using Box = std::array<float, 6>;
static void require(bool b, const char *m)
{
  if (!b)
    throw std::runtime_error(m);
}
struct Image
{
  std::vector<RGBA> color;
  std::vector<RGB> albedo, normal;
};
struct Scene
{
  ANARIDevice d{};
  ANARIWorld world{};
  ANARIRenderer renderer{};
  ANARICamera camera{};
  std::vector<ANARIObject> objects;
  std::mutex mutex;
  int errors{0};
  bool fallback{false};
  Scene(const char *subtype = "interactive");
  ~Scene();
  template <typename T>
  T keep(T o);
  template <typename T>
  void set(ANARIObject o, const char *n, ANARIDataType t, const T &v);
  void commit(ANARIObject o);
  ANARIVolume volume(
      RGB transmission, RGB scattering, RGB emission, float density = 1.f);
  void volumes(const std::vector<ANARIVolume> &);
  ANARISurface surface(float z, bool glass, float opacity, RGB color);
  Image render(int spp = 1, bool denoise = false);
  ANARIFrame frame();
  Image renderFrame(ANARIFrame frame);
  static void status(const void *,
      ANARIDevice,
      ANARIObject,
      ANARIDataType,
      ANARIStatusSeverity,
      ANARIStatusCode,
      const char *);
};
void Scene::status(const void *p,
    ANARIDevice,
    ANARIObject,
    ANARIDataType,
    ANARIStatusSeverity s,
    ANARIStatusCode,
    const char *m)
{
  auto &self = *const_cast<Scene *>(static_cast<const Scene *>(p));
  std::lock_guard<std::mutex> l(self.mutex);
  if (std::getenv("VISRTX_TEST_PROGRESS"))
    std::fprintf(stderr, "[%d] %s\n", int(s), m);
  if (s <= ANARI_SEVERITY_ERROR) {
    ++self.errors;
    std::fprintf(stderr, "ANARI: %s\n", m);
  }
  if (std::strstr(m, "effective denoiseMode is color"))
    self.fallback = true;
}
template <typename T>
T Scene::keep(T o)
{
  objects.push_back(reinterpret_cast<ANARIObject>(o));
  return o;
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
Scene::Scene(const char *subtype)
{
  d = makeVisRTXDevice(status, this);
  commit(d);
  world = keep(anariNewWorld(d));
  commit(world);
  renderer = keep(anariNewRenderer(d, subtype));
  set(renderer, "background", ANARI_FLOAT32_VEC4, RGBA{0, 0, 0, 0});
  anariSetParameter(d, renderer, "fireflyFilterMode", ANARI_STRING, "none");
  set(renderer, "volumeSamplingRate", ANARI_FLOAT32, 8.f);
  set(renderer, "volumeSamplingRateShadows", ANARI_FLOAT32, 8.f);
  set(renderer, "ambientSamples", ANARI_INT32, 0);
  commit(renderer);
  camera = keep(anariNewCamera(d, "orthographic"));
  set(camera, "position", ANARI_FLOAT32_VEC3, RGB{.5f, .5f, 2.f});
  set(camera, "direction", ANARI_FLOAT32_VEC3, RGB{0, 0, -1});
  set(camera, "height", ANARI_FLOAT32, .5f);
  commit(camera);
}
Scene::~Scene()
{
  for (auto i = objects.rbegin(); i != objects.rend(); ++i)
    anariRelease(d, *i);
  anariRelease(d, d);
}
ANARIVolume Scene::volume(RGB c, RGB a, RGB j, float density)
{
  auto v = keep(anariNewVolume(d, "principledVolume"));
  set(v, "bounds", ANARI_FLOAT32_BOX3, Box{0, 0, 0, 1, 1, 1});
  set(v, "density", ANARI_FLOAT32, density);
  set(v, "transmissionColor", ANARI_FLOAT32_VEC3, c);
  set(v, "scatteringColor", ANARI_FLOAT32_VEC3, a);
  set(v, "emission", ANARI_FLOAT32_VEC3, j);
  commit(v);
  return v;
}
void Scene::volumes(const std::vector<ANARIVolume> &volumes)
{
  if (volumes.empty())
    anariUnsetParameter(d, world, "volume");
  else {
    auto a = keep(anariNewArray1D(
        d, nullptr, nullptr, nullptr, ANARI_VOLUME, volumes.size()));
    std::memcpy(anariMapArray(d, a),
        volumes.data(),
        volumes.size() * sizeof(ANARIVolume));
    anariUnmapArray(d, a);
    set(world, "volume", ANARI_ARRAY1D, a);
  }
  commit(world);
}
ANARISurface Scene::surface(float z, bool glass, float opacity, RGB color)
{
  auto data = keep(
      anariNewArray1D(d, nullptr, nullptr, nullptr, ANARI_FLOAT32_VEC3, 4));
  const RGB vertices[4] = {{-1, -1, z}, {2, -1, z}, {2, 2, z}, {-1, 2, z}};
  std::memcpy(anariMapArray(d, data), vertices, sizeof(vertices));
  anariUnmapArray(d, data);
  auto geometry = keep(anariNewGeometry(d, "quad"));
  set(geometry, "vertex.position", ANARI_ARRAY1D, data);
  commit(geometry);
  auto material =
      keep(anariNewMaterial(d, glass ? "physicallyBased" : "matte"));
  set(material, glass ? "baseColor" : "color", ANARI_FLOAT32_VEC3, color);
  set(material, "opacity", ANARI_FLOAT32, opacity);
  if (glass) {
    set(material, "metallic", ANARI_FLOAT32, 0.f);
    set(material, "transmission", ANARI_FLOAT32, 1.f);
    set(material, "ior", ANARI_FLOAT32, 1.f);
    set(material, "specular", ANARI_FLOAT32, 0.f);
  }
  commit(material);
  auto s = keep(anariNewSurface(d));
  set(s, "geometry", ANARI_GEOMETRY, geometry);
  set(s, "material", ANARI_MATERIAL, material);
  commit(s);
  auto a =
      keep(anariNewArray1D(d, nullptr, nullptr, nullptr, ANARI_SURFACE, 1));
  *static_cast<ANARISurface *>(anariMapArray(d, a)) = s;
  anariUnmapArray(d, a);
  set(world, "surface", ANARI_ARRAY1D, a);
  commit(world);
  return s;
}
Image Scene::render(int spp, bool denoise)
{
  set(renderer, "pixelSamples", ANARI_INT32, spp);
  set(renderer, "denoise", ANARI_BOOL, denoise);
  anariSetParameter(
      d, renderer, "denoiseMode", ANARI_STRING, "colorAlbedoNormal");
  commit(renderer);
  return renderFrame(this->frame());
}
ANARIFrame Scene::frame()
{
  auto frame = keep(anariNewFrame(d));
  set(frame, "size", ANARI_UINT32_VEC2, std::array<uint32_t, 2>{8, 8});
  set(frame, "channel.color", ANARI_DATA_TYPE, ANARI_FLOAT32_VEC4);
  set(frame, "channel.albedo", ANARI_DATA_TYPE, ANARI_FLOAT32_VEC3);
  set(frame, "channel.normal", ANARI_DATA_TYPE, ANARI_FLOAT32_VEC3);
  set(frame, "renderer", ANARI_RENDERER, renderer);
  set(frame, "camera", ANARI_CAMERA, camera);
  set(frame, "world", ANARI_WORLD, world);
  commit(frame);
  return frame;
}
Image Scene::renderFrame(ANARIFrame frame)
{
  anariRenderFrame(d, frame);
  anariFrameReady(d, frame, ANARI_WAIT);
  uint32_t w, h;
  ANARIDataType type;
  const auto *c = static_cast<const RGBA *>(
      anariMapFrame(d, frame, "channel.color", &w, &h, &type));
  require(c && w == 8 && h == 8, "color map");
  Image out;
  out.color.assign(c, c + 64);
  anariUnmapFrame(d, frame, "channel.color");
  for (auto pair : {std::pair<const char *, std::vector<RGB> *>{
                        "channel.albedo", &out.albedo},
           {"channel.normal", &out.normal}}) {
    const auto *p = static_cast<const RGB *>(
        anariMapFrame(d, frame, pair.first, &w, &h, &type));
    require(p && w == 8 && h == 8, "AOV map");
    pair.second->assign(p, p + 64);
    anariUnmapFrame(d, frame, pair.first);
  }
  require(errors == 0, "unexpected ANARI errors");
  return out;
}
static void expect(const Image &image,
    const std::array<double, 3> &rgb,
    double alpha,
    std::array<double, 3> albedo = {0, 0, 0},
    std::array<double, 3> normal = {0, 0, 0},
    double tolerance = 2e-5)
{
  for (size_t i = 0; i < image.color.size(); ++i) {
    for (int c = 0; c < 3; ++c) {
      const double got = image.color[i][c];
      if (!std::isfinite(got)
          || std::abs(got - rgb[c]) > tolerance * (std::abs(rgb[c]) + 1e-35)) {
        std::fprintf(stderr,
            "pixel %zu channel %d: %.9g expected %.9g\n",
            i,
            c,
            got,
            rgb[c]);
        throw std::runtime_error("radiance oracle");
      }
      require(
          std::abs(image.albedo[i][c] - albedo[c]) < 2e-5, "albedo event law");
      require(
          std::abs(image.normal[i][c] - normal[c]) < 2e-5, "arithmetic normal");
    }
    require(std::abs(image.color[i][3] - alpha) < 2e-5, "coverage alpha");
  }
}
static std::array<double, 3> blackbody(double temperature)
{
  std::ifstream file(BLACKBODY_ORACLE);
  require(bool(file), "thermal oracle missing");
  std::string line;
  std::getline(file, line);
  std::getline(file, line);
  double t, r, g, b, y;
  while (file >> t >> r >> g >> b >> y)
    if (t == temperature)
      return {r, g, b};
  throw std::runtime_error("thermal temperature absent");
}
#include "PrincipledInteractiveProviders.h"

int main()
try {
  const RGB expOne{
      .36787944117144233f, .36787944117144233f, .36787944117144233f};
  const double sigma = -std::log(double(expOne[0]));
  {
    Scene s;
    auto v = s.volume(expOne, {0, 0, 0}, {2, 4, 8}, 0);
    s.volumes({v});
    expect(s.render(4), {2, 4, 8}, 0);
    const auto denoised = s.render(4, true);
    for (const auto &p : denoised.color)
      require(p[0] > 0 && p[1] > 0 && p[2] > 0 && p[3] == 0,
          "denoiser erased zero-alpha source");
    require(s.fallback, "missing color-only fallback warning");
  }
  {
    Scene s("default");
    auto v = s.volume(expOne, {0, 0, 0}, {2, 4, 8}, 0);
    s.volumes({v});
    expect(s.render(), {2, 4, 8}, 0);
  }
  {
    Scene s;
    auto v = s.volume({.5f, .25f, .125f}, {0, 0, 0}, {0, 0, 0});
    s.volumes({v});
    s.set(s.renderer, "background", ANARI_FLOAT32_VEC4, RGBA{2, 4, 8, 0});
    s.commit(s.renderer);
    const double u = .2126 * .5 + .7152 * .25 + .0722 * .125;
    expect(s.render(), {1, 1, 1}, 1 - u);
    s.set(v, "emission", ANARI_FLOAT32_VEC3, RGB{2, 4, 8});
    s.commit(v);
    expect(s.render(),
        {1 + 2 * .5 / std::log(2.),
            1 + 4 * .75 / std::log(4.),
            1 + 8 * .875 / std::log(8.)},
        1 - u);
  }
  {
    Scene s;
    auto v = s.volume(expOne, {1, 1, 1}, {0, 0, 0});
    s.volumes({v});
    const double mass = -std::expm1(-sigma);
    expect(s.render(), {0, 0, 0}, mass, {mass, mass, mass});
  }
  {
    Scene s;
    auto v = s.volume(expOne, {.3f, .6f, .9f}, {0, 0, 0}, .1f);
    s.volumes({v});
    s.set(v, "anisotropy", ANARI_FLOAT32, .2f);
    s.set(v, "secondaryAnisotropy", ANARI_FLOAT32, -.3f);
    s.set(v, "secondaryWeight", ANARI_FLOAT32, .4f);
    s.commit(v);
    auto light = s.keep(anariNewLight(s.d, "directional"));
    s.set(light, "direction", ANARI_FLOAT32_VEC3, RGB{0, 0, -1});
    s.set(light, "irradiance", ANARI_FLOAT32, 4.f);
    s.commit(light);
    auto a =
        s.keep(anariNewArray1D(s.d, nullptr, nullptr, nullptr, ANARI_LIGHT, 1));
    *static_cast<ANARILight *>(anariMapArray(s.d, a)) = light;
    anariUnmapArray(s.d, a);
    s.set(s.world, "light", ANARI_ARRAY1D, a);
    s.commit(s.world);
    auto hg = [](double g) {
      return (1 - g * g)
          / (4 * std::acos(-1.) * std::pow(1 + g * g + 2 * g, 1.5));
    };
    const double phase = (1 - double(.4f)) * hg(.2f) + double(.4f) * hg(-.3f),
                 t = double(.1f) * sigma;
    const double f = phase * 4 * (-std::expm1(-2 * t)) / 2;
    const double mass = -std::expm1(-t);
    expect(s.render(),
        {double(.3f) * f, double(.6f) * f, double(.9f) * f},
        mass,
        {double(.3f) * mass, double(.6f) * mass, double(.9f) * mass},
        {0, 0, 0},
        3e-4);
  }
  {
    Scene s;
    auto a = s.volume(expOne, {0, 0, 0}, {2, 0, 0});
    auto b = s.volume(expOne, {0, 0, 0}, {0, 4, 0}, 2);
    s.volumes({a, b});
    const double f = -std::expm1(-3 * sigma) / (3 * sigma),
                 alpha = -std::expm1(-3 * sigma);
    expect(s.render(), {2 * f, 4 * f, 0}, alpha);
    s.volumes({b, a});
    expect(s.render(), {2 * f, 4 * f, 0}, alpha);
  }
  {
    Scene s;
    auto v = s.volume({.5f, .5f, .5f}, {0, 0, 0}, {0, 0, 0});
    s.volumes({v});
    s.surface(-.5f, false, .5f, {.4f, .6f, .8f});
    s.set(s.renderer, "ambientRadiance", ANARI_FLOAT32, 1.f);
    s.commit(s.renderer);
    expect(s.render(4), {.1, .15, .2}, .75, {.1, .15, .2}, {0, 0, .25});
  }
  {
    Scene s;
    auto v = s.volume({.5f, .25f, .125f}, {0, 0, 0}, {0, 0, 0});
    s.volumes({v});
    s.surface(1.5f, true, 1.f, {.5f, .25f, .75f});
    s.set(s.renderer, "background", ANARI_FLOAT32_VEC4, RGBA{2, 4, 8, 0});
    s.commit(s.renderer);
    expect(s.render(), {.5, .25, .75}, 1, {.5, .25, .75}, {0, 0, 1});
    anariSetParameter(s.d, s.renderer, "fogMode", ANARI_STRING, "linear");
    s.set(s.renderer, "fogStart", ANARI_FLOAT32, 0.f);
    s.set(s.renderer, "fogEnd", ANARI_FLOAT32, 1.f);
    s.set(s.renderer, "fogColor", ANARI_FLOAT32_VEC3, RGB{1, 2, 3});
    s.commit(s.renderer);
    expect(s.render(), {.75, 1.125, 1.875}, 1, {.5, .25, .75}, {0, 0, 1});
  }
  {
    // Exercise both OptiX reference-field callables and an exact ROI split.
    Scene s;
    auto v = s.volume(expOne, {0, 0, 0}, {1, 1, 1});
    auto scalar = s.keep(anariNewArray3D(
        s.d, nullptr, nullptr, nullptr, ANARI_FLOAT32, 2, 2, 2));
    auto rgb = s.keep(anariNewArray3D(
        s.d, nullptr, nullptr, nullptr, ANARI_FLOAT32_VEC3, 2, 2, 2));
    auto *density = static_cast<float *>(anariMapArray(s.d, scalar));
    auto *emission = static_cast<RGB *>(anariMapArray(s.d, rgb));
    for (int i = 0; i < 8; ++i) {
      density[i] = .5f;
      emission[i] = {2, 4, 8};
    }
    anariUnmapArray(s.d, scalar);
    anariUnmapArray(s.d, rgb);
    auto makeField = [&](ANARIArray3D data) {
      auto f = s.keep(anariNewSpatialField(s.d, "structuredRegular"));
      s.set(f, "data", ANARI_ARRAY3D, data);
      anariSetParameter(s.d, f, "dataCentering", ANARI_STRING, "cell");
      s.set(f, "origin", ANARI_FLOAT32_VEC3, RGB{0, 0, 0});
      s.set(f, "spacing", ANARI_FLOAT32_VEC3, RGB{.5f, .5f, .5f});
      s.commit(f);
      return f;
    };
    auto d = makeField(scalar), e = makeField(rgb);
    s.set(v, "density.field", ANARI_SPATIAL_FIELD, d);
    s.set(v, "emission.field", ANARI_SPATIAL_FIELD, e);
    s.commit(v);
    s.volumes({v});
    const double k = .5 * sigma, mass = -std::expm1(-k), f = mass / k;
    expect(s.render(), {2 * f, 4 * f, 8 * f}, mass);
    s.set(e, "roi", ANARI_FLOAT32_BOX3, Box{0, 0, 0, 1, 1, .5f});
    s.commit(e);
    const double roi = std::exp(-.5 * k) * (-std::expm1(-.5 * k)) / k;
    expect(s.render(), {2 * roi, 4 * roi, 8 * roi}, mass);
  }
  {
    // A spatially varying HDR image, composed once through colored extinction.
    Scene s;
    auto v = s.volume({.5f, .25f, .125f}, {0, 0, 0}, {0, 0, 0});
    s.volumes({v});
    auto image = s.keep(anariNewArray2D(
        s.d, nullptr, nullptr, nullptr, ANARI_FLOAT32_VEC4, 2, 2));
    const RGBA texels[4] = {
        {2, 4, 8, .5f}, {4, 4, 12, .5f}, {2, 8, 12, .5f}, {4, 8, 16, .5f}};
    std::memcpy(anariMapArray(s.d, image), texels, sizeof(texels));
    anariUnmapArray(s.d, image);
    s.set(s.renderer, "background", ANARI_ARRAY2D, image);
    s.commit(s.renderer);
    auto frame = s.frame();
    const double u = .2126 * .5 + .7152 * .25 + .0722 * .125;
    constexpr double weights[8] = {0, 0, .125, .375, .625, .875, 1, 1};
    for (int policy = 0; policy < 4; ++policy) {
      const bool premultiply = policy & 1;
      const bool denoise = policy & 2;
      s.set(s.renderer, "denoise", ANARI_BOOL, denoise);
      s.set(s.renderer, "premultiplyBackground", ANARI_BOOL, premultiply);
      s.commit(s.renderer);
      const auto output = s.renderFrame(frame);
      const double factor = premultiply ? .5 : 1.;
      for (size_t p = 0; p < 64; ++p) {
        const double x = weights[p % 8], y = weights[p / 8];
        Image pixel{{output.color[p]}, {output.albedo[p]}, {output.normal[p]}};
        expect(pixel,
            {factor * (1 + x),
                factor * (1 + y),
                factor * (1 + .5 * x + .5 * y)},
            1 - .5 * u);
      }
    }
    // Array update must invalidate the reused frame and its backdrop history.
    auto *pixels = static_cast<RGBA *>(anariMapArray(s.d, image));
    std::fill(pixels, pixels + 4, RGBA{4, 8, 16, .5f});
    anariUnmapArray(s.d, image);
    expect(s.renderFrame(frame), {1, 1, 1}, 1 - .5 * u);
  }
  {
    // TF is a source/absorber here, not a scattering material or a winner in
    // overlap.
    Scene s("default");
    auto data = s.keep(anariNewArray3D(
        s.d, nullptr, nullptr, nullptr, ANARI_FLOAT32, 2, 2, 2));
    auto *values = static_cast<float *>(anariMapArray(s.d, data));
    std::fill(values, values + 8, .5f);
    anariUnmapArray(s.d, data);
    auto field = s.keep(anariNewSpatialField(s.d, "structuredRegular"));
    s.set(field, "data", ANARI_ARRAY3D, data);
    s.set(field, "spacing", ANARI_FLOAT32_VEC3, RGB{.5f, .5f, .5f});
    anariSetParameter(s.d, field, "dataCentering", ANARI_STRING, "cell");
    s.commit(field);
    auto tf = s.keep(anariNewVolume(s.d, "transferFunction1D"));
    s.set(tf, "value", ANARI_SPATIAL_FIELD, field);
    s.set(tf, "color", ANARI_FLOAT32_VEC3, RGB{0, 2, 4});
    s.set(tf, "opacity", ANARI_FLOAT32, .5f);
    s.set(tf, "unitDistance", ANARI_FLOAT32, 1.f);
    s.commit(tf);
    auto p = s.volume({.5f, .5f, .5f}, {0, 0, 0}, {1, 0, 0});
    auto frame = s.frame();
    s.volumes({tf});
    expect(s.renderFrame(frame), {0, 1, 2}, .5);
    for (auto volumes :
        {std::vector<ANARIVolume>{tf, p}, std::vector<ANARIVolume>{p, tf}}) {
      s.volumes(volumes);
      expect(s.renderFrame(frame), {.75 / (2 * std::log(2.)), .75, 1.5}, .75);
    }
    s.set(tf, "opacity", ANARI_FLOAT32, 0.f);
    s.commit(tf);
    expect(s.renderFrame(frame), {.5 / std::log(2.), 0, 0}, .5);
  }
  {
    // The absorbing slab is ABOVE the camera: only surface-light connections
    // cross it. No camera attenuation or source can mask a missing shadow edge.
    Scene s;
    s.surface(1.5f, false, 1.f, {.4f, .6f, .8f});
    auto light = s.keep(anariNewLight(s.d, "directional"));
    s.set(light, "direction", ANARI_FLOAT32_VEC3, RGB{0, 0, -1});
    s.set(light, "irradiance", ANARI_FLOAT32, float(std::acos(-1.)));
    s.commit(light);
    auto lights =
        s.keep(anariNewArray1D(s.d, nullptr, nullptr, nullptr, ANARI_LIGHT, 1));
    *static_cast<ANARILight *>(anariMapArray(s.d, lights)) = light;
    anariUnmapArray(s.d, lights);
    s.set(s.world, "light", ANARI_ARRAY1D, lights);
    s.commit(s.world);
    auto frame = s.frame();
    expect(s.renderFrame(frame), {.4, .6, .8}, 1, {.4, .6, .8}, {0, 0, 1});
    auto v = s.volume({.5f, .25f, .125f}, {0, 0, 0}, {0, 0, 0});
    s.set(v, "bounds", ANARI_FLOAT32_BOX3, Box{0, 0, 3, 1, 1, 4});
    s.commit(v);
    s.volumes({v});
    expect(s.renderFrame(frame), {.2, .15, .1}, 1, {.4, .6, .8}, {0, 0, 1});
    s.volumes({v, v});
    expect(
        s.renderFrame(frame), {.1, .0375, .0125}, 1, {.4, .6, .8}, {0, 0, 1});
  }
  for (bool checkerboard : {false, true}) {
    Scene s;
    auto v = s.volume({.5f, .5f, .5f}, {1, 1, 1}, {2, 4, 8});
    s.volumes({v});
    s.set(s.renderer, "checkerboarding", ANARI_BOOL, checkerboard);
    s.set(s.renderer, "pixelSamples", ANARI_INT32, 3);
    s.commit(s.renderer);
    auto frame = s.frame();
    const double f = .5 / std::log(2.);
    s.set(frame, "accumulationVersion", ANARI_UINT64, uint64_t(1));
    s.commit(frame);
    // Match the inherited Frame lifecycle: initial allocation/reset precedes
    // the first version latch (also covered by TestSampleAddressing).
    expect(s.renderFrame(frame), {2 * f, 4 * f, 8 * f}, .5, {.5, .5, .5});
    bool pending = false;
    require(anariGetProperty(s.d,
                frame,
                "nextFrameReset",
                ANARI_BOOL,
                &pending,
                sizeof(pending),
                ANARI_WAIT)
            && pending,
        "initial version latch");
    for (int repeat = 0; repeat < 2; ++repeat) {
      if (repeat) {
        s.set(frame, "accumulationVersion", ANARI_UINT64, uint64_t(2));
        s.commit(frame);
      }
      for (int launch = 0; launch < 8; ++launch) {
        expect(s.renderFrame(frame), {2 * f, 4 * f, 8 * f}, .5, {.5, .5, .5});
        int samples = -1;
        require(anariGetProperty(s.d,
                    frame,
                    "numSamples",
                    ANARI_INT32,
                    &samples,
                    sizeof(samples),
                    ANARI_WAIT),
            "sample count query");
        if (samples != (checkerboard ? (launch + 1) / 4 : 3 * (launch + 1)))
          std::fprintf(stderr,
              "checkerboard=%d repeat=%d launch=%d samples=%d\n",
              checkerboard,
              repeat,
              launch,
              samples);
        require(samples == (checkerboard ? (launch + 1) / 4 : 3 * (launch + 1)),
            "reused volume frame sample count/reset");
      }
    }
    // Restore automatic reset before testing observer-driven changes.
    s.set(frame, "accumulationVersion", ANARI_UINT64, uint64_t(0));
    s.commit(frame);
    // Observer-driven volume update must discard old source AND AOV history.
    s.set(v, "scatteringColor", ANARI_FLOAT32_VEC3, RGB{0, 0, 0});
    s.set(v, "emission", ANARI_FLOAT32_VEC3, RGB{1, 2, 3});
    s.commit(v);
    expect(s.renderFrame(frame), {f, 2 * f, 3 * f}, .5);
    s.volumes({});
    expect(s.renderFrame(frame), {0, 0, 0}, 0);
    s.volumes({v});
    expect(s.renderFrame(frame), {f, 2 * f, 3 * f}, .5);
  }
  for (float temperature : {0.f, 100.f, 250.f, 2000.f}) {
    Scene s;
    auto v = s.volume(expOne, {0, 0, 0}, {0, 0, 0});
    s.volumes({v});
    const float intensity = temperature == 100 ? 1e30f : 1.f;
    s.set(v, "temperature", ANARI_FLOAT32, temperature);
    s.set(v, "blackbodyIntensity", ANARI_FLOAT32, intensity);
    s.set(v,
        "blackbodyTint",
        ANARI_FLOAT32_VEC3,
        RGB{intensity, intensity, intensity});
    s.commit(v);
    auto rgb = blackbody(temperature);
    for (auto &c : rgb)
      c *= double(intensity) * double(intensity) * (-std::expm1(-sigma));
    expect(s.render(), rgb, -std::expm1(-sigma), {0, 0, 0}, {0, 0, 0}, .004);
  }
  provider_controls::run();
  std::puts(
      "Interactive principled: public radiance, alpha, lighting, AOV, fog and thermal controls pass");
  return 0;
} catch (const std::exception &e) {
  std::fprintf(stderr, "FAIL: %s\n", e.what());
  return 1;
}
