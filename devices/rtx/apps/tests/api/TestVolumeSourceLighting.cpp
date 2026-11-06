// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#include <anari/ext/visrtx/makeVisRTXDevice.h>
#include <anari/anari_cpp.hpp>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
using V3 = std::array<float, 3>;
using V4 = std::array<float, 4>;
constexpr unsigned SIZE = 32;
struct Scene
{
  ANARIDevice d{};
  ANARIWorld world{};
  ANARICamera camera{};
  ANARIRenderer renderer{};
  ANARIVolume lastEmitter{};
  std::vector<ANARIObject> objects;
  int errors = 0;
  Scene();
  ~Scene();
  template <typename T>
  T keep(T o);
  template <typename T>
  void set(ANARIObject o, const char *name, ANARIDataType type, const T &v);
  void commit(ANARIObject o);
  ANARISurface plane(bool receiver, bool native, float opacity, float tint);
  ANARIVolume emitter();
  void setup(bool native,
      int duplicates,
      float paneOpacity = -1.f,
      float paneTint = 0.f,
      float mediumSigma = 0.f);
  std::vector<V4> render();
};
static void require(bool ok, const char *message)
{
  if (!ok)
    throw std::runtime_error(message);
}
static void status(const void *ptr,
    ANARIDevice,
    ANARIObject,
    ANARIDataType,
    ANARIStatusSeverity severity,
    ANARIStatusCode,
    const char *message)
{
  if (severity <= ANARI_SEVERITY_ERROR) {
    ++const_cast<Scene *>(static_cast<const Scene *>(ptr))->errors;
    std::fprintf(stderr, "ANARI: %s\n", message);
  }
}
template <typename T>
T Scene::keep(T o)
{
  objects.push_back((ANARIObject)o);
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
Scene::Scene()
{
  d = makeVisRTXDevice(status, this);
  commit(d);
  world = keep(anariNewWorld(d));
  commit(world);
  camera = keep(anariNewCamera(d, "orthographic"));
  set(camera, "position", ANARI_FLOAT32_VEC3, V3{0, 0, 3});
  set(camera, "direction", ANARI_FLOAT32_VEC3, V3{0, 0, -1});
  set(camera, "up", ANARI_FLOAT32_VEC3, V3{0, 1, 0});
  set(camera, "height", ANARI_FLOAT32, 1.f);
  commit(camera);
  renderer = keep(anariNewRenderer(d, "quality"));
  set(renderer, "pixelSamples", ANARI_INT32, 512);
  set(renderer, "maxRayDepth", ANARI_INT32, 32);
  set(renderer, "sampleLimit", ANARI_INT32, 0);
  set(renderer, "background", ANARI_FLOAT32_VEC4, V4{0, 0, 0, 0});
  anariSetParameter(d, renderer, "fireflyFilterMode", ANARI_STRING, "none");
  commit(renderer);
}
Scene::~Scene()
{
  for (auto i = objects.rbegin(); i != objects.rend(); ++i)
    anariRelease(d, *i);
  anariRelease(d, d);
}
ANARISurface Scene::plane(bool receiver, bool native, float opacity, float tint)
{
  auto data = keep(
      anariNewArray1D(d, nullptr, nullptr, nullptr, ANARI_FLOAT32_VEC3, 4));
  const std::array<V3, 4> points = receiver
      ? std::array<V3, 4>{V3{-2, -2, 0},
            V3{2, -2, 0},
            V3{2, 2, 0},
            V3{-2, 2, 0}}
      : std::array<V3, 4>{
            V3{.6f, -2, -1}, V3{.6f, -2, 3}, V3{.6f, 2, 3}, V3{.6f, 2, -1}};
  std::memcpy(anariMapArray(d, data), points.data(), sizeof(points));
  anariUnmapArray(d, data);
  auto geometry = keep(anariNewGeometry(d, "quad"));
  set(geometry, "vertex.position", ANARI_ARRAY1D, data);
  commit(geometry);
  auto material =
      keep(anariNewMaterial(d, native ? "physicallyBased" : "matte"));
  set(material,
      native ? "baseColor" : "color",
      ANARI_FLOAT32_VEC3,
      V3{tint, tint, tint});
  set(material, "opacity", ANARI_FLOAT32, opacity);
  if (native) {
    set(material, "specular", ANARI_FLOAT32, 0.f);
    set(material, "metallic", ANARI_FLOAT32, 0.f);
    set(material, "roughness", ANARI_FLOAT32, .02f);
    set(material, "ior", ANARI_FLOAT32, 1.f);
    set(material, "transmission", ANARI_FLOAT32, receiver ? 0.f : 1.f);
  }
  commit(material);
  auto surface = keep(anariNewSurface(d));
  set(surface, "geometry", ANARI_GEOMETRY, geometry);
  set(surface, "material", ANARI_MATERIAL, material);
  commit(surface);
  return surface;
}
ANARIVolume Scene::emitter()
{
  auto v = keep(anariNewVolume(d, "principledVolume"));
  set(v,
      "bounds",
      ANARI_FLOAT32_BOX3,
      std::array<float, 6>{.75f, -.25f, .75f, 1.25f, .25f, 1.25f});
  set(v, "transmissionColor", ANARI_FLOAT32_VEC3, V3{1, 1, 1});
  set(v, "emission", ANARI_FLOAT32_VEC3, V3{1, 1, 1});
  commit(v);
  lastEmitter = v;
  return v;
}
void Scene::setup(bool native,
    int duplicates,
    float paneOpacity,
    float paneTint,
    float mediumSigma)
{
  const auto v = emitter();
  std::vector<ANARIVolume> volumes(size_t(duplicates), v);
  if (mediumSigma > 0.f) {
    auto absorber = keep(anariNewVolume(d, "principledVolume"));
    set(absorber,
        "bounds",
        ANARI_FLOAT32_BOX3,
        std::array<float, 6>{.55f, -2, -1, .65f, 2, 3});
    const float t = std::exp(-mediumSigma);
    set(absorber, "transmissionColor", ANARI_FLOAT32_VEC3, V3{t, t, t});
    set(absorber, "scatteringColor", ANARI_FLOAT32_VEC3, V3{0, 0, 0});
    commit(absorber);
    volumes.push_back(absorber);
  }
  auto a = keep(anariNewArray1D(
      d, nullptr, nullptr, nullptr, ANARI_VOLUME, volumes.size()));
  std::memcpy(anariMapArray(d, a), volumes.data(), volumes.size() * sizeof(v));
  anariUnmapArray(d, a);
  set(world, "volume", ANARI_ARRAY1D, a);
  std::vector<ANARISurface> surfaces{plane(true, native, 1.f, .7f)};
  if (paneOpacity >= 0.f)
    surfaces.push_back(plane(false, true, paneOpacity, paneTint));
  auto b = keep(anariNewArray1D(
      d, nullptr, nullptr, nullptr, ANARI_SURFACE, surfaces.size()));
  std::memcpy(anariMapArray(d, b),
      surfaces.data(),
      surfaces.size() * sizeof(ANARISurface));
  anariUnmapArray(d, b);
  set(world, "surface", ANARI_ARRAY1D, b);
  commit(world);
}
std::vector<V4> Scene::render()
{
  auto f = keep(anariNewFrame(d));
  set(f, "size", ANARI_UINT32_VEC2, std::array<unsigned, 2>{SIZE, SIZE});
  set(f, "channel.color", ANARI_DATA_TYPE, ANARI_FLOAT32_VEC4);
  set(f, "renderer", ANARI_RENDERER, renderer);
  set(f, "camera", ANARI_CAMERA, camera);
  set(f, "world", ANARI_WORLD, world);
  commit(f);
  anariRenderFrame(d, f);
  anariFrameReady(d, f, ANARI_WAIT);
  require(errors == 0, "render errors");
  uint32_t w, h;
  ANARIDataType type;
  const auto *data = static_cast<const V4 *>(
      anariMapFrame(d, f, "channel.color", &w, &h, &type));
  require(data && w == SIZE && h == SIZE, "frame map");
  std::vector<V4> out(data, data + w * h);
  anariUnmapFrame(d, f, "channel.color");
  return out;
}
static std::vector<double> oracle(float mediumSigma = 0.f)
{
  // Independent 8-point Gauss rule in each volume dimension and a 2x2 rule
  // over the pixel footprint. Source and receiver are strictly separated.
  const double nodes[8]{-.9602898564975363,
      -.7966664774136267,
      -.5255324099163290,
      -.1834346424956498,
      .1834346424956498,
      .5255324099163290,
      .7966664774136267,
      .9602898564975363};
  const double weights[8]{.1012285362903763,
      .2223810344533745,
      .3137066458778873,
      .3626837833783620,
      .3626837833783620,
      .3137066458778873,
      .2223810344533745,
      .1012285362903763};
  std::vector<double> result(SIZE * SIZE);
  for (unsigned py = 0; py < SIZE; ++py)
    for (unsigned px = 0; px < SIZE; ++px) {
      double value = 0.;
      for (double u : {-1. / std::sqrt(3.), 1. / std::sqrt(3.)})
        for (double v : {-1. / std::sqrt(3.), 1. / std::sqrt(3.)}) {
          const double x = (px + .5 + .5 * u) / SIZE - .5,
                       y = (py + .5 + .5 * v) / SIZE - .5;
          for (int i = 0; i < 8; ++i)
            for (int j = 0; j < 8; ++j)
              for (int k = 0; k < 8; ++k) {
                const double dx = 1 + .25 * nodes[i] - x,
                             dy = .25 * nodes[j] - y, z = 1 + .25 * nodes[k];
                const double r2 = dx * dx + dy * dy + z * z;
                const double sigma = mediumSigma > 0.f
                    ? -std::log(double(std::exp(-mediumSigma)))
                    : 0.;
                const double attenuation = std::exp(-sigma
                    * (double(.65f) - double(.55f)) * std::sqrt(r2) / dx);
                value += weights[i] * weights[j] * weights[k] * z
                    / (r2 * std::sqrt(r2)) * .015625 * .25 * attenuation;
              }
        }
      result[py * SIZE + px] = .7 / std::acos(-1.) * value;
    }
  return result;
}
static void check(const char *label,
    const std::vector<V4> &image,
    const std::vector<double> &reference,
    double factor)
{
  long double sum = 0., squares = 0., error2 = 0., target2 = 0.;
  for (size_t i = 0; i < image.size(); ++i) {
    for (int c = 0; c < 4; ++c)
      require(std::isfinite(image[i][c]), "nonfinite source output");
    require(image[i][3] == 1.f, "source changed receiver coverage");
    const double target = factor * reference[i], error = image[i][0] - target;
    sum += error;
    squares += error * error;
    error2 += error * error;
    target2 += target * target;
    require(std::abs(image[i][0] - image[i][1]) < 1e-6
            && std::abs(image[i][0] - image[i][2]) < 1e-6,
        "neutral source channel weights");
  }
  const double mean = double(sum / image.size());
  const double variance = std::max(
      0., double((squares - sum * sum / image.size()) / (image.size() - 1)));
  const double tolerance = 6. * std::sqrt(variance / image.size()) + 2e-5;
  std::printf("%s: mean error %.9g tolerance %.9g NRMS %.6g\n",
      label,
      mean,
      tolerance,
      target2 ? std::sqrt(double(error2 / target2)) : 0.);
  require(std::abs(mean) <= tolerance, "source contribution ownership/mean");
  if (factor == 0.)
    require(squares == 0., "opaque coverage leaked source");
}
int main()
try {
  Scene scene;
  const auto reference = oracle();
  const auto *defaultMode =
      static_cast<const int *>(anariGetParameterInfo(scene.d,
          ANARI_RENDERER,
          "quality",
          "volumeSourceResampling",
          ANARI_BOOL,
          "default",
          ANARI_BOOL));
  require(defaultMode && *defaultMode == 0, "source resampling must be opt-in");
  scene.setup(false, 1);
  const auto original = scene.render();
  scene.set(scene.renderer, "volumeSourceResampling", ANARI_BOOL, 1);
  scene.commit(scene.renderer);
  check("resampled source", scene.render(), reference, 1.);
  scene.set(scene.renderer, "volumeSourceResampling", ANARI_BOOL, 0);
  scene.commit(scene.renderer);
  require(
      scene.render() == original, "source resampling altered default state");
  for (int resample : {0, 1}) {
    scene.set(scene.renderer, "volumeSourceResampling", ANARI_BOOL, resample);
    scene.commit(scene.renderer);
    scene.setup(false, 1);
    check("matte direct source", scene.render(), reference, 1.);
#ifdef VISRTX_NATIVE_PBR
    scene.setup(true, 1);
    check("native replacement", scene.render(), reference, 1.);
    scene.setup(true, 2);
    check("duplicate labeled emitters", scene.render(), reference, 2.);
    scene.setup(true, 1, .5f, .8f);
    check(
        "half holes plus covered transmission", scene.render(), reference, .9);
    scene.setup(true, 1, 1.f, 1.f);
    check("covered delta restores owner", scene.render(), reference, 1.);
#endif
    scene.setup(false, 1, .5f, .8f);
    check("matte direct holes only", scene.render(), reference, .5);
    scene.setup(false, 1, 1.f, 0.f);
    check("opaque source connection", scene.render(), reference, 0.);
    scene.setup(false, 1, -1.f, 0.f, 4.f);
    check("non-emitting medium attenuates source",
        scene.render(),
        oracle(4.f),
        1.);
  }
  scene.set(scene.lastEmitter, "emission", ANARI_FLOAT32_VEC3, V3{2, 2, 2});
  scene.commit(scene.lastEmitter);
  check("resampled source update", scene.render(), oracle(4.f), 2.);
  scene.set(scene.lastEmitter, "visible", ANARI_BOOL, 0);
  scene.commit(scene.lastEmitter);
  check("resampled empty source set", scene.render(), reference, 0.);
  scene.set(scene.lastEmitter, "visible", ANARI_BOOL, 1);
  scene.commit(scene.lastEmitter);
  check("resampled source restoration", scene.render(), oracle(4.f), 2.);
  return 0;
} catch (const std::exception &e) {
  std::fprintf(stderr, "FAIL: %s\n", e.what());
  return 1;
}
