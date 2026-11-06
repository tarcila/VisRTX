// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// Public-ANARI measurement driver. No private renderer switch or seed mutation.
#include <anari/anari.h>
#include <cuda_runtime_api.h>
#include <nanovdb/io/IO.h>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using F3 = std::array<float, 3>;
using F4 = std::array<float, 4>;
using Box = std::array<float, 6>;
namespace fs = std::filesystem;
struct Scene
{
  ANARILibrary lib{};
  ANARIDevice d{};
  ANARIMaterial lastMaterial{};
  std::vector<ANARIObject> objects;
  int errors = 0;
  explicit Scene(const char *library);
  ~Scene();
  static void status(const void *,
      ANARIDevice,
      ANARIObject,
      ANARIDataType,
      ANARIStatusSeverity,
      ANARIStatusCode,
      const char *);
  template <typename T>
  T keep(T object);
  template <typename T>
  void set(
      ANARIObject object, const char *name, ANARIDataType type, const T &value);
  void str(ANARIObject object, const char *name, const char *value);
  void commit(ANARIObject object);
  ANARIArray1D array(
      ANARIDataType type, const void *src, size_t count, size_t bytes);
  ANARISpatialField field(const fs::path &path);
  ANARIVolume volume(Box bounds, F3 color, F3 albedo, F3 emission);
  ANARISurface mesh(const std::array<F3, 4> &vertices,
      float reflectance,
      bool native = false);
};
void Scene::status(const void *ptr,
    ANARIDevice,
    ANARIObject,
    ANARIDataType,
    ANARIStatusSeverity severity,
    ANARIStatusCode,
    const char *message)
{
  auto &s = *static_cast<Scene *>(const_cast<void *>(ptr));
  if (severity <= ANARI_SEVERITY_ERROR)
    ++s.errors;
  if (severity <= ANARI_SEVERITY_WARNING)
    std::cerr << message << '\n';
}
Scene::Scene(const char *library)
{
  lib = anariLoadLibrary(library, status, this);
  if (!lib)
    throw std::runtime_error("library load");
  d = anariNewDevice(lib, "default");
  commit(d);
}
Scene::~Scene()
{
  for (auto i = objects.rbegin(); i != objects.rend(); ++i)
    anariRelease(d, *i);
  anariRelease(d, d);
  anariUnloadLibrary(lib);
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
void Scene::str(ANARIObject o, const char *n, const char *v)
{
  anariSetParameter(d, o, n, ANARI_STRING, v);
}
void Scene::commit(ANARIObject o)
{
  anariCommitParameters(d, o);
}
ANARIArray1D Scene::array(
    ANARIDataType type, const void *src, size_t count, size_t bytes)
{
  auto a = keep(anariNewArray1D(d, nullptr, nullptr, nullptr, type, count));
  std::memcpy(anariMapArray(d, a), src, bytes);
  anariUnmapArray(d, a);
  return a;
}
ANARISpatialField Scene::field(const fs::path &path)
{
  auto grid = nanovdb::io::readGrid<>(path.string());
  auto a = array(ANARI_UINT8, grid.data(), grid.size(), grid.size());
  auto f = keep(anariNewSpatialField(d, "nanovdb"));
  set(f, "data", ANARI_ARRAY1D, a);
  str(f, "dataCentering", "cell");
  str(f, "filter", "linear");
  commit(f);
  return f;
}
ANARIVolume Scene::volume(Box bounds, F3 color, F3 albedo, F3 emission)
{
  auto v = keep(anariNewVolume(d, "principledVolume"));
  set(v, "bounds", ANARI_FLOAT32_BOX3, bounds);
  set(v, "transmissionColor", ANARI_FLOAT32_VEC3, color);
  set(v, "scatteringColor", ANARI_FLOAT32_VEC3, albedo);
  set(v, "emission", ANARI_FLOAT32_VEC3, emission);
  commit(v);
  return v;
}
ANARISurface Scene::mesh(
    const std::array<F3, 4> &points, float gray, bool native)
{
  auto g = keep(anariNewGeometry(d, "triangle"));
  auto a = array(ANARI_FLOAT32_VEC3, points.data(), 4, sizeof(points));
  set(g, "vertex.position", ANARI_ARRAY1D, a);
  const std::array<std::array<unsigned, 3>, 2> triangles{
      {{0, 1, 2}, {0, 2, 3}}};
  a = array(ANARI_UINT32_VEC3, triangles.data(), 2, sizeof(triangles));
  set(g, "primitive.index", ANARI_ARRAY1D, a);
  commit(g);
  auto m = keep(anariNewMaterial(d, native ? "physicallyBased" : "matte"));
  lastMaterial = m;
  set(m,
      native ? "baseColor" : "color",
      ANARI_FLOAT32_VEC3,
      F3{gray, gray, gray});
  if (native) {
    set(m, "specular", ANARI_FLOAT32, 0.f);
    set(m, "metallic", ANARI_FLOAT32, 0.f);
    set(m, "roughness", ANARI_FLOAT32, 1.f);
  }
  commit(m);
  auto surface = keep(anariNewSurface(d));
  set(surface, "geometry", ANARI_GEOMETRY, g);
  set(surface, "material", ANARI_MATERIAL, m);
  commit(surface);
  return surface;
}
static double seconds(std::chrono::steady_clock::time_point start)
{
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
      .count();
}
int main(int argc, char **argv)
try {
  if (argc < 11 || argc > 15)
    throw std::runtime_error(
        "quality_baseline library renderer scene assets output size spp launches depth rate [static|camera|light|instance|density|temperature|material] [randomSeed] [shadowRate] [reference|fast]");
  const std::string renderer = argv[2], name = argv[3];
  const std::string motion = argc >= 12 ? argv[11] : "static";
  const auto seedValue = argc >= 13 ? std::stoull(argv[12]) : 0ull;
  if (seedValue > UINT32_MAX)
    throw std::runtime_error("randomSeed exceeds uint32");
  const auto randomSeed = uint32_t(seedValue);
  if (motion != "static" && motion != "camera" && motion != "light"
      && motion != "instance" && motion != "density" && motion != "temperature"
      && motion != "material")
    throw std::runtime_error("unknown motion mode");
  if (motion != "static" && name != "cloud" && name != "plume")
    throw std::runtime_error("motion fixtures require cloud or plume");
  if ((motion == "temperature" || motion == "material") && name != "plume")
    throw std::runtime_error("temperature/material motion requires plume");
  const fs::path assets = argv[4], output = argv[5];
  const unsigned size = std::stoul(argv[6]);
  const int spp = std::stoi(argv[7]), launches = std::stoi(argv[8]),
            depth = std::stoi(argv[9]);
  const float rate = std::stof(argv[10]);
  const float shadowRate = argc >= 14 ? std::stof(argv[13]) : rate;
  const std::string fieldSampling = argc >= 15 ? argv[14] : "reference";
  if (fieldSampling != "reference" && fieldSampling != "fast")
    throw std::runtime_error("invalid Quality field sampling");
  if (fieldSampling == "fast" && renderer != "quality")
    throw std::runtime_error("fast field sampling requires Quality");
  if (!std::isfinite(rate) || rate < 1e-3f || !std::isfinite(shadowRate)
      || shadowRate < 1e-4f)
    throw std::runtime_error(
        "sampling rates must be finite and above renderer minima");
  fs::create_directories(output);
  const auto start = std::chrono::steady_clock::now();
  Scene s(argv[1]);
  std::vector<ANARIVolume> volumes;
  std::vector<ANARISurface> surfaces;
  std::vector<ANARILight> lights;
  const bool receiver = name == "receiver-matte" || name == "receiver-native",
             overlap = name == "overlap";
  const Box domain{-1, -1, 0, 1, 1, 2};
  const F3 unit{1, 1, 1}, zero{0, 0, 0},
      neutral{.36787944117144233f, .36787944117144233f, .36787944117144233f};
  if (name == "cloud" || name == "plume" || name == "support") {
    auto density = s.field(assets
        / (name == "plume" ? "plume-128" : "cloud-128") / "density.nvdb");
    const F3 albedo = name == "cloud" ? F3{.9f, .9f, .9f}
        : name == "plume"             ? F3{.2f, .2f, .2f}
                                      : zero;
    auto v = s.volume(domain, neutral, albedo, zero);
    s.set(v, "density.field", ANARI_SPATIAL_FIELD, density);
    if (name == "cloud")
      s.set(v, "anisotropy", ANARI_FLOAT32, .65f);
    if (name == "plume") {
      auto temperature = s.field(assets / "plume-128" / "temperature.nvdb");
      s.set(v, "temperature.field", ANARI_SPATIAL_FIELD, temperature);
      s.set(v, "blackbodyIntensity", ANARI_FLOAT32, 1.f);
      surfaces.push_back(s.mesh(
          {F3{-2, -2, 0}, F3{2, -2, 0}, F3{2, 2, 0}, F3{-2, 2, 0}}, .65f));
      surfaces.push_back(s.mesh({F3{-.55f, -.35f, .05f},
                                    F3{-.55f, .35f, .05f},
                                    F3{-.55f, .35f, 1.25f},
                                    F3{-.55f, -.35f, 1.25f}},
          .7f));
    } else if (name == "support") {
      s.set(density, "roi", ANARI_FLOAT32_BOX3, Box{-1, -1, 0, -.25f, 1, 2});
      s.commit(density);
      auto emission = s.field(assets / "plume-256" / "emission_offset.nvdb");
      s.set(v, "emission", ANARI_FLOAT32_VEC3, unit);
      s.set(v, "emission.field", ANARI_SPATIAL_FIELD, emission);
    }
    s.commit(v);
    volumes.push_back(v);
    auto light = s.keep(anariNewLight(s.d, "directional"));
    s.set(light, "direction", ANARI_FLOAT32_VEC3, F3{1, -1, -1});
    s.set(light, "irradiance", ANARI_FLOAT32, 1.f);
    s.commit(light);
    lights.push_back(light);
  } else if (overlap) {
    volumes.push_back(
        s.volume(domain, F3{.7f, .5f, .9f}, zero, F3{.2f, .4f, .1f}));
    volumes.push_back(s.volume(Box{-1, -.5f, 0, 1, 1.5f, 2},
        F3{.6f, .8f, .4f},
        zero,
        F3{.3f, .05f, .2f}));
  } else if (receiver) {
    volumes.push_back(
        s.volume(Box{.75f, -.25f, .75f, 1.25f, .25f, 1.25f}, unit, zero, unit));
    // The emitter is outside the entire primary-camera footprint.
    surfaces.push_back(
        s.mesh({F3{-2, -2, 0}, F3{2, -2, 0}, F3{2, 2, 0}, F3{-2, 2, 0}},
            .7f,
            name == "receiver-native"));
  } else
    throw std::runtime_error("unknown scene");
  auto world = s.keep(anariNewWorld(s.d));
  auto a = s.array(ANARI_VOLUME,
      volumes.data(),
      volumes.size(),
      volumes.size() * sizeof(ANARIVolume));
  ANARIInstance movingInstance{};
  if (motion == "instance") {
    auto group = s.keep(anariNewGroup(s.d));
    s.set(group, "volume", ANARI_ARRAY1D, a);
    s.commit(group);
    movingInstance = s.keep(anariNewInstance(s.d, "transform"));
    s.set(movingInstance, "group", ANARI_GROUP, group);
    s.commit(movingInstance);
    auto instances =
        s.array(ANARI_INSTANCE, &movingInstance, 1, sizeof(movingInstance));
    s.set(world, "instance", ANARI_ARRAY1D, instances);
  } else
    s.set(world, "volume", ANARI_ARRAY1D, a);
  if (!surfaces.empty()) {
    a = s.array(ANARI_SURFACE,
        surfaces.data(),
        surfaces.size(),
        surfaces.size() * sizeof(ANARISurface));
    s.set(world, "surface", ANARI_ARRAY1D, a);
  }
  if (!lights.empty()) {
    a = s.array(ANARI_LIGHT,
        lights.data(),
        lights.size(),
        lights.size() * sizeof(ANARILight));
    s.set(world, "light", ANARI_ARRAY1D, a);
  }
  s.commit(world);
  auto camera = s.keep(anariNewCamera(
      s.d, receiver || overlap ? "orthographic" : "perspective"));
  s.set(camera,
      "position",
      ANARI_FLOAT32_VEC3,
      receiver      ? F3{0, 0, 3}
          : overlap ? F3{0, -3, 1}
                    : F3{0, -4.8f, 1.6f});
  s.set(camera,
      "direction",
      ANARI_FLOAT32_VEC3,
      receiver      ? F3{0, 0, -1}
          : overlap ? F3{0, 1, 0}
                    : F3{0, 4.8f, -.6f});
  s.set(camera, "up", ANARI_FLOAT32_VEC3, receiver ? F3{0, 1, 0} : F3{0, 0, 1});
  s.set(camera, "aspect", ANARI_FLOAT32, 1.f);
  if (receiver || overlap)
    s.set(camera, "height", ANARI_FLOAT32, 1.f);
  else
    s.set(camera, "fovy", ANARI_FLOAT32, 32.f * float(std::acos(-1.)) / 180.f);
  s.commit(camera);
  auto r = s.keep(anariNewRenderer(s.d, renderer.c_str()));
  s.str(r, "fireflyFilterMode", "none");
  if (fieldSampling == "fast")
    s.set(r, "volumeFastSampling", ANARI_BOOL, true);
  if (renderer == "quality")
    s.set(r, "randomSeed", ANARI_UINT32, randomSeed);
  else if (randomSeed != 0)
    throw std::runtime_error("randomSeed requires Quality");
  s.set(r, "pixelSamples", ANARI_INT32, spp);
  s.set(r, "sampleLimit", ANARI_INT32, 0);
  s.set(r, "maxRayDepth", ANARI_INT32, depth);
  s.set(r, "denoise", ANARI_BOOL, false);
  s.set(r, "checkerboarding", ANARI_BOOL, false);
  s.set(r, "ambientRadiance", ANARI_FLOAT32, 0.f);
  s.set(r, "ambientSamples", ANARI_INT32, 0);
  s.set(r, "volumeSamplingRate", ANARI_FLOAT32, rate);
  s.set(r, "volumeSamplingRateShadows", ANARI_FLOAT32, shadowRate);
  s.set(r,
      "background",
      ANARI_FLOAT32_VEC4,
      receiver ? F4{0, 0, 0, 0} : F4{.05f, .05f, .05f, 0});
  s.commit(r);
  auto frame = s.keep(anariNewFrame(s.d));
  s.set(frame, "size", ANARI_UINT32_VEC2, std::array<unsigned, 2>{size, size});
  s.set(frame, "channel.color", ANARI_DATA_TYPE, ANARI_FLOAT32_VEC4);
  s.set(frame, "world", ANARI_WORLD, world);
  s.set(frame, "camera", ANARI_CAMERA, camera);
  s.set(frame, "renderer", ANARI_RENDERER, r);
  s.commit(frame);
  std::cout << "setup_seconds " << seconds(start) << " motion " << motion
            << " random_seed " << randomSeed << " volume_rate " << rate
            << " shadow_rate " << shadowRate << " field_sampling "
            << fieldSampling << '\n';
  std::ofstream timings(output / "timings.csv");
  timings << "frame,samples,gpu_ms,wall_ms,map_ms,gpu_used_bytes\n";
  for (int i = 0; i < launches; ++i) {
    const auto t = std::chrono::steady_clock::now();
    // Deterministic per-launch edits. Wall timing includes commits/preparation;
    // GPU duration alone must not hide source-guide or acceleration rebuilds.
    const float delta = .1f * std::sin(.2f * float(i));
    if (motion == "camera") {
      s.set(camera, "position", ANARI_FLOAT32_VEC3, F3{delta, -4.8f, 1.6f});
      s.commit(camera);
    } else if (motion == "light") {
      s.set(lights[0], "direction", ANARI_FLOAT32_VEC3, F3{1 + delta, -1, -1});
      s.commit(lights[0]);
    } else if (motion == "instance") {
      const std::array<float, 16> transform{
          1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, delta, 0, 0, 1};
      s.set(movingInstance, "transform", ANARI_FLOAT32_MAT4, transform);
      s.commit(movingInstance);
    } else if (motion == "density" || motion == "temperature") {
      s.set(volumes[0], motion.c_str(), ANARI_FLOAT32, 1.f + delta);
      s.commit(volumes[0]);
    } else if (motion == "material") {
      s.set(s.lastMaterial,
          "color",
          ANARI_FLOAT32_VEC3,
          F3{.7f + delta, .7f + delta, .7f + delta});
      s.commit(s.lastMaterial);
    }
    anariRenderFrame(s.d, frame);
    anariFrameReady(s.d, frame, ANARI_WAIT);
    const double wall = seconds(t) * 1000.;
    float duration = 0;
    int samples = 0;
    if (!anariGetProperty(s.d,
            frame,
            "duration",
            ANARI_FLOAT32,
            &duration,
            sizeof(duration),
            ANARI_WAIT)
        || !anariGetProperty(s.d,
            frame,
            "numSamples",
            ANARI_INT32,
            &samples,
            sizeof(samples),
            ANARI_WAIT))
      throw std::runtime_error("frame properties");
    const auto m = std::chrono::steady_clock::now();
    uint32_t w, h;
    ANARIDataType type;
    const auto *p = static_cast<const float *>(
        anariMapFrame(s.d, frame, "channel.color", &w, &h, &type));
    if (!p || w != size || h != size || type != ANARI_FLOAT32_VEC4)
      throw std::runtime_error("frame map");
    const double mapTime = seconds(m) * 1000.;
    if (size <= 128 || i == launches - 1) {
      std::ofstream image(
          output / ("frame-" + std::to_string(i) + ".rgba"), std::ios::binary);
      image.write(reinterpret_cast<const char *>(p),
          size_t(size) * size * 4 * sizeof(float));
    }
    anariUnmapFrame(s.d, frame, "channel.color");
    size_t free = 0, total = 0;
    const auto err = cudaMemGetInfo(&free, &total);
    if (err != cudaSuccess)
      throw std::runtime_error(cudaGetErrorString(err));
    timings << i << "," << samples << "," << 1000. * duration << "," << wall
            << "," << mapTime << "," << (total - free) << '\n';
    timings.flush();
    std::cout << "frame " << i << " samples " << samples << " gpu_ms "
              << 1000. * duration << " wall_ms " << wall << '\n';
    std::cout.flush();
    if (s.errors)
      throw std::runtime_error("ANARI errors");
  }
#ifdef __linux__
  std::ifstream maps("/proc/self/maps");
  std::ofstream loaded(output / "loader.txt");
  std::string line;
  while (std::getline(maps, line))
    if (line.find("libanari_library_visrtx") != std::string::npos)
      loaded << line << '\n';
#endif
  return 0;
} catch (const std::exception &e) {
  std::cerr << "FAIL: " << e.what() << '\n';
  return 1;
}
