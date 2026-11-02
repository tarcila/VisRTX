// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <anari/ext/visrtx/makeVisRTXDevice.h>
#include <anari/anari_cpp.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Box = std::array<float, 6>;
using RGB = std::array<float, 3>;
using RGBA = std::array<float, 4>;
const Box domain{0, 0, 0, 1, 1, 1};

void require(bool ok, const char *message)
{
  if (!ok)
    throw std::runtime_error(message);
}

struct Messages
{
  std::mutex mutex;
  std::vector<std::string> warnings;
  std::atomic<int> errors{0};
  static void callback(const void *,
      ANARIDevice,
      ANARIObject,
      ANARIDataType,
      ANARIStatusSeverity,
      ANARIStatusCode,
      const char *);
};

void Messages::callback(const void *ptr,
    ANARIDevice,
    ANARIObject,
    ANARIDataType,
    ANARIStatusSeverity severity,
    ANARIStatusCode,
    const char *message)
{
  auto &self = *const_cast<Messages *>(static_cast<const Messages *>(ptr));
  std::lock_guard<std::mutex> lock(self.mutex);
  if (severity <= ANARI_SEVERITY_ERROR) {
    ++self.errors;
    std::fprintf(stderr, "ANARI error: %s\n", message);
  } else if (severity == ANARI_SEVERITY_WARNING)
    self.warnings.emplace_back(message);
}

struct Scene
{
  Messages messages;
  ANARIDevice d{};
  std::vector<ANARIObject> objects;
  Scene();
  ~Scene();
  template <typename T>
  T keep(T object);
  void release(ANARIObject object);
  ANARIVolume volume();
  ANARISpatialField field(bool rgb, ANARIArray3D &array);
};
Scene::Scene()
{
  d = makeVisRTXDevice(Messages::callback, &messages);
  require(d != nullptr, "device creation");
  anariCommitParameters(d, d);
}
Scene::~Scene()
{
  for (auto it = objects.rbegin(); it != objects.rend(); ++it)
    anariRelease(d, *it);
  anariRelease(d, d);
}
template <typename T>
T Scene::keep(T object)
{
  objects.push_back(reinterpret_cast<ANARIObject>(object));
  return object;
}
void Scene::release(ANARIObject object)
{
  const auto it = std::find(objects.begin(), objects.end(), object);
  require(it != objects.end(), "test releasing unknown object");
  objects.erase(it);
  anariRelease(d, object);
}
ANARIVolume Scene::volume()
{
  auto v = keep(anariNewVolume(d, "principledVolume"));
  anariSetParameter(d, v, "bounds", ANARI_FLOAT32_BOX3, domain.data());
  anariCommitParameters(d, v);
  return v;
}
ANARISpatialField Scene::field(bool rgb, ANARIArray3D &array)
{
  array = keep(anariNewArray3D(d,
      nullptr,
      nullptr,
      nullptr,
      rgb ? ANARI_FLOAT32_VEC3 : ANARI_FLOAT32,
      2,
      2,
      2));
  auto *p = static_cast<float *>(anariMapArray(d, array));
  std::fill(p, p + (rgb ? 24 : 8), .5f);
  anariUnmapArray(d, array);
  auto f = keep(anariNewSpatialField(d, "structuredRegular"));
  anariSetParameter(d, f, "data", ANARI_ARRAY3D, &array);
  anariCommitParameters(d, f);
  return f;
}

bool valid(Scene &s, ANARIObject object)
{
  bool result = false;
  require(anariGetProperty(s.d,
              object,
              "valid",
              ANARI_BOOL,
              &result,
              sizeof(result),
              ANARI_WAIT),
      "valid property missing");
  return result;
}

void structure(Scene &s)
{
  auto v = s.keep(anariNewVolume(s.d, "principledVolume"));
  anariCommitParameters(s.d, v);
  require(!valid(s, v), "missing required bounds accepted");
  float wrong = 1.f;
  anariSetParameter(s.d, v, "bounds", ANARI_FLOAT32, &wrong);
  anariCommitParameters(s.d, v);
  require(!valid(s, v), "wrong bounds type accepted");
  for (Box box : {Box{0, 0, 0, 0, 1, 1},
           Box{1, 0, 0, 0, 1, 1},
           Box{0, 0, 0, INFINITY, 1, 1},
           Box{NAN, 0, 0, 1, 1, 1}}) {
    anariSetParameter(s.d, v, "bounds", ANARI_FLOAT32_BOX3, box.data());
    anariCommitParameters(s.d, v);
    require(!valid(s, v), "invalid bounds accepted");
  }
  anariSetParameter(s.d, v, "bounds", ANARI_FLOAT32_BOX3, domain.data());
  anariCommitParameters(s.d, v);
  require(valid(s, v), "default homogeneous volume rejected");
  Box result{};
  require(anariGetProperty(s.d,
              v,
              "bounds",
              ANARI_FLOAT32_BOX3,
              result.data(),
              sizeof(result),
              ANARI_WAIT)
          && result == domain,
      "committed bounds property");
  // Uncommitted staging must not leak into a buffered commit.
  const Box captured{-1, -1, -1, 2, 2, 2};
  const Box staged{-3, -3, -3, 4, 4, 4};
  anariSetParameter(s.d, v, "bounds", ANARI_FLOAT32_BOX3, captured.data());
  anariCommitParameters(s.d, v);
  anariSetParameter(s.d, v, "bounds", ANARI_FLOAT32_BOX3, staged.data());
  require(valid(s, v), "captured state invalid");
  anariGetProperty(s.d,
      v,
      "bounds",
      ANARI_FLOAT32_BOX3,
      result.data(),
      sizeof(result),
      ANARI_WAIT);
  require(result == captured, "staging leaked into commit");
  anariCommitParameters(s.d, v);
  anariGetProperty(s.d,
      v,
      "bounds",
      ANARI_FLOAT32_BOX3,
      result.data(),
      sizeof(result),
      ANARI_WAIT);
  require(result == staged, "next commit lost staged bounds");
  for (const char *name :
      {"value", "valueRange", "color", "opacity", "unitDistance"}) {
    anariSetParameter(s.d, v, name, ANARI_FLOAT32, &wrong);
    anariCommitParameters(s.d, v);
    require(!valid(s, v), "TF parameter silently accepted");
    anariUnsetParameter(s.d, v, name);
    anariCommitParameters(s.d, v);
    require(valid(s, v), "TF parameter removal did not recover");
  }
}

void typesAndBindings(Scene &s)
{
  ANARIArray3D scalarArray{}, rgbArray{};
  auto scalar = s.field(false, scalarArray);
  auto rgb = s.field(true, rgbArray);
  auto v = s.volume();
  struct Role
  {
    const char *name;
    bool rgb;
  };
  const Role roles[] = {{"density", false},
      {"scatteringColor", true},
      {"transmissionColor", true},
      {"transmissionDistance", false},
      {"anisotropy", false},
      {"secondaryAnisotropy", false},
      {"secondaryWeight", false},
      {"emission", true},
      {"temperature", false},
      {"blackbodyIntensity", false},
      {"blackbodyTint", true}};
  const float one = 1.f;
  const RGB white{1, 1, 1};
  for (const auto &role : roles) {
    const auto wrongType = role.rgb ? ANARI_FLOAT32 : ANARI_FLOAT32_VEC3;
    const void *wrongValue =
        role.rgb ? static_cast<const void *>(&one) : white.data();
    for (const auto name :
        {std::string(role.name), std::string(role.name) + ".outside"}) {
      anariSetParameter(s.d, v, name.c_str(), wrongType, wrongValue);
      anariCommitParameters(s.d, v);
      require(!valid(s, v), "wrong numeric property type accepted");
      anariUnsetParameter(s.d, v, name.c_str());
    }
    const auto name = std::string(role.name) + ".field";
    anariSetParameter(s.d, v, name.c_str(), ANARI_FLOAT32, &one);
    anariCommitParameters(s.d, v);
    require(!valid(s, v), "wrong field parameter type accepted");
    auto wrongField = role.rgb ? scalar : rgb;
    anariSetParameter(s.d, v, name.c_str(), ANARI_SPATIAL_FIELD, &wrongField);
    anariCommitParameters(s.d, v);
    require(!valid(s, v), "wrong field arity accepted");
    auto field = role.rgb ? rgb : scalar;
    anariSetParameter(s.d, v, name.c_str(), ANARI_SPATIAL_FIELD, &field);
    anariCommitParameters(s.d, v);
    require(valid(s, v), "matching typed binding failed");
    anariUnsetParameter(s.d, v, name.c_str());
    anariCommitParameters(s.d, v);
    require(valid(s, v), "unbinding did not restore constant");
    ANARISpatialField nullField = nullptr;
    anariSetParameter(s.d, v, name.c_str(), ANARI_SPATIAL_FIELD, &nullField);
    anariCommitParameters(s.d, v);
    require(valid(s, v), "null optional field did not select constant");
    anariUnsetParameter(s.d, v, name.c_str());
  }
  for (const char *name : {"id", "visible", "name"}) {
    anariSetParameter(s.d, v, name, ANARI_FLOAT32, &one);
    anariCommitParameters(s.d, v);
    require(!valid(s, v), "wrong common parameter type accepted");
    anariUnsetParameter(s.d, v, name);
  }
  anariCommitParameters(s.d, v);
  require(valid(s, v), "common parameter type recovery");
  // Required bounds for extinction are construction checks, not sample scans.
  const RGB zero{0, 0, 0};
  anariSetParameter(
      s.d, v, "transmissionColor", ANARI_FLOAT32_VEC3, zero.data());
  anariCommitParameters(s.d, v);
  require(!valid(s, v), "unusable extinction bound accepted");
  anariUnsetParameter(s.d, v, "transmissionColor");
  anariSetParameter(s.d, v, "density.field", ANARI_SPATIAL_FIELD, &scalar);
  anariSetParameter(s.d, v, "temperature.field", ANARI_SPATIAL_FIELD, &scalar);
  anariCommitParameters(s.d, v);
  require(valid(s, v), "shared scalar field failed");
  anariUnsetParameter(s.d, v, "temperature.field");
  anariCommitParameters(s.d, v);
  require(valid(s, v), "removing one shared binding failed");
  // Arity change invalidates the remaining density binding, and recovery must
  // happen via the observer, without recommitting the volume.
  anariSetParameter(s.d, scalar, "data", ANARI_ARRAY3D, &rgbArray);
  anariCommitParameters(s.d, scalar);
  require(!valid(s, v), "remaining shared-field observer was lost");
  anariSetParameter(s.d, scalar, "data", ANARI_ARRAY3D, &scalarArray);
  anariCommitParameters(s.d, scalar);
  require(valid(s, v), "observer-only recovery failed");
  const Box emptyROI{2, 2, 2, 3, 3, 3};
  anariSetParameter(s.d, scalar, "roi", ANARI_FLOAT32_BOX3, emptyROI.data());
  anariCommitParameters(s.d, scalar);
  require(valid(s, v), "empty field ROI invalidated outside-only volume");
  s.release(v);
  // Deleting a consumer must detach its one observer before this notification.
  auto *data = static_cast<float *>(anariMapArray(s.d, scalarArray));
  std::fill(data, data + 8, .75f);
  anariUnmapArray(s.d, scalarArray);
  require(valid(s, scalar), "field update after volume deletion failed");

  auto retained = s.volume();
  anariSetParameter(
      s.d, retained, "density.field", ANARI_SPATIAL_FIELD, &scalar);
  anariCommitParameters(s.d, retained);
  s.release(scalar); // only volume/snapshot references remain
  require(
      valid(s, retained), "bound field not retained through buffered commit");
  const bool hidden = false;
  anariSetParameter(s.d, retained, "visible", ANARI_BOOL, &hidden);
  const uint32_t id = 123;
  anariSetParameter(s.d, retained, "id", ANARI_UINT32, &id);
  anariSetParameter(s.d, retained, "name", ANARI_STRING, "retained medium");
  anariCommitParameters(s.d, retained);
  require(valid(s, retained), "common volume parameters failed");
}

std::vector<RGBA> image(Scene &s, ANARIWorld world, const char *subtype)
{
  auto camera = s.keep(anariNewCamera(s.d, "orthographic"));
  const RGB position{.5f, .5f, 3.f}, direction{0, 0, -1};
  const float height = 2.f;
  anariSetParameter(
      s.d, camera, "position", ANARI_FLOAT32_VEC3, position.data());
  anariSetParameter(
      s.d, camera, "direction", ANARI_FLOAT32_VEC3, direction.data());
  anariSetParameter(s.d, camera, "height", ANARI_FLOAT32, &height);
  anariCommitParameters(s.d, camera);
  auto renderer = s.keep(anariNewRenderer(s.d, subtype));
  const RGBA background{.1f, .2f, .3f, 0.f};
  const bool no = false;
  anariSetParameter(
      s.d, renderer, "background", ANARI_FLOAT32_VEC4, background.data());
  anariSetParameter(s.d, renderer, "denoise", ANARI_BOOL, &no);
  anariSetParameter(s.d, renderer, "fireflyFilterMode", ANARI_STRING, "none");
  anariCommitParameters(s.d, renderer);
  auto frame = s.keep(anariNewFrame(s.d));
  const std::array<uint32_t, 2> size{8, 8};
  const ANARIDataType color = ANARI_FLOAT32_VEC4;
  anariSetParameter(s.d, frame, "size", ANARI_UINT32_VEC2, size.data());
  anariSetParameter(s.d, frame, "channel.color", ANARI_DATA_TYPE, &color);
  anariSetParameter(s.d, frame, "world", ANARI_WORLD, &world);
  anariSetParameter(s.d, frame, "camera", ANARI_CAMERA, &camera);
  anariSetParameter(s.d, frame, "renderer", ANARI_RENDERER, &renderer);
  anariCommitParameters(s.d, frame);
  anariRenderFrame(s.d, frame);
  anariFrameReady(s.d, frame, ANARI_WAIT);
  uint32_t width = 0, heightOut = 0;
  ANARIDataType type = ANARI_UNKNOWN;
  const auto *pixels = static_cast<const RGBA *>(
      anariMapFrame(s.d, frame, "channel.color", &width, &heightOut, &type));
  require(
      pixels && width == 8 && heightOut == 8 && type == color, "map failed");
  const std::vector<RGBA> result(pixels, pixels + 64);
  anariUnmapFrame(s.d, frame, "channel.color");
  return result;
}

void renderSafety(Scene &s)
{
  auto v = s.volume();
  require(valid(s, v), "render-safety volume invalid");
  ANARIArray3D array{};
  auto field = s.field(false, array);
  auto tf = s.keep(anariNewVolume(s.d, "transferFunction1D"));
  anariSetParameter(s.d, tf, "value", ANARI_SPATIAL_FIELD, &field);
  const float opacity = .2f;
  anariSetParameter(s.d, tf, "opacity", ANARI_FLOAT32, &opacity);
  anariCommitParameters(s.d, tf);
  auto world = [&](const std::vector<ANARIVolume> &volumes) {
    auto w = s.keep(anariNewWorld(s.d));
    if (!volumes.empty()) {
      auto a = s.keep(anariNewArray1D(
          s.d, nullptr, nullptr, nullptr, ANARI_VOLUME, volumes.size()));
      std::memcpy(anariMapArray(s.d, a),
          volumes.data(),
          volumes.size() * sizeof(ANARIVolume));
      anariUnmapArray(s.d, a);
      anariSetParameter(s.d, w, "volume", ANARI_ARRAY1D, &a);
    }
    anariCommitParameters(s.d, w);
    return w;
  };
  const auto empty = world({}), only = world({v}), legacy = world({tf}),
             mixed = world({tf, v});
  require(image(s, empty, "quality") != image(s, only, "quality"),
      "Quality principled activation missing");
  for (const char *renderer : {"fast", "debug", "test"}) {
    require(image(s, empty, renderer) == image(s, only, renderer),
        "unsupported principled contribution was not safely omitted");
    require(image(s, legacy, renderer) == image(s, mixed, renderer),
        "principled payload corrupted TF transport");
    require(valid(s, v), "renderer omission invalidated shared volume");
  }
  bool warned = false;
  {
    std::lock_guard<std::mutex> lock(s.messages.mutex);
    for (const auto &m : s.messages.warnings)
      warned |= m.find("principledVolume unsupported by renderer")
          != std::string::npos;
  }
  require(warned, "missing transport diagnostic");
  Box bounds{};
  require(anariGetProperty(s.d,
              only,
              "bounds",
              ANARI_FLOAT32_BOX3,
              bounds.data(),
              sizeof(bounds),
              ANARI_WAIT),
      "world bounds property");
  // World bounds come from OptiX's emitted acceleration bounds, which may be
  // conservatively expanded. Intrinsic volume bounds above are exact; these
  // need to be finite, nonempty and enclose the authored medium.
  for (int c = 0; c < 3; ++c)
    require(std::isfinite(bounds[c]) && std::isfinite(bounds[c + 3])
            && bounds[c] <= domain[c] && bounds[c + 3] >= domain[c + 3],
        "intrinsic world bounds lost by omission");
  const Box expanded{-2, -2, -2, 3, 3, 3};
  anariSetParameter(s.d, v, "bounds", ANARI_FLOAT32_BOX3, expanded.data());
  anariCommitParameters(s.d, v);
  require(anariGetProperty(s.d,
              only,
              "bounds",
              ANARI_FLOAT32_BOX3,
              bounds.data(),
              sizeof(bounds),
              ANARI_WAIT),
      "updated world bounds property");
  for (int c = 0; c < 3; ++c)
    require(bounds[c] <= expanded[c] && bounds[c + 3] >= expanded[c + 3],
        "changed domain did not invalidate BLAS");
  const bool hidden = false;
  anariSetParameter(s.d, v, "visible", ANARI_BOOL, &hidden);
  anariCommitParameters(s.d, v);
  anariGetProperty(s.d,
      only,
      "bounds",
      ANARI_FLOAT32_BOX3,
      bounds.data(),
      sizeof(bounds),
      ANARI_WAIT);
  require(valid(s, v) && bounds[0] > bounds[3],
      "visibility did not update membership independently of validity");
  const bool visible = true;
  anariSetParameter(s.d, v, "visible", ANARI_BOOL, &visible);
  anariUnsetParameter(s.d, v, "bounds");
  anariCommitParameters(s.d, v);
  anariGetProperty(s.d,
      only,
      "bounds",
      ANARI_FLOAT32_BOX3,
      bounds.data(),
      sizeof(bounds),
      ANARI_WAIT);
  require(!valid(s, v) && bounds[0] > bounds[3],
      "invalid volume remained in world bounds");
  anariSetParameter(s.d, v, "bounds", ANARI_FLOAT32_BOX3, domain.data());
  anariCommitParameters(s.d, v);
  anariGetProperty(s.d,
      only,
      "bounds",
      ANARI_FLOAT32_BOX3,
      bounds.data(),
      sizeof(bounds),
      ANARI_WAIT);
  require(valid(s, v) && bounds[0] <= 0.f && bounds[3] >= 1.f,
      "recovered volume missing from world bounds");
}

} // namespace

int main()
try {
  Scene scene;
  structure(scene);
  typesAndBindings(scene);
  renderSafety(scene);
  const char **subtypes = anariGetObjectSubtypes(scene.d, ANARI_VOLUME);
  for (size_t i = 0; subtypes && subtypes[i]; ++i)
    require(std::strcmp(subtypes[i], "principledVolume") != 0,
        "unfinished subtype advertised");
  require(scene.messages.errors == 0, "unexpected status errors");
  std::puts(
      "principledVolume: public structure, bindings, recovery and render safety pass");
  return 0;
} catch (const std::exception &e) {
  std::fprintf(stderr, "FAIL: %s\n", e.what());
  return 1;
}
