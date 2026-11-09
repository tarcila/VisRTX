// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#include <anari/anari_cpp/ext/std.h>
#include <anari/ext/visrtx/makeVisRTXDevice.h>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <thread>
#include <vector>
#include "MdlVolumeTestChecks.h"

namespace {
using V3 = std::array<float, 3>;
using V4 = std::array<float, 4>;
using Box = std::array<V3, 2>;
void require(bool value, const char *why)
{
  if (!value)
    throw std::runtime_error(why);
}
constexpr const char *FIELD_SOURCE = R"mdl(mdl 1.7;
import ::scene::*;
export material main(uniform float scale=0.25) =
 material(volume: material_volume(absorption_coefficient:
 color(scale * scene::data_lookup_float("density", 0.125))));
)mdl";
struct Fixture
{
  struct Outcome
  {
    unsigned errors{}, completions{};
    bool inspectFailure{};
    std::string cause, callbackError;
  };
  std::array<Outcome, 2> outcomes;
  anari::Device d{};
  anari::Material material{};
  anari::Camera camera{};
  anari::Renderer renderer{};
  std::array<anari::Volume, 2> volumes{};
  std::array<anari::SpatialField, 2> fields{};
  std::array<anari::Array3D, 2> data{};
  std::array<anari::Instance, 2> instances{};
  std::array<anari::World, 2> worlds{};
  std::array<anari::Frame, 2> frames{};
  std::vector<anari::Object> owned;
  bool quality;
  void releaseApplicationHandle(anari::Object object)
  {
    for (auto &entry : owned)
      if (entry == object) {
        anari::release(d, object);
        entry = nullptr;
        return;
      }
    throw std::runtime_error("unowned application handle");
  }
  template <typename T>
  T own(T object)
  {
    owned.push_back(object);
    return object;
  }
  static void status(const void *user,
      ANARIDevice d,
      ANARIObject object,
      ANARIDataType,
      ANARIStatusSeverity severity,
      ANARIStatusCode,
      const char *message)
  {
    auto &s = *static_cast<Fixture *>(const_cast<void *>(user));
    if (severity > ANARI_SEVERITY_ERROR)
      return;
    fprintf(stderr, "%s\n", message);
    for (int i = 0; i < 2; ++i) {
      if (object != s.frames[i])
        continue;
      auto &o = s.outcomes[i];
      ++o.errors;
      o.cause = message;
      if (o.inspectFailure && o.errors == 1) {
        try {
          checkFailedMdlFrame(d, s.frames[i]);
        } catch (const std::exception &e) {
          o.callbackError = e.what();
        }
      }
    }
  }
  static void completed(const void *user, ANARIDevice d, ANARIFrame frame)
  {
    auto &s = *static_cast<Fixture *>(const_cast<void *>(user));
    for (int i = 0; i < 2; ++i) {
      if (frame != s.frames[i])
        continue;
      auto &o = s.outcomes[i];
      ++o.completions;
      if (o.inspectFailure) {
        try {
          require(o.errors == 1, "failure precedes completion exactly once");
          checkFailedMdlFrame(d, frame);
        } catch (const std::exception &e) {
          o.callbackError = e.what();
        }
      }
    }
  }
  explicit Fixture(const char *kind) : quality(std::string(kind) == "quality")
  {
    d = makeVisRTXDevice(status, this);
    material = own(anari::newObject<anari::Material>(d, "mdl"));
    source(FIELD_SOURCE);
    camera = own(anari::newObject<anari::Camera>(d, "orthographic"));
    anari::setParameter(d, camera, "position", V3{0, 0, 0});
    anari::setParameter(d, camera, "direction", V3{0, 0, 1});
    anari::setParameter(d, camera, "up", V3{0, 1, 0});
    anari::setParameter(d, camera, "height", .25f);
    anari::commitParameters(d, camera);
    renderer = own(anari::newObject<anari::Renderer>(d, kind));
    anari::setParameter(d, renderer, "denoise", false);
    anari::setParameter(d, renderer, "fireflyFilterMode", "none");
    anari::setParameter(d, renderer, "ambientRadiance", 0.f);
    anari::setParameter(d, renderer, "background", V4{1, 1, 1, 1});
    anari::setParameter(d, renderer, "pixelSamples", quality ? 2048 : 1);
    anari::setParameter(d, renderer, "sampleLimit", 0);
    anari::setParameter(d, renderer, "volumeSamplingRate", 1.f);
    anari::commitParameters(d, renderer);
    // Reverse construction order is deliberate: no shared binding may depend
    // on the last Volume finalized.
    for (int i : {1, 0}) {
      data[i] = own(anari::newArray3D(
          d, nullptr, nullptr, nullptr, ANARI_FLOAT32, 2, 2, 2));
      fill(i, float(i + 1));
      fields[i] =
          own(anari::newObject<anari::SpatialField>(d, "structuredRegular"));
      anari::setParameter(d, fields[i], "data", data[i]);
      anari::setParameter(d, fields[i], "origin", V3{-1, -1, 2});
      anari::setParameter(d, fields[i], "spacing", V3{2, 2, 2});
      anari::commitParameters(d, fields[i]);
      volumes[i] = own(anari::newObject<anari::Volume>(d, "mdl"));
      anari::setParameter(d, volumes[i], "material", material);
      anari::setParameter(d, volumes[i], "value", fields[i]);
      anari::setParameter(d, volumes[i], "fieldName", "density");
      majorant(i, 4.f);
      auto group = own(anari::newObject<anari::Group>(d));
      anari::setParameterArray1D(d, group, "volume", &volumes[i], 1);
      anari::commitParameters(d, group);
      auto instance = instances[i] =
          own(anari::newObject<anari::Instance>(d, "transform"));
      anari::setParameter(d, instance, "group", group);
      anari::commitParameters(d, instance);
      worlds[i] = own(anari::newObject<anari::World>(d));
      anari::setParameterArray1D(d, worlds[i], "instance", &instance, 1);
      anari::commitParameters(d, worlds[i]);
      frames[i] = own(anari::newObject<anari::Frame>(d));
      setupFrame(frames[i], worlds[i]);
    }
  }
  ~Fixture()
  {
    for (auto f : frames)
      anariFrameReady(d, f, ANARI_WAIT);
    for (auto it = owned.rbegin(); it != owned.rend(); ++it)
      if (*it)
        anari::release(d, *it);
    anari::release(d, d);
  }
  void setupFrame(anari::Frame f, anari::World w)
  {
    anari::setParameter(d, f, "size", std::array<unsigned, 2>{8, 8});
    for (const char *c : {"channel.color", "channel.normal", "channel.albedo"})
      anari::setParameter(d,
          f,
          c,
          std::string(c) == "channel.color" ? ANARI_FLOAT32_VEC4
                                            : ANARI_FLOAT32_VEC3);
    anari::setParameter(d, f, "channel.depth", ANARI_FLOAT32);
    for (const char *c :
        {"channel.primitiveId", "channel.objectId", "channel.instanceId"})
      anari::setParameter(d, f, c, ANARI_UINT32);
    anari::setParameter(d, f, "world", w);
    anari::setParameter(d, f, "camera", camera);
    anari::setParameter(d, f, "renderer", renderer);
    ANARIFrameCompletionCallback cb = completed;
    anariSetParameter(
        d, f, "frameCompletionCallback", ANARI_FRAME_COMPLETION_CALLBACK, &cb);
    anari::setParameter(d, f, "frameCompletionCallbackUserData", (void *)this);
    anari::commitParameters(d, f);
  }
  void source(const char *code)
  {
    anari::setParameter(d, material, "sourceType", "code");
    anari::setParameter(d, material, "source", code);
    anari::commitParameters(d, material);
  }
  void fill(int i, float value)
  {
    auto *p = static_cast<float *>(anariMapArray(d, data[i]));
    for (int j = 0; j < 8; ++j)
      p[j] = value;
    anariUnmapArray(d, data[i]);
  }
  void majorant(int i, float m)
  {
    anari::setParameter(d, volumes[i], "extinctionMajorant", m);
    anari::commitParameters(d, volumes[i]);
  }
  void extent(int i, Box b)
  {
    anariSetParameter(d, volumes[i], "extent", ANARI_FLOAT32_BOX3, b.data());
    anari::commitParameters(d, volumes[i]);
  }
  int count(anari::Frame f)
  {
    int n = -1;
    require(anariGetProperty(
                d, f, "numSamples", ANARI_INT32, &n, sizeof(n), ANARI_WAIT),
        "numSamples query");
    return n;
  }
  void render(int i)
  {
    const auto before = outcomes[i].completions;
    anari::render(d, frames[i]);
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(60);
    while (!anariFrameReady(d, frames[i], ANARI_NO_WAIT)) {
      require(std::chrono::steady_clock::now() < deadline,
          "NO_WAIT eventually ready");
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    require(anariFrameReady(d, frames[i], ANARI_WAIT), "WAIT completes");
    if (outcomes[i].completions != before + 1)
      fprintf(stderr,
          "completion %d: before %u after %u\n",
          i,
          before,
          outcomes[i].completions);
    require(
        outcomes[i].completions == before + 1, "one completion per attempt");
  }
  std::array<double, 4> image(anari::Frame f)
  {
    auto map = anari::map<V4>(d, f, "channel.color");
    require(map.data && map.width == 8 && map.height == 8, "published color");
    std::array<double, 4> mean{};
    for (int p = 0; p < 64; ++p)
      for (int c = 0; c < 4; ++c)
        mean[c] += map.data[p][c] / 64.;
    anari::unmap(d, f, "channel.color");
    return mean;
  }
  void check(int i, const char *label, float opticalDepth, bool reset = false)
  {
    // Fixed budget, shared with the quality absorption fixture: 16 retained
    // batches of 8x8x2048 samples. No lucky retries or internal RNG seam.
    const int batches = quality ? 16 : 1;
    std::array<double, 4> previous{}, mean{}, m2{};
    int previousCount = 0;
    for (int b = 0; b < batches; ++b) {
      render(i);
      require(outcomes[i].errors == 0, "unexpected medium error");
      int n = count(frames[i]);
      if (b == 0 && reset)
        require(n == (quality ? 2048 : 1),
            "dependency change resets automatic accumulation");
      if (b == 0) {
        // Consume the existing initial allocation/reset transition.
        render(i);
        n = count(frames[i]);
        previous = image(frames[i]);
        previousCount = n;
        render(i);
        n = count(frames[i]);
      }
      auto accumulated = image(frames[i]);
      require(n > previousCount, "retained batches must advance");
      for (int c = 0; c < 4; ++c) {
        double x = (accumulated[c] * n - previous[c] * previousCount)
            / (n - previousCount);
        double delta = x - mean[c];
        mean[c] += delta / (b + 1);
        m2[c] += delta * (x - mean[c]);
      }
      previous = accumulated;
      previousCount = n;
    }
    for (int c = 0; c < 4; ++c) {
      const double ref = c == 3 ? 1. : std::exp(-double(opticalDepth));
      const double se =
          batches > 1 ? std::sqrt(m2[c] / (batches * (batches - 1))) : 0.;
      printf("%s c%d mean=%.9g ref=%.9g SE=%.9g\n", label, c, mean[c], ref, se);
      require(se <= .0016 && std::abs(mean[c] - ref) <= .01
              && std::abs(mean[c] - ref) <= 5 * se + .002,
          "independent Beer-Lambert/statistical rule");
    }
  }
};
void sharedArguments(Fixture &s)
{
  s.check(0, "initial field one", .5f);
  s.check(1, "initial field two", 1.f);
  anari::setParameter(s.d, s.material, "scale", .5f);
  // Pending parameters must not enter the launch.
  s.check(0, "uncommitted argument", .5f);
  anari::commitParameters(s.d, s.material);
  s.check(1, "shared committed argument two", 2.f, true);
  s.check(0, "shared committed argument one", 1.f, true);
  s.source(R"mdl(mdl 1.7; import ::scene::*;
export material main(uniform float scale=0.25) = material(volume: material_volume(
 absorption_coefficient: color(scale*(scene::data_isvalid("left") ?
 scene::data_lookup_float("left",0.125) : scene::data_lookup_float("right",0.125)))));
)mdl");
  for (int i : {1, 0}) {
    anari::setParameter(
        s.d, s.volumes[i], "fieldName", i == 0 ? "left" : "right");
    anari::commitParameters(s.d, s.volumes[i]);
  }
  s.check(1, "shared source distinct name right", 2.f, true);
  s.check(0, "shared source distinct name left", 1.f, true);
  s.source(FIELD_SOURCE);
  for (int i : {0, 1}) {
    anari::setParameter(s.d, s.volumes[i], "fieldName", "density");
    anari::commitParameters(s.d, s.volumes[i]);
  }
}
void fieldAndSourceUpdates(Fixture &s)
{
  // All prior renders have completed and all mappings have been unmapped.
  s.fill(0, 3.f);
  s.check(0, "array unmap propagates", 3.f, true);
  s.check(1, "other field remains isolated", 2.f);
  anari::setParameter(s.d, s.fields[0], "origin", V3{-1, -1, 3});
  anari::setParameter(s.d, s.fields[0], "spacing", V3{2, 2, 1});
  anari::commitParameters(s.d, s.fields[0]);
  s.check(0, "field shrink updates traversal", 1.5f, true);
  s.extent(0, Box{V3{-1, -1, 2}, V3{1, 1, 4}});
  // One unit supported (density 3), one authored default (density .125).
  s.check(0, "explicit extent preserves support/default split", 1.5625f, true);
  anari::setParameter(s.d, s.fields[0], "origin", V3{4, -1, 3});
  anari::commitParameters(s.d, s.fields[0]);
  s.check(0, "field move changes support not extent", .125f, true);
  anariUnsetParameter(s.d, s.volumes[0], "extent");
  anari::commitParameters(s.d, s.volumes[0]);
  s.check(0, "unset extent restores moved field domain", 0.f, true);
  anari::setParameter(s.d, s.fields[0], "origin", V3{-1, -1, 2});
  anari::setParameter(s.d, s.fields[0], "spacing", V3{2, 2, 2});
  anari::commitParameters(s.d, s.fields[0]);
  // Rename the source lookup and both bindings together; no redundant Volume
  // commit is used for ordinary source edits, only for the actual name edit.
  s.source(R"mdl(mdl 1.7;
import ::scene::*;
export material main(uniform float scale=0.25) = material(
 volume: material_volume(absorption_coefficient:
 color(scale * scene::data_lookup_float("fresh",0.125))));
)mdl");
  for (int i : {0, 1}) {
    anari::setParameter(s.d, s.volumes[i], "fieldName", "fresh");
    anari::commitParameters(s.d, s.volumes[i]);
  }
  s.check(0, "replacement source/string table one", 3.f, true);
  s.check(1, "replacement source/string table two", 2.f, true);
  s.source(R"mdl(mdl 1.7;
import ::scene::*;
export material main(uniform float scale=0.125) = material(
 volume: material_volume(absorption_coefficient:
 color(0.25 * scale * scene::data_lookup_float("fresh",0.125))));
)mdl");
  s.check(1, "source-only replacement two", .5f, true);
  s.check(0, "source-only replacement one", .75f, true);
  s.extent(0, Box{V3{-1, -1, 2}, V3{1, 1, 4}});
  anariUnsetParameter(s.d, s.volumes[0], "value");
  anari::commitParameters(s.d, s.volumes[0]);
  s.source(R"mdl(mdl 1.7;
import ::scene::*;
export material main() = material(volume: material_volume(
 absorption_coefficient: color(scene::data_isvalid("fresh") ?
 scene::data_lookup_float("fresh",0.125)*0.125 : 0.25)));
)mdl");
  s.check(0, "retained name has no stale binding", .5f, true);
  s.check(1, "other volume keeps its binding", .5f, true);
  anari::setParameter(s.d, s.volumes[0], "value", s.fields[1]);
  anari::commitParameters(s.d, s.volumes[0]);
  s.check(0, "restore different field reference", .5f, true);
  auto replacement = s.own(anari::newArray3D(
      s.d, nullptr, nullptr, nullptr, ANARI_FLOAT32, 2, 2, 2));
  auto *values = static_cast<float *>(anariMapArray(s.d, replacement));
  for (int j = 0; j < 8; ++j)
    values[j] = 1.f;
  anariUnmapArray(s.d, replacement);
  anari::setParameter(s.d, s.fields[1], "data", replacement);
  anari::commitParameters(s.d, s.fields[1]);
  for (int i : {1, 0})
    s.check(i, "shared field data-array replacement", .25f, true);
}
void manualEdits(Fixture &s)
{
  for (int i : {0, 1}) {
    anari::setParameter(s.d, s.frames[i], "accumulationVersion", uint64_t(7));
    anari::commitParameters(s.d, s.frames[i]);
    s.render(i);
    s.render(i);
    s.render(i);
  }
  const int before = s.count(s.frames[0]);
  s.fill(0, 1.f);
  // Both Volumes are bound to field 1; replace its contents through the
  // currently committed array, not the obsolete original application array.
  auto array = s.own(anari::newArray3D(
      s.d, nullptr, nullptr, nullptr, ANARI_FLOAT32, 2, 2, 2));
  auto *values = static_cast<float *>(anariMapArray(s.d, array));
  for (int j = 0; j < 8; ++j)
    values[j] = 2.f;
  anariUnmapArray(s.d, array);
  anari::setParameter(s.d, s.fields[1], "data", array);
  anari::commitParameters(s.d, s.fields[1]);
  s.render(0);
  require(s.count(s.frames[0]) == before + (s.quality ? 2048 : 1),
      "ordinary edit retains manual accumulation without version bump");
  for (int i : {0, 1}) {
    anari::setParameter(s.d, s.frames[i], "accumulationVersion", uint64_t(8));
    anari::commitParameters(s.d, s.frames[i]);
    s.check(i, "explicit manual accumulation version", .5f, true);
    anariUnsetParameter(s.d, s.frames[i], "accumulationVersion");
    anari::commitParameters(s.d, s.frames[i]);
  }
}
void resourceAndLifetime(Fixture &s)
{
  anari::setParameter(s.d, s.volumes[1], "value", s.fields[0]);
  anari::commitParameters(s.d, s.volumes[1]);
  s.source(R"mdl(mdl 1.7; import ::tex::*; import ::state::*; import ::scene::*;
export material main(uniform texture_2d image=texture_2d()) = material(
 volume: material_volume(absorption_coefficient:
 tex::lookup_color(image,float2(state::position().z/8,0.5))*
 scene::data_lookup_float("density",0.125)));)mdl");
  auto image = s.own(anari::newArray2D(
      s.d, nullptr, nullptr, nullptr, ANARI_FLOAT32_VEC3, 1, 1));
  auto write = [&](float value) {
    auto *pixel = static_cast<V3 *>(anariMapArray(s.d, image));
    *pixel = {value, value, value};
    anariUnmapArray(s.d, image);
  };
  write(.125f);
  auto sampler = s.own(anari::newObject<anari::Sampler>(s.d, "image2D"));
  anari::setParameter(s.d, sampler, "image", image);
  anari::commitParameters(s.d, sampler);
  anari::setParameter(s.d, s.material, "image", sampler);
  anari::commitParameters(s.d, s.material);
  for (int i : {0, 1})
    s.check(i, "shared texture argument", .25f, true);
  write(.25f);
  for (int i : {1, 0})
    s.check(i, "shared texture array unmap", .5f, true);
  auto replacement = s.own(anari::newArray2D(
      s.d, nullptr, nullptr, nullptr, ANARI_FLOAT32_VEC3, 1, 1));
  *static_cast<V3 *>(anariMapArray(s.d, replacement)) = V3{.5f, .5f, .5f};
  anariUnmapArray(s.d, replacement);
  anari::setParameter(s.d, sampler, "image", replacement);
  anari::commitParameters(s.d, sampler);
  for (int i : {0, 1})
    s.check(i, "shared sampler resource replacement", 1.f, true);
  // A source replacement changes the resource table and Argument Block.
  // Keep the same authored texture argument while adding an unrelated name.
  s.source(R"mdl(mdl 1.7; import ::tex::*; import ::state::*; import ::scene::*;
export material main(uniform texture_2d image=texture_2d()) = material(
 volume: material_volume(absorption_coefficient:
 tex::lookup_color(image,float2(state::position().z/8,0.5))*
 scene::data_lookup_float("density",0.125)*
 (scene::data_isvalid("unrelated") ? 0.75 : 0.5)),
 surface: material_surface());)mdl");
  for (int i : {1, 0})
    s.check(i, "resource-bearing source replacement", .5f, true);
  // Both Volumes retain the same field, and the Material retains the Sampler.
  // Releasing application references during a launch must not revoke readers.
  anari::render(s.d, s.frames[0]);
  s.releaseApplicationHandle(s.material);
  s.releaseApplicationHandle(sampler);
  s.releaseApplicationHandle(image);
  s.releaseApplicationHandle(replacement);
  for (int i : {0, 1}) {
    s.releaseApplicationHandle(s.fields[i]);
    s.releaseApplicationHandle(s.data[i]);
  }
  require(anariFrameReady(s.d, s.frames[0], ANARI_WAIT),
      "in-flight retained resources complete");
  for (int i : {0, 1})
    s.check(i, "retained resource/object lifetime", .5f);
}
void failAndRecover(Fixture &s, bool manual)
{
  if (manual)
    for (auto f : s.frames) {
      anari::setParameter(s.d, f, "accumulationVersion", uint64_t(17));
      anari::commitParameters(s.d, f);
    }
  // Both Frames now reference the exact same Volume/field/Material graph.
  anari::setParameter(s.d, s.frames[1], "world", s.worlds[0]);
  anari::commitParameters(s.d, s.frames[1]);
  s.extent(0, Box{V3{-1, -1, 2}, V3{1, 1, 4}});
  anari::setParameter(s.d, s.volumes[0], "value", s.fields[0]);
  anari::setParameter(s.d, s.volumes[0], "fieldName", "density");
  s.majorant(0, 4.f);
  s.fill(0, 1.f);
  const std::array<const char *, 5> sources{{
      R"mdl(mdl 1.7; import ::state::*;
export material main()=material(volume: material_volume(
 absorption_coefficient: color(0.25-state::animation_time())));)mdl",
      R"mdl(mdl 1.7; export material main()=material(volume:
 material_volume(absorption_coefficient: color(0.25)));)mdl",
      R"mdl(mdl 1.7; import ::state::*; import ::df::*;
export material main()=material(volume: material_volume(
 absorption_coefficient: color(0.25), scattering_coefficient: color(0.125),
 scattering: df::anisotropic_vdf(directional_bias: state::animation_time())));)mdl",
      R"mdl(mdl 1.8; import ::state::*;
export material main()=material(volume: material_volume(
 absorption_coefficient: color(0.25),
 emission_intensity: color(state::animation_time())));)mdl",
      R"mdl(mdl 1.7; import ::state::*;
export material main()=material(volume: material_volume(
 absorption_coefficient: color(0.25+state::animation_time())));)mdl"}};
  const char *causes[]{
      "coefficient", "field value", "directional bias", "emission", "majorant"};
  for (int k = 0; k < 5; ++k) {
    s.source(sources[k]);
    s.majorant(0, k == 4 ? .5f : 4.f);
    s.fill(0, 1.f);
    for (int i : {0, 1}) {
      anari::setParameter(s.d, s.frames[i], "mdl.animationTime", 0.f);
      anari::commitParameters(s.d, s.frames[i]);
      s.outcomes[i] = {};
      s.render(i);
      s.render(i);
      s.render(i);
      require(s.count(s.frames[i]) > 0, "valid accumulation precedes failure");
    }
    for (const char *channel : {"channel.colorCUDA",
             "channel.depthCUDA",
             "channel.normalCUDA",
             "channel.albedoCUDA",
             "channel.primitiveIdCUDA",
             "channel.objectIdCUDA",
             "channel.instanceIdCUDA"}) {
      unsigned w = 0, h = 0;
      ANARIDataType type = ANARI_UNKNOWN;
      require(anariMapFrame(s.d, s.frames[0], channel, &w, &h, &type) && w == 8
              && h == 8 && type != ANARI_UNKNOWN,
          "enabled CUDA channel is published before failure");
      anariUnmapFrame(s.d, s.frames[0], channel);
    }
    const auto otherImage = s.image(s.frames[1]);
    const int otherCount = s.count(s.frames[1]);
    if (k == 1)
      s.fill(0, std::numeric_limits<float>::quiet_NaN());
    else {
      anari::setParameter(s.d, s.frames[0], "mdl.animationTime", 1.f);
      anari::commitParameters(s.d, s.frames[0]);
    }
    s.outcomes[0] = {};
    s.outcomes[0].inspectFailure = true;
    s.render(0);
    require(s.outcomes[0].errors == 1, "one Frame-sourced failure");
    require(s.outcomes[0].cause.find(causes[k]) != std::string::npos,
        "real MDL failure cause");
    require(s.outcomes[0].callbackError.empty(),
        "reentrant failed access from both callbacks");
    checkFailedMdlFrame(s.d, s.frames[0]);
    require(s.count(s.frames[1]) == otherCount
            && s.image(s.frames[1]) == otherImage,
        "shared failure must not mutate another published Frame");
    // Correct underlying data without changing accumulationVersion. In the
    // field case even Frame time is unchanged, so only failure forces reset.
    s.outcomes[0] = {};
    if (k == 1)
      s.fill(0, 1.f);
    else {
      anari::setParameter(s.d, s.frames[0], "mdl.animationTime", 0.f);
      anari::commitParameters(s.d, s.frames[0]);
    }
    s.render(0);
    require(s.outcomes[0].errors == 0
            && s.count(s.frames[0]) == (s.quality ? 2048 : 1),
        "failure forces fresh accumulation even in manual mode");
    // Bias fixture has scattering, for which white-environment transmission
    // alone is not a Beer-Lambert radiance reference. Compare recovered and
    // fresh Frames with exactly the same public configuration/sample budget.
    const auto recovered = s.image(s.frames[0]);
    auto fresh = s.own(anari::newObject<anari::Frame>(s.d));
    s.setupFrame(fresh, s.worlds[0]);
    if (manual) {
      anari::setParameter(s.d, fresh, "accumulationVersion", uint64_t(17));
      anari::commitParameters(s.d, fresh);
    }
    anari::render(s.d, fresh);
    require(anariFrameReady(s.d, fresh, ANARI_WAIT),
        "fresh recovery reference ready");
    const auto reference = s.image(fresh);
    require(recovered == reference,
        "recovery matches fresh Frame sample-for-sample");
    require(s.count(s.frames[1]) == otherCount
            && s.image(s.frames[1]) == otherImage,
        "recovery preserves other Frame publication");
    if (k == 1) {
      s.render(1);
      require(s.count(s.frames[1])
              == (manual ? otherCount + (s.quality ? 2048 : 1)
                         : (s.quality ? 2048 : 1)),
          "later dependency invalidation follows ordinary automatic/manual rules");
    }
    printf("%s %s failure/recovery and callback isolation: ok\n",
        manual ? "manual" : "automatic",
        causes[k]);
  }
}
void domainAndTransformBounds(Fixture &s)
{
  for (int i : {0, 1}) {
    anariUnsetParameter(s.d, s.volumes[i], "value");
    s.extent(i, Box{V3{-1, -1, 2}, V3{1, 1, 4}});
    s.majorant(i, .25f);
  }
  s.source(R"mdl(mdl 1.7; import ::state::*;
export material main()=material(volume:material_volume(
 absorption_coefficient:color(0.01*state::position().z*state::position().z)));)mdl");
  s.check(0, "field-free automatic step", float((64. - 8.) / 300.), true);
  s.extent(0, Box{V3{-1, -1, 2}, V3{1, 1, 6}});
  s.outcomes[0] = {};
  s.outcomes[0].inspectFailure = true;
  s.render(0);
  require(s.outcomes[0].errors == 1
          && s.outcomes[0].cause.find("majorant") != std::string::npos,
      "extent expansion requires a larger bound");
  checkFailedMdlFrame(s.d, s.frames[0]);
  s.majorant(0, .5f);
  s.outcomes[0] = {};
  s.check(0,
      "expanded extent and recalculated step after bound correction",
      float((216. - 8.) / 300.),
      true);
  s.source(R"mdl(mdl 1.7; import ::state::*;
export material main()=material(volume:material_volume(absorption_coefficient:
 color(0.125*state::transform_vector(state::coordinate_object,
 state::coordinate_world,float3(0,0,1)).z)));)mdl");
  s.extent(0, Box{V3{-1, -1, 2}, V3{1, 1, 4}});
  s.majorant(0, .125f);
  s.check(0, "transform-dependent initial coefficient", .25f, true);
  const std::array<float, 16> scaled{
      1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1};
  anariSetParameter(
      s.d, s.instances[0], "transform", ANARI_FLOAT32_MAT4, scaled.data());
  anari::commitParameters(s.d, s.instances[0]);
  s.outcomes[0] = {};
  s.outcomes[0].inspectFailure = true;
  s.render(0);
  require(s.outcomes[0].errors == 1
          && s.outcomes[0].cause.find("majorant") != std::string::npos,
      "transform-dependent coefficient requires a larger bound");
  checkFailedMdlFrame(s.d, s.frames[0]);
  s.majorant(0, .25f);
  s.outcomes[0] = {};
  s.check(0, "scaled output and world-distance attenuation", 1.f, true);
}
} // namespace
int main()
{
  try {
    for (const char *renderer : {"interactive", "quality"}) {
      Fixture s(renderer);
      sharedArguments(s);
      fieldAndSourceUpdates(s);
      manualEdits(s);
      failAndRecover(s, false);
      failAndRecover(s, true);
      // Lifetime releases are terminal; use an independently initialized graph.
      Fixture retained(renderer);
      resourceAndLifetime(retained);
      Fixture domains(renderer);
      domainAndTransformBounds(domains);
    }
  } catch (const std::exception &e) {
    fprintf(stderr, "TestMdlVolumeUpdates: %s\n", e.what());
    return 1;
  }
  return 0;
}
