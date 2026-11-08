// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <anari/anari_cpp/ext/std.h>
#include <anari/ext/visrtx/makeVisRTXDevice.h>
#include <anari/anari_cpp.hpp>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include "MdlVolumeTestChecks.h"

namespace {
using V3 = std::array<float, 3>;
using V4 = std::array<float, 4>;
using Box = std::array<V3, 2>;
void require(bool ok, const char *why)
{
  if (!ok)
    throw std::runtime_error(why);
}
struct Scene
{
  unsigned errors{};
  std::string cause;
  static void status(const void *user,
      ANARIDevice,
      ANARIObject source,
      ANARIDataType,
      ANARIStatusSeverity severity,
      ANARIStatusCode,
      const char *message)
  {
    auto &s = *const_cast<Scene *>(static_cast<const Scene *>(user));
    if (severity <= ANARI_SEVERITY_ERROR) {
      fprintf(stderr, "%s\n", message);
      if (source == s.frame) {
        ++s.errors;
        s.cause = message;
      }
    }
  }
  anari::Device d = makeVisRTXDevice(status, this);
  anari::Material material = anari::newObject<anari::Material>(d, "mdl");
  anari::Volume volume = anari::newObject<anari::Volume>(d, "mdl");
  anari::SpatialField field =
      anari::newObject<anari::SpatialField>(d, "structuredRegular");
  anari::Group group = anari::newObject<anari::Group>(d);
  anari::Instance instance = anari::newObject<anari::Instance>(d, "transform");
  anari::World world = anari::newObject<anari::World>(d);
  anari::Camera camera = anari::newObject<anari::Camera>(d, "orthographic");
  anari::Renderer renderer =
      anari::newObject<anari::Renderer>(d, "interactive");
  anari::Frame frame = anari::newObject<anari::Frame>(d);
  Scene()
  {
    source("color(scene::data_lookup_float(\"density\", 0.125))");
    fieldData(2.f);
    anari::setParameter(d, volume, "material", material);
    anari::setParameter(d, volume, "value", field);
    anari::setParameter(d, volume, "fieldName", "density");
    anari::setParameter(d, volume, "extinctionMajorant", 4.f);
    anari::commitParameters(d, volume);
    anari::setParameterArray1D(d, group, "volume", &volume, 1);
    anari::commitParameters(d, group);
    anari::setParameter(d, instance, "group", group);
    anari::commitParameters(d, instance);
    anari::setParameterArray1D(d, world, "instance", &instance, 1);
    anari::commitParameters(d, world);
    anari::setParameter(d, camera, "position", V3{0, 0, 0});
    anari::setParameter(d, camera, "direction", V3{0, 0, 1});
    anari::setParameter(d, camera, "up", V3{0, 1, 0});
    anari::setParameter(d, camera, "height", 0.25f);
    anari::commitParameters(d, camera);
    anari::setParameter(d, renderer, "denoise", false);
    anari::setParameter(d, renderer, "fireflyFilterMode", "none");
    anari::setParameter(d, renderer, "ambientRadiance", 0.f);
    anari::setParameter(d, renderer, "background", V4{1, 1, 1, 1});
    anari::setParameter(d, renderer, "pixelSamples", 1);
    anari::setParameter(d, renderer, "volumeSamplingRate", 1.f);
    anari::commitParameters(d, renderer);
    anari::setParameter(d, frame, "size", std::array<unsigned, 2>{8, 8});
    anari::setParameter(d, frame, "channel.color", ANARI_FLOAT32_VEC4);
    anari::setParameter(d, frame, "world", world);
    anari::setParameter(d, frame, "camera", camera);
    anari::setParameter(d, frame, "renderer", renderer);
    anari::commitParameters(d, frame);
  }
  ~Scene()
  {
    for (auto o : {anari::Object(frame),
             anari::Object(renderer),
             anari::Object(camera),
             anari::Object(world),
             anari::Object(instance),
             anari::Object(group),
             anari::Object(volume),
             anari::Object(field),
             anari::Object(material)})
      anari::release(d, o);
    anari::release(d, d);
  }
  void source(const std::string &expression)
  {
    const std::string code =
        "mdl 1.8; import ::scene::*; import ::state::*; "
        "export material main(uniform bool choose = true) = material(volume: material_volume("
        "absorption_coefficient: "
        + expression + "));";
    anari::setParameter(d, material, "sourceType", "code");
    anari::setParameter(d, material, "source", code.c_str());
    anari::commitParameters(d, material);
  }
  void fieldData(float value)
  {
    std::vector<float> values(125, value);
    anari::setParameterArray3D(d, field, "data", values.data(), 5, 5, 5);
    anari::setParameter(d, field, "origin", V3{-1, -1, 2});
    anari::setParameter(d, field, "spacing", V3{0.5, 0.5, 0.5});
    anari::commitParameters(d, field);
  }
  void extent(float lo, float hi)
  {
    Box box{V3{-1, -1, lo}, V3{1, 1, hi}};
    anariSetParameter(d, volume, "extent", ANARI_FLOAT32_BOX3, box.data());
    anari::commitParameters(d, volume);
  }
  void render()
  {
    anari::render(d, frame);
    require(anariFrameReady(d, frame, ANARI_WAIT), "WAIT must complete");
  }
  void fail(const char *message)
  {
    const auto before = errors;
    render();
    require(errors > before, "expected Frame error");
    require(
        cause.find(message) != std::string::npos, "wrong Frame error category");
    checkFailedMdlFrame(d, frame);
    printf("expected failure: %s\n", message);
  }
  void bind(bool present)
  {
    if (present)
      anari::setParameter(d, volume, "value", field);
    else
      anari::unsetParameter(d, volume, "value");
    anari::commitParameters(d, volume);
  }
  void check(const char *label, double opticalDepth, double tolerance = .002)
  {
    const auto before = errors;
    render();
    require(errors == before, "unexpected Frame error");
    unsigned w{}, h{};
    ANARIDataType type{};
    auto p = static_cast<const V4 *>(
        anariMapFrame(d, frame, "channel.color", &w, &h, &type));
    require(p && w == 8 && h == 8 && type == ANARI_FLOAT32_VEC4, "valid map");
    for (unsigned i = 0; i < 64; ++i) {
      for (unsigned c = 0; c < 3; ++c) {
        if (std::abs(p[i][c] - std::exp(-opticalDepth)) > tolerance) {
          fprintf(stderr,
              "%s: got %g expected %g\n",
              label,
              p[i][c],
              std::exp(-opticalDepth));
          throw std::runtime_error("Beer-Lambert mismatch");
        }
      }
      require(p[i][3] == 1.f, "opaque background alpha");
    }
    anari::unmap(d, frame, "channel.color");
    printf("%s: ok\n", label);
  }
};

void lookups(Scene &s)
{
  s.fieldData(.5f);
  anari::unsetParameter(s.d, s.volume, "stepSize");
  anari::commitParameters(s.d, s.volume);
  s.extent(2.f, 4.f);
  s.source("color(scene::data_lookup_float(\"density\", 3.0))");
  s.check("finite raw sample, not authored lookup default", 1.);
  for (const char *expression :
      {"color(scene::data_lookup_float(\"wrong\", 0.25))",
          "color(scene::data_lookup_uniform_float(\"density\", 0.25))",
          "color(scene::data_lookup_float2(\"density\", float2(0.25)).x)",
          "color(scene::data_lookup_float3(\"density\", float3(0.25)))",
          "color(scene::data_lookup_float4(\"density\", float4(0.25)).x)",
          "scene::data_lookup_color(\"density\", color(0.25))",
          "color(float(scene::data_lookup_int(\"density\", 1)))",
          "color(float(scene::data_lookup_int2(\"density\", int2(1)).x))",
          "color(float(scene::data_lookup_int3(\"density\", int3(1)).x))",
          "color(float(scene::data_lookup_int4(\"density\", int4(1)).x))",
          "color(scene::data_lookup_float4x4(\"density\", float4x4(1.0))[0][0])"}) {
    s.source(expression);
    s.fail("scene-data lookup");
  }
  s.source(
      "color(scene::data_isvalid(\"wrong\") ? "
      "scene::data_lookup_float(\"wrong\", 1.0) : 0.25)");
  s.check("unexecuted lookup of unbound name is not an error", .5);
  s.source(
      "color(scene::data_lookup_float(choose ? \"density\" : \"wrong\", 0.25))");
  s.check("executed dynamic bound name", 1.);
  anari::setParameter(s.d, s.material, "choose", false);
  anari::commitParameters(s.d, s.material);
  s.fail("scene-data lookup");
  anari::unsetParameter(s.d, s.material, "choose");
  anari::commitParameters(s.d, s.material);
  s.check("dynamic name argument recovery", 1.);
  // Material sceneData.* names remain surface-only, in both presence queries
  // and lookups. A Volume has no second generic binding parser.
  anari::setParameter(s.d, s.material, "sceneData.density", "attribute0");
  anari::commitParameters(s.d, s.material);
  s.bind(false);
  s.source("color(scene::data_isvalid(\"density\") ? 1.0 : 0.25)");
  s.check("surface binding cannot satisfy volume isvalid", .5);
  s.source("color(scene::data_lookup_float(\"density\", 0.25))");
  s.fail("scene-data lookup");
  s.bind(true);
  s.check("volume binding is independent of surface binding", 1.);
  s.fieldData(0.f);
  s.check("authored zero is data", 0.);
  s.fieldData(std::numeric_limits<float>::infinity());
  s.fail("nonfinite field");
  s.source("color(0.25)");
  s.fail("nonfinite field");
  s.fieldData(.5f);
  s.check("ignored finite field recovers", .5);
}

void sharing(Scene &s)
{
  s.fieldData(.5f);
  s.source("color(scene::data_lookup_float(\"density\", 0.125))");
  s.extent(2.f, 4.f);
  auto original = s.volume;
  auto originalField = s.field;
  auto second = anari::newObject<anari::Volume>(s.d, "mdl");
  auto secondField =
      anari::newObject<anari::SpatialField>(s.d, "structuredRegular");
  s.field = secondField;
  s.fieldData(1.f);
  s.field = originalField;
  anari::setParameter(s.d, second, "material", s.material);
  anari::setParameter(s.d, second, "value", secondField);
  anari::setParameter(s.d, second, "fieldName", "density");
  anari::setParameter(s.d, second, "extinctionMajorant", 4.f);
  auto select = [&](anari::Volume volume) {
    s.volume = volume;
    anari::setParameterArray1D(s.d, s.group, "volume", &volume, 1);
    anari::commitParameters(s.d, s.group);
  };
  // Keep both Volumes alive while selecting one transport-active domain at a
  // time: multi-volume overlap/traversal policy is a separate ticket.
  for (bool reverse : {false, true}) {
    anari::commitParameters(s.d, reverse ? original : second);
    anari::commitParameters(s.d, reverse ? second : original);
    select(second);
    s.check("shared Material, second field", 2.);
    select(original);
    s.check("shared Material, first field", 1.);
  }
  // Releasing the application's field reference must not break its Volume.
  anari::release(s.d, secondField);
  select(second);
  s.check("Volume retains field after application release", 2.);
  select(original);
  s.fieldData(2.f);
  s.check("shared Material field edit stays Volume-local", 4.);
  select(second);
  s.check("other field unaffected by first field edit", 2.);
  anari::setParameter(s.d, second, "value", originalField);
  anari::commitParameters(s.d, second);
  s.extent(3.f, 5.f);
  s.check("shared field, expanded second extent", 2.125);
  select(original);
  s.check("second extent does not mutate shared field bounds", 4.);
  s.extent(2.f, 3.f);
  s.check("shared field, cropped first extent", 2.);
  select(second);
  s.check("first crop does not mutate second domain", 2.125);
  // Source edits must refresh the volume-bundle string mapping independently
  // of surface sceneData and without leaking the second Volume's binding.
  s.source("color(scene::data_lookup_float(\"fresh\", 0.125))");
  anari::setParameter(s.d, second, "fieldName", "fresh");
  anari::commitParameters(s.d, second);
  s.check("new source and second Volume name", 2.125);
  select(original);
  s.fail("scene-data lookup");
  anari::setParameter(s.d, original, "fieldName", "fresh");
  anari::commitParameters(s.d, original);
  s.check("restore first Volume name after shared source edit", 2.);
  anari::release(s.d, second);
  anari::setParameter(s.d, original, "fieldName", "density");
  anari::commitParameters(s.d, original);
}

void varyingFieldAndUpdates(Scene &s)
{
  std::vector<float> values(125);
  for (unsigned z = 0; z < 5; ++z)
    for (unsigned i = 0; i < 25; ++i)
      values[z * 25 + i] = .5f * z;
  anari::setParameterArray3D(s.d, s.field, "data", values.data(), 5, 5, 5);
  anari::commitParameters(s.d, s.field);
  s.source("color(scene::data_lookup_float(\"density\", 0.125))");
  anari::unsetParameter(s.d, s.volume, "extent");
  anari::commitParameters(s.d, s.volume);
  s.check("linearly interpolated raw scalar", 2.);
  s.extent(2.f, 3.f);
  s.check("crop keeps original field coordinates", .5);
  s.extent(2.f, 6.f);
  s.check("expansion does not stretch varying field", 2.25);
  anari::setParameter(s.d, s.field, "origin", V3{-1, -1, 4});
  anari::commitParameters(s.d, s.field);
  s.check("field move changes support, not explicit extent", 2.25);
  s.extent(2.f, 4.f);
  s.check("explicit extent survives disjoint field-domain edit", .25);
  anari::unsetParameter(s.d, s.volume, "extent");
  anari::commitParameters(s.d, s.volume);
  s.check("unset extent follows moved field bounds", 2.);
  anari::setParameter(s.d, s.field, "spacing", V3{.5f, .5f, .25f});
  anari::commitParameters(s.d, s.field);
  s.check("field domain shrink updates traversal without Volume commit", 1.);
  anari::setParameter(s.d, s.field, "spacing", V3{.5f, .5f, 1.f});
  anari::commitParameters(s.d, s.field);
  s.check("field domain expansion updates traversal without Volume commit", 4.);
  s.fieldData(.5f);
}

void samplingScales(Scene &s)
{
  s.fieldData(.5f);
  s.source("color((state::position().z - 3.0) * (state::position().z - 3.0))");
  s.extent(2.f, 4.f);
  // Independent midpoint sums for the quadratic: integral error is L*h^2/12.
  // The .5-spaced field suggests h=.25, even with an explicit extent.
  s.check("field suggested step wins over extent default", .65625, 2e-6);
  anari::setParameter(s.d, s.volume, "stepSize", .5f);
  anari::commitParameters(s.d, s.volume);
  s.check("explicit step wins over suggested field scale", .625, 2e-6);
  anari::unsetParameter(s.d, s.volume, "stepSize");
  anari::commitParameters(s.d, s.volume);
  s.extent(2.875f, 3.125f);
  s.check("tiny crop intentionally uses one clipped field step", 0., 2e-6);
  s.extent(1.f, 5.f);
  s.check("expansion retains field step", 5.3125, 2e-6);
  s.bind(false);
  s.extent(2.f, 4.f);
  s.check("field-free default recalculates after removing value",
      .6666259765625,
      2e-6);
  s.extent(2.875f, 3.125f);
  s.check("extent-only step recalculates after domain crop",
      .0013020038604736328,
      2e-6);
  s.bind(true);
  s.check("restoring value restores field scale", 0., 2e-6);
  s.source("color(0.25)");
  s.extent(2.f, 4.f);
}

void metadata(Scene &s)
{
  s.extent(2.f, 4.f);
  s.source("color(0.25)");
  for (bool present : {false, true}) {
    s.bind(present);
    anari::setParameter(s.d, s.volume, "fieldName", 7.f);
    anari::commitParameters(s.d, s.volume);
    s.fail("fieldName");
    anari::setParameter(s.d, s.volume, "fieldName", "");
    anari::commitParameters(s.d, s.volume);
    if (present)
      s.fail("fieldName");
    else
      s.check("empty retained string is metadata only", .5);
    anari::unsetParameter(s.d, s.volume, "fieldName");
    anari::commitParameters(s.d, s.volume);
    if (present)
      s.fail("fieldName");
    else
      s.check("no field needs no name", .5);
    anari::setParameter(s.d, s.volume, "fieldName", "density");
    anari::commitParameters(s.d, s.volume);
    s.check("restore metadata", .5);
  }
  // Invalid explicit extent never falls back to a valid field.
  for (Box invalid :
      {Box{V3{-1, -1, 4}, V3{1, 1, 2}},
          Box{V3{-1, -1, 2}, V3{1, 1, 2}},
          Box{V3{-1, -1, 2}, V3{1, 1, std::numeric_limits<float>::quiet_NaN()}},
          Box{V3{-std::numeric_limits<float>::max(), -1, 2},
              V3{std::numeric_limits<float>::max(), 1, 4}}}) {
    anariSetParameter(
        s.d, s.volume, "extent", ANARI_FLOAT32_BOX3, invalid.data());
    anari::commitParameters(s.d, s.volume);
    s.fail("extent");
  }
  anari::setParameter(s.d, s.volume, "extent", V3{1, 1, 1});
  anari::commitParameters(s.d, s.volume);
  s.fail("extent");
  s.extent(2.f, 4.f);
  anari::setParameter(s.d, s.volume, "value", 2.f);
  anari::commitParameters(s.d, s.volume);
  s.fail("value");
  auto invalid =
      anari::newObject<anari::SpatialField>(s.d, "structuredRegular");
  anari::commitParameters(s.d, invalid);
  anari::setParameter(s.d, s.volume, "value", invalid);
  anari::commitParameters(s.d, s.volume);
  s.fail("field");
  s.bind(true);
  anari::release(s.d, invalid);
  s.check("invalid field reference recovery", .5);
}
} // namespace

int main()
{
  try {
    Scene s;
    s.check("field bounds and raw scalar", 4.);
    s.extent(2.5f, 3.5f);
    s.check("smaller extent clips, does not stretch field", 2.);
    s.extent(1.f, 5.f);
    s.check("expanded extent uses authored lookup default", 4.25);
    s.extent(3.f, 5.f);
    s.check("partial overlap", 2.125);
    s.extent(5.f, 7.f);
    s.check("disjoint extent", .25);
    s.source("color(scene::data_isvalid(\"density\") ? 0.25 : 1.0)");
    s.check("isvalid is uniform presence outside support", .5);
    s.source(
        "color(scene::data_lookup_float(\"density\", state::position().z * 0.125))");
    s.check("procedural authored default outside support", 1.5);
    s.source("color(0.25)");
    s.check("ignored field does not truncate procedural medium", .5);
    s.fieldData(std::numeric_limits<float>::quiet_NaN());
    s.check("unsupported positions never probe ignored NaN field", .5);
    s.extent(2.f, 4.f);
    s.fail("nonfinite field");
    anari::setParameter(s.d, s.volume, "extinctionMajorant", 0.f);
    anari::commitParameters(s.d, s.volume);
    s.check("zero majorant skips unevaluated NaN", 0.);
    anari::setParameter(s.d, s.volume, "extinctionMajorant", 4.f);
    anari::commitParameters(s.d, s.volume);
    s.fieldData(2.f);
    s.check("field edit recovers without volume recommit", .5);
    s.bind(false);
    s.source("color(scene::data_isvalid(\"density\") ? 1.0 : 0.25)");
    s.check("retained fieldName without value creates no binding", .5);
    s.source("color(scene::data_lookup_float(\"density\", 0.125))");
    s.fail("scene-data lookup");
    s.bind(true);
    s.check("restored field binding", 4.);
    anari::unsetParameter(s.d, s.volume, "extent");
    anari::commitParameters(s.d, s.volume);
    s.check("unset extent restores field-derived domain", 4.);
    s.bind(false);
    s.fail("extent or value");
    s.bind(true);
    s.check("restore field-only domain", 4.);
    // A valid extent and explicit step cannot conceal an invalid supplied
    // field.
    s.extent(2.f, 4.f);
    anari::setParameter(s.d, s.field, "spacing", V3{.5f, .5f, -.5f});
    anari::commitParameters(s.d, s.field);
    anari::setParameter(s.d, s.volume, "stepSize", .1f);
    anari::commitParameters(s.d, s.volume);
    s.fail("field");
    lookups(s);
    metadata(s);
    sharing(s);
    varyingFieldAndUpdates(s);
    samplingScales(s);
  } catch (const std::exception &e) {
    fprintf(stderr, "%s\n", e.what());
    return 1;
  }
  return 0;
}
