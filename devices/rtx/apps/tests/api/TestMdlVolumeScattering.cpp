// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#include <anari/anari_cpp/ext/std.h>
#include <anari/ext/visrtx/makeVisRTXDevice.h>
#include <algorithm>
#include <anari/anari_cpp.hpp>
#include <array>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include "MdlScatteringReference.h"

namespace {
using V3 = std::array<float, 3>;
using V4 = std::array<float, 4>;
using Box = std::array<V3, 2>;
void require(bool ok, const char *message)
{
  if (!ok)
    throw std::runtime_error(message);
}
struct Scene
{
  unsigned errors{};
  std::vector<anari::Object> owned;
  std::vector<anari::Light> lights;
  std::vector<anari::Surface> surfaces;
  static void status(const void *user,
      ANARIDevice,
      ANARIObject,
      ANARIDataType,
      ANARIStatusSeverity severity,
      ANARIStatusCode,
      const char *message)
  {
    if (severity <= ANARI_SEVERITY_ERROR) {
      ++static_cast<Scene *>(const_cast<void *>(user))->errors;
      fprintf(stderr, "%s\n", message);
    }
  }
  anari::Device d = makeVisRTXDevice(status, this);
  anari::Material material = anari::newObject<anari::Material>(d, "mdl");
  anari::Volume volume = anari::newObject<anari::Volume>(d, "mdl");
  anari::Group group = anari::newObject<anari::Group>(d);
  anari::Instance instance = anari::newObject<anari::Instance>(d, "transform");
  anari::World world = anari::newObject<anari::World>(d);
  anari::Camera camera = anari::newObject<anari::Camera>(d, "orthographic");
  anari::Renderer renderer = anari::newObject<anari::Renderer>(d, "quality");
  anari::Frame frame = anari::newObject<anari::Frame>(d);
  anari::Light env = anari::newObject<anari::Light>(d, "hdri");
  Scene()
  {
    anari::setParameter(d, material, "sourceType", "code");
    anari::setParameter(d, material, "source", R"mdl(mdl 1.7;
export material main(uniform color absorption=color(0),
 uniform color scattering=color(0,.5,1.5)) = material(volume: material_volume(
 absorption_coefficient: absorption, scattering_coefficient: scattering));
)mdl");
    anari::commitParameters(d, material);
    anari::setParameter(d, volume, "material", material);
    Box box{V3{-1, -1, 2}, V3{1, 1, 4}};
    anariSetParameter(d, volume, "extent", ANARI_FLOAT32_BOX3, box.data());
    anari::setParameter(d, volume, "extinctionMajorant", 1.5f);
    anari::commitParameters(d, volume);
    anari::setParameterArray1D(d, group, "volume", &volume, 1);
    anari::commitParameters(d, group);
    anari::setParameter(d, instance, "group", group);
    anari::commitParameters(d, instance);
    anari::setParameterArray1D(d, world, "instance", &instance, 1);
    const V3 white{1, 1, 1};
    anari::setParameterArray2D(d, env, "radiance", &white, 1, 1);
    anari::commitParameters(d, env);
    lights.push_back(env);
    anari::setParameterArray1D(d, world, "light", lights.data(), lights.size());
    anari::commitParameters(d, world);
    anari::setParameter(d, camera, "position", V3{0, 0, 0});
    anari::setParameter(d, camera, "direction", V3{0, 0, 1});
    anari::setParameter(d, camera, "up", V3{0, 1, 0});
    anari::setParameter(d, camera, "height", .001f);
    anari::commitParameters(d, camera);
    anari::setParameter(d, renderer, "denoise", false);
    anari::setParameter(d, renderer, "fireflyFilterMode", "none");
    anari::setParameter(d, renderer, "ambientRadiance", 0.f);
    anari::setParameter(d, renderer, "pixelSamples", 2048);
    anari::setParameter(d, renderer, "sampleLimit", 0);
    depth(32);
    anari::setParameter(d, frame, "size", std::array<unsigned, 2>{8, 8});
    anari::setParameter(d, frame, "channel.color", ANARI_FLOAT32_VEC4);
    anari::setParameter(d, frame, "world", world);
    anari::setParameter(d, frame, "camera", camera);
    anari::setParameter(d, frame, "renderer", renderer);
    anari::commitParameters(d, frame);
  }
  ~Scene()
  {
    for (auto o : owned)
      anari::release(d, o);
    for (auto o : {anari::Object(frame),
             anari::Object(renderer),
             anari::Object(camera),
             anari::Object(world),
             anari::Object(instance),
             anari::Object(group),
             anari::Object(volume),
             anari::Object(material),
             anari::Object(env)})
      anari::release(d, o);
    anari::release(d, d);
  }
  void coefficients(V3 a, V3 s, float majorant)
  {
    anari::setParameter(d, material, "absorption", a);
    anari::setParameter(d, material, "scattering", s);
    anari::commitParameters(d, material);
    anari::setParameter(d, volume, "extinctionMajorant", majorant);
    anari::commitParameters(d, volume);
  }
  void environment(V3 radiance)
  {
    anari::setParameterArray2D(d, env, "radiance", &radiance, 1, 1);
    anari::commitParameters(d, env);
  }
  void plane(
      float z, float half, V3 radiance, bool covered, float opacity = 1.f)
  {
    {
      auto geom =
          anari::newObject<anari::Geometry>(d, covered ? "triangle" : "quad");
      auto mat = anari::newObject<anari::Material>(d, "physicallyBased");
      auto surface = anari::newObject<anari::Surface>(d);
      owned.insert(owned.end(), {geom, mat, surface});
      surfaces.push_back(surface);
      const std::array<V3, 4> vertices{V3{-half, -half, z},
          V3{-half, half, z},
          V3{half, half, z},
          V3{half, -half, z}};
      anari::setParameterArray1D(
          d, geom, "vertex.position", vertices.data(), 4);
      if (covered) {
        const std::array<std::array<unsigned, 3>, 2> indices{
            {{0, 1, 2}, {0, 2, 3}}};
        anari::setParameterArray1D(
            d, geom, "primitive.index", indices.data(), 2);
      }
      anari::commitParameters(d, geom);
      anari::setParameter(d, mat, "baseColor", V3{0, 0, 0});
      anari::setParameter(d, mat, "specular", 0.f);
      anari::setParameter(d, mat, "emissive", radiance);
      anari::setParameter(d, mat, "opacity", opacity);
      anari::setParameter(d, mat, "alphaMode", "blend");
      anari::commitParameters(d, mat);
      anari::setParameter(d, surface, "geometry", geom);
      anari::setParameter(d, surface, "material", mat);
      anari::commitParameters(d, surface);
      anari::setParameterArray1D(
          d, world, "surface", surfaces.data(), surfaces.size());
    }
    anari::commitParameters(d, world);
  }
  void analyticPlane(float z, float half, V3 radiance)
  {
    auto light = anari::newObject<anari::Light>(d, "quad");
    owned.push_back(light);
    lights.push_back(light);
    anari::setParameter(d, light, "position", V3{-half, -half, z});
    anari::setParameter(d, light, "edge1", V3{2 * half, 0, 0});
    anari::setParameter(d, light, "edge2", V3{0, 2 * half, 0});
    anari::setParameter(d, light, "side", "both");
    anari::setParameter(d, light, "color", radiance);
    anari::setParameter(d, light, "intensity", 1.f);
    anari::commitParameters(d, light);
    anari::setParameterArray1D(d, world, "light", lights.data(), lights.size());
    anari::commitParameters(d, world);
  }
  void secondDomain(float extinction, bool scatter = false)
  {
    auto mat = anari::newObject<anari::Material>(d, "mdl");
    auto vol = anari::newObject<anari::Volume>(d, "mdl");
    owned.insert(owned.end(), {mat, vol});
    anari::setParameter(d, mat, "sourceType", "code");
    anari::setParameter(d, mat, "source", R"mdl(mdl 1.7;
export material main(uniform color absorption=color(0),uniform color scattering=color(0))
 = material(volume:material_volume(absorption_coefficient:absorption,
 scattering_coefficient:scattering));)mdl");
    anari::setParameter(d,
        mat,
        scatter ? "scattering" : "absorption",
        V3{extinction, extinction, extinction});
    anari::commitParameters(d, mat);
    anari::setParameter(d, vol, "material", mat);
    Box box{V3{-1, -1, 4.5f}, V3{1, 1, 5.5f}};
    anariSetParameter(d, vol, "extent", ANARI_FLOAT32_BOX3, box.data());
    // Large bound and zero coefficients exercise many nulls in another domain.
    anari::setParameter(
        d, vol, "extinctionMajorant", std::max(8.f, extinction));
    anari::commitParameters(d, vol);
    std::array<anari::Volume, 2> volumes{volume, vol};
    anari::setParameterArray1D(
        d, group, "volume", volumes.data(), volumes.size());
    anari::commitParameters(d, group);
  }
  void depth(int n)
  {
    anari::setParameter(d, renderer, "maxRayDepth", n);
    anari::commitParameters(d, renderer);
  }
  V3 check(const char *name, V3 expected, int batches = 16)
  {
    // White conservative tests use 64 consecutive batches (8388608 camera
    // samples), fixed after the initial tracer's B SE=.00208 was inconclusive.
    // Lower-radiance analog/emitter cases use a predeclared 16 batches
    // (2097152 camera samples); the same uncertainty ceiling still applies.
    // No retries on statistical failures; the initial automatic-reset image
    // may repeat once and is discarded, never counted as another batch.
    constexpr int spp = 2048;
    printf("begin %s (%d batches)\n", name, batches);
    fflush(stdout);
    std::array<double, 3> previous{}, mean{}, m2{};
    bool discarded = false;
    for (int b = 0; b < batches; ++b) {
      anari::render(d, frame);
      require(anariFrameReady(d, frame, ANARI_WAIT), "WAIT did not finish");
      require(errors == 0, "unexpected API error");
      int count = 0;
      require(anariGetProperty(d,
                  frame,
                  "numSamples",
                  ANARI_INT32,
                  &count,
                  sizeof(count),
                  ANARI_WAIT),
          "sample count unavailable");
      if (b == 1 && count == spp && !discarded) {
        discarded = true;
        b = 0;
        previous = {};
        mean = {};
        m2 = {};
      }
      require(count == (b + 1) * spp, "nonconsecutive accumulation batch");
      auto p = anari::map<V4>(d, frame, "channel.color");
      require(p.data && p.width == 8 && p.height == 8, "failed color map");
      std::array<double, 3> cumulative{};
      for (int i = 0; i < 64; ++i)
        for (int c = 0; c < 3; ++c) {
          require(std::isfinite(p.data[i][c]), "nonfinite radiance");
          cumulative[c] += p.data[i][c] / 64.;
        }
      anari::unmap(d, frame, "channel.color");
      for (int c = 0; c < 3; ++c) {
        double batch = cumulative[c] * (b + 1) - previous[c] * b;
        previous[c] = cumulative[c];
        double delta = batch - mean[c];
        mean[c] += delta / (b + 1);
        m2[c] += delta * (batch - mean[c]);
      }
    }
    V3 result{};
    for (int c = 0; c < 3; ++c) {
      double se = std::sqrt(m2[c] / (batches * (batches - 1)));
      double error = std::abs(mean[c] - expected[c]);
      printf("%s c%d: %.8f reference %.8f SE %.8f\n",
          name,
          c,
          mean[c],
          expected[c],
          se);
      require(se <= .0016, "inconclusive: batch uncertainty exceeds ceiling");
      require(error <= .01 && error <= 5 * se + .002,
          "scattering reference mismatch");
      result[c] = float(mean[c]);
    }
    return result;
  }
};
} // namespace
int main(int argc, char **argv)
{
  const std::string focus = argc > 1 ? argv[1] : "all";
  try {
    using scattering_reference::Reference;
    using scattering_reference::zeroPlusOne;
    if (focus == "all" || focus == "conservative") {
      const double coarse = zeroPlusOne(1.5, 64), fine = zeroPlusOne(1.5, 128);
      printf(
          "zero+one: %.8f -> %.8f; separation %.8f\n", coarse, fine, 1. - fine);
      require(std::abs(coarse - fine) < .0005,
          "single-scatter quadrature unconverged");
      require(1. - fine > 5 * .01,
          "fixture cannot distinguish multiple scattering");
      {
        Scene s;
        s.check("conservative depth32", {1, 1, 1}, 64);
        s.depth(64);
        const auto d64 = s.check("conservative depth64", {1, 1, 1}, 64);
        s.depth(128);
        const auto d128 = s.check("conservative depth128", {1, 1, 1}, 64);
        for (int c = 0; c < 3; ++c)
          require(std::abs(d64[c] - d128[c]) < .002,
              "last two depths must converge");
      }
    }
    if (focus == "all" || focus == "analog") {
      Scene s;
      Reference ref;
      ref.media[0].a = {.2f, .3f, 0};
      ref.media[0].s = {.8f, 1.2f, 0};
      ref.environment = {.25f, .4f, .6f};
      s.coefficients(ref.media[0].a, ref.media[0].s, 2.f);
      s.environment(ref.environment);
      s.depth(ref.depth);
      const auto expected = ref.analog();
      s.check("absorbing-scattering analog", expected);
      s.coefficients(ref.media[0].a, ref.media[0].s, 4.f);
      s.check("twice majorant scattering", expected);
      s.coefficients(ref.media[0].a, ref.media[0].s, 8.f);
      s.secondDomain(0.f);
      s.plane(4.25f, 4.f, {0, 0, 0}, false, 0.f);
      s.check(
          "phase through many nulls, vacuum domain and cutout to environment",
          expected);
    }
    if (focus == "all" || focus == "emitters") {
      Scene s;
      Reference ref;
      ref.environment = {.1f, .2f, .3f};
      ref.media[0].a = {.25f, .125f, 0};
      ref.media[0].s = {.75f, 1.375f, 0};
      s.coefficients(ref.media[0].a, ref.media[0].s, 2.f);
      s.environment(ref.environment);
      // Two covered components compete with the environment. The coverage
      // set includes BOTH even though NEE picks only one. A quad (not an
      // area-samplable triangle) is deliberately continuation-only.
      for (auto plane : {scattering_reference::Plane{6, 1, {.7f, .2f, .1f}},
               scattering_reference::Plane{-2, 2, {.15f, .5f, .2f}},
               scattering_reference::Plane{7, 4, {.2f, .3f, .6f}}}) {
        ref.planes.push_back(plane);
        s.plane(
            float(plane.z), float(plane.half), plane.emission, plane.z != 7);
      }
      ref.planes.push_back({4.25, 4, {0, 0, 0}, 0.});
      s.plane(4.25f, 4.f, {0, 0, 0}, false, 0.f);
      s.secondDomain(0.f);
      ref.media.push_back({{-1, -1, 4.5}, {1, 1, 5.5}, {0, 0, 0}, {0, 0, 0}});
      for (int depth : {1, 2, 32}) {
        s.depth(depth);
        ref.depth = depth;
        s.check(depth == 1   ? "terminal phase emitter ownership"
                : depth == 2 ? "two-event emitter ownership"
                             : "multiple-light analog",
            ref.analog());
      }
      s.secondDomain(.5f, true);
      ref.media.back().s = {.5f, .5f, .5f};
      s.depth(4);
      ref.depth = 4;
      s.check("real phase replaces ownership in second domain", ref.analog());
    }
    if (focus == "all" || focus == "terminal") {
      for (bool covered : {false, true}) {
        Scene s;
        Reference ref;
        ref.environment = {0, 0, 0};
        ref.media[0].a = {.25f, .125f, 0};
        ref.media[0].s = {.75f, 1.375f, 0};
        ref.planes.push_back({6, 4, {.6f, .6f, .6f}});
        ref.depth = 1;
        s.environment(ref.environment);
        s.coefficients(ref.media[0].a, ref.media[0].s, 2.f);
        if (covered)
          s.analyticPlane(6, 4, {.6f, .6f, .6f});
        else
          s.plane(6, 4, {.6f, .6f, .6f}, false);
        s.depth(1);
        s.check(covered ? "terminal NEE-covered analytic endpoint"
                        : "terminal continuation-only endpoint",
            ref.analog());
      }
    }
    if (focus == "all" || focus == "delta") {
      Scene s;
      s.environment({0, 0, 0});
      s.coefficients({.25f, .5f, 0}, {.5f, 1.5f, 0}, 2.f);
      s.depth(1);
      auto light = anari::newObject<anari::Light>(s.d, "directional");
      s.owned.push_back(light);
      s.lights.push_back(light);
      anari::setParameter(s.d, light, "direction", V3{0, 0, -1});
      anari::setParameter(s.d, light, "irradiance", 4.f);
      anari::commitParameters(s.d, light);
      anari::setParameterArray1D(
          s.d, s.world, "light", s.lights.data(), s.lights.size());
      anari::commitParameters(s.d, s.world);
      // Camera and light distances sum to two everywhere on the center ray.
      // This is the exact zero-plus-one-scatter directional-light solution.
      V3 expected{
          float(4 * .5 * 2 * std::exp(-.75 * 2) / (4 * 3.141592653589793)),
          float(4 * 1.5 * 2 * std::exp(-2. * 2) / (4 * 3.141592653589793)),
          0};
      s.check("delta emitter is NEE-only at terminal phase", expected);
    }
  } catch (const std::exception &e) {
    fprintf(stderr, "TestMdlVolumeScattering: %s\n", e.what());
    return 1;
  }
}
