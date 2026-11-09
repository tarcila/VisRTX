// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#include "MdlCompositingReference.h"

#include <anari/anari_cpp/ext/std.h>
#include <anari/ext/visrtx/makeVisRTXDevice.h>
#include <anari/anari_cpp.hpp>
#include <array>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using V3 = std::array<float, 3>;
using V4 = std::array<float, 4>;
using Box = std::array<V3, 2>;
void require(bool ok, const char *message)
{
  if (!ok)
    throw std::runtime_error(message);
}
// Public ANARI seam. References below are straight-track Beer--Lambert
// products, not calls to the renderer's evaluators or tracking implementation.
struct Scene
{
  unsigned errors{};
  static void status(const void *user,
      ANARIDevice,
      ANARIObject,
      ANARIDataType,
      ANARIStatusSeverity severity,
      ANARIStatusCode,
      const char *message);
  anari::Device d = makeVisRTXDevice(status, this);
  std::vector<anari::Object> owned;
  std::vector<anari::Volume> volumes;
  std::vector<anari::Surface> surfaces;
  anari::World world = anari::newObject<anari::World>(d);
  anari::Camera camera = anari::newObject<anari::Camera>(d, "orthographic");
  anari::Renderer renderer;
  anari::Frame frame = anari::newObject<anari::Frame>(d);
  bool quality;
  explicit Scene(const char *type);
  ~Scene();
  Scene(const Scene &) = delete;
  Scene &operator=(const Scene &) = delete;
  Scene(Scene &&) = delete;
  Scene &operator=(Scene &&) = delete;
  void background(float alpha);
  void medium(V3 absorption, V3 scattering, float lo, float hi);
  void tf(V3 color, float lo);
  void environment(V3 radiance);
  anari::Material surface(float z,
      float opacity,
      bool transmitting = false,
      V3 emission = {0, 0, 0});
  void check(const char *name, V4 reference, bool alphaOnly = false);
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
    ++static_cast<Scene *>(const_cast<void *>(user))->errors;
    fprintf(stderr, "%s\n", message);
  }
}
Scene::Scene(const char *type)
    : renderer(anari::newObject<anari::Renderer>(d, type)),
      quality(std::string(type) == "quality")
{
  anari::setParameter(d, camera, "position", V3{0, 0, 0});
  anari::setParameter(d, camera, "direction", V3{0, 0, 1});
  anari::setParameter(d, camera, "up", V3{0, 1, 0});
  anari::setParameter(d, camera, "height", .1f);
  anari::commitParameters(d, camera);
  anari::setParameter(d, renderer, "denoise", false);
  anari::setParameter(d, renderer, "fireflyFilterMode", "none");
  anari::setParameter(d, renderer, "ambientRadiance", 0.f);
  anari::setParameter(d, renderer, "sampleLimit", 0);
  anari::setParameter(d, renderer, "pixelSamples", quality ? 2048 : 1);
  background(0.f);
  anari::setParameter(d, frame, "size", std::array<unsigned, 2>{8, 8});
  anari::setParameter(d, frame, "channel.color", ANARI_FLOAT32_VEC4);
  anari::setParameter(d, frame, "world", world);
  anari::setParameter(d, frame, "camera", camera);
  anari::setParameter(d, frame, "renderer", renderer);
  anari::commitParameters(d, frame);
}
Scene::~Scene()
{
  for (auto o : owned)
    anari::release(d, o);
  for (auto o : {anari::Object(frame),
           anari::Object(renderer),
           anari::Object(camera),
           anari::Object(world)})
    anari::release(d, o);
  anari::release(d, d);
}
void Scene::background(float alpha)
{
  anari::setParameter(d, renderer, "background", V4{.2f, .3f, .4f, alpha});
  anari::commitParameters(d, renderer);
}
void Scene::medium(V3 absorption, V3 scattering, float lo, float hi)
{
  auto m = anari::newObject<anari::Material>(d, "mdl");
  auto v = anari::newObject<anari::Volume>(d, "mdl");
  owned.insert(owned.end(), {m, v});
  anari::setParameter(d, m, "sourceType", "code");
  anari::setParameter(d, m, "source", R"mdl(mdl 1.7;
export material main(uniform color a=color(0), uniform color s=color(0)) =
material(volume: material_volume(absorption_coefficient:a,
 scattering_coefficient:s));
)mdl");
  anari::setParameter(d, m, "a", absorption);
  anari::setParameter(d, m, "s", scattering);
  anari::commitParameters(d, m);
  anari::setParameter(d, v, "material", m);
  Box box{V3{-2, -2, lo}, V3{2, 2, hi}};
  anariSetParameter(d, v, "extent", ANARI_FLOAT32_BOX3, box.data());
  anari::setParameter(d, v, "extinctionMajorant", 4.f);
  anari::setParameter(d, v, "stepSize", .03125f);
  anari::commitParameters(d, v);
  volumes.push_back(v);
  anari::setParameterArray1D(
      d, world, "volume", volumes.data(), volumes.size());
  anari::commitParameters(d, world);
}
void Scene::tf(V3 color, float lo)
{
  auto field = anari::newObject<anari::SpatialField>(d, "structuredRegular");
  auto v = anari::newObject<anari::Volume>(d, "transferFunction1D");
  owned.insert(owned.end(), {field, v});
  std::vector<float> data(5 * 5 * 3, .5f);
  anari::setParameterArray3D(d, field, "data", data.data(), 5, 5, 3);
  anari::setParameter(d, field, "origin", V3{-2, -2, lo});
  anari::setParameter(d, field, "spacing", V3{1, 1, 1});
  anari::commitParameters(d, field);
  anari::setParameter(d, v, "value", field);
  anari::setParameter(d, v, "color", color);
  anari::setParameter(d, v, "opacity", .25f);
  anari::setParameter(d, v, "unitDistance", 1.f);
  anari::commitParameters(d, v);
  volumes.push_back(v);
  anari::setParameterArray1D(
      d, world, "volume", volumes.data(), volumes.size());
  anari::commitParameters(d, world);
  anari::setParameter(d, renderer, "volumeSamplingRate", 1.f);
  anari::setParameter(d, renderer, "volumeSamplingRateShadows", 1.f);
  anari::commitParameters(d, renderer);
}
void Scene::environment(V3 radiance)
{
  auto light = anari::newObject<anari::Light>(d, "hdri");
  owned.push_back(light);
  anari::setParameterArray2D(d, light, "radiance", &radiance, 1, 1);
  anari::commitParameters(d, light);
  anari::setParameterArray1D(d, world, "light", &light, 1);
  anari::commitParameters(d, world);
}
anari::Material Scene::surface(
    float z, float opacity, bool transmitting, V3 emission)
{
  auto g = anari::newObject<anari::Geometry>(d, "triangle");
  const bool pbr = transmitting || emission != V3{0, 0, 0};
  auto m =
      anari::newObject<anari::Material>(d, pbr ? "physicallyBased" : "matte");
  auto s = anari::newObject<anari::Surface>(d);
  owned.insert(owned.end(), {g, m, s});
  const V3 vertices[] = {{-4, -4, z}, {0, 4, z}, {4, -4, z}};
  anari::setParameterArray1D(d, g, "vertex.position", vertices, 3);
  anari::commitParameters(d, g);
  anari::setParameter(d,
      m,
      pbr ? "baseColor" : "color",
      transmitting ? V3{1, 1, 1} : V3{0, 0, 0});
  anari::setParameter(d, m, "opacity", opacity);
  if (pbr)
    anari::setParameter(d, m, "emissive", emission);
  if (transmitting) {
    anari::setParameter(d, m, "transmission", 1.f);
    anari::setParameter(d, m, "specular", 0.f);
    anari::setParameter(d, m, "metallic", 0.f);
  }
  anari::commitParameters(d, m);
  anari::setParameter(d, s, "geometry", g);
  anari::setParameter(d, s, "material", m);
  anari::setParameter(d, s, "id", 77u);
  anari::commitParameters(d, s);
  surfaces.push_back(s);
  anari::setParameterArray1D(
      d, world, "surface", surfaces.data(), surfaces.size());
  anari::commitParameters(d, world);
  return m;
}
void Scene::check(const char *name, V4 reference, bool alphaOnly)
{
  const int batches = quality ? 16 : 1;
  std::array<double, 4> previous{}, mean{}, m2{};
  bool discardedReset = false;
  for (int b = 0; b < batches; ++b) {
    anari::render(d, frame);
    anari::wait(d, frame);
    require(!errors, "unexpected ANARI error");
    int count{};
    require(anariGetProperty(d,
                frame,
                "numSamples",
                ANARI_INT32,
                &count,
                sizeof(count),
                ANARI_WAIT),
        "sample count query");
    if (b == 1 && count == 2048 && !discardedReset) {
      discardedReset = true;
      b = 0;
      previous = {};
      mean = {};
      m2 = {};
    }
    require(count == (b + 1) * (quality ? 2048 : 1),
        "consecutive nonoverlapping sample batches");
    auto p = anari::map<V4>(d, frame, "channel.color");
    require(p.data && p.width == 8 && p.height == 8, "color map");
    std::array<double, 4> cumulative{};
    for (int i = 0; i < 64; ++i)
      for (int c = 0; c < 4; ++c) {
        require(std::isfinite(p.data[i][c]), "finite color");
        cumulative[c] += p.data[i][c] / 64.;
      }
    anari::unmap(d, frame, "channel.color");
    for (int c = 0; c < 4; ++c) {
      double batch = cumulative[c] * (b + 1) - previous[c] * b;
      previous[c] = cumulative[c];
      double delta = batch - mean[c];
      mean[c] += delta / (b + 1);
      m2[c] += delta * (batch - mean[c]);
    }
  }
  for (int c = alphaOnly ? 3 : 0; c < 4; ++c) {
    double se = quality ? std::sqrt(m2[c] / (batches * (batches - 1))) : 0.;
    double error = std::abs(mean[c] - reference[c]);
    printf("%s %s c%d mean %.9g ref %.9g SE %.9g\n",
        quality ? "quality" : "interactive",
        name,
        c,
        mean[c],
        reference[c],
        se);
    require(se <= .0016, "inconclusive: batch SE exceeds ceiling");
    require(error <= .01 && error <= 5 * se + .002,
        "independent compositing reference mismatch");
  }
}
void complementary(const char *type)
{
  Scene s(type);
  const float k = std::log(10.f);
  s.medium({k, 0, 0}, {0, 0, 0}, 2, 3);
  s.medium({0, k, 0}, {0, 0, 0}, 4, 5);
  // T1=(.1,1,1), T2=(1,.1,1): mean(product)=.4, NOT .7*.7.
  s.check("complementary transparent", {.02f, .03f, .4f, .6f});
  s.surface(3.5f, .25f);
  s.check("partial between", {.015f, .0225f, .3f, .7f});
  s.surface(2.5f, .5f);
  s.check("partial inside", {.0075f, .01125f, .15f, .85f});
  s.background(1.f);
  s.check("opaque screen", {.0075f, .01125f, .15f, 1.f});
}
void aovRegression(const char *type)
{
  Scene s(type);
  s.medium({0, 0, 0}, {0, 0, 0}, 2, 3);
  auto m = s.surface(4.f, 1.f);
  const V3 tint{.2f, .5f, .8f};
  anari::setParameter(s.d, m, "color", tint);
  anari::commitParameters(s.d, m);
  anari::setParameter(s.d, s.camera, "far", 9.f);
  anari::commitParameters(s.d, s.camera);
  anari::setParameter(s.d, s.frame, "channel.depth", ANARI_FLOAT32);
  anari::setParameter(s.d, s.frame, "channel.objectId", ANARI_UINT32);
  anari::setParameter(s.d, s.frame, "channel.primitiveId", ANARI_UINT32);
  anari::setParameter(s.d, s.frame, "channel.normal", ANARI_FLOAT32_VEC3);
  anari::setParameter(s.d, s.frame, "channel.albedo", ANARI_FLOAT32_VEC3);
  anari::commitParameters(s.d, s.frame);
  s.check("vacuum surface coverage/AOV", {0, 0, 0, 1});

  // Preserve the existing multi-iteration AOV reduction, independently of
  // selected-channel transport. Compare with the same public-API surface-only
  // scene rather than assuming a new normalization rule.
  Scene baseline(type);
  auto baselineMaterial = baseline.surface(4.f, 1.f);
  anari::setParameter(baseline.d, baselineMaterial, "color", tint);
  anari::commitParameters(baseline.d, baselineMaterial);
  anari::setParameter(
      baseline.d, baseline.frame, "channel.albedo", ANARI_FLOAT32_VEC3);
  anari::commitParameters(baseline.d, baseline.frame);
  baseline.check("surface-only AOV reference", {0, 0, 0, 1});
  auto baselineAlbedo =
      anari::map<V3>(baseline.d, baseline.frame, "channel.albedo");
  require(baselineAlbedo.data, "baseline albedo map");
  auto depth = anari::map<float>(s.d, s.frame, "channel.depth");
  auto object = anari::map<unsigned>(s.d, s.frame, "channel.objectId");
  auto primitive = anari::map<unsigned>(s.d, s.frame, "channel.primitiveId");
  auto normal = anari::map<V3>(s.d, s.frame, "channel.normal");
  auto albedo = anari::map<V3>(s.d, s.frame, "channel.albedo");
  require(
      depth.data && object.data && primitive.data && normal.data && albedo.data,
      "AOV maps");
  for (int p = 0; p < 64; ++p) {
    require(std::abs(depth.data[p] - 4.f) < 1e-5 && object.data[p] == 77u
            && primitive.data[p] == 0u,
        "first surface depth and IDs");
    for (int c = 0; c < 3; ++c) {
      require(std::abs(normal.data[p][c] - (c == 2 ? -1.f : 0.f)) < 1e-5,
          "normal is not channel-compensated");
      require(std::abs(albedo.data[p][c] - baselineAlbedo.data[p][c]) < 1e-5,
          "multi-iteration albedo preserves surface-only reduction");
    }
  }
  for (auto channel : {"channel.depth",
           "channel.objectId",
           "channel.primitiveId",
           "channel.normal",
           "channel.albedo"})
    anari::unmap(s.d, s.frame, channel);
  anari::unmap(baseline.d, baseline.frame, "channel.albedo");

  // A single sample also has an analytic RGB albedo reference, with no
  // accumulated-frame normalization involved. The radiance channel's factor
  // three must not affect ANY albedo component, even on one camera sample.
  anari::setParameter(s.d, s.renderer, "pixelSamples", 1);
  anari::commitParameters(s.d, s.renderer);
  anari::render(s.d, s.frame);
  anari::wait(s.d, s.frame);
  require(!s.errors, "single-sample AOV render");
  int samples{};
  require(anariGetProperty(s.d,
              s.frame,
              "numSamples",
              ANARI_INT32,
              &samples,
              sizeof(samples),
              ANARI_WAIT)
          && samples == 1,
      "single-sample AOV reset");
  albedo = anari::map<V3>(s.d, s.frame, "channel.albedo");
  require(albedo.data, "single-sample albedo map");
  for (int p = 0; p < 64; ++p)
    for (int c = 0; c < 3; ++c)
      require(std::abs(albedo.data[p][c] - tint[c]) < 1e-5,
          "single-sample albedo is not channel-compensated");
  anari::unmap(s.d, s.frame, "channel.albedo");
  anari::setParameter(s.d, s.renderer, "pixelSamples", s.quality ? 2048 : 1);
  anari::commitParameters(s.d, s.renderer);

  // Cutting at z=1 excludes all geometry and media. Miss Depth remains the
  // camera Far Limit, not the renderer's shorter traversal interval.
  anari::setParameter(s.d, s.renderer, "cutPlane", V4{0, 0, -1, 1});
  anari::commitParameters(s.d, s.renderer);
  s.check("clipped miss", {.2f, .3f, .4f, 0});
  depth = anari::map<float>(s.d, s.frame, "channel.depth");
  require(depth.data, "Miss depth maps");
  for (int p = 0; p < 64; ++p)
    require(depth.data[p] == 9.f, "Miss Depth equals Far Limit");
  anari::unmap(s.d, s.frame, "channel.depth");
}
void mixedSurfaceReference()
{
  Scene s("quality");
  s.medium({.1f, .2f, 0}, {.3f, .1f, 0}, 2, 4);
  s.tf({0, 0, 0}, 5);
  auto m = s.surface(8.f, 1.f);
  anari::setParameter(s.d, m, "color", V3{.2f, .5f, .8f});
  anari::commitParameters(s.d, m);
  s.environment({.2f, .3f, .4f});
  const auto rgb = compositing_reference::mixedSurface();
  s.check("phase surface TF1D MDL MIS", {rgb[0], rgb[1], rgb[2], 1.f});
}
void mixedTransmission(const char *type)
{
  Scene s(type);
  s.medium({.2f, .4f, 0}, {0, 0, 0}, 2, 4);
  s.tf({0, 0, 0}, 5);
  auto filter = s.surface(4.5f, 1.f, true);
  anari::setParameter(s.d, filter, "baseColor", V3{.4f, .7f, .2f});
  anari::setParameter(s.d, filter, "roughness", 0.f);
  anari::setParameter(s.d, filter, "ior", 1.f);
  anari::commitParameters(s.d, filter);
  s.surface(8.f, 1.f, false, {.2f, .3f, .4f});
  s.check("colored transmission TF1D MDL",
      {.2f * .4f * std::exp(-.4f) * .5625f,
          .3f * .7f * std::exp(-.8f) * .5625f,
          .4f * .2f * .5625f,
          1.f});
}
void mixedReflection(const char *type)
{
  Scene s(type);
  s.medium({.2f, .4f, 0}, {0, 0, 0}, 2, 4);
  s.tf({0, 0, 0}, 5);
  auto m = s.surface(8.f, 1.f);
  anari::setParameter(s.d, m, "color", V3{.2f, .5f, .8f});
  anari::commitParameters(s.d, m);
  auto light = anari::newObject<anari::Light>(s.d, "directional");
  s.owned.push_back(light);
  anari::setParameter(s.d, light, "direction", V3{0, 0, 1});
  anari::setParameter(s.d, light, "irradiance", 1.f);
  anari::commitParameters(s.d, light);
  anari::setParameterArray1D(s.d, s.world, "light", &light, 1);
  anari::commitParameters(s.d, s.world);
  // Coaxial delta illumination crosses each medium twice. Black TF1D has
  // no source. The surface is Lambertian: f*cos = tint/pi at normal incidence.
  constexpr float pi = 3.14159265358979323846f;
  anari::unsetParameter(s.d, s.world, "volume");
  anari::commitParameters(s.d, s.world);
  s.check("colored reflection control", {.2f / pi, .5f / pi, .8f / pi, 1.f});
  anari::setParameterArray1D(
      s.d, s.world, "volume", s.volumes.data(), s.volumes.size());
  anari::commitParameters(s.d, s.world);
  s.check("colored reflection TF1D MDL",
      {.2f / pi * std::exp(-.8f) * std::pow(.75f, 4),
          .5f / pi * std::exp(-1.6f) * std::pow(.75f, 4),
          .8f / pi * std::pow(.75f, 4),
          1.f});
}
void backgroundAndClipping(const char *type)
{
  Scene s(type);
  const float k = std::log(10.f);
  s.medium({k, 0, 0}, {0, 0, 0}, 2, 3);
  s.medium({0, k, 0}, {0, 0, 0}, 4, 5);
  s.background(.4f);
  anari::setParameter(s.d, s.renderer, "premultiplyBackground", true);
  anari::commitParameters(s.d, s.renderer);
  s.check("premultiplied screen", {.008f, .012f, .16f, .76f});
  anari::setParameter(s.d, s.renderer, "premultiplyBackground", false);
  anari::commitParameters(s.d, s.renderer);
  s.check("straight screen", {.02f, .03f, .4f, .76f});
  s.environment({.2f, .3f, .4f});
  s.check("HDRI background", {.02f, .03f, .4f, 1.f});
  anari::unsetParameter(s.d, s.world, "light");
  anari::commitParameters(s.d, s.world);
  s.surface(6.f, 1.f, false, {.2f, .3f, .4f});
  s.check("emissive endpoint", {.02f, .03f, .4f, 1.f});
  // Near/far crop both media to half thickness and exclude the emitter.
  anari::setParameter(s.d, s.camera, "near", 2.5f);
  anari::setParameter(s.d, s.camera, "far", 4.5f);
  anari::commitParameters(s.d, s.camera);
  const float t = std::sqrt(.1f);
  s.check("clipped primary",
      {.2f * t, .3f * t, .4f, 1.f - (2.f * t + 1.f) / 3.f * .6f});
}
void coverageIndependence()
{
  Scene preview("interactive");
  preview.medium({.4f, .2f, 0}, {.3f, .8f, 0}, 2, 4);
  const float previewAlpha =
      1.f - (std::exp(-1.4f) + std::exp(-2.f) + 1.f) / 3.f;
  preview.check("scattering coverage independent of preview lighting",
      {0, 0, 0, previewAlpha},
      true);
  Scene s("quality");
  s.medium({.4f, .2f, 0}, {.3f, .8f, 0}, 2, 4);
  const float alpha = 1.f - (std::exp(-1.4f) + std::exp(-2.f) + 1.f) / 3.f;
  anari::setParameter(s.d, s.renderer, "maxRayDepth", 1);
  anari::commitParameters(s.d, s.renderer);
  s.check("scattering coverage depth 1", {0, 0, 0, alpha}, true);
  anari::setParameter(s.d, s.renderer, "maxRayDepth", 12);
  anari::commitParameters(s.d, s.renderer);
  s.check("scattering coverage depth 12", {0, 0, 0, alpha}, true);
  for (int i = 0; i < 10; ++i)
    s.surface(4.5f + .1f * i, .2f, true);
  const float layeredAlpha = 1.f - (1.f - alpha) * std::pow(.8f, 10);
  // Short radiance transparency/depth budgets deliberately abandon paths.
  // Long paths take BSDF transmission and encounter surface roulette.
  anari::setParameter(s.d, s.renderer, "maxRayDepth", 1);
  anari::setParameter(s.d, s.renderer, "maxTransparencyDepth", 0);
  anari::commitParameters(s.d, s.renderer);
  s.check("coverage independent of terminated radiance",
      {0, 0, 0, layeredAlpha},
      true);
  anari::setParameter(s.d, s.renderer, "maxRayDepth", 32);
  anari::setParameter(s.d, s.renderer, "maxTransparencyDepth", 32);
  anari::commitParameters(s.d, s.renderer);
  s.check(
      "coverage independent of BSDF roulette", {0, 0, 0, layeredAlpha}, true);
}
void mixedPreview()
{
  Scene s("interactive");
  s.medium({std::log(2.f), std::log(4.f), 0}, {0, 0, 0}, 2, 3);
  s.tf({.4f, .2f, .1f}, 4);
  // TF1D has thickness 2 and T=.75^2, with its existing preview source.
  // Its source AND the screen must be attenuated by the earlier MDL box.
  s.check("MDL then emitting TF1D",
      {.5f * (.4375f * .4f + .5625f * .2f),
          .25f * (.4375f * .2f + .5625f * .3f),
          .4375f * .1f + .5625f * .4f,
          1.f - .5625f * 1.75f / 3.f});
  Scene reverse("interactive");
  reverse.tf({.4f, .2f, .1f}, 2);
  reverse.medium({std::log(2.f), std::log(4.f), 0}, {0, 0, 0}, 5, 6);
  reverse.check("emitting TF1D then MDL",
      {.4375f * .4f + .5625f * .5f * .2f,
          .4375f * .2f + .5625f * .25f * .3f,
          .4375f * .1f + .5625f * .4f,
          1.f - .5625f * 1.75f / 3.f});
}
void transmissionCoverage(const char *type)
{
  Scene s(type);
  s.medium({0, 0, 0}, {0, 0, 0}, 2, 3);
  s.surface(4.f, 1.f, true);
  // Refraction is radiance transport, NOT uncovered geometric area.
  s.check("fully covered transmitting surface", {0, 0, 0, 1}, true);
}
} // namespace
int main(int argc, char **argv)
{
  try {
    if (argc > 1 && std::string(argv[1]) == "aov") {
      aovRegression("interactive");
      aovRegression("quality");
      return 0;
    }
    if (argc > 1 && std::string(argv[1]) == "mixed") {
      mixedReflection("interactive");
      mixedReflection("quality");
      mixedTransmission("interactive");
      mixedTransmission("quality");
      mixedSurfaceReference();
      return 0;
    }
    complementary("interactive");
    complementary("quality");
    transmissionCoverage("interactive");
    transmissionCoverage("quality");
    mixedPreview();
    backgroundAndClipping("interactive");
    backgroundAndClipping("quality");
    coverageIndependence();
    mixedReflection("interactive");
    mixedReflection("quality");
    mixedTransmission("interactive");
    mixedTransmission("quality");
    mixedSurfaceReference();
    aovRegression("interactive");
    aovRegression("quality");
    return 0;
  } catch (const std::exception &e) {
    fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
