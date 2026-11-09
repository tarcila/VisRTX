// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#include <anari/anari_cpp/ext/std.h>
#include <anari/ext/visrtx/makeVisRTXDevice.h>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>
#include "MdlVolumeTestHelpers.h"

namespace {
using V3 = std::array<float, 3>;
using V4 = std::array<float, 4>;
using Box = std::array<V3, 2>;
using Matrix = std::array<float, 16>;
constexpr Matrix IDENTITY{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
void require(bool value, const char *message)
{
  if (!value)
    throw std::runtime_error(message);
}
struct Scene
{
  unsigned errors{};
  std::string cause;
  ANARIObject frameSource{};
  static void status(const void *user,
      ANARIDevice,
      ANARIObject source,
      ANARIDataType,
      ANARIStatusSeverity severity,
      ANARIStatusCode,
      const char *message)
  {
    auto &s = *static_cast<Scene *>(const_cast<void *>(user));
    if (severity <= ANARI_SEVERITY_ERROR) {
      fprintf(stderr, "%s\n", message);
      if (source == s.frameSource) {
        ++s.errors;
        s.cause = message;
      }
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
  Scene()
  {
    frameSource = frame;
    anari::setParameter(d, material, "sourceType", "code");
    anari::setParameter(d, material, "source", R"mdl(mdl 1.7;
export material main(uniform color absorption=color(0.25,0.5,0),
    uniform color scattering=color(0)) = material(volume: material_volume(
    absorption_coefficient: absorption, scattering_coefficient: scattering));
)mdl");
    anari::commitParameters(d, material);
    anari::setParameter(d, volume, "material", material);
    Box box{V3{-2, -2, 2}, V3{2, 2, 6}};
    anariSetParameter(d, volume, "extent", ANARI_FLOAT32_BOX3, box.data());
    majorant(.5f);
    anari::setParameterArray1D(d, group, "volume", &volume, 1);
    anari::commitParameters(d, group);
    anari::setParameter(d, instance, "group", group);
    anari::commitParameters(d, instance);
    anari::setParameterArray1D(d, world, "instance", &instance, 1);
    anari::commitParameters(d, world);
    anari::setParameter(d, camera, "position", V3{0, 0, 0});
    anari::setParameter(d, camera, "direction", V3{0, 0, 1});
    anari::setParameter(d, camera, "up", V3{0, 1, 0});
    anari::setParameter(d, camera, "height", .5f);
    anari::commitParameters(d, camera);
    anari::setParameter(d, renderer, "denoise", false);
    anari::setParameter(d, renderer, "fireflyFilterMode", "none");
    anari::setParameter(d, renderer, "ambientRadiance", 0.f);
    anari::setParameter(d, renderer, "pixelSamples", 2048);
    anari::setParameter(d, renderer, "sampleLimit", 0);
    background({1, 1, 1, 1});
    anari::setParameter(d, frame, "size", std::array<unsigned, 2>{8, 8});
    anari::setParameter(d, frame, "channel.color", ANARI_FLOAT32_VEC4);
    anari::setParameter(d, frame, "channel.depth", ANARI_FLOAT32);
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
             anari::Object(material)})
      anari::release(d, o);
    anari::release(d, d);
  }
  void majorant(float m)
  {
    anari::setParameter(d, volume, "extinctionMajorant", m);
    anari::commitParameters(d, volume);
  }
  void coefficients(V3 a, V3 s = {0, 0, 0})
  {
    anari::setParameter(d, material, "absorption", a);
    anari::setParameter(d, material, "scattering", s);
    anari::commitParameters(d, material);
  }
  void background(V4 bg, bool premultiply = true)
  {
    anari::setParameter(d, renderer, "background", bg);
    anari::setParameter(d, renderer, "premultiplyBackground", premultiply);
    anari::commitParameters(d, renderer);
  }
  void extent(Box box)
  {
    anariSetParameter(d, volume, "extent", ANARI_FLOAT32_BOX3, box.data());
    anari::commitParameters(d, volume);
  }
  void transform(Matrix matrix)
  {
    anariSetParameter(
        d, instance, "transform", ANARI_FLOAT32_MAT4, matrix.data());
    anari::commitParameters(d, instance);
  }
  void render()
  {
    anari::render(d, frame);
    require(anariFrameReady(d, frame, ANARI_WAIT), "WAIT must complete");
  }
  void check(const char *name, V4 reference)
  {
    // Fixed, predeclared budget: 16 nonoverlapping batches of 8x8x2048
    // samples (32768 spp). Channel compensation needs more than the spec's
    // initial 4096 spp to meet its uncertainty ceiling for a white vacuum.
    constexpr int batches = 16;
    std::array<double, 4> previous{}, mean{}, m2{};
    const auto before = errors;
    bool discardedInitialReset = false;
    for (int b = 0; b < batches; ++b) {
      render();
      require(errors == before, "unexpected Frame error");
      int count = 0;
      require(anariGetProperty(d,
                  frame,
                  "numSamples",
                  ANARI_INT32,
                  &count,
                  sizeof(count),
                  ANARI_WAIT),
          "numSamples query");
      // The existing automatic reset lifecycle may consume the first
      // render after initial setup/recovery twice. Discard that first image,
      // then require strictly increasing counts for every retained batch.
      if (b == 1 && count == 2048 && !discardedInitialReset) {
        discardedInitialReset = true;
        b = 0;
        previous = {};
        mean = {};
        m2 = {};
      }
      if (count != (b + 1) * 2048)
        fprintf(stderr,
            "%s batch %d: sample count %d expected %d\n",
            name,
            b,
            count,
            (b + 1) * 2048);
      require(count == (b + 1) * 2048,
          "every batch must add camera samples, not repeat a capped image");
      const auto p = anari::map<V4>(d, frame, "channel.color");
      require(
          p.data && p.width == 8 && p.height == 8, "quality output must map");
      std::array<double, 4> cumulative{};
      for (int i = 0; i < 64; ++i)
        for (int c = 0; c < 4; ++c) {
          require(std::isfinite(p.data[i][c]), "nonfinite result");
          cumulative[c] += p.data[i][c] / 64.;
        }
      anari::unmap(d, frame, "channel.color");
      for (int c = 0; c < 4; ++c) {
        const double batch = cumulative[c] * (b + 1) - previous[c] * b;
        previous[c] = cumulative[c];
        const double delta = batch - mean[c];
        mean[c] += delta / (b + 1);
        m2[c] += delta * (batch - mean[c]);
      }
    }
    for (int c = 0; c < 4; ++c) {
      const double se = std::sqrt(m2[c] / (batches * (batches - 1)));
      const double error = std::abs(mean[c] - reference[c]);
      printf("%s c%d: %.8f reference %.8f SE %.8f\n",
          name,
          c,
          mean[c],
          reference[c],
          se);
      require(se <= .0016, "inconclusive: batch uncertainty exceeds ceiling");
      require(
          error <= .01 && error <= 5 * se + .002, "quality reference mismatch");
    }
  }
  void fail(const char *message)
  {
    const auto before = errors;
    render();
    require(errors > before && cause.find(message) != std::string::npos,
        "expected Frame failure category");
    requireFailedMdlFrame(d, frame);
  }
};
V4 beer(V3 a, double length, V4 bg = {1, 1, 1, 1}, bool premultiply = true)
{
  V4 r{};
  double mean = 0.;
  for (int c = 0; c < 3; ++c) {
    const double t = std::exp(-a[c] * length);
    r[c] = float(t * bg[c] * (premultiply ? bg[3] : 1.f));
    mean += t / 3.;
  }
  r[3] = float(1. - mean * (1. - bg[3]));
  return r;
}

void geometryAndClipping(Scene &s)
{
  Matrix matrix = IDENTITY;
  matrix[0] = 0.f;
  matrix[1] = 2.f;
  matrix[4] = -.75f;
  matrix[5] = 0.f;
  matrix[10] = 1.5f;
  matrix[14] = 1.f;
  s.transform(matrix);
  s.check(
      "rotated nonuniform Instance / world distance", beer({.25f, .5f, 0}, 6));
  s.transform(IDENTITY);
  anari::setParameter(s.d, s.camera, "near", 3.f);
  anari::setParameter(s.d, s.camera, "far", 5.f);
  anari::commitParameters(s.d, s.camera);
  s.check("clipped field-free Interval", beer({.25f, .5f, 0}, 2));
  anari::unsetParameter(s.d, s.camera, "near");
  anari::unsetParameter(s.d, s.camera, "far");
  anari::commitParameters(s.d, s.camera);
  auto perspective = anari::newObject<anari::Camera>(s.d, "perspective");
  anari::setParameter(s.d, perspective, "position", V3{0, 0, 0});
  anari::setParameter(s.d, perspective, "direction", V3{0, 0, 1});
  anari::setParameter(s.d, perspective, "up", V3{0, 1, 0});
  // Pinhole limit: variation in world path length is <1e-10.
  anari::setParameter(s.d, perspective, "fovy", 1e-6f);
  anari::setParameter(s.d, perspective, "near", 3.f);
  anari::setParameter(s.d, perspective, "far", 5.f);
  anari::commitParameters(s.d, perspective);
  anari::setParameter(s.d, s.frame, "camera", perspective);
  anari::commitParameters(s.d, s.frame);
  s.check("perspective clipped Interval", beer({.25f, .5f, 0}, 2));
  auto rayBuffer = anari::newObject<anari::Camera>(s.d, "rayBuffer");
  std::array<V3, 64> directions;
  std::array<float, 64> limits;
  directions.fill({0, 0, 1});
  limits.fill(3.f);
  anari::setParameter(s.d, rayBuffer, "position", V3{0, 0, 0});
  anari::setAndReleaseParameter(s.d,
      rayBuffer,
      "ray.dir",
      anari::newArray2D(s.d, directions.data(), 8, 8));
  anari::setAndReleaseParameter(
      s.d, rayBuffer, "ray.tmax", anari::newArray2D(s.d, limits.data(), 8, 8));
  anari::commitParameters(s.d, rayBuffer);
  anari::setParameter(s.d, s.frame, "camera", rayBuffer);
  anari::commitParameters(s.d, s.frame);
  s.check("ray-buffer tmax clips inside medium", beer({.25f, .5f, 0}, 1));
  s.coefficients({0, 0, 0});
  s.majorant(0.f);
  anari::setParameter(s.d, s.renderer, "cutPlane", V4{0, 0, -1, 1});
  anari::commitParameters(s.d, s.renderer);
  s.check("vacuum ray-buffer Far Limit", {1, 1, 1, 1});
  const auto depths = anari::map<float>(s.d, s.frame, "channel.depth");
  require(depths.data, "depth map");
  for (unsigned i = 0; i < 64; ++i)
    require(depths.data[i] == 3.f, "Miss Depth remains camera Far Limit");
  anari::unmap(s.d, s.frame, "channel.depth");
  anari::unsetParameter(s.d, s.renderer, "cutPlane");
  anari::commitParameters(s.d, s.renderer);
  s.coefficients({.25f, .5f, 0});
  s.majorant(.5f);
  anari::setParameter(s.d, s.frame, "camera", s.camera);
  anari::commitParameters(s.d, s.frame);
  anari::release(s.d, perspective);
  anari::release(s.d, rayBuffer);
  for (float step : {.7f, std::numeric_limits<float>::min(), 100.f}) {
    anari::setParameter(s.d, s.volume, "stepSize", step);
    anari::commitParameters(s.d, s.volume);
    s.check("quality ignores valid marching step", beer({.25f, .5f, 0}, 4));
  }
  anari::setParameter(s.d, s.volume, "stepSize", 0.f);
  anari::commitParameters(s.d, s.volume);
  s.fail("stepSize");
  anari::unsetParameter(s.d, s.volume, "stepSize");
  anari::commitParameters(s.d, s.volume);
  // This valid extent's automatic FP32 /128 scale would round to zero.
  // Quality must not derive or reject that unused marching heuristic.
  s.coefficients({0, 0, 0});
  s.majorant(0.f);
  s.extent({V3{0, -2, 2}, V3{std::numeric_limits<float>::denorm_min(), 2, 6}});
  s.check(
      "unused automatic marching scale cannot reject quality", {1, 1, 1, 1});
  s.extent({V3{-2, -2, 2}, V3{2, 2, 6}});
  s.coefficients({.25f, .5f, 0});
  s.majorant(.5f);
}

void failureAndRecovery(Scene &s)
{
  for (bool manual : {false, true}) {
    if (manual)
      anari::setParameter(s.d, s.frame, "accumulationVersion", uint64_t(17));
    else
      anari::unsetParameter(s.d, s.frame, "accumulationVersion");
    anari::commitParameters(s.d, s.frame);
    // Force a valid fresh accumulation before each failure.
    s.majorant(.5f);
    s.check("valid before evaluated failure", beer({.25f, .5f, 0}, 4));
    s.coefficients({.25f, .75f, 0});
    s.fail("majorant violation");
    s.coefficients({.25f, .5f, 0});
    s.check(
        "recovery resets even manual accumulation", beer({.25f, .5f, 0}, 4));
  }
  anari::unsetParameter(s.d, s.frame, "accumulationVersion");
  anari::commitParameters(s.d, s.frame);
  for (float bad : {-1.f,
           std::numeric_limits<float>::infinity(),
           std::numeric_limits<float>::quiet_NaN()}) {
    s.coefficients({0, 0, bad}); // validate B even on R/G transport paths
    s.fail("coefficient");
  }
  s.coefficients({0, 0, 1});
  s.fail("majorant violation");
  s.majorant(1.f);
  for (float tiny : {std::ldexp(1.f, -25), std::ldexp(1.f, -80)}) {
    s.coefficients({1, 0, 0}, {tiny, 0, 0});
    s.fail("majorant violation");
    s.coefficients({tiny, 0, 0}, {1, 0, 0});
    s.fail("majorant violation");
    s.majorant(std::nextafter(1.f, 2.f));
    const auto before = s.errors;
    s.render();
    require(s.errors == before, "valid widened scattering bound must render");
    s.majorant(1.f);
  }
  s.coefficients({.25f, .5f, 0});
  s.majorant(std::numeric_limits<float>::max());
  s.fail("numerical failure"); // a candidate cannot advance world t near 2
  s.majorant(.5f);
  s.check("recover after non-progress", beer({.25f, .5f, 0}, 4));
}

void surfaceAndEnvironment(Scene &s)
{
  auto geometry = anari::newObject<anari::Geometry>(s.d, "quad");
  auto material = anari::newObject<anari::Material>(s.d, "mdl");
  auto surface = anari::newObject<anari::Surface>(s.d);
  auto light = anari::newObject<anari::Light>(s.d, "hdri");
  const V3 texel{.8f, .6f, .4f};
  auto image = anari::newArray2D(s.d, &texel, 1, 1);
  anari::setParameter(s.d, light, "radiance", image);
  anari::commitParameters(s.d, light);
  anari::setParameterArray1D(s.d, s.world, "light", &light, 1);
  anari::commitParameters(s.d, s.world);
  s.check("visible HDRI through absorption",
      beer({.25f, .5f, 0}, 4, {.8f, .6f, .4f, 1}));
  auto source = [&](const char *code) {
    anari::setParameter(s.d, material, "sourceType", "code");
    anari::setParameter(s.d, material, "source", code);
    anari::commitParameters(s.d, material);
  };
  auto place = [&](const std::array<V3, 4> &vertices) {
    anari::setParameterArray1D(
        s.d, geometry, "vertex.position", vertices.data(), 4);
    anari::commitParameters(s.d, geometry);
  };
  place({V3{-2, -2, 8}, V3{-2, 2, 8}, V3{2, 2, 8}, V3{2, -2, 8}});
  source(R"mdl(mdl 1.7; import ::df::*;
export material main() = material(surface: material_surface(
  emission: material_emission(emission: df::diffuse_edf(),
    intensity: color(0.8,0.6,0.4)*3.141592653589793)));
)mdl");
  anari::setParameter(s.d, surface, "geometry", geometry);
  anari::setParameter(s.d, surface, "material", material);
  anari::commitParameters(s.d, surface);
  anari::setParameterArray1D(s.d, s.world, "surface", &surface, 1);
  anari::commitParameters(s.d, s.world);
  s.check("finite emissive surface background",
      beer({.25f, .5f, 0}, 4, {.8f, .6f, .4f, 1}));
  source(R"mdl(mdl 1.7; import ::df::*;
export material main() = material(surface: material_surface(
  scattering: df::specular_bsdf(tint: color(0.6,0.3,0.8),mode:df::scatter_reflect)));
)mdl");
  place({V3{-2, -2, -1}, V3{-2, 2, -1}, V3{2, 2, 3}, V3{2, -2, 3}});
  s.extent({V3{2, -2, -2}, V3{6, 2, 2}});
  s.check("channel stays fixed through colored mirror",
      beer({.25f, .5f, 0}, 4, {.8f * .6f, .6f * .3f, .4f * .8f, 1}));
  source(R"mdl(mdl 1.7; import ::df::*;
export material main() = material(surface: material_surface(
  scattering: df::diffuse_reflection_bsdf(tint:color(0.6,0.3,0.8))));
)mdl");
  place({V3{-2, -2, 8}, V3{-2, 2, 8}, V3{2, 2, 8}, V3{2, -2, 8}});
  s.extent({V3{-1000000, -1000000, 2}, V3{1000000, 1000000, 6}});
  // Independent hemispherical integral for a Lambert reflector behind a
  // slab: exp(-tau)*2*integral_0^1 mu*exp(-tau/mu) dmu. The huge transverse
  // domain changes only a negligible grazing-angle region (mu < 1e-5).
  auto reference = [](unsigned n) {
    V4 result{0, 0, 0, 1};
    const double tau[3] = {1, 2, 0}, tint[3] = {.6, .3, .8},
                 env[3] = {.8, .6, .4};
    for (int c = 0; c < 3; ++c) {
      double integral = 0.;
      for (unsigned i = 0; i < n; ++i) {
        const double mu = (i + .5) / n;
        integral += 2 * mu * std::exp(-tau[c] / mu) / n;
      }
      result[c] = float(env[c] * tint[c] * std::exp(-tau[c]) * integral);
    }
    return result;
  };
  const auto coarse = reference(4096), fine = reference(8192);
  for (int c = 0; c < 3; ++c)
    require(std::abs(coarse[c] - fine[c]) < .0005f, "quadrature refinement");
  s.check("full RGB diffuse proposal / actual MIS PDFs", fine);
  anari::unsetParameter(s.d, s.world, "surface");
  anari::unsetParameter(s.d, s.world, "light");
  anari::commitParameters(s.d, s.world);
  s.extent({V3{-2, -2, 2}, V3{2, 2, 6}});
  for (auto o : {anari::Object(image),
           anari::Object(light),
           anari::Object(surface),
           anari::Object(material),
           anari::Object(geometry)})
    anari::release(s.d, o);
}

void surfaceShadow(Scene &s)
{
  auto geometry = anari::newObject<anari::Geometry>(s.d, "quad");
  const std::array<V3, 4> vertices{
      V3{-2, -2, -7}, V3{-2, 2, -7}, V3{2, 2, 9}, V3{2, -2, 9}};
  anari::setParameterArray1D(
      s.d, geometry, "vertex.position", vertices.data(), 4);
  anari::commitParameters(s.d, geometry);
  auto material = anari::newObject<anari::Material>(s.d, "matte");
  anari::setParameter(s.d, material, "color", V3{1, 1, 1});
  anari::commitParameters(s.d, material);
  auto surface = anari::newObject<anari::Surface>(s.d);
  anari::setParameter(s.d, surface, "geometry", geometry);
  anari::setParameter(s.d, surface, "material", material);
  anari::commitParameters(s.d, surface);
  auto light = anari::newObject<anari::Light>(s.d, "directional");
  anari::setParameter(s.d, light, "direction", V3{-.5f, 0, -1});
  anari::setParameter(s.d, light, "irradiance", 10.f);
  anari::commitParameters(s.d, light);
  anari::setParameterArray1D(s.d, s.world, "surface", &surface, 1);
  anari::setParameterArray1D(s.d, s.world, "light", &light, 1);
  anari::commitParameters(s.d, s.world);
  s.extent({V3{-20, -20, 2}, V3{20, 20, 6}});
  // Independent Lambert cosine from plane normal (4,0,-1)/sqrt(17)
  // and light direction (.5,0,1)/sqrt(1.25).
  const float L = float(10. / (3.141592653589793 * std::sqrt(21.25)));
  for (float M : {.5f, 1.f, 2.f}) {
    s.majorant(M);
    s.check("ratio-tracked surface light / actual RGB proposal",
        beer({.25f, .5f, 0}, 4 * std::sqrt(1.25), {L, L, L, 1}));
  }
  auto blockerGeometry = anari::newObject<anari::Geometry>(s.d, "quad");
  const std::array<V3, 4> blockerVertices{
      V3{-20, -20, 3}, V3{-20, 20, 3}, V3{20, 20, 3}, V3{20, -20, 3}};
  anari::setParameterArray1D(
      s.d, blockerGeometry, "vertex.position", blockerVertices.data(), 4);
  anari::commitParameters(s.d, blockerGeometry);
  auto blocker = anari::newObject<anari::Surface>(s.d);
  anari::setParameter(s.d, blocker, "geometry", blockerGeometry);
  anari::setParameter(s.d, blocker, "material", material);
  anari::commitParameters(s.d, blocker);
  const std::array<anari::Surface, 2> surfaces{surface, blocker};
  anari::setParameterArray1D(s.d, s.world, "surface", surfaces.data(), 2);
  anari::commitParameters(s.d, s.world);
  s.check("opaque shadow visibility", {0, 0, 0, 1});
  anari::unsetParameter(s.d, s.world, "surface");
  anari::unsetParameter(s.d, s.world, "light");
  anari::commitParameters(s.d, s.world);
  s.extent({V3{-2, -2, 2}, V3{2, 2, 6}});
  s.majorant(.5f);
  for (auto o : {anari::Object(blocker),
           anari::Object(blockerGeometry),
           anari::Object(light),
           anari::Object(surface),
           anari::Object(material),
           anari::Object(geometry)})
    anari::release(s.d, o);
}
} // namespace
int main()
{
  try {
    Scene s;
    s.check("RGB Beer-Lambert / equality", beer({.25f, .5f, 0}, 4));
    for (float M : {1.f, 2.f}) {
      s.majorant(M);
      s.check("2x/4x majorant", beer({.25f, .5f, 0}, 4));
    }
    s.coefficients({0, 0, 0});
    for (float M : {0.f, .5f}) {
      s.majorant(M);
      s.check("vacuum / single factor three", {1, 1, 1, 1});
    }
    s.coefficients({.25f, .5f, 0});
    s.majorant(.5f);
    for (bool premultiply : {false, true})
      for (float alpha : {0.f, .35f, 1.f}) {
        V4 bg{.8f, .6f, .4f, alpha};
        s.background(bg, premultiply);
        s.check("screen chromatic weighting",
            beer({.25f, .5f, 0}, 4, bg, premultiply));
      }
    s.background({1, 1, 1, 1});
    s.coefficients({.005f, .02f, 0});
    s.majorant(.02f);
    s.check("optically thin / zero channel", beer({.005f, .02f, 0}, 4));
    s.coefficients({5.f, .5f, 0});
    s.majorant(5.f);
    s.check("optically thick / zero channel", beer({5.f, .5f, 0}, 4));
    s.coefficients({.25f, .5f, 0});
    s.majorant(.5f);
    geometryAndClipping(s);
    failureAndRecovery(s);
    surfaceShadow(s);
    surfaceAndEnvironment(s);
  } catch (const std::exception &e) {
    fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
