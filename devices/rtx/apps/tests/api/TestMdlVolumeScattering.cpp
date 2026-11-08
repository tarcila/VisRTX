// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#include <anari/anari_cpp/ext/std.h>
#include <anari/ext/visrtx/makeVisRTXDevice.h>
#include <anari/anari_cpp.hpp>
#include <array>
#include <cmath>
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <vector>

namespace {
using V3 = std::array<float, 3>;
using V4 = std::array<float, 4>;
using D3 = std::array<double, 3>;
using Box = std::array<V3, 2>;
constexpr Box BOX{V3{-2, -2, 2}, V3{2, 2, 6}};
constexpr V3 ABS{.15f, .3f, 0}, SCATTER{.35f, .1f, 0};
constexpr double PI = 3.14159265358979323846;
const char *HOMOGENEOUS = R"mdl(mdl 1.7;
export material main(uniform color a=color(0.15,0.3,0),
    uniform color s=color(0.35,0.1,0)) = material(volume: material_volume(
    absorption_coefficient:a, scattering_coefficient:s));
)mdl";
void require(bool b, const char *message)
{
  if (!b)
    throw std::runtime_error(message);
}
// Independent composite Simpson reference. Refine until well below the
// specification's 0.0005 reference allowance, not a production midpoint loop.
double quadrature(const std::function<double(double)> &f, double lo, double hi)
{
  double previous = 0;
  for (unsigned n = 32; n <= 65536; n *= 2) {
    const double h = (hi - lo) / n;
    double sum = f(lo) + f(hi);
    for (unsigned j = 1; j < n; ++j)
      sum += (j % 2 ? 4 : 2) * f(lo + j * h);
    const double result = sum * h / 3;
    if (n > 32 && std::abs(result - previous) < 1e-9)
      return result;
    previous = result;
  }
  throw std::runtime_error("reference quadrature did not converge");
}
struct Scene
{
  unsigned errors = 0;
  static void status(const void *u,
      ANARIDevice,
      ANARIObject,
      ANARIDataType,
      ANARIStatusSeverity severity,
      ANARIStatusCode,
      const char *message)
  {
    if (severity <= ANARI_SEVERITY_ERROR) {
      ++static_cast<Scene *>(const_cast<void *>(u))->errors;
      fprintf(stderr, "%s\n", message);
    }
  }
  anari::Device d = makeVisRTXDevice(status, this);
  anari::Material m = anari::newObject<anari::Material>(d, "mdl");
  anari::Volume v = anari::newObject<anari::Volume>(d, "mdl");
  anari::Group g = anari::newObject<anari::Group>(d);
  anari::Instance i = anari::newObject<anari::Instance>(d, "transform");
  anari::World w = anari::newObject<anari::World>(d);
  anari::Camera c = anari::newObject<anari::Camera>(d, "orthographic");
  anari::Renderer r = anari::newObject<anari::Renderer>(d, "interactive");
  anari::Frame f = anari::newObject<anari::Frame>(d);
  std::vector<anari::Light> lights;
  uint64_t accumulationVersion = 0;
  Scene()
  {
    source(HOMOGENEOUS);
    anari::setParameter(d, v, "material", m);
    extent(BOX);
    anari::setParameter(d, v, "extinctionMajorant", 0.81f);
    anari::setParameter(d, v, "stepSize", 0.125f);
    anari::commitParameters(d, v);
    anari::setParameterArray1D(d, g, "volume", &v, 1);
    anari::commitParameters(d, g);
    anari::setParameter(d, i, "group", g);
    anari::commitParameters(d, i);
    anari::setParameterArray1D(d, w, "instance", &i, 1);
    anari::commitParameters(d, w);
    anari::setParameter(d, c, "position", V3{0, 0, 0});
    anari::setParameter(d, c, "direction", V3{0, 0, 1});
    anari::setParameter(d, c, "up", V3{0, 1, 0});
    // Negligible transverse jitter; all controlled optical paths are axial.
    anari::setParameter(d, c, "height", 0.00001f);
    anari::commitParameters(d, c);
    anari::setParameter(d, r, "denoise", false);
    anari::setParameter(d, r, "fireflyFilterMode", "none");
    anari::setParameter(d, r, "ambientRadiance", 0.f);
    anari::setParameter(d, r, "sampleLimit", 0);
    anari::setParameter(d, r, "background", V4{0, 0, 0, 1});
    rates(1, 1, 512);
    anari::setParameter(d, f, "size", std::array<unsigned, 2>{8, 8});
    anari::setParameter(d, f, "channel.color", ANARI_FLOAT32_VEC4);
    anari::setParameter(d, f, "world", w);
    anari::setParameter(d, f, "camera", c);
    anari::setParameter(d, f, "renderer", r);
    anari::commitParameters(d, f);
  }
  ~Scene()
  {
    for (auto light : lights)
      anari::release(d, light);
    for (auto o : {anari::Object(f),
             anari::Object(r),
             anari::Object(c),
             anari::Object(w),
             anari::Object(i),
             anari::Object(g),
             anari::Object(v),
             anari::Object(m)})
      anari::release(d, o);
    anari::release(d, d);
  }
  void source(const char *code)
  {
    anari::setParameter(d, m, "sourceType", "code");
    anari::setParameter(d, m, "source", code);
    anari::commitParameters(d, m);
  }
  void extent(Box box)
  {
    anariSetParameter(d, v, "extent", ANARI_FLOAT32_BOX3, box.data());
    anari::commitParameters(d, v);
  }
  void rates(float primary, float shadow, int samples)
  {
    anari::setParameter(d, r, "volumeSamplingRate", primary);
    anari::setParameter(d, r, "volumeSamplingRateShadows", shadow);
    anari::setParameter(d, r, "pixelSamples", samples);
    anari::commitParameters(d, r);
  }
  anari::Light addLight(const char *type)
  {
    auto light = anari::newObject<anari::Light>(d, type);
    lights.push_back(light);
    return light;
  }
  void publishLights()
  {
    anari::setParameterArray1D(d, w, "light", lights.data(), lights.size());
    anari::commitParameters(d, w);
  }
  anari::Light directional(V3 direction, float irradiance)
  {
    auto light = addLight("directional");
    anari::setParameter(d, light, "direction", direction);
    anari::setParameter(d, light, "irradiance", irradiance);
    anari::commitParameters(d, light);
    publishLights();
    return light;
  }
  D3 render()
  {
    anari::render(d, f);
    anari::wait(d, f);
    require(errors == 0, "unexpected render error");
    auto image = anari::map<V4>(d, f, "channel.color");
    require(image.data != nullptr, "color must map");
    D3 mean{};
    for (unsigned p = 0; p < 64; ++p)
      for (unsigned ch = 0; ch < 3; ++ch) {
        require(std::isfinite(image.data[p][ch]), "finite color required");
        mean[ch] += double(image.data[p][ch]) / 64;
      }
    anari::unmap(d, f, "channel.color");
    return mean;
  }
  void check(const char *label, D3 expected, double tolerance = .002)
  {
    auto value = render();
    for (unsigned ch = 0; ch < 3; ++ch) {
      printf("%s channel %u: %.9g reference %.9g\n",
          label,
          ch,
          value[ch],
          expected[ch]);
      require(std::abs(value[ch] - expected[ch]) < tolerance,
          "quadrature reference mismatch");
    }
  }
  void batches(const char *label, D3 expected)
  {
    // Eight consecutive nonoverlapping batches of 512 spp: 8x8x4096 samples.
    ++accumulationVersion;
    anariSetParameter(
        d, f, "accumulationVersion", ANARI_UINT64, &accumulationVersion);
    anari::commitParameters(d, f);
    // Initial Frame allocation already has a reset pending; consume it before
    // the explicit manual-version reset starts the measured accumulation.
    render();
    std::array<D3, 8> batch{};
    D3 previous{}, final{};
    for (unsigned b = 0; b < 8; ++b) {
      const auto accumulated = render();
      int samples = 0;
      require(anariGetProperty(d,
                  f,
                  "numSamples",
                  ANARI_INT32,
                  &samples,
                  sizeof(samples),
                  ANARI_WAIT),
          "sample count property");
      printf("batch %u samples %d\n", b, samples);
      require(samples == int((b + 1) * 512),
          "each batch must add 512 camera samples");
      for (unsigned ch = 0; ch < 3; ++ch) {
        batch[b][ch] = (b + 1) * accumulated[ch] - b * previous[ch];
        final[ch] += batch[b][ch] / 8;
      }
      previous = accumulated;
    }
    for (unsigned ch = 0; ch < 3; ++ch) {
      double variance = 0;
      for (auto value : batch)
        variance += std::pow(value[ch] - final[ch], 2) / 7;
      const double se = std::sqrt(variance / 8);
      const double error = std::abs(final[ch] - expected[ch]);
      printf("%s channel %u: mean %.9g reference %.9g SE %.9g (4096 spp)\n",
          label,
          ch,
          final[ch],
          expected[ch],
          se);
      require(se <= .0016, "inconclusive sampling uncertainty");
      require(error <= .01 && error <= 5 * se + .002,
          "statistical reference mismatch");
    }
    anari::unsetParameter(d, f, "accumulationVersion");
    anari::commitParameters(d, f);
  }
};
D3 homogeneous(const std::function<double(double, double)> &incident,
    double lo = 2,
    double hi = 6)
{
  D3 reference{};
  for (unsigned ch = 0; ch < 3; ++ch) {
    const double t = double(ABS[ch]) + SCATTER[ch];
    reference[ch] = quadrature(
        [&](double z) {
          return SCATTER[ch] / (4 * PI) * std::exp(-t * (z - lo))
              * incident(z, t);
        },
        lo,
        hi);
  }
  return reference;
}
void emptyLightsAndThinSource()
{
  Scene s;
  s.check("empty eligible set", {0, 0, 0});
  anari::setParameter(s.d, s.r, "background", V4{1, 1, 1, 1});
  anari::commitParameters(s.d, s.r);
  anari::unsetParameter(s.d, s.v, "stepSize");
  anari::commitParameters(s.d, s.v);
  s.check("empty set retains RGB extinction and zero-channel transmission",
      {std::exp(-4 * (double(ABS[0]) + SCATTER[0])),
          std::exp(-4 * (double(ABS[1]) + SCATTER[1])),
          1});
  anari::setParameter(s.d, s.r, "background", V4{0, 0, 0, 1});
  anari::commitParameters(s.d, s.r);
  anari::setParameter(s.d, s.m, "a", V3{0, 0, 0});
  anari::setParameter(s.d, s.m, "s", V3{1e-20f, 0, 0});
  anari::commitParameters(s.d, s.m);
  s.directional({0, 0, 1}, 1e20f);
  const double expected = double(1e-20f) * double(1e20f) / PI;
  s.check("expm1 source at near-zero extinction", {expected, 0, 0});
}

void coloredMultipleLights()
{
  Scene s;
  s.directional({0, 0, 1}, 1);
  s.directional(
      {0, 0, -1}, 3); // unequal pick powers and distinct incident paths
  auto reference = homogeneous([](double z, double t) {
    return std::exp(-t * (z - 2)) + 3 * std::exp(-t * (6 - z));
  });
  s.batches("two unequal-power lights", reference);
}
void visibilityAndSelfAttenuation()
{
  Scene s;
  s.directional({1, 0, 0}, 4);
  const auto reference =
      homogeneous([](double, double t) { return 4 * std::exp(-2 * t); });
  s.batches("side-lit box includes self attenuation", reference);
  s.extent({V3{-4, -2, 2}, V3{2, 2, 6}});
  const auto deeper =
      homogeneous([](double, double t) { return 4 * std::exp(-4 * t); });
  require(
      reference[0] - deeper[0] > .02, "shadow-depth fixture must discriminate");
  s.check("greater shadow optical depth", deeper);
  s.extent(BOX);
  auto geometry = anari::newObject<anari::Geometry>(s.d, "quad");
  const std::array<V3, 4> vertices{
      V3{-3, -3, 0}, V3{-3, -3, 8}, V3{-3, 3, 8}, V3{-3, 3, 0}};
  anari::setParameterArray1D(
      s.d, geometry, "vertex.position", vertices.data(), 4);
  anari::commitParameters(s.d, geometry);
  auto material = anari::newObject<anari::Material>(s.d, "matte");
  anari::commitParameters(s.d, material);
  auto surface = anari::newObject<anari::Surface>(s.d);
  anari::setParameter(s.d, surface, "geometry", geometry);
  anari::setParameter(s.d, surface, "material", material);
  anari::commitParameters(s.d, surface);
  anari::setParameterArray1D(s.d, s.w, "surface", &surface, 1);
  anari::commitParameters(s.d, s.w);
  s.check("opaque side blocker", {0, 0, 0}, 1e-6);
  for (auto o : {anari::Object(surface),
           anari::Object(material),
           anari::Object(geometry)})
    anari::release(s.d, o);
}
// The camera sees only the scattering box. A separate domain, with a unit
// vacuum gap, lies solely on the -X light path. Its analytical transmission
// multiplies the independent single-scattering quadrature, not primary alpha.
void secondShadowDomain(bool tf1d)
{
  Scene s;
  s.directional({1, 0, 0}, 4);
  const auto unoccluded =
      homogeneous([](double, double t) { return 4 * std::exp(-2 * t); });
  auto second =
      anari::newObject<anari::Volume>(s.d, tf1d ? "transferFunction1D" : "mdl");
  anari::Material material = nullptr;
  anari::SpatialField field = nullptr;
  if (tf1d) {
    field = anari::newObject<anari::SpatialField>(s.d, "structuredRegular");
    std::vector<float> values(3 * 5 * 5, .5f);
    anari::setParameterArray3D(s.d, field, "data", values.data(), 3, 5, 5);
    anari::setParameter(s.d, field, "origin", V3{-5, -2, 2});
    anari::setParameter(s.d, field, "spacing", V3{1, 1, 1});
    anari::commitParameters(s.d, field);
    anari::setParameter(s.d, second, "value", field);
    anari::setParameter(s.d, second, "color", V3{0, 0, 0});
    anari::setParameter(s.d, second, "opacity", .25f);
    anari::setParameter(s.d, second, "unitDistance", 1.f);
  } else {
    material = anari::newObject<anari::Material>(s.d, "mdl");
    anari::setParameter(s.d, material, "sourceType", "code");
    anari::setParameter(s.d, material, "source", HOMOGENEOUS);
    anari::setParameter(s.d, material, "a", V3{.4f, .2f, 0});
    anari::setParameter(s.d, material, "s", V3{0, 0, 0});
    anari::commitParameters(s.d, material);
    anari::setParameter(s.d, second, "material", material);
    const Box box{V3{-5, -2, 2}, V3{-3, 2, 6}};
    anariSetParameter(s.d, second, "extent", ANARI_FLOAT32_BOX3, box.data());
    anari::setParameter(s.d, second, "extinctionMajorant", .5f);
    anari::setParameter(s.d, second, "stepSize", .125f);
  }
  anari::commitParameters(s.d, second);
  const std::array<anari::Volume, 2> volumes{s.v, second};
  anari::setParameterArray1D(s.d, s.g, "volume", volumes.data(), 2);
  anari::commitParameters(s.d, s.g);
  D3 expected = unoccluded;
  for (unsigned ch = 0; ch < 3; ++ch) {
    const double transmission =
        tf1d ? .75 * .75 : std::exp(-2 * double(V3{.4f, .2f, 0}[ch]));
    expected[ch] *= transmission;
  }
  require(unoccluded[0] - expected[0] > .02,
      "second-domain fixture must detect omitted shadow attenuation");
  s.batches(tf1d ? "second TF1D shadow domain" : "second MDL shadow domain",
      expected);
  // Removing only the off-camera domain must restore the original lighting.
  anari::setParameterArray1D(s.d, s.g, "volume", &s.v, 1);
  anari::commitParameters(s.d, s.g);
  s.check(tf1d ? "remove TF1D shadow domain" : "remove MDL shadow domain",
      unoccluded);
  anari::release(s.d, second);
  if (material)
    anari::release(s.d, material);
  if (field)
    anari::release(s.d, field);
}

void finiteLightClipping()
{
  Scene s;
  auto light = s.addLight("point");
  anari::setParameter(s.d, light, "position", V3{-1, 0, 4});
  anari::setParameter(s.d, light, "intensity", 1.f);
  anari::commitParameters(s.d, light);
  s.publishLights();
  s.rates(4, 4, 512);
  const auto reference = homogeneous([](double z, double t) {
    const double distance = std::sqrt(1 + (z - 4) * (z - 4));
    return std::exp(-t * distance) / (distance * distance);
  });
  s.batches("shadow stops at point inside medium", reference);
}
void transformsAndClipping()
{
  Scene s;
  s.directional({0, 1, 0}, 4);
  // Rz(90) * scale(2,.75,1.5), translated in z. The light travels along
  // transformed object x; coefficients remain inverse WORLD distance.
  const std::array<float, 16> matrix{
      0, 2, 0, 0, -.75f, 0, 0, 0, 0, 0, 1.5f, 0, 0, 0, 1, 1};
  anariSetParameter(s.d, s.i, "transform", ANARI_FLOAT32_MAT4, matrix.data());
  anari::commitParameters(s.d, s.i);
  anari::setParameter(s.d, s.c, "near", 5.f);
  anari::setParameter(s.d, s.c, "far", 8.f);
  anari::commitParameters(s.d, s.c);
  const auto reference =
      homogeneous([](double, double t) { return 4 * std::exp(-4 * t); }, 5, 8);
  s.batches("nonuniform rotated Instance and primary clipping", reference);
}
void proceduralRefinement()
{
  Scene s;
  s.source(R"mdl(mdl 1.7;
import ::math::*; import ::state::*;
export material main() = let {
    float density=math::exp(0.5*(state::position().z-2)+0.5*state::position().x);
} in material(volume: material_volume(
    absorption_coefficient:color(0.012,0.02,0)*density,
    scattering_coefficient:color(0.028,0.01,0)*density));
)mdl");
  s.directional({1, 0, 0}, 4);
  for (bool transformed : {false, true}) {
    const double scaleX = transformed ? 2 : 1, scaleZ = transformed ? 1.5 : 1;
    const double lo = transformed ? 8. / 3 : 2, hi = transformed ? 14. / 3 : 6;
    if (transformed) {
      const std::array<float, 16> matrix{
          0, 2, 0, 0, -.75f, 0, 0, 0, 0, 0, 1.5f, 0, 0, 0, 1, 1};
      anariSetParameter(
          s.d, s.i, "transform", ANARI_FLOAT32_MAT4, matrix.data());
      anari::commitParameters(s.d, s.i);
      anari::setParameter(s.d, s.c, "near", 5.f);
      anari::setParameter(s.d, s.c, "far", 8.f);
      anari::commitParameters(s.d, s.c);
      anari::setParameter(s.d, s.lights[0], "direction", V3{0, 1, 0});
      anari::commitParameters(s.d, s.lights[0]);
    }
    D3 reference{};
    for (unsigned ch = 0; ch < 2; ++ch) {
      const double a = ch == 0 ? double(.012f) : double(.02f);
      const double scatter = ch == 0 ? double(.028f) : double(.01f);
      const double t = a + scatter;
      reference[ch] = quadrature(
          [&](double z) {
            const double density = std::exp(.5 * (z - 2));
            const double cameraDepth =
                scaleZ * t * 2 * (density - std::exp(.5 * (lo - 2)));
            const double lightDepth =
                scaleX * t * density * 2 * (1 - std::exp(-1.));
            return scaleZ * 4 * scatter * density / (4 * PI)
                * std::exp(-cameraDepth - lightDepth);
          },
          lo,
          hi);
    }
    // h=.125 is <=1/8 of the shortest controlled density/optical scale.
    // One eligible delta light and negligible xy jitter make refinement
    // deterministic; one spp avoids FP32 accumulation hiding small errors.
    for (unsigned mode = 0; mode < 3; ++mode) {
      double previous = 1;
      for (float rate : {1.f, 2.f, 4.f}) {
        s.rates(mode == 1 ? 4 : rate, mode == 0 ? 4 : rate, 1);
        const auto value = s.render();
        const double error = std::abs(value[0] - reference[0]);
        printf(
            "procedural transformed %d mode %u rate %g: %.9g reference %.9g error %.9g\n",
            transformed,
            mode,
            rate,
            value[0],
            reference[0],
            error);
        require(error < previous * .8 + 2e-8,
            "primary/shadow refinement must converge");
        previous = error;
        for (unsigned ch = 0; ch < 3; ++ch)
          require(std::abs(value[ch] - reference[ch]) < .01,
              "procedural RGB reference mismatch");
      }
      require(previous < .0005, "finest refinement below reference allowance");
    }
  }
}
} // namespace
int main()
{
  try {
    emptyLightsAndThinSource();
    bool domainsPassed = true;
    for (bool tf1d : {false, true}) {
      try {
        secondShadowDomain(tf1d);
      } catch (const std::exception &e) {
        fprintf(
            stderr, "%s shadow domain: %s\n", tf1d ? "TF1D" : "MDL", e.what());
        domainsPassed = false;
      }
    }
    require(domainsPassed, "second-domain shadow acceptance failed");
    coloredMultipleLights();
    visibilityAndSelfAttenuation();
    finiteLightClipping();
    transformsAndClipping();
    proceduralRefinement();
  } catch (const std::exception &e) {
    fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
