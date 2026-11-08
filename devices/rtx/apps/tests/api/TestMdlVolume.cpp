// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <anari/anari_cpp/ext/std.h>
#include <anari/ext/visrtx/makeVisRTXDevice.h>
#include <anari/anari_cpp.hpp>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using V3 = std::array<float, 3>;
using V4 = std::array<float, 4>;
using Box = std::array<V3, 2>;
using Matrix = std::array<float, 16>;
constexpr Box EXTENT{V3{-2, -2, 2}, V3{2, 2, 6}};
constexpr Matrix IDENTITY{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
const char *PARAMETRIC = R"mdl(mdl 1.7;
export material main(uniform color absorption = color(0.25, 0.5, 0),
    uniform color scattering = color(0), uniform color emission = color(0)) =
    material(volume: material_volume(absorption_coefficient: absorption,
        scattering_coefficient: scattering, emission_intensity: emission));
)mdl";
void require(bool ok, const char *why)
{
  if (!ok)
    throw std::runtime_error(why);
}
struct Log
{
  ANARIObject frame{};
  unsigned frameErrors{};
  std::string cause;
};
void status(const void *user,
    ANARIDevice,
    ANARIObject source,
    ANARIDataType,
    ANARIStatusSeverity severity,
    ANARIStatusCode,
    const char *message)
{
  auto &log = *const_cast<Log *>(static_cast<const Log *>(user));
  if (severity <= ANARI_SEVERITY_ERROR) {
    fprintf(stderr, "%s\n", message);
    if (source == log.frame) {
      ++log.frameErrors;
      log.cause = message;
    }
  }
}
struct Scene
{
  Log log;
  anari::Device d = makeVisRTXDevice(status, &log);
  anari::Material material = anari::newObject<anari::Material>(d, "mdl");
  anari::Volume volume = anari::newObject<anari::Volume>(d, "mdl");
  anari::Group group = anari::newObject<anari::Group>(d);
  anari::Instance instance = anari::newObject<anari::Instance>(d, "transform");
  anari::World world = anari::newObject<anari::World>(d);
  anari::Camera camera = anari::newObject<anari::Camera>(d, "orthographic");
  anari::Renderer renderer =
      anari::newObject<anari::Renderer>(d, "interactive");
  anari::Frame frame = anari::newObject<anari::Frame>(d);
  Scene()
  {
    log.frame = frame;
    source(PARAMETRIC);
    anari::setParameter(d, volume, "material", material);
    extent(EXTENT);
    majorant(0.5f);
    anari::setParameter(d, volume, "id", 71u);
    anari::commitParameters(d, volume);
    anari::setParameterArray1D(d, group, "volume", &volume, 1);
    anari::commitParameters(d, group);
    anari::setParameter(d, instance, "group", group);
    anari::setParameter(d, instance, "id", 93u);
    anari::commitParameters(d, instance);
    anari::setParameterArray1D(d, world, "instance", &instance, 1);
    anari::commitParameters(d, world);
    anari::setParameter(d, camera, "position", V3{0, 0, 0});
    anari::setParameter(d, camera, "direction", V3{0, 0, 1});
    anari::setParameter(d, camera, "up", V3{0, 1, 0});
    anari::setParameter(d, camera, "height", 0.5f);
    anari::commitParameters(d, camera);
    anari::setParameter(d, renderer, "denoise", false);
    anari::setParameter(d, renderer, "fireflyFilterMode", "none");
    anari::setParameter(d, renderer, "ambientRadiance", 0.f);
    background({1, 1, 1, 1});
    anari::setParameter(d, renderer, "pixelSamples", 4096);
    anari::commitParameters(d, renderer);
    anari::setParameter(d, frame, "size", std::array<unsigned, 2>{8, 8});
    anari::setParameter(d, frame, "channel.color", ANARI_FLOAT32_VEC4);
    anari::setParameter(d, frame, "channel.normal", ANARI_FLOAT32_VEC3);
    anari::setParameter(d, frame, "channel.albedo", ANARI_FLOAT32_VEC3);
    anari::setParameter(d, frame, "channel.depth", ANARI_FLOAT32);
    for (const char *name :
        {"channel.primitiveId", "channel.objectId", "channel.instanceId"})
      anari::setParameter(d, frame, name, ANARI_UINT32);
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
  void source(const char *code)
  {
    anari::setParameter(d, material, "sourceType", "code");
    anari::setParameter(d, material, "source", code);
    anari::commitParameters(d, material);
  }
  void coefficients(V3 a, V3 s = {0, 0, 0}, V3 e = {0, 0, 0})
  {
    anari::setParameter(d, material, "absorption", a);
    anari::setParameter(d, material, "scattering", s);
    anari::setParameter(d, material, "emission", e);
    anari::commitParameters(d, material);
  }
  void extent(Box box)
  {
    anariSetParameter(d, volume, "extent", ANARI_FLOAT32_BOX3, box.data());
    anari::commitParameters(d, volume);
  }
  void majorant(float value)
  {
    anari::setParameter(d, volume, "extinctionMajorant", value);
    anari::commitParameters(d, volume);
  }
  void background(V4 value, bool premultiply = true)
  {
    anari::setParameter(d, renderer, "background", value);
    anari::setParameter(d, renderer, "premultiplyBackground", premultiply);
    anari::commitParameters(d, renderer);
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
  void check(const char *label, V4 expected, float tolerance = 0.002f)
  {
    const auto before = log.frameErrors;
    render();
    require(log.frameErrors == before, "unexpected Frame error");
    unsigned w = 0, h = 0;
    anari::DataType type = ANARI_UNKNOWN;
    auto p = static_cast<const V4 *>(
        anariMapFrame(d, frame, "channel.color", &w, &h, &type));
    require(p && w == 8 && h == 8 && type == ANARI_FLOAT32_VEC4,
        "successful color map");
    for (unsigned i = 0; i < 64; i++)
      for (unsigned c = 0; c < 4; c++) {
        if (!std::isfinite(p[i][c])
            || std::abs(p[i][c] - expected[c]) > tolerance) {
          fprintf(stderr,
              "%s channel %u: got %.9g expected %.9g\n",
              label,
              c,
              p[i][c],
              expected[c]);
          throw std::runtime_error("analytic reference mismatch");
        }
      }
    anari::unmap(d, frame, "channel.color");
    printf("%s: ok\n", label);
  }
  void fail(const char *cause)
  {
    const auto before = log.frameErrors;
    render();
    require(log.frameErrors > before,
        "failed attempt must report a Frame-sourced error");
    require(log.cause.find(cause) != std::string::npos,
        "wrong Frame error category");
    int count = -1;
    require(anariGetProperty(d,
                frame,
                "numSamples",
                ANARI_INT32,
                &count,
                sizeof(count),
                ANARI_WAIT)
            && count == 0,
        "failed accumulation must have zero samples");
    for (const char *channel : {"channel.color",
             "channel.depth",
             "channel.normal",
             "channel.albedo",
             "channel.primitiveId",
             "channel.objectId",
             "channel.instanceId",
             "channel.colorCUDA",
             "channel.depthCUDA",
             "channel.normalCUDA",
             "channel.albedoCUDA",
             "channel.primitiveIdCUDA",
             "channel.objectIdCUDA",
             "channel.instanceIdCUDA"}) {
      unsigned w = 9, h = 9;
      ANARIDataType type = ANARI_FLOAT32;
      require(!anariMapFrame(d, frame, channel, &w, &h, &type) && w == 0
              && h == 0 && type == ANARI_UNKNOWN,
          "failed channel must be null/zero/UNKNOWN");
    }
    printf("expected failure: %s\n", cause);
  }
};

V4 beer(V3 a, double length, V4 bg = {1, 1, 1, 1}, bool premultiply = true)
{
  // Independent homogeneous solution, in world distance. Alpha is tested
  // separately from radiance and preserves chromatic uncovered fractions.
  V4 result{};
  double mean = 0.;
  for (unsigned c = 0; c < 3; c++) {
    const double t = std::exp(-double(a[c]) * length);
    result[c] = float(t * bg[c] * (premultiply ? bg[3] : 1.f));
    mean += t / 3.;
  }
  result[3] = float(1. - mean * (1. - bg[3]));
  return result;
}

void backgroundsAndVacuum(Scene &s)
{
  const V3 a{0.25f, 0.5f, 0};
  for (bool premultiply : {false, true})
    for (float alpha : {0.f, 0.35f, 1.f}) {
      const V4 bg{0.8f, 0.6f, 0.4f, alpha};
      s.background(bg, premultiply);
      s.check("chromatic screen background / coverage",
          beer(a, 4, bg, premultiply));
    }
  s.coefficients({0, 0, 0});
  s.majorant(0.f);
  s.background({0.8f, 0.6f, 0.4f, 0.35f}, false);
  s.check("zero-bound vacuum", {0.8f, 0.6f, 0.4f, 0.35f});
  s.majorant(0.5f);
  s.check("positive-bound vacuum", {0.8f, 0.6f, 0.4f, 0.35f});
  s.coefficients(a);
  s.background({1, 1, 1, 1});

  const std::array<V3, 4> texels{V3{0.8f, 0.6f, 0.4f},
      V3{0.8f, 0.6f, 0.4f},
      V3{0.8f, 0.6f, 0.4f},
      V3{0.8f, 0.6f, 0.4f}};
  auto light = anari::newObject<anari::Light>(s.d, "hdri");
  auto image = anari::newArray2D(s.d, texels.data(), 2, 2);
  anari::setParameter(s.d, light, "radiance", image);
  anari::commitParameters(s.d, light);
  anari::setParameterArray1D(s.d, s.world, "light", &light, 1);
  anari::commitParameters(s.d, s.world);
  s.check("chromatic HDRI background", beer(a, 4, {0.8f, 0.6f, 0.4f, 1}));
  anari::unsetParameter(s.d, s.world, "light");
  anari::commitParameters(s.d, s.world);
  anari::release(s.d, image);
  anari::release(s.d, light);
}

void surfaceBackground(Scene &s)
{
  auto geometry = anari::newObject<anari::Geometry>(s.d, "quad");
  auto emitter = anari::newObject<anari::Material>(s.d, "mdl");
  auto surface = anari::newObject<anari::Surface>(s.d);
  anari::setParameter(s.d, emitter, "sourceType", "code");
  anari::setParameter(s.d, emitter, "source", R"mdl(mdl 1.7;
import ::df::*;
export material main() = material(surface: material_surface(
    emission: material_emission(emission: df::diffuse_edf(),
        intensity: color(0.8,0.6,0.4) * 3.141592653589793)));
)mdl");
  anari::commitParameters(s.d, emitter);
  auto place = [&](float z) {
    const std::array<V3, 4> vertices{
        V3{-2, -2, z}, V3{-2, 2, z}, V3{2, 2, z}, V3{2, -2, z}};
    anari::setParameterArray1D(
        s.d, geometry, "vertex.position", vertices.data(), 4);
    anari::commitParameters(s.d, geometry);
  };
  place(8.f);
  anari::setParameter(s.d, surface, "geometry", geometry);
  anari::setParameter(s.d, surface, "material", emitter);
  anari::commitParameters(s.d, surface);
  anari::setParameterArray1D(s.d, s.world, "surface", &surface, 1);
  anari::commitParameters(s.d, s.world);
  s.check("chromatic attenuation reaches emissive surface once",
      beer({.25f, .5f, 0}, 4, {.8f, .6f, .4f, 1}));
  place(4.f);
  s.check("opaque surface clips medium interval",
      beer({.25f, .5f, 0}, 2, {.8f, .6f, .4f, 1}));
  anari::unsetParameter(s.d, s.world, "surface");
  anari::commitParameters(s.d, s.world);
  anari::release(s.d, surface);
  anari::release(s.d, emitter);
  anari::release(s.d, geometry);
}

void reflectedBackground(Scene &s)
{
  auto geometry = anari::newObject<anari::Geometry>(s.d, "quad");
  const std::array<V3, 4> vertices{
      V3{-2, -2, -1}, V3{-2, 2, -1}, V3{2, 2, 3}, V3{2, -2, 3}};
  anari::setParameterArray1D(
      s.d, geometry, "vertex.position", vertices.data(), 4);
  anari::commitParameters(s.d, geometry);
  auto material = anari::newObject<anari::Material>(s.d, "mdl");
  anari::setParameter(s.d, material, "sourceType", "code");
  anari::setParameter(s.d, material, "source", R"mdl(mdl 1.7;
import ::df::*;
export material main() = material(surface: material_surface(
    scattering: df::specular_bsdf(tint: color(1),mode: df::scatter_reflect)));
)mdl");
  anari::commitParameters(s.d, material);
  auto surface = anari::newObject<anari::Surface>(s.d);
  anari::setParameter(s.d, surface, "geometry", geometry);
  anari::setParameter(s.d, surface, "material", material);
  anari::commitParameters(s.d, surface);
  auto light = anari::newObject<anari::Light>(s.d, "hdri");
  const V3 texel{1, 1, 1};
  auto image = anari::newArray2D(s.d, &texel, 1, 1);
  anari::setParameter(s.d, light, "radiance", image);
  anari::commitParameters(s.d, light);
  anari::setParameterArray1D(s.d, s.world, "surface", &surface, 1);
  anari::setParameterArray1D(s.d, s.world, "light", &light, 1);
  anari::commitParameters(s.d, s.world);
  s.extent({V3{2, -2, -2}, V3{6, 2, 2}});
  s.check("mirror continuation sees chromatic medium before HDRI",
      beer({.25f, .5f, 0}, 4));
  s.extent(EXTENT);
  anari::unsetParameter(s.d, s.world, "surface");
  anari::unsetParameter(s.d, s.world, "light");
  anari::commitParameters(s.d, s.world);
  for (auto o : {anari::Object(image),
           anari::Object(light),
           anari::Object(surface),
           anari::Object(material),
           anari::Object(geometry)})
    anari::release(s.d, o);
}

void surfaceShadow(Scene &s)
{
  // The tilted receiver is in front of the medium. Only its light ray crosses
  // the box; a vacuum control isolates the source from analytic transmission.
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
  s.coefficients({0, 0, 0});
  s.majorant(0.f);
  s.render();
  const auto mapped = anari::map<V4>(s.d, s.frame, "channel.color");
  require(mapped.data != nullptr, "vacuum shadow reference must map");
  const auto control = mapped.data[0];
  anari::unmap(s.d, s.frame, "channel.color");
  require(control[0] > .05f && control[1] > .05f && control[2] > .05f,
      "shadow control must be nonzero");
  s.coefficients({.25f, .5f, 0});
  s.majorant(.5f);
  for (float rate : {.25f, 1.f, 4.f}) {
    anari::setParameter(s.d, s.renderer, "volumeSamplingRateShadows", rate);
    anari::commitParameters(s.d, s.renderer);
    s.check("RGB surface-light shadow transmission",
        beer({.25f, .5f, 0}, 4 * std::sqrt(1.25), control, false));
  }
  s.extent(EXTENT);
  anari::unsetParameter(s.d, s.world, "surface");
  anari::unsetParameter(s.d, s.world, "light");
  anari::commitParameters(s.d, s.world);
  for (auto o : {anari::Object(light),
           anari::Object(surface),
           anari::Object(material),
           anari::Object(geometry)})
    anari::release(s.d, o);
}

void transformsAndClipping(Scene &s)
{
  const V3 a{0.25f, 0.5f, 0};
  Matrix matrix = IDENTITY;
  matrix[0] = 2.f;
  matrix[5] = 0.75f;
  matrix[10] = 1.5f;
  matrix[14] = 1.f;
  s.transform(matrix);
  s.check("nonuniform Instance scale and translation", beer(a, 6));
  // Rotate about the viewing axis while retaining the thickness and footprint.
  matrix[0] = 0.f;
  matrix[1] = 2.f;
  matrix[4] = -0.75f;
  matrix[5] = 0.f;
  s.transform(matrix);
  s.check("rotated nonuniform Instance", beer(a, 6));
  s.transform(IDENTITY);
  matrix = IDENTITY;
  matrix[0] = 0.f;
  s.transform(matrix);
  s.fail("transform");
  matrix = IDENTITY;
  matrix[10] = std::numeric_limits<float>::infinity();
  s.transform(matrix);
  s.fail("transform");
  s.transform(IDENTITY);
  anari::setParameter(s.d, s.camera, "near", 3.f);
  anari::setParameter(s.d, s.camera, "far", 5.f);
  anari::commitParameters(s.d, s.camera);
  s.check("near/far clipped interval", beer(a, 2));
  anari::unsetParameter(s.d, s.camera, "near");
  anari::unsetParameter(s.d, s.camera, "far");
  anari::commitParameters(s.d, s.camera);
  anari::setParameter(s.d, s.volume, "stepSize", 0.7f);
  anari::commitParameters(s.d, s.volume);
  s.check("explicit step with clipped last segment", beer(a, 4));
  anari::unsetParameter(s.d, s.volume, "stepSize");
  anari::commitParameters(s.d, s.volume);
}

void errorsAndRecovery(Scene &s)
{
  const V3 valid{0.25f, 0.5f, 0};
  for (bool manual : {false, true}) {
    if (manual)
      anari::setParameter(s.d, s.frame, "accumulationVersion", 17u);
    else
      anari::unsetParameter(s.d, s.frame, "accumulationVersion");
    anari::commitParameters(s.d, s.frame);
    s.check("valid accumulation", beer(valid, 4));
    s.coefficients({0.25f, 0.75f, 0});
    s.fail("majorant violation");
    s.coefficients(valid);
    s.check(
        "recovery without Volume recommit / version increment", beer(valid, 4));
  }
  anari::unsetParameter(s.d, s.frame, "accumulationVersion");
  anari::commitParameters(s.d, s.frame);
  s.majorant(1.f);
  s.coefficients({1, 0, 0});
  s.check("majorant equality", beer({1, 0, 0}, 4));
  for (float small : {std::ldexp(1.f, -25), std::ldexp(1.f, -80)}) {
    s.coefficients({1, 0, 0}, {small, 0, 0});
    s.fail("majorant violation");
    s.coefficients({small, 0, 0}, {1, 0, 0});
    s.fail("majorant violation");
    s.majorant(std::nextafter(1.f, 2.f));
    // With no eligible lights, scattering contributes extinction only.
    s.check("valid widened bound with scattering", beer({1, 0, 0}, 4));
    s.majorant(1.f);
  }
  s.coefficients({0.5f, 0, 0}, {0.5f, 0, 0});
  s.check("equal absorption/scattering sum", beer({1, 0, 0}, 4));
  s.coefficients({1, 0, 0});
  s.majorant(std::nextafter(1.f, 0.f));
  s.fail("majorant violation");
  const float maximum = std::numeric_limits<float>::max();
  s.majorant(maximum);
  s.coefficients({maximum, 0, 0}, {maximum, 0, 0});
  s.fail("majorant violation");
  s.majorant(1.f);
  for (float invalid : {-0.1f,
           std::numeric_limits<float>::infinity(),
           std::numeric_limits<float>::quiet_NaN()}) {
    s.coefficients({invalid, 0, 0});
    s.fail("coefficient");
  }
  s.coefficients({0, 0, 0}, {0, 0, 0}, {0.1f, 0, 0});
  s.fail("unsupported medium");
  s.majorant(std::ldexp(1.f, -80));
  s.coefficients({std::ldexp(1.f, -80), 0, 0});
  s.check("small positive majorant equality", {1, 1, 1, 1});
  s.majorant(100.f);
  s.coefficients({100.f, 0.5f, 0});
  s.check("thick chromatic absorption keeps zero channel transparent",
      beer({100.f, .5f, 0}, 4));
  s.coefficients(valid);
  s.majorant(0.5f);
  s.check("fresh output after numerical failures", beer(valid, 4));
  anari::setParameter(
      s.d, s.volume, "stepSize", std::numeric_limits<float>::min());
  anari::commitParameters(s.d, s.volume);
  s.fail("numerical failure");
  anari::unsetParameter(s.d, s.volume, "stepSize");
  anari::commitParameters(s.d, s.volume);
  s.check("recover from a nonadvancing marching step", beer(valid, 4));
}

void compilationAndState(Scene &s)
{
  s.source(R"mdl(mdl 1.7;
import ::df::*;
import ::state::*;
export material main() = material(
    surface: material_surface(scattering: df::diffuse_reflection_bsdf(
        tint: color(state::texture_coordinate(0)))),
    geometry: material_geometry(normal: state::texture_coordinate(0)),
    volume: material_volume(absorption_coefficient: color(0.25,0.5,0)));
)mdl");
  s.check("constant bundle ignores surface geometry.normal / UV dependencies",
      beer({.25f, .5f, 0}, 4));
  auto sampler = anari::newObject<anari::Sampler>(s.d, "image2D");
  const V3 texel{.5f, .5f, .5f};
  auto image = anari::newArray2D(s.d, &texel, 1, 1);
  anari::setParameter(s.d, sampler, "image", image);
  anari::commitParameters(s.d, sampler);
  s.source(R"mdl(mdl 1.7;
import ::df::*; import ::tex::*; import ::state::*;
export material main(uniform texture_2d tex = texture_2d()) = material(
    surface: material_surface(scattering: df::diffuse_reflection_bsdf(
        tint: tex::lookup_color(tex,float2(state::texture_coordinate(0).x,state::texture_coordinate(0).y)))),
    volume: material_volume(absorption_coefficient: color(0.25,0.5,0)));
)mdl");
  anari::setParameter(s.d, s.material, "tex", sampler);
  anari::commitParameters(s.d, s.material);
  s.check("surface-only texture argument does not restrict volume bundle",
      beer({.25f, .5f, 0}, 4));
  s.source(R"mdl(mdl 1.7;
import ::tex::*;
export material main(uniform texture_2d tex = texture_2d()) = material(
    volume: material_volume(absorption_coefficient: tex::lookup_color(tex,float2(0))));
)mdl");
  s.fail("unsupported medium");
  anari::unsetParameter(s.d, s.material, "tex");
  anari::commitParameters(s.d, s.material);
  anari::release(s.d, sampler);
  anari::release(s.d, image);
  s.source(R"mdl(mdl 1.7;
import ::state::*;
export material main() = material(volume: material_volume(
    absorption_coefficient: color(0.125 * state::position().z,
        state::normal().x * state::normal().x,
        state::geometry_normal().y * state::geometry_normal().y)));
)mdl");
  s.majorant(.75f);
  s.check("procedural object position and zero volume normals",
      {float(std::exp(-2.)), 1, 1, 1});
  Matrix xfm = IDENTITY;
  xfm[10] = 1.5f;
  xfm[14] = 1.f;
  s.transform(xfm);
  s.check("position follows object; attenuation follows world distance",
      {float(std::exp(-3.)), 1, 1, 1});
  s.transform(IDENTITY);
  s.source(R"mdl(mdl 1.7;
import ::state::*;
export material main() = material(volume: material_volume(
    absorption_coefficient: color(state::texture_coordinate(0))));
)mdl");
  s.fail("unsupported MDL volume state");
  s.source(R"mdl(mdl 1.7;
import ::df::*;
export material main() = material(volume: material_volume(
    scattering: df::anisotropic_vdf(directional_bias: 0.25)));
)mdl");
  s.fail("unsupported VDF expression");
  s.source(R"mdl(mdl 1.7;
import ::scene::*;
export material main() = material(volume: material_volume(
    absorption_coefficient: color(scene::data_isvalid("retainedMetadata") ? 1 : 0.25)));
)mdl");
  s.check("field-free data_isvalid is false", beer({.25f, .25f, .25f}, 4));
  s.source(R"mdl(mdl 1.7;
import ::scene::*;
export material main() = material(volume: material_volume(
    absorption_coefficient: color(scene::data_lookup_float("unbound",0.25))));
)mdl");
  s.fail("scene-data lookup");
  s.source("mdl 1.7; this is not a material");
  s.fail("invalid MDL material source");
  s.source(PARAMETRIC);
  s.majorant(.5f);
  s.check("source replacement restores independent Argument Block",
      beer({.25f, .5f, 0}, 4));
}

void marchingConvergence(Scene &s)
{
  s.source(R"mdl(mdl 1.7;
import ::math::*; import ::state::*;
export material main() = material(volume: material_volume(
    absorption_coefficient: color(0.1 * math::exp(0.5 * state::position().z),0,0)));
)mdl");
  s.majorant(2.1f);
  // exp(z/2) varies on length scale 2. Default h=4/128, with rates
  // 1/8,1/4,1/2, gives world segments 1/4,1/8,1/16: coarse <= scale/8.
  // This fixture is deterministic (xy jitter cannot change the integral), so
  // use one sample to avoid accumulation roundoff masking the refinement.
  anari::setParameter(s.d, s.renderer, "pixelSamples", 1);
  const double reference = std::exp(-0.2 * (std::exp(3.) - std::exp(1.)));
  for (bool explicitStep : {false, true}) {
    if (explicitStep) {
      anari::setParameter(s.d, s.volume, "stepSize", .25f);
      anari::commitParameters(s.d, s.volume);
    }
    double previous = 1.;
    for (float rate : {.125f, .25f, .5f}) {
      const float actualRate = explicitStep ? rate * 8 : rate;
      anari::setParameter(s.d, s.renderer, "volumeSamplingRate", actualRate);
      anari::commitParameters(s.d, s.renderer);
      s.check("procedural marching refinement", {float(reference), 1, 1, 1});
      const auto mapped = anari::map<V4>(s.d, s.frame, "channel.color");
      const double error = std::abs(mapped.data[0][0] - reference);
      anari::unmap(s.d, s.frame, "channel.color");
      printf("explicit step %d, primary rate %g: analytic error %.9g\n",
          explicitStep,
          actualRate,
          error);
      require(error < previous * .4,
          "midpoint refinement must reduce deterministic error");
      previous = error;
    }
    require(previous < .0005,
        "finest marching error must be below reference allowance");
  }
  anari::unsetParameter(s.d, s.volume, "stepSize");
  anari::commitParameters(s.d, s.volume);
  anari::setParameter(s.d, s.renderer, "pixelSamples", 4096);
  anari::unsetParameter(s.d, s.renderer, "volumeSamplingRate");
  anari::commitParameters(s.d, s.renderer);
  s.source(PARAMETRIC);
  s.majorant(.5f);
}

void rendererRejection(Scene &s)
{
  for (const char *type : {"fast", "quality", "debug", "test"}) {
    auto renderer = anari::newObject<anari::Renderer>(s.d, type);
    anari::commitParameters(s.d, renderer);
    anari::setParameter(s.d, s.frame, "renderer", renderer);
    anari::commitParameters(s.d, s.frame);
    s.fail("interactive Renderer");
    anari::release(s.d, renderer);
  }
  anari::setParameter(s.d, s.frame, "renderer", s.renderer);
  anari::commitParameters(s.d, s.frame);
  s.check(
      "restore interactive after renderer rejection", beer({.25f, .5f, 0}, 4));
}

void configuration(Scene &s)
{
  for (float invalid : {-1.f,
           std::numeric_limits<float>::infinity(),
           std::numeric_limits<float>::quiet_NaN()}) {
    s.majorant(invalid);
    s.fail("extinctionMajorant");
  }
  anari::unsetParameter(s.d, s.volume, "extinctionMajorant");
  anari::commitParameters(s.d, s.volume);
  s.fail("extinctionMajorant");
  anari::setParameter(s.d, s.volume, "extinctionMajorant", V3{1, 1, 1});
  anari::commitParameters(s.d, s.volume);
  s.fail("extinctionMajorant");
  s.majorant(0.5f);
  anari::unsetParameter(s.d, s.volume, "extent");
  anari::commitParameters(s.d, s.volume);
  s.fail("extent");
  constexpr float maximumFloat = std::numeric_limits<float>::max();
  for (Box invalid :
      {Box{V3{0, 0, 0}, V3{1, 1, 0}},
          Box{V3{2, 0, 0}, V3{1, 1, 1}},
          Box{V3{-maximumFloat, 0, 0}, V3{maximumFloat, 1, 1}},
          Box{V3{0, 0, 0}, V3{1, 1, std::numeric_limits<float>::infinity()}}}) {
    s.extent(invalid);
    s.fail("extent");
  }
  anari::setParameter(s.d, s.volume, "extent", V3{1, 1, 1});
  anari::commitParameters(s.d, s.volume);
  s.fail("extent");
  s.extent(EXTENT);
  for (float invalid : {0.f,
           -1.f,
           std::numeric_limits<float>::infinity(),
           std::numeric_limits<float>::quiet_NaN()}) {
    anari::setParameter(s.d, s.volume, "stepSize", invalid);
    anari::commitParameters(s.d, s.volume);
    s.fail("stepSize");
  }
  anari::setParameter(s.d, s.volume, "stepSize", V3{1, 1, 1});
  anari::commitParameters(s.d, s.volume);
  s.fail("stepSize");
  anari::unsetParameter(s.d, s.volume, "stepSize");
  anari::unsetParameter(s.d, s.volume, "material");
  anari::commitParameters(s.d, s.volume);
  s.fail("Material");
  auto matte = anari::newObject<anari::Material>(s.d, "matte");
  anari::commitParameters(s.d, matte);
  anari::setParameter(s.d, s.volume, "material", matte);
  anari::commitParameters(s.d, s.volume);
  s.fail("Material");
  anari::setParameter(s.d, s.volume, "material", s.material);
  anari::setParameter(s.d, s.volume, "fieldName", "retainedMetadata");
  anari::commitParameters(s.d, s.volume);
  s.check("retained name without field / recovered configuration",
      beer({.25f, .5f, 0}, 4));
  anari::setParameter(s.d, s.volume, "value", anari::SpatialField(nullptr));
  anari::commitParameters(s.d, s.volume);
  s.check("explicit null field is absent", beer({.25f, .5f, 0}, 4));
  Log otherLog;
  auto otherDevice = makeVisRTXDevice(status, &otherLog);
  auto foreignMaterial = anari::newObject<anari::Material>(otherDevice, "mdl");
  anari::commitParameters(otherDevice, foreignMaterial);
  anari::setParameter(s.d, s.volume, "material", foreignMaterial);
  anari::commitParameters(s.d, s.volume);
  s.fail("same-device");
  anari::setParameter(s.d, s.volume, "material", s.material);
  anari::commitParameters(s.d, s.volume);
  s.check("restore same-device material", beer({.25f, .5f, 0}, 4));
  anari::release(otherDevice, foreignMaterial);
  anari::release(otherDevice, otherDevice);
  anari::release(s.d, matte);
}
} // namespace
int main()
{
  try {
    Scene s;
    s.check("homogeneous RGB absorption", beer({.25f, .5f, 0}, 4));
    const auto normals = anari::map<V3>(s.d, s.frame, "channel.normal");
    require(normals.data != nullptr, "volume normals must map");
    for (unsigned i = 0; i < 64; ++i)
      for (float c : normals.data[i])
        require(c == 0.f, "isolated volume normal must be zero");
    anari::unmap(s.d, s.frame, "channel.normal");
    const auto ids = anari::map<unsigned>(s.d, s.frame, "channel.objectId");
    require(ids.data && ids.data[0] == 71u, "Volume ID must survive traversal");
    anari::unmap(s.d, s.frame, "channel.objectId");
    backgroundsAndVacuum(s);
    surfaceBackground(s);
    reflectedBackground(s);
    surfaceShadow(s);
    transformsAndClipping(s);
    errorsAndRecovery(s);
    configuration(s);
    compilationAndState(s);
    marchingConvergence(s);
    rendererRejection(s);
    for (const char **p = anariGetObjectSubtypes(s.d, ANARI_VOLUME); p && *p;
        ++p)
      require(std::strcmp(*p, "mdl") != 0,
          "experimental subtype must remain unadvertised");
  } catch (const std::exception &e) {
    fprintf(stderr, "%s\n", e.what());
    return 1;
  }
  return 0;
}
