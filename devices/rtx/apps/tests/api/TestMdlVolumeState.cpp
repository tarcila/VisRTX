// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <anari/anari_cpp/ext/std.h>
#include <anari/ext/visrtx/makeVisRTXDevice.h>
#include <anari/anari_cpp.hpp>
#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <limits>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stdexcept>
#include <string>
#include "stb_image_write.h"

namespace {
using V3 = std::array<float, 3>;
using V4 = std::array<float, 4>;
void require(bool ok, const char *why)
{
  if (!ok)
    throw std::runtime_error(why);
}
struct Scene
{
  ANARIObject failedFrame{};
  std::string error;
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
      if (source == s.failedFrame)
        s.error = message;
    }
  }
  anari::Device d = makeVisRTXDevice(status, this);
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
    failedFrame = frame;
    anari::setParameter(d, volume, "material", material);
    const std::array<V3, 2> box{V3{-2, -2, 2}, V3{2, 2, 6}};
    anariSetParameter(d, volume, "extent", ANARI_FLOAT32_BOX3, box.data());
    anari::setParameter(d, volume, "extinctionMajorant", 2.f);
    anari::setParameter(d, volume, "id", 0x80000047u);
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
    anari::setParameter(d, camera, "height", .5f);
    anari::commitParameters(d, camera);
    anari::setParameter(d, renderer, "denoise", false);
    anari::setParameter(d, renderer, "fireflyFilterMode", "none");
    anari::setParameter(d, renderer, "ambientRadiance", 0.f);
    anari::setParameter(d, renderer, "background", V4{1, 1, 1, 1});
    // Deterministic optical paths; no stochastic-transport claim.
    anari::setParameter(d, renderer, "pixelSamples", 1);
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
  void render()
  {
    error.clear();
    anari::render(d, frame);
    require(anariFrameReady(d, frame, ANARI_WAIT), "WAIT completes");
  }
  void check(const char *label, V3 opticalDepth)
  {
    render();
    require(error.empty(), "unexpected Frame error");
    auto image = anari::map<V4>(d, frame, "channel.color");
    require(image.data && image.width == 8 && image.height == 8, "color map");
    for (unsigned i = 0; i < 64; ++i)
      for (unsigned c = 0; c < 4; ++c) {
        const float expected = c == 3 ? 1.f : std::exp(-opticalDepth[c]);
        if (!std::isfinite(image.data[i][c])
            || std::abs(image.data[i][c] - expected) > .002f) {
          fprintf(stderr,
              "%s channel %u: %.9g expected %.9g\n",
              label,
              c,
              image.data[i][c],
              expected);
          throw std::runtime_error("independent optical depth mismatch");
        }
      }
    anari::unmap(d, frame, "channel.color");
    printf("%s: ok\n", label);
  }
  void fail(const char *cause)
  {
    render();
    require(
        error.find(cause) != std::string::npos, "expected Frame diagnostic");
    int samples = -1;
    require(anariGetProperty(d,
                frame,
                "numSamples",
                ANARI_INT32,
                &samples,
                sizeof(samples),
                ANARI_WAIT)
            && samples == 0,
        "failed samples reset");
    for (const char *channel : {"channel.color", "channel.colorCUDA"}) {
      unsigned w = 9, h = 9;
      ANARIDataType type = ANARI_FLOAT32;
      require(!anariMapFrame(d, frame, channel, &w, &h, &type) && w == 0
              && h == 0 && type == ANARI_UNKNOWN,
          "failed mapping");
    }
  }
};
void vdf(Scene &s)
{
  s.source(R"mdl(mdl 1.7;
import ::df::*;
export material main(uniform float bias = 0) = material(volume: material_volume(
    scattering: df::anisotropic_vdf(directional_bias: bias),
    absorption_coefficient: color(0.25,0.5,0)));
)mdl");
  s.check("editable zero-bias isotropic VDF", {1, 2, 0});
  for (float bias : {.25f,
           std::numeric_limits<float>::infinity(),
           std::numeric_limits<float>::quiet_NaN()}) {
    anari::setParameter(s.d, s.material, "bias", bias);
    anari::commitParameters(s.d, s.material);
    s.fail("directional bias");
  }
  anari::setParameter(s.d, s.material, "bias", 0.f);
  anari::commitParameters(s.d, s.material);
  s.check("bias edit recovery", {1, 2, 0});
  anari::unsetParameter(s.d, s.material, "bias");
  s.source(R"mdl(mdl 1.7;
import ::df::*;
export material main() = let { vdf phase = df::anisotropic_vdf(directional_bias: 0.0); }
in material(volume: material_volume(scattering: phase,
    absorption_coefficient: color(0.25,0.5,0)));
)mdl");
  s.check("constant zero bias through compiler temporary", {1, 2, 0});
  s.source(R"mdl(mdl 1.7;
import ::df::*;
export material main() = material(volume: material_volume(
    scattering: true ? df::anisotropic_vdf(directional_bias: 0) : df::anisotropic_vdf(directional_bias: 0.5),
    absorption_coefficient: color(0.25,0.5,0)));
)mdl");
  s.check("folded supported VDF conditional", {1, 2, 0});
  s.source(R"mdl(mdl 1.7;
import ::df::*; import ::state::*;
export material main() = material(volume: material_volume(
    scattering: df::anisotropic_vdf(directional_bias: state::animation_time()),
    absorption_coefficient: color(0.25,0.5,0)));
)mdl");
  s.check("compiled varying zero-bias expression", {1, 2, 0});
  anari::setParameter(s.d, s.frame, "mdl.animationTime", .25f);
  anari::commitParameters(s.d, s.frame);
  s.fail("directional bias");
  anari::setParameter(s.d, s.frame, "mdl.animationTime", 0.f);
  anari::commitParameters(s.d, s.frame);
  s.check("varying bias recovery", {1, 2, 0});
  s.source(R"mdl(mdl 1.8;
import ::df::*;
export material main() = material(volume: material_volume(scattering: df::fog_vdf()));
)mdl");
  s.fail("unsupported VDF expression");
  s.source(R"mdl(mdl 1.7;
import ::df::*;
export material main(uniform bool choose = true) = material(volume: material_volume(
    scattering: choose ? df::anisotropic_vdf(directional_bias: 0.25) : vdf()));
)mdl");
  s.fail("unsupported VDF expression");
  s.source(R"mdl(mdl 1.7;
import ::df::*;
export material main(uniform float weight = 0.5) = material(volume: material_volume(
    scattering: df::normalized_mix(df::vdf_component[2](
        df::vdf_component(weight, df::anisotropic_vdf(directional_bias: 0.25)),
        df::vdf_component(1-weight, df::anisotropic_vdf(directional_bias: -0.25))))));
)mdl");
  s.fail("unsupported VDF expression");
  anari::setParameter(s.d, s.volume, "extinctionMajorant", 0.f);
  anari::commitParameters(s.d, s.volume);
  s.source(R"mdl(mdl 1.7;
import ::df::*;
export material main() = material(volume: material_volume(
    scattering: df::anisotropic_vdf(directional_bias: 0.25)));
)mdl");
  s.fail("directional bias");
  anari::setParameter(s.d, s.volume, "extinctionMajorant", 2.f);
  anari::commitParameters(s.d, s.volume);
}
void supportedState(Scene &s)
{
  s.source(R"mdl(mdl 1.7;
import ::df::*; import ::scene::*; import ::state::*;
export material main() = material(
    surface: material_surface(scattering: df::diffuse_reflection_bsdf(
        tint: color(scene::data_lookup_float("surfaceOnly",state::texture_coordinate(0).x)))),
    geometry: material_geometry(normal: state::texture_coordinate(0)),
    volume: material_volume(absorption_coefficient: color(0.25,0.5,0)));
)mdl");
  s.check("surface-only state and scene-data names do not poison volume",
      {1, 2, 0});
  s.source(R"mdl(mdl 1.7;
import ::state::*; import ::math::*;
export material main() = material(volume: material_volume(
    absorption_coefficient: color(
        0.125 * state::position().z + 0.25 * state::animation_time(),
        math::dot(state::normal(), state::normal()) +
            math::dot(state::geometry_normal(), state::geometry_normal()),
        state::object_id() == -2147483577 ? 0.25 * state::meters_per_scene_unit() : 1)));
)mdl");
  s.check("position / zero normals / signed Volume ID / meters", {2, 0, 1});
  anari::setParameter(s.d, s.frame, "accumulationVersion", 7);
  anari::commitParameters(s.d, s.frame);
  s.check("manual accumulation initial time", {2, 0, 1});
  s.check("manual accumulation same time", {2, 0, 1});
  int samples = 0;
  anariGetProperty(s.d,
      s.frame,
      "numSamples",
      ANARI_INT32,
      &samples,
      sizeof(samples),
      ANARI_WAIT);
  require(samples == 2, "unchanged time accumulates");
  anari::setParameter(s.d, s.frame, "mdl.animationTime", 1.f);
  anari::commitParameters(s.d, s.frame);
  s.check("Frame time edit changes coefficients", {3, 0, 1});
  anariGetProperty(s.d,
      s.frame,
      "numSamples",
      ANARI_INT32,
      &samples,
      sizeof(samples),
      ANARI_WAIT);
  require(samples == 1, "time edit forces manual accumulation reset");
  anari::unsetParameter(s.d, s.frame, "accumulationVersion");
  anari::setParameter(s.d, s.frame, "mdl.animationTime", 0.f);
  anari::commitParameters(s.d, s.frame);
  s.source(R"mdl(mdl 1.7;
import ::state::*;
export material main() = material(volume: material_volume(
    absorption_coefficient: color(
        0.125 * state::transform_point(state::coordinate_object, state::coordinate_world, float3(0)).z,
        0.25 * state::transform_vector(state::coordinate_object, state::coordinate_world, float3(0,0,1)).z,
        -0.125 * state::transform_point(state::coordinate_world, state::coordinate_object, float3(0)).z)));
)mdl");
  const std::array<float, 16> transform{
      0, 2, 0, 0, -3, 0, 0, 0, 0, 0, 1.5f, 0, 0, 0, 3, 1};
  anariSetParameter(
      s.d, s.instance, "transform", ANARI_FLOAT32_MAT4, transform.data());
  anari::commitParameters(s.d, s.instance);
  s.check("actual translated rotated nonuniform Instance transforms",
      {2.25f, 2.25f, 1.5f});
  const std::array<float, 16> identity{
      1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
  anariSetParameter(
      s.d, s.instance, "transform", ANARI_FLOAT32_MAT4, identity.data());
  anari::commitParameters(s.d, s.instance);
}
void unsupportedState(Scene &s)
{
  for (const char *expression : {"state::texture_coordinate(0).x",
           "state::texture_tangent_u(0).x",
           "state::texture_tangent_v(0).y",
           "state::geometry_tangent_u(0).x",
           "state::geometry_tangent_v(0).y",
           "state::motion().x",
           "state::direction().z",
           "state::rounded_corner_normal(0.25).x",
           "state::tangent_space(0)[0].x"}) {
    const std::string code =
        std::string(
            "mdl 1.7; import ::state::*; export material main() = material(volume: material_volume(absorption_coefficient: color(")
        + expression + ")));";
    s.source(code.c_str());
    s.fail("unsupported MDL volume state");
  }
  s.source(R"mdl(mdl 1.7;
import ::df::*; import ::state::*;
export material main() = material(volume: material_volume(
    scattering: df::anisotropic_vdf(directional_bias: state::texture_coordinate(0).x)));
)mdl");
  s.fail("unsupported MDL volume state");
}
void textures(Scene &s)
{
  const char *definition = R"mdl(mdl 1.7;
import ::tex::*; import ::state::*;
export material main(uniform texture_2d image = texture_2d()) = material(
    volume: material_volume(absorption_coefficient:
        tex::lookup_color(image, float2(state::position().z/8,0.5))));
)mdl";
  s.source(definition);
  const V3 texel{.25f, .5f, 0};
  auto image = anari::newArray2D(s.d, &texel, 1, 1);
  auto sampler = anari::newObject<anari::Sampler>(s.d, "image2D");
  anari::setParameter(s.d, sampler, "image", image);
  anari::commitParameters(s.d, sampler);
  anari::setParameter(s.d, s.material, "image", sampler);
  anari::commitParameters(s.d, s.material);
  s.check("explicit-position texture coordinates", {1, 2, 0});
  const std::array<V3, 2> pattern{V3{.25f, 0, 0}, V3{0, .5f, 0}};
  auto patternImage = anari::newArray2D(s.d, pattern.data(), 2, 1);
  anari::setParameter(s.d, sampler, "image", patternImage);
  anari::setParameter(s.d, sampler, "filter", "nearest");
  anari::commitParameters(s.d, sampler);
  s.check("position chooses distinct texels", {.5f, 1, 0});
  const V3 replacement{0, 0, .25f};
  auto replacementImage = anari::newArray2D(s.d, &replacement, 1, 1);
  auto replacementSampler = anari::newObject<anari::Sampler>(s.d, "image2D");
  anari::setParameter(s.d, replacementSampler, "image", replacementImage);
  anari::commitParameters(s.d, replacementSampler);
  anari::setParameter(s.d, s.material, "image", replacementSampler);
  anari::commitParameters(s.d, s.material);
  s.check("texture argument replacement", {0, 0, 1});
  anari::setParameter(s.d, s.material, "image", sampler);
  anari::commitParameters(s.d, s.material);
  s.check("restore original texture mapping", {.5f, 1, 0});
  // Traverse only one domain at a time: this tests bundle isolation, not the
  // separate multi-volume transport capability.
  auto otherMaterial = anari::newObject<anari::Material>(s.d, "mdl");
  anari::setParameter(s.d, otherMaterial, "sourceType", "code");
  anari::setParameter(s.d, otherMaterial, "source", definition);
  anari::setParameter(s.d, otherMaterial, "image", replacementSampler);
  anari::commitParameters(s.d, otherMaterial);
  auto otherVolume = anari::newObject<anari::Volume>(s.d, "mdl");
  const std::array<V3, 2> extent{V3{-2, -2, 2}, V3{2, 2, 6}};
  anari::setParameter(s.d, otherVolume, "material", otherMaterial);
  anariSetParameter(
      s.d, otherVolume, "extent", ANARI_FLOAT32_BOX3, extent.data());
  anari::setParameter(s.d, otherVolume, "extinctionMajorant", 2.f);
  anari::commitParameters(s.d, otherVolume);
  anari::setParameterArray1D(s.d, s.group, "volume", &otherVolume, 1);
  anari::commitParameters(s.d, s.group);
  s.check("shared code retains second Material resource indices", {0, 0, 1});
  anari::setParameterArray1D(s.d, s.group, "volume", &s.volume, 1);
  anari::commitParameters(s.d, s.group);
  s.check("second Volume does not mutate first resource mapping", {.5f, 1, 0});
  anari::release(s.d, otherVolume);
  anari::release(s.d, otherMaterial);
  anari::unsetParameter(s.d, s.material, "image");
  anari::commitParameters(s.d, s.material);
  s.check("unset texture argument restores invalid-texture default", {0, 0, 0});
  anari::release(s.d, replacementSampler);
  anari::release(s.d, replacementImage);
  anari::release(s.d, patternImage);
  anari::release(s.d, sampler);
  anari::release(s.d, image);
}
void bodyResources(Scene &s)
{
  struct Resources
  {
    std::filesystem::path path =
        std::filesystem::current_path() / "mdl_volume_state_resources";
    Resources()
    {
      std::filesystem::create_directories(path);
    }
    ~Resources()
    {
      std::filesystem::remove_all(path);
    }
  } resources;
  const std::array<unsigned char, 3> surface{0, 0, 192};
  const std::array<unsigned char, 3> medium{64, 128, 0};
  require(stbi_write_png((resources.path / "surface.png").string().c_str(),
              1,
              1,
              3,
              surface.data(),
              3),
      "write surface resource");
  require(stbi_write_png((resources.path / "medium.png").string().c_str(),
              1,
              1,
              3,
              medium.data(),
              3),
      "write volume resource");
  anari::setParameter(
      s.d, s.d, "mdlResourceSearchPaths", resources.path.string());
  anari::commitParameters(s.d, s.d);
  s.source(R"mdl(mdl 1.7;
import ::tex::*; import ::df::*; import ::state::*;
export material main() = material(
    surface: material_surface(scattering: df::diffuse_reflection_bsdf(
        tint: tex::lookup_color(texture_2d("/surface.png", gamma: tex::gamma_linear), float2(state::texture_coordinate(0).x,0)))),
    volume: material_volume(absorption_coefficient:
        tex::lookup_color(texture_2d("/medium.png", gamma: tex::gamma_linear), float2(state::position().z/8,0.5))));
)mdl");
  s.check("body resources use volume bundle indices, ignore surface UV",
      {4 * 64.f / 255, 4 * 128.f / 255, 0});
  s.source(R"mdl(mdl 1.7;
import ::tex::*;
export material main(uniform texture_2d image = texture_2d("/surface.png", gamma: tex::gamma_linear)) = material(
    volume: material_volume(absorption_coefficient: tex::lookup_color(image,float2(0.5))));
)mdl");
  s.check("source replacement refreshes default texture mapping",
      {0, 0, 4 * 192.f / 255});
  s.source(R"mdl(mdl 1.7;
import ::tex::*;
export material main(uniform texture_2d image = texture_2d("/medium.png", gamma: tex::gamma_linear)) = material(
    volume: material_volume(absorption_coefficient: tex::lookup_color(image,float2(0.5))));
)mdl");
  s.check("default texture resource replacement",
      {4 * 64.f / 255, 4 * 128.f / 255, 0});
  auto aliasedDefaults = [&](bool alias) {
    return std::string(R"mdl(mdl 1.7; import ::tex::*;
export material main(
    uniform texture_2d a = texture_2d("/medium.png", gamma: tex::gamma_linear),
    uniform texture_2d b = texture_2d(")mdl")
        + (alias ? "/medium.png" : "/surface.png")
        + R"mdl(", gamma: tex::gamma_linear)) = material(
    volume: material_volume(absorption_coefficient: 0.125 * (
        tex::lookup_color(a,float2(0.5)) + tex::lookup_color(b,float2(0.5)) +
        tex::lookup_color(texture_2d("/medium.png",gamma:tex::gamma_linear),float2(0.5)))));
)mdl";
  };
  s.source(aliasedDefaults(true).c_str());
  s.check("aliased defaults and body texture",
      {1.5f * 64 / 255, 1.5f * 128 / 255, 0});
  s.source(aliasedDefaults(false).c_str());
  s.check("shared code with different resource-table cardinality",
      {64.f / 255, 128.f / 255, .5f * 192 / 255});
  s.source(aliasedDefaults(true).c_str());
  s.check("restore cached source-specific default mappings",
      {1.5f * 64 / 255, 1.5f * 128 / 255, 0});
}
void emission(Scene &s)
{
  s.source(R"mdl(mdl 1.7;
import ::df::*; import ::state::*;
export material main(uniform color emission = color(0)) = material(
    surface: material_surface(emission: material_emission(emission: df::diffuse_edf(), intensity: color(10))),
    volume: material_volume(absorption_coefficient: color(0.25,0.5,0), emission_intensity: emission));
)mdl");
  s.check("surface emission is not volume emission", {1, 2, 0});
  anari::setParameter(s.d, s.material, "emission", V3{.1f, 0, 0});
  anari::commitParameters(s.d, s.material);
  s.fail("volume emission");
  for (float value : {std::numeric_limits<float>::infinity(),
           std::numeric_limits<float>::quiet_NaN()}) {
    anari::setParameter(s.d, s.material, "emission", V3{value, 0, 0});
    anari::commitParameters(s.d, s.material);
    s.fail("volume emission");
  }
  anari::setParameter(s.d, s.material, "emission", V3{0, 0, 0});
  anari::commitParameters(s.d, s.material);
  s.check("zero emission recovery", {1, 2, 0});
  anari::unsetParameter(s.d, s.material, "emission");
  anari::setParameter(s.d, s.volume, "extinctionMajorant", 0.f);
  anari::commitParameters(s.d, s.volume);
  s.source(R"mdl(mdl 1.7;
export material main() = material(volume: material_volume(emission_intensity: color(0.1)));
)mdl");
  s.fail("volume emission");
  anari::setParameter(s.d, s.volume, "extinctionMajorant", 2.f);
  anari::commitParameters(s.d, s.volume);
}
} // namespace
int main()
{
  try {
    Scene s;
    vdf(s);
    supportedState(s);
    unsupportedState(s);
    textures(s);
    bodyResources(s);
    emission(s);
  } catch (const std::exception &e) {
    fprintf(stderr, "%s\n", e.what());
    return 1;
  }
  return 0;
}
