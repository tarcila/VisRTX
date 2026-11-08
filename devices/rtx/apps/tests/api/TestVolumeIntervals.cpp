// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "RendererFogSupport.h"

#include <anari/anari_cpp/ext/std.h>
#include <anari/ext/visrtx/makeVisRTXDevice.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <string>

namespace {

using namespace visrtx::fogtest;
using Box3 = std::array<Vec3, 2>;
using Mat4 = std::array<float, 16>;
constexpr UVec2 SIZE{8, 8};
constexpr Box3 FULL{Vec3{-2.f, -2.f, 2.f}, Vec3{2.f, 2.f, 6.f}};
constexpr Box3 CROP{Vec3{-2.f, -2.f, 3.f}, Vec3{2.f, 2.f, 5.f}};

// A uniform field clipped by ROI and a physically cropped field describe the
// same medium. Their lattice spacing is identical, but their acceleration
// grids and reported field bounds differ. Compare public output, not layouts
// or callable indices. The uncut medium is a non-vacuous control.
struct Scene
{
  Scene(anari::Device device, const char *type);
  ~Scene();
  Scene(const Scene &) = delete;
  Scene &operator=(const Scene &) = delete;
  void fieldBounds(Box3 bounds, const Box3 *roi);
  Image render();

  anari::Device d;
  anari::SpatialField field;
  anari::Instance instance;
  anari::World world;
  anari::Camera camera;
  anari::Renderer renderer;
  anari::Frame frame;
};

Scene::Scene(anari::Device device, const char *type) : d(device)
{
  field = anari::newObject<anari::SpatialField>(d, "structuredRegular");
  fieldBounds(FULL, nullptr);
  auto volume = anari::newObject<anari::Volume>(d, "transferFunction1D");
  const ObjectOwner volumeOwner(d, volume);
  anari::setParameter(d, volume, "value", field);
  anari::setParameter(d, volume, "color", Vec3{0.6f, 0.3f, 0.1f});
  anari::setParameter(d, volume, "opacity", 0.25f);
  anari::setParameter(d, volume, "unitDistance", 1.f);
  anari::setParameter(d, volume, "id", 71u);
  anari::commitParameters(d, volume);
  auto group = anari::newObject<anari::Group>(d);
  const ObjectOwner groupOwner(d, group);
  anari::setParameterArray1D(d, group, "volume", &volume, 1);
  anari::commitParameters(d, group);
  instance = anari::newObject<anari::Instance>(d, "transform");
  anari::setParameter(d, instance, "group", group);
  anari::setParameter(d, instance, "id", 93u);
  anari::commitParameters(d, instance);
  world = anari::newObject<anari::World>(d);
  anari::setParameterArray1D(d, world, "instance", &instance, 1);
  anari::commitParameters(d, world);
  camera = anari::newObject<anari::Camera>(d, "orthographic");
  anari::setParameter(d, camera, "position", Vec3{0.f, 0.f, 0.f});
  anari::setParameter(d, camera, "direction", Vec3{0.f, 0.f, 1.f});
  anari::setParameter(d, camera, "up", Vec3{0.f, 1.f, 0.f});
  anari::setParameter(d, camera, "height", 0.5f);
  anari::commitParameters(d, camera);
  renderer = anari::newObject<anari::Renderer>(d, type);
  anari::setParameter(d, renderer, "denoise", false);
  anari::setParameter(d, renderer, "fireflyFilterMode", "none");
  anari::setParameter(d, renderer, "ambientSamples", 0);
  anari::setParameter(d, renderer, "ambientRadiance", 1.f);
  anari::setParameter(d, renderer, "background", Vec4{});
  anari::setParameter(d, renderer, "pixelSamples", 4096);
  anari::setParameter(d, renderer, "maxRayDepth", 4);
  anari::commitParameters(d, renderer);
  frame = anari::newObject<anari::Frame>(d);
  anari::setParameter(d, frame, "size", SIZE);
  anari::setParameter(d, frame, "channel.color", ANARI_FLOAT32_VEC4);
  anari::setParameter(d, frame, "channel.depth", ANARI_FLOAT32);
  anari::setParameter(d, frame, "channel.normal", ANARI_FLOAT32_VEC3);
  anari::setParameter(d, frame, "channel.albedo", ANARI_FLOAT32_VEC3);
  for (const char *name :
      {"channel.primitiveId", "channel.objectId", "channel.instanceId"})
    anari::setParameter(d, frame, name, ANARI_UINT32);
  anari::setParameter(d, frame, "world", world);
  anari::setParameter(d, frame, "camera", camera);
  anari::setParameter(d, frame, "renderer", renderer);
  anari::commitParameters(d, frame);
}

Scene::~Scene()
{
  anari::release(d, frame);
  anari::release(d, renderer);
  anari::release(d, camera);
  anari::release(d, world);
  anari::release(d, instance);
  anari::release(d, field);
}

void Scene::fieldBounds(Box3 bounds, const Box3 *roi)
{
  const unsigned nz = unsigned((bounds[1][2] - bounds[0][2]) * 8) + 1;
  auto data = anari::newArray3D(d, ANARI_FLOAT32, 33, 33, nz);
  const ObjectOwner dataOwner(d, data);
  {
    const ArrayMapping<float> values(d, data);
    std::fill(values.data(), values.data() + 33 * 33 * nz, 0.5f);
  }
  anari::setParameter(d, field, "data", data);
  anari::setParameter(d, field, "origin", bounds[0]);
  anari::setParameter(d, field, "spacing", Vec3{0.125f, 0.125f, 0.125f});
  if (roi)
    anariSetParameter(d, field, "roi", ANARI_FLOAT32_BOX3, roi->data());
  else
    anari::unsetParameter(d, field, "roi");
  anari::commitParameters(d, field);
}

Image Scene::render()
{
  anari::render(d, frame);
  anari::wait(d, frame);
  return readImage(d, frame, SIZE);
}

bool equivalent(
    const Image &a, const Image &b, const char *label, bool stochastic = false)
{
  // A cropped acceleration grid partitions control flights differently, so
  // Quality's samples are not paired. Compare means over 262144 camera
  // samples, with a fixed .01 ceiling (over seven worst-case binomial SEs
  // for the difference). Marching and identical-grid restoration stay exact.
  bool ok = stochastic ? a.object == b.object && a.instance == b.instance
                       : sameAuxiliary(a, b, label);
  double meanError[4] = {};
  for (unsigned p = 0; p < a.color.size(); ++p) {
    for (unsigned c = 0; c < 4; ++c) {
      ok &= std::isfinite(a.color[p][c]) && std::isfinite(b.color[p][c]);
      meanError[c] += (a.color[p][c] - b.color[p][c]) / a.color.size();
      if (!stochastic)
        ok &= std::abs(a.color[p][c] - b.color[p][c]) < 2e-5f;
    }
    for (unsigned c = 0; c < 3; ++c)
      ok &= std::isfinite(a.normal[p][c]) && std::isfinite(b.normal[p][c])
          && std::isfinite(a.albedo[p][c]) && std::isfinite(b.albedo[p][c]);
  }
  for (double error : meanError)
    ok &= std::abs(error) < .01;
  printf("%s: %s\n", label, ok ? "ok" : "FAILED");
  return ok;
}

int testShadows(anari::Device d, const char *type)
{
  Scene scene(d, type);
  // The camera sees a tilted receiver before the field. A directional light
  // reaches its front face through the off-axis Volume, isolating shadow
  // integration from primary volume compositing.
  auto geometry = anari::newObject<anari::Geometry>(d, "quad");
  const ObjectOwner geometryOwner(d, geometry);
  const std::array<Vec3, 4> positions{Vec3{-2.f, -2.f, -7.f},
      Vec3{-2.f, 2.f, -7.f},
      Vec3{2.f, 2.f, 9.f},
      Vec3{2.f, -2.f, 9.f}};
  anari::setParameterArray1D(
      d, geometry, "vertex.position", positions.data(), positions.size());
  anari::commitParameters(d, geometry);
  auto material = anari::newObject<anari::Material>(d, "matte");
  const ObjectOwner materialOwner(d, material);
  anari::setParameter(d, material, "color", Vec3{1.f, 1.f, 1.f});
  anari::commitParameters(d, material);
  auto surface = anari::newObject<anari::Surface>(d);
  const ObjectOwner surfaceOwner(d, surface);
  anari::setParameter(d, surface, "geometry", geometry);
  anari::setParameter(d, surface, "material", material);
  anari::commitParameters(d, surface);
  auto light = anari::newObject<anari::Light>(d, "directional");
  const ObjectOwner lightOwner(d, light);
  anari::setParameter(d, light, "direction", Vec3{-0.5f, 0.f, -1.f});
  anari::setParameter(d, light, "irradiance", 10.f);
  anari::commitParameters(d, light);
  anari::setParameterArray1D(d, scene.world, "surface", &surface, 1);
  anari::setParameterArray1D(d, scene.world, "light", &light, 1);
  anari::commitParameters(d, scene.world);
  anari::setParameter(d, scene.renderer, "ambientRadiance", 0.f);
  // The default preview shadow step exceeds this fixture's optical length.
  anari::setParameter(d, scene.renderer, "volumeSamplingRateShadows", 1.f);
  anari::setParameter(d, scene.renderer, "maxRayDepth", 1);
  anari::commitParameters(d, scene.renderer);
  anari::setParameter(d, scene.camera, "height", 0.125f);
  anari::commitParameters(d, scene.camera);
  const auto full = scene.render();
  scene.fieldBounds(FULL, &CROP);
  const auto roi = scene.render();
  scene.fieldBounds(CROP, nullptr);
  const auto crop = scene.render();
  int failures = !equivalent(roi,
      crop,
      (std::string(type) + " shadow ROI/crop").c_str(),
      std::string(type) == "quality");
  double fullMean = 0., cropMean = 0.;
  for (unsigned p = 0; p < crop.color.size(); ++p) {
    fullMean += full.color[p][0] / crop.color.size();
    cropMean += crop.color[p][0] / crop.color.size();
  }
  printf("%s: full shadow red=%g cropped shadow red=%g\n",
      type,
      fullMean,
      cropMean);
  failures += !(cropMean > fullMean + .03 && fullMean > .01);
  return failures;
}

int testIntervals(anari::Device d, const char *type)
{
  Scene scene(d, type);
  int failures = 0;
  const auto full = scene.render();
  scene.fieldBounds(FULL, &CROP);
  const auto roi = scene.render();
  scene.fieldBounds(CROP, nullptr);
  const auto crop = scene.render();
  const bool quality = std::string(type) == "quality";
  failures += !equivalent(
      roi, crop, (std::string(type) + " ROI/crop").c_str(), quality);
  double alphaFull = 0., alphaCrop = 0.;
  for (unsigned p = 0; p < crop.color.size(); ++p) {
    alphaFull += full.color[p][3] / crop.color.size();
    alphaCrop += crop.color[p][3] / crop.color.size();
    failures += crop.object[p] != 71 || crop.instance[p] != 93
        || !std::isfinite(crop.depth[p]) || crop.depth[p] < 3.f
        || crop.depth[p] > 5.f;
  }
  printf("%s: full alpha=%g cropped alpha=%g\n", type, alphaFull, alphaCrop);
  failures += !(alphaFull > alphaCrop + 0.1 && alphaCrop > 0.2);
  // The homogeneous coverage oracle is independent of TF implementation:
  // opacity .25 per unit distance transmits .75^length.
  failures += std::abs(alphaCrop - (1. - std::pow(.75, 2.))) > .01;
  failures += std::abs(alphaFull - (1. - std::pow(.75, 4.))) > .01;

  // Nonuniform scale, rotation around the camera axis and translation retain
  // the object-space ROI, while near/far clip the world-space interval.
  constexpr Mat4 xfm{0.f,
      2.f,
      0.f,
      0.f,
      -1.f,
      0.f,
      0.f,
      0.f,
      0.f,
      0.f,
      1.5f,
      0.f,
      0.f,
      0.f,
      1.f,
      1.f};
  anariSetParameter(
      d, scene.instance, "transform", ANARI_FLOAT32_MAT4, xfm.data());
  anari::commitParameters(d, scene.instance);
  anari::setParameter(d, scene.camera, "near", 6.f);
  anari::setParameter(d, scene.camera, "far", 8.f);
  anari::commitParameters(d, scene.camera);
  const auto transformedCrop = scene.render();
  scene.fieldBounds(FULL, &CROP);
  const auto transformedROI = scene.render();
  failures += !equivalent(transformedROI,
      transformedCrop,
      (std::string(type) + " transformed/clipped ROI/crop").c_str(),
      quality);

  // Editing ROI alone must update dependent Volume state without recommitting
  // that Volume. Disjoint ROI removes it, but a Miss still reports camera far.
  constexpr Box3 DISJOINT{Vec3{8.f, 8.f, 8.f}, Vec3{9.f, 9.f, 9.f}};
  anariSetParameter(d, scene.field, "roi", ANARI_FLOAT32_BOX3, DISJOINT.data());
  anari::commitParameters(d, scene.field);
  const auto miss = scene.render();
  for (unsigned p = 0; p < miss.color.size(); ++p)
    failures += miss.object[p] != ~0u || miss.instance[p] != ~0u
        || miss.depth[p] != 8.f || miss.color[p] != Vec4{};
  anariSetParameter(d, scene.field, "roi", ANARI_FLOAT32_BOX3, CROP.data());
  anari::commitParameters(d, scene.field);
  failures += !equivalent(scene.render(),
      transformedROI,
      (std::string(type) + " restored ROI").c_str());
  return failures;
}

} // namespace

int main()
{
  auto d = makeVisRTXDevice(exitOnErrorStatus);
  const ObjectOwner deviceOwner(d, d);
  int failures = 0;
  for (const char *type : {"interactive", "quality"}) {
    failures += testIntervals(d, type);
    failures += testShadows(d, type);
  }
  printf("volume interval regression: %d failure(s)\n", failures);
  return failures ? 1 : 0;
}
