// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#include <anari/anari_cpp/ext/std.h>
#include <anari/ext/visrtx/makeVisRTXDevice.h>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include "MdlVolumeTestHelpers.h"

namespace {
using V3 = std::array<float, 3>;
using V4 = std::array<float, 4>;
using Box = std::array<V3, 2>;
using Matrix = std::array<float, 16>;
constexpr Matrix IDENTITY{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
void require(bool b, const char *message)
{
  if (!b)
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
  std::array<anari::Volume, 2> volumes{
      anari::newObject<anari::Volume>(d, "mdl"),
      anari::newObject<anari::Volume>(d, "mdl")};
  anari::World world = anari::newObject<anari::World>(d);
  anari::Camera camera = anari::newObject<anari::Camera>(d, "orthographic");
  anari::Renderer renderer;
  bool quality;
  anari::Frame frame = anari::newObject<anari::Frame>(d);
  std::vector<anari::Object> owned;
  Scene(const char *type)
      : renderer(anari::newObject<anari::Renderer>(d, type)),
        quality(std::string(type) == "quality")
  {
    frameSource = frame;
    anari::setParameter(d, material, "sourceType", "code");
    anari::setParameter(d, material, "source", R"mdl(mdl 1.7;
export material main() = material(volume: material_volume(
  absorption_coefficient: color(0.25,0.5,0)));
)mdl");
    anari::commitParameters(d, material);
    for (auto v : volumes) {
      anari::setParameter(d, v, "material", material);
      anari::setParameter(d, v, "extinctionMajorant", .5f);
    }
    extent(0, {V3{-2, -2, 2}, V3{2, 2, 4}});
    extent(1, {V3{-2, -2, 5}, V3{2, 2, 7}});
    direct();
    anari::setParameter(d, camera, "position", V3{0, 0, 0});
    anari::setParameter(d, camera, "direction", V3{0, 0, 1});
    anari::setParameter(d, camera, "up", V3{0, 1, 0});
    anari::setParameter(d, camera, "height", .25f);
    anari::commitParameters(d, camera);
    anari::setParameter(d, renderer, "denoise", false);
    anari::setParameter(d, renderer, "fireflyFilterMode", "none");
    anari::setParameter(d, renderer, "ambientRadiance", 0.f);
    anari::setParameter(d, renderer, "background", V4{1, 1, 1, 1});
    anari::setParameter(d, renderer, "sampleLimit", 0);
    anari::setParameter(d, renderer, "pixelSamples", 1);
    anari::commitParameters(d, renderer);
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
    for (auto o : owned)
      anari::release(d, o);
    for (auto v : volumes)
      anari::release(d, v);
    for (auto o : {anari::Object(frame),
             anari::Object(renderer),
             anari::Object(camera),
             anari::Object(world),
             anari::Object(material)})
      anari::release(d, o);
    anari::release(d, d);
  }
  void extent(int i, Box box)
  {
    anariSetParameter(d, volumes[i], "extent", ANARI_FLOAT32_BOX3, box.data());
    anari::commitParameters(d, volumes[i]);
  }
  void direct()
  {
    direct(volumes);
  }
  void direct(std::array<anari::Volume, 2> list)
  {
    anari::unsetParameter(d, world, "instance");
    anari::setParameterArray1D(d, world, "volume", list.data(), list.size());
    anari::commitParameters(d, world);
  }
  void render()
  {
    anari::render(d, frame);
    require(anariFrameReady(d, frame, ANARI_WAIT), "WAIT must complete");
  }
  void check(const char *name, V4 reference)
  {
    anari::setParameter(d, renderer, "pixelSamples", quality ? 2048 : 1);
    anari::commitParameters(d, renderer);
    const int batches = quality ? 16 : 1;
    std::array<double, 4> previous{}, mean{}, m2{};
    bool discardedReset = false;
    const auto before = errors;
    for (int b = 0; b < batches; ++b) {
      render();
      require(errors == before, "unexpected Frame error in numerical scene");
      int count = 0;
      require(anariGetProperty(d,
                  frame,
                  "numSamples",
                  ANARI_INT32,
                  &count,
                  sizeof(count),
                  ANARI_WAIT),
          "numSamples query");
      if (b == 1 && count == 2048 && !discardedReset) {
        discardedReset = true;
        b = 0;
        previous = {};
        mean = {};
        m2 = {};
      }
      require(count == (b + 1) * (quality ? 2048 : 1),
          "nonoverlapping sample batches");
      auto p = anari::map<V4>(d, frame, "channel.color");
      require(p.data && p.width == 8 && p.height == 8, "numerical scene maps");
      std::array<double, 4> cumulative{};
      for (int i = 0; i < 64; ++i)
        for (int c = 0; c < 4; ++c) {
          require(std::isfinite(p.data[i][c]), "finite output");
          cumulative[c] += p.data[i][c] / 64.;
        }
      anari::unmap(d, frame, "channel.color");
      for (int c = 0; c < 4; ++c) {
        const double sample = cumulative[c] * (b + 1) - previous[c] * b;
        previous[c] = cumulative[c];
        const double delta = sample - mean[c];
        mean[c] += delta / (b + 1);
        m2[c] += delta * (sample - mean[c]);
      }
    }
    for (int c = 0; c < 4; ++c) {
      const double se =
          quality ? std::sqrt(m2[c] / (batches * (batches - 1))) : 0.;
      const double error = std::abs(mean[c] - reference[c]);
      printf("%s %s c%d mean %.9g ref %.9g SE %.9g\n",
          quality ? "quality" : "interactive",
          name,
          c,
          mean[c],
          reference[c],
          se);
      require(se <= .0016, "inconclusive: batch uncertainty exceeds ceiling");
      require(quality ? error <= .01 && error <= 5 * se + .002 : error <= 2e-5,
          "multi-volume independent absorption reference mismatch");
    }
    anari::setParameter(d, renderer, "pixelSamples", 1);
    anari::commitParameters(d, renderer);
  }
  void fail(const char *category = "positive-volume overlap")
  {
    const auto before = errors;
    render();
    require(errors > before && cause.find(category) != std::string::npos,
        "expected overlap failure category");
    require(cause.find("Volume registry") != std::string::npos
            && cause.find("and Volume registry") != std::string::npos
            && cause.find("Instance") != std::string::npos
            && cause.find("transform") != std::string::npos,
        "diagnostic must identify both placements");
    require(anariFrameReady(d, frame, ANARI_NO_WAIT), "failed Frame is ready");
    requireFailedMdlFrame(d, frame);
  }
  anari::SpatialField field(V3 origin, V3 spacing = {1, 1, 1})
  {
    auto f = anari::newObject<anari::SpatialField>(d, "structuredRegular");
    owned.push_back(f);
    std::vector<float> values(5 * 5 * 3, .5f);
    anari::setParameterArray3D(d, f, "data", values.data(), 5, 5, 3);
    anari::setParameter(d, f, "origin", origin);
    anari::setParameter(d, f, "spacing", spacing);
    anari::commitParameters(d, f);
    return f;
  }
  anari::Instance instance(anari::Volume volume, Matrix matrix = IDENTITY)
  {
    auto g = anari::newObject<anari::Group>(d);
    owned.push_back(g);
    anari::setParameterArray1D(d, g, "volume", &volume, 1);
    anari::commitParameters(d, g);
    auto i = anari::newObject<anari::Instance>(d, "transform");
    owned.push_back(i);
    anari::setParameter(d, i, "group", g);
    transform(i, matrix);
    return i;
  }
  void transform(anari::Instance i, Matrix matrix)
  {
    anariSetParameter(d, i, "transform", ANARI_FLOAT32_MAT4, matrix.data());
    anari::commitParameters(d, i);
  }
  void instances(std::vector<anari::Instance> list)
  {
    anari::unsetParameter(d, world, "volume");
    anari::setParameterArray1D(d, world, "instance", list.data(), list.size());
    anari::commitParameters(d, world);
  }
  void accept(const char *name)
  {
    const auto before = errors;
    render();
    printf("accepted: %s\n", name);
    require(errors == before, "unexpected Frame failure in accepted scene");
    auto p = anari::map<V4>(d, frame, "channel.color");
    require(p.data && p.width == 8 && p.height == 8, "accepted scene must map");
    anari::unmap(d, frame, "channel.color");
  }
};
V4 beer(double length)
{
  return {float(std::exp(-.25 * length)), float(std::exp(-.5 * length)), 1, 1};
}
void domainEdits(Scene &s)
{
  s.check("disjoint extent-only sequential media", beer(4));
  s.extent(1, {V3{-2, -2, 3}, V3{2, 2, 5}});
  s.fail();
  s.extent(1, {V3{-2, -2, 4}, V3{2, 2, 6}});
  s.check("exact face contact and recovery", beer(4));
  s.extent(1, {V3{-2, -2, std::nextafter(4.f, 5.f)}, V3{2, 2, 6}});
  s.accept("one-FP32-ULP gap is not an expanded overlap epsilon");
  s.extent(1, {V3{-2, -2, std::nextafter(4.f, 3.f)}, V3{2, 2, 6}});
  s.fail();
  // Contact at an edge and a corner also has zero intersection volume.
  s.extent(1, {V3{2, 2, 2}, V3{4, 4, 4}});
  s.accept("exact edge contact");
  s.extent(1, {V3{2, 2, 4}, V3{4, 4, 6}});
  s.accept("exact corner contact");
  // Neither medium is visible to primary rays, but both remain active in
  // secondary/shadow transport. Camera culling must not remove this pair.
  s.extent(0, {V3{10, -2, 2}, V3{14, 2, 4}});
  s.extent(1, {V3{11, -2, 2}, V3{15, 2, 4}});
  s.fail();
  // In this device visible=false removes a Volume from ALL traversal. It is
  // not a camera-only mask; keep that established parameter meaning intact.
  anari::setParameter(s.d, s.volumes[1], "visible", false);
  anari::commitParameters(s.d, s.volumes[1]);
  s.accept("globally disabled Volume is not transport active");
  anari::setParameter(s.d, s.volumes[1], "visible", true);
  anari::commitParameters(s.d, s.volumes[1]);
  s.fail();
  s.extent(0, {V3{-2, -2, 2}, V3{2, 2, 4}});
  s.extent(1, {V3{-2, -2, 5}, V3{2, 2, 7}});
}
void fieldDomains(Scene &s)
{
  const auto shared = s.field({-2, -2, 2}, {1, 1, 4}); // field covers z=[2,10]
  for (auto v : s.volumes) {
    anari::setParameter(s.d, v, "value", shared);
    anari::setParameter(s.d, v, "fieldName", "density");
    anari::commitParameters(s.d, v);
  }
  s.check("shared field bounds overlap but explicit domains do not", beer(4));
  for (auto v : s.volumes) {
    anari::unsetParameter(s.d, v, "extent");
    anari::commitParameters(s.d, v);
  }
  s.fail();
  s.extent(0, {V3{-2, -2, 2}, V3{2, 2, 4}});
  s.extent(1, {V3{-2, -2, 5}, V3{2, 2, 7}});
  // Domain changes of a field-derived Volume propagate without recommitting
  // that Volume, while the other Volume retains its explicit extent.
  auto moving = s.field({-2, -2, 5});
  anari::setParameter(s.d, s.volumes[1], "value", moving);
  anari::unsetParameter(s.d, s.volumes[1], "extent");
  anari::commitParameters(s.d, s.volumes[1]);
  s.check("field-derived disjoint domain", beer(4));
  anari::setParameter(s.d, moving, "origin", V3{-2, -2, 3});
  anari::commitParameters(s.d, moving);
  s.fail();
  anari::setParameter(s.d, moving, "origin", V3{-2, -2, 5});
  anari::commitParameters(s.d, moving);
  s.check("field-domain edit removes overlap", beer(4));
  s.extent(1, {V3{-2, -2, 5}, V3{2, 2, 7}});
  for (auto v : s.volumes) {
    anari::unsetParameter(s.d, v, "value");
    anari::commitParameters(s.d, v);
  }
}
void transformedDomains(Scene &s)
{
  const auto a = s.instance(s.volumes[0]);
  Matrix shift = IDENTITY;
  shift[14] = 3;
  const auto b =
      s.instance(s.volumes[0], shift); // SAME Volume, second Instance
  s.instances({a, b});
  s.check("shared Volume and Material in disjoint Instances", beer(4));
  shift[14] = 1;
  s.transform(b, shift);
  s.fail();
  shift[14] = 2;
  s.transform(b, shift);
  s.check("Instance edit gives exact contact", beer(4));
  // Two nonuniformly scaled, rotated parallel bars. Their world AABBs overlap
  // over most of their area; the actual transformed boxes have a positive gap.
  s.extent(0, {V3{-2, -.125f, 2}, V3{2, .125f, 4}});
  Matrix rotate = IDENTITY;
  rotate[0] = 2;
  rotate[1] = 2;
  rotate[4] = -1;
  rotate[5] = 1;
  s.transform(a, rotate);
  rotate[12] = -.5f;
  rotate[13] = .5f;
  s.transform(b, rotate);
  anari::setParameter(s.d, s.camera, "height", .05f);
  anari::commitParameters(s.d, s.camera);
  s.check("separated rotated boxes despite overlapping world AABBs", beer(2));
  rotate[12] = -.125f;
  rotate[13] = .125f;
  s.transform(b, rotate);
  s.fail();
  // Sheared parallelepipeds are not orthogonal OBBs. SAT uses transformed
  // face normals, not incorrectly treating transformed edges as normals.
  Matrix shear = IDENTITY;
  shear[4] = .5f;
  s.transform(a, shear);
  shear[13] = 1;
  s.transform(b, shear);
  s.accept("nonoverlapping affine sheared boxes");
  // Multiple transforms on one Instance must be checked individually, even
  // though they share the same authored Instance/Volume identity.
  s.extent(0, {V3{-2, -2, 2}, V3{2, 2, 4}});
  s.transform(a, IDENTITY);
  s.instances({a});
  std::array<Matrix, 2> transforms{IDENTITY, IDENTITY};
  transforms[1][14] = 3;
  auto array = anariNewArray1D(
      s.d, nullptr, nullptr, nullptr, ANARI_FLOAT32_MAT4, transforms.size());
  auto initial = static_cast<Matrix *>(anariMapArray(s.d, array));
  initial[0] = transforms[0];
  initial[1] = transforms[1];
  anariUnmapArray(s.d, array);
  s.owned.push_back(array);
  anari::setParameter(s.d, a, "transform", array);
  anari::commitParameters(s.d, a);
  s.check("Instance transform array", beer(4));
  auto mapped = static_cast<Matrix *>(anariMapArray(s.d, array));
  mapped[1][14] = 1;
  anariUnmapArray(s.d, array);
  s.fail();
  mapped = static_cast<Matrix *>(anariMapArray(s.d, array));
  mapped[1][14] = 3;
  anariUnmapArray(s.d, array);
  s.accept("transform array edit removes overlap");
  s.direct();
  anari::setParameter(s.d, s.camera, "height", .25f);
  anari::commitParameters(s.d, s.camera);
}
void mixedDomains(Scene &s)
{
  const auto tf = anari::newObject<anari::Volume>(s.d, "transferFunction1D");
  s.owned.push_back(tf);
  auto field = s.field({-2, -2, 5});
  anari::setParameter(s.d, tf, "value", field);
  anari::setParameter(s.d, tf, "color", V3{0, 0, 0});
  anari::setParameter(s.d, tf, "opacity", .25f);
  anari::setParameter(s.d, tf, "unitDistance", 1.f);
  anari::commitParameters(s.d, tf);
  s.direct({s.volumes[0], tf});
  anari::setParameter(s.d, field, "origin", V3{-2, -2, 3});
  anari::commitParameters(s.d, field);
  s.fail();
  anari::setParameter(s.d, field, "origin", V3{-2, -2, 4});
  anari::commitParameters(s.d, field);
  s.accept("MDL/TF1D exact contact");
  // TF1D's ROI participates in its effective traversal domain.
  anari::setParameter(s.d, field, "origin", V3{-2, -2, 3});
  Box roi{V3{-2, -2, 4}, V3{2, 2, 5}};
  anariSetParameter(s.d, field, "roi", ANARI_FLOAT32_BOX3, roi.data());
  anari::commitParameters(s.d, field);
  s.accept("TF1D effective ROI contact, not original field overlap");
  // TF1D-only overlap stays on its existing path (no new gate).
  auto tf2 = anari::newObject<anari::Volume>(s.d, "transferFunction1D");
  s.owned.push_back(tf2);
  anari::setParameter(s.d, tf2, "value", field);
  anari::setParameter(s.d, tf2, "color", V3{0, 0, 0});
  anari::setParameter(s.d, tf2, "opacity", .25f);
  anari::commitParameters(s.d, tf2);
  s.direct({tf2, tf});
  s.accept("TF1D-only positive overlap retains existing behavior");
  anari::setParameter(s.d, field, "origin", V3{-2, -2, 5});
  anari::unsetParameter(s.d, field, "roi");
  anari::commitParameters(s.d, field);
  s.direct({s.volumes[0], tf});
  const auto t = beer(2);
  s.check("nonoverlapping MDL and nonzero TF1D absorption",
      {t[0] * .5625f, t[1] * .5625f, .5625f, 1});
  s.direct();
}
void ambiguity(Scene &s)
{
  // Exactly represented FP32 inputs whose transformed corner sums cannot be
  // resolved even by the wider host predicate: 1 + 2^-100. The two domains
  // touch at that plane. Acceptance is permitted ONLY if established; an
  // explicit indeterminate-overlap failure is the conservative alternative.
  const float tiny = std::ldexp(1.f, -100);
  s.extent(0, {V3{0, -1, 2}, V3{tiny, 1, 4}});
  s.extent(1, {V3{tiny, -1, 2}, V3{2 * tiny, 1, 4}});
  for (auto v : s.volumes) {
    anari::setParameter(s.d, v, "stepSize", .1f);
    anari::commitParameters(s.d, v);
  }
  Matrix shift = IDENTITY;
  shift[12] = 1;
  auto a = s.instance(s.volumes[0], shift);
  auto b = s.instance(s.volumes[1], shift);
  s.instances({a, b});
  const auto before = s.errors;
  s.render();
  if (s.errors > before) {
    require(s.cause.find("indeterminate overlap") != std::string::npos,
        "ambiguous contact must not be mislabeled positive overlap");
    requireFailedMdlFrame(s.d, s.frame);
  } else {
    s.accept("higher precision established near-contact classification");
  }
}
} // namespace
int main()
{
  int failures = 0;
  // Independent scenes keep one failing integration case from hiding the
  // other Renderer's preflight evidence. ANY failure still fails the test.
  for (const char *type : {"interactive", "quality"})
    for (auto test : {domainEdits,
             fieldDomains,
             transformedDomains,
             mixedDomains,
             ambiguity}) {
      try {
        Scene s(type);
        test(s);
      } catch (const std::exception &e) {
        fprintf(stderr, "FAIL (%s): %s\n", type, e.what());
        ++failures;
      }
    }
  return failures ? 1 : 0;
}
