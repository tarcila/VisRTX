// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "gpu/aggregateVolume.h"
#include "world/World.h"

#include <anari/ext/visrtx/makeVisRTXDevice.h>
#include <anari/anari_cpp.hpp>

#include <atomic>
#include <cstdio>
#include <cstring>
#include <set>
#include <stdexcept>
#include <vector>

using namespace visrtx;
namespace av = aggregate_volume;
namespace ref = optical_reference;
static std::atomic<int> errors{0}, unavailable{0}, omitted{0};
static void require(bool ok, const char *message)
{
  if (!ok)
    throw std::runtime_error(message);
}
static void checked(cudaError_t error)
{
  if (error != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(error));
}
static void status(const void *,
    ANARIDevice,
    ANARIObject,
    ANARIDataType,
    ANARIStatusSeverity severity,
    ANARIStatusCode,
    const char *message)
{
  if (severity <= ANARI_SEVERITY_ERROR) {
    ++errors;
    std::fprintf(stderr, "ANARI: %s\n", message);
  }
  if (std::strstr(message, "no partial scene view published"))
    ++unavailable;
  if (std::strstr(message, "interactive/default omits this volume"))
    ++omitted;
}
struct Objects
{
  ANARIDevice d{};
  std::vector<ANARIObject> handles;
  ~Objects();
  template <typename T>
  T keep(T value);
};
Objects::~Objects()
{
  for (auto i = handles.rbegin(); i != handles.rend(); ++i)
    anariRelease(d, *i);
  anariRelease(d, d);
}
template <typename T>
T Objects::keep(T value)
{
  handles.push_back(reinterpret_cast<ANARIObject>(value));
  return value;
}
static bool valid(Objects &o, ANARIObject object)
{
  bool value = false;
  require(
      anariGetProperty(
          o.d, object, "valid", ANARI_BOOL, &value, sizeof(value), ANARI_WAIT),
      "valid property");
  return value;
}
static void volumes(
    Objects &o, ANARIObject parent, const std::vector<ANARIVolume> &values)
{
  auto array = o.keep(anariNewArray1D(
      o.d, nullptr, nullptr, nullptr, ANARI_VOLUME, values.size()));
  std::memcpy(anariMapArray(o.d, array),
      values.data(),
      values.size() * sizeof(ANARIVolume));
  anariUnmapArray(o.d, array);
  anariSetParameter(o.d, parent, "volume", ANARI_ARRAY1D, &array);
}
static AggregateVolumeSceneGPUData prepare(
    Objects &o, ANARIWorld world, bool omitUnavailable = false)
{
  box3 bounds;
  require(anariGetProperty(o.d,
              world,
              "bounds",
              ANARI_FLOAT32_BOX3,
              &bounds,
              sizeof(bounds),
              ANARI_WAIT),
      "world prepare");
  // Private prepared-view seam after public commits/WAIT, not a new test API.
  return reinterpret_cast<World *>(world)->referenceVolumes(omitUnavailable);
}
struct Blackbody
{
  VISRTX_HOST_DEVICE vec3 operator()(float) const;
};
VISRTX_HOST_DEVICE vec3 Blackbody::operator()(float) const
{
  return vec3(0.f);
}
struct Result
{
  av::SegmentIntegral integral;
  size_t spans{0}, peak{0};
  bool bounded{true};
};
VISRTX_GLOBAL void consumeWorld(
    AggregateVolumeSceneGPUData scene, Result *result)
{
  const av::Ray ray =
      av::worldDistanceRay({vec3(.5f, .5f, -1.f), vec3(0, 0, 1), {0, 4}});
  Result r;
  double cursor = ray.lower;
  av::Interval span;
  Blackbody blackbody;
  while (av::nextSpan(scene, ray, cursor, span)) {
    const auto sample = av::evaluate(
        scene, ray, span, (span.lower + span.upper) * .5, .25f, blackbody);
    ++r.spans;
    r.peak = r.peak > sample.activeCount ? r.peak : sample.activeCount;
    for (int c = 0; c < 3; ++c)
      r.bounded &= ref::extinction(sample.coefficients, c)
          <= ref::majorant(sample.bounds);
    av::integrateSegment(sample.coefficients,
        sample.coefficients.emission,
        span.upper - span.lower,
        r.integral);
  }
  *result = r;
}
static Result consume(AggregateVolumeSceneGPUData scene)
{
  require(scene.complete, "cannot consume incomplete scene");
  DeviceBuffer output;
  output.reserve(sizeof(Result));
  consumeWorld<<<1, 1>>>(scene, output.ptrAs<Result>());
  checked(cudaGetLastError());
  checked(cudaDeviceSynchronize());
  Result result;
  checked(cudaMemcpy(
      &result, output.ptr(), sizeof(result), cudaMemcpyDeviceToHost));
  require(result.bounded, "aggregate exceeds matching bound");
  return result;
}
static void close(double a, double b)
{
  require(std::isfinite(a) && std::abs(a - b) <= 1e-10 * std::abs(b) + 1e-300,
      "world analytic oracle");
}
static void expect(Result r,
    double a,
    double b,
    int pv1,
    int tf1,
    int pv2 = 0,
    int tf2 = 0,
    double blue = 0.)
{
  double transmission = 1.;
  double color[3]{};
  auto segment = [&](int pv, int tf) {
    const double sigma = pv * a + tf * b;
    if (pv + tf == 0)
      return;
    const double f = sigma > 0 ? -std::expm1(-sigma) / sigma : 1.;
    color[0] += transmission * 2. * pv * f;
    color[1] += transmission * 2. * tf * b * f;
    color[2] += transmission * blue * f;
    transmission *= std::exp(-sigma);
  };
  segment(pv1, tf1);
  segment(pv2, tf2);
  require(r.spans == size_t(1 + (pv2 + tf2 > 0)), "wrong world span count");
  for (int c = 0; c < 3; ++c) {
    close(r.integral.transmittance[c], transmission);
    close(r.integral.radiance[c], color[c]);
  }
}

int main()
try {
  Objects o;
  o.d = makeVisRTXDevice(status);
  anariCommitParameters(o.d, o.d);
  auto pv = o.keep(anariNewVolume(o.d, "principledVolume"));
  const box3 domain(vec3(0), vec3(1));
  const vec3 black(0), red(2, 0, 0), green(0, 2, 0);
  anariSetParameter(o.d, pv, "bounds", ANARI_FLOAT32_BOX3, &domain);
  anariSetParameter(o.d, pv, "scatteringColor", ANARI_FLOAT32_VEC3, &black);
  anariSetParameter(o.d, pv, "emission", ANARI_FLOAT32_VEC3, &red);
  anariCommitParameters(o.d, pv);
  auto array = o.keep(
      anariNewArray3D(o.d, nullptr, nullptr, nullptr, ANARI_FLOAT32, 2, 2, 2));
  auto *values = static_cast<float *>(anariMapArray(o.d, array));
  std::fill(values, values + 8, .5f);
  anariUnmapArray(o.d, array);
  auto field = o.keep(anariNewSpatialField(o.d, "structuredRegular"));
  anariSetParameter(o.d, field, "data", ANARI_ARRAY3D, &array);
  anariCommitParameters(o.d, field);
  auto tf = o.keep(anariNewVolume(o.d, "transferFunction1D"));
  anariSetParameter(o.d, tf, "value", ANARI_SPATIAL_FIELD, &field);
  anariSetParameter(o.d, tf, "color", ANARI_FLOAT32_VEC3, &green);
  float opacity = .5f, distance = 2.f;
  anariSetParameter(o.d, tf, "opacity", ANARI_FLOAT32, &opacity);
  anariSetParameter(o.d, tf, "unitDistance", ANARI_FLOAT32, &distance);
  anariCommitParameters(o.d, tf);
  auto group = o.keep(anariNewGroup(o.d));
  volumes(o, group, {pv, tf, pv});
  anariCommitParameters(o.d, group);
  auto instance = o.keep(anariNewInstance(o.d, "transform"));
  anariSetParameter(o.d, instance, "group", ANARI_GROUP, &group);
  auto transforms = o.keep(
      anariNewArray1D(o.d, nullptr, nullptr, nullptr, ANARI_FLOAT32_MAT4, 2));
  auto *matrices = static_cast<mat4 *>(anariMapArray(o.d, transforms));
  matrices[0] = mat4(1);
  matrices[1] = mat4(1);
  matrices[1][2][2] = 2.f;
  anariUnmapArray(o.d, transforms);
  anariSetParameter(o.d, instance, "transform", ANARI_ARRAY1D, &transforms);
  const uint32_t sameID = 7;
  anariSetParameter(o.d, instance, "id", ANARI_UINT32, &sameID);
  anariCommitParameters(o.d, instance);
  auto world = o.keep(anariNewWorld(o.d));
  volumes(o, world, {pv});
  auto instances = o.keep(
      anariNewArray1D(o.d, nullptr, nullptr, nullptr, ANARI_INSTANCE, 2));
  auto *handles = static_cast<ANARIInstance *>(anariMapArray(o.d, instances));
  handles[0] = handles[1] = instance;
  anariUnmapArray(o.d, instances);
  anariSetParameter(o.d, world, "instance", ANARI_ARRAY1D, &instances);
  anariCommitParameters(o.d, world);
  auto scene = prepare(o, world);
  require(scene.complete && scene.count == 13,
      "authored occurrences omitted/deduplicated");
  const auto unchanged = prepare(o, world);
  require(unchanged.complete && unchanged.generation == scene.generation
          && unchanged.occurrences == scene.occurrences,
      "read-only preparation replaced an in-use publication");
  std::vector<VolumeOccurrenceGPUData> records(scene.count);
  checked(cudaMemcpy(records.data(),
      scene.occurrences,
      records.size() * sizeof(records[0]),
      cudaMemcpyDeviceToHost));
  std::set<std::pair<size_t, size_t>> keys;
  for (const auto &r : records)
    keys.emplace(r.placement, r.groupSlot);
  require(keys.size() == 13, "occurrence keys alias");
  const double a = -std::log(double(.36787944117144233f));
  double b = std::log(2.) / 2.;
  auto result = consume(scene);
  require(result.peak == 13, "overlap lost");
  expect(result, a, b, 9, 4, 4, 2);
  const auto generation = scene.generation;
  opacity = .25f;
  anariSetParameter(o.d, tf, "opacity", ANARI_FLOAT32, &opacity);
  anariCommitParameters(o.d, tf);
  scene = prepare(o, world);
  require(scene.complete && scene.generation != generation,
      "new publication identity");
  b = -std::log(.75) / 2.;
  expect(consume(scene), a, b, 9, 4, 4, 2);
  const bool hidden = false, visible = true;
  anariSetParameter(o.d, pv, "visible", ANARI_BOOL, &hidden);
  anariCommitParameters(o.d, pv);
  scene = prepare(o, world);
  require(scene.complete && scene.count == 4, "visibility filtering");
  records.resize(scene.count);
  checked(cudaMemcpy(records.data(),
      scene.occurrences,
      records.size() * sizeof(records[0]),
      cudaMemcpyDeviceToHost));
  for (const auto &r : records)
    require(
        r.groupSlot == 1, "authored slot compressed after visibility change");
  expect(consume(scene), a, b, 0, 4, 0, 2);
  anariSetParameter(o.d, pv, "visible", ANARI_BOOL, &visible);
  anariCommitParameters(o.d, pv);
  anariUnsetParameter(o.d, instance, "transform");
  anariCommitParameters(o.d, instance);
  scene = prepare(o, world);
  require(scene.complete && scene.count == 7, "transform-array update");
  expect(consume(scene), a, b, 5, 2);

  // Unavailable reference metadata must not publish just the preceding PV.
  distance = INFINITY;
  anariSetParameter(o.d, tf, "unitDistance", ANARI_FLOAT32, &distance);
  anariCommitParameters(o.d, tf);
  scene = prepare(o, world);
  require(!scene.complete && !scene.occurrences && scene.count == 0,
      "partial aggregate published");
  require(valid(o, tf) && valid(o, pv) && unavailable > 0,
      "reference failure poisoned intrinsic objects or lacked diagnostic");
  // Interactive explicitly opts into diagnosed renderer-local eligibility.
  // Its cached eligible view must never satisfy a subsequent strict request.
  scene = prepare(o, world, true);
  require(scene.complete && scene.count == 5 && omitted > 0,
      "renderer-local omission or diagnostic missing");
  expect(consume(scene), a, b, 5, 0);
  const int omissionWarnings = omitted;
  const auto eligibleAgain = prepare(o, world, true);
  require(eligibleAgain.generation == scene.generation
          && omitted == omissionWarnings,
      "unchanged eligible request rebuilt or repeated diagnostics");
  scene = prepare(o, world);
  require(!scene.complete && !scene.occurrences && scene.count == 0,
      "strict request reused an incomplete eligible subset");
  require(valid(o, tf) && valid(o, pv), "omission poisoned intrinsic validity");
  distance = 2;
  anariSetParameter(o.d, tf, "unitDistance", ANARI_FLOAT32, &distance);
  anariCommitParameters(o.d, tf);
  scene = prepare(o, world);
  require(scene.complete && scene.count == 7, "reference recovery");
  expect(consume(scene), a, b, 5, 2);

  auto many = o.keep(anariNewWorld(o.d));
  volumes(o, many, std::vector<ANARIVolume>(257, pv));
  anariCommitParameters(o.d, many);
  scene = prepare(o, many);
  require(scene.complete && scene.count == 257, "hidden occurrence cap");
  result = consume(scene);
  require(result.peak == 257, "active overlap cap");
  expect(result, a, b, 257, 0);

  // TF-only consumers use independent per-volume classifications on one field.
  auto other = o.keep(anariNewVolume(o.d, "transferFunction1D"));
  const vec3 blueColor(0, 0, 3);
  const float alphaOther = .75f;
  anariSetParameter(o.d, other, "value", ANARI_SPATIAL_FIELD, &field);
  anariSetParameter(o.d, other, "color", ANARI_FLOAT32_VEC3, &blueColor);
  anariSetParameter(o.d, other, "opacity", ANARI_FLOAT32, &alphaOther);
  anariCommitParameters(o.d, other);
  auto tfWorld = o.keep(anariNewWorld(o.d));
  volumes(o, tfWorld, {tf, other});
  anariCommitParameters(o.d, tfWorld);
  scene = prepare(o, tfWorld);
  require(scene.complete && scene.count == 2, "TF-only reference");
  result = consume(scene);
  const double otherT = std::log(4.), sum = b + otherT,
               factor = -std::expm1(-sum) / sum;
  for (int c = 0; c < 3; ++c)
    close(result.integral.transmittance[c], std::exp(-sum));
  close(result.integral.radiance[0], 0);
  close(result.integral.radiance[1], 2 * b * factor);
  close(result.integral.radiance[2], 3 * otherT * factor);
  // A nonconstant classification must follow field edits, not just cached
  // optical bounds. The existing 256-bin TF interpolation is an approximation
  // to this linear ramp; use a declared float classification tolerance.
  auto colors = o.keep(
      anariNewArray1D(o.d, nullptr, nullptr, nullptr, ANARI_FLOAT32_VEC3, 2));
  auto *ramp = static_cast<vec3 *>(anariMapArray(o.d, colors));
  ramp[0] = vec3(0);
  ramp[1] = green;
  anariUnmapArray(o.d, colors);
  anariSetParameter(o.d, tf, "color", ANARI_ARRAY1D, &colors);
  anariCommitParameters(o.d, tf);
  auto rampWorld = o.keep(anariNewWorld(o.d));
  volumes(o, rampWorld, {tf});
  anariCommitParameters(o.d, rampWorld);
  scene = prepare(o, rampWorld);
  require(scene.complete && scene.count == 1, "ramp reference");
  result = consume(scene);
  require(std::abs(result.integral.radiance[1] - (-std::expm1(-b))) < 2e-6,
      "TF ramp classification");
  values = static_cast<float *>(anariMapArray(o.d, array));
  std::fill(values, values + 8, 1.f);
  anariUnmapArray(o.d, array);
  scene = prepare(o, rampWorld);
  result = consume(scene);
  require(std::abs(result.integral.radiance[1] - 2 * (-std::expm1(-b))) < 2e-6,
      "TF field update lost without volume recommit");
  auto empty = o.keep(anariNewWorld(o.d));
  anariCommitParameters(o.d, empty);
  scene = prepare(o, empty);
  require(scene.complete && scene.count == 0, "empty scene not complete");
  result = consume(scene);
  require(result.spans == 0 && result.integral.transmittance[0] == 1.
          && result.integral.radiance[0] == 0.,
      "empty scene transport");
  require(errors == 0, "unexpected ANARI errors");
  std::puts(
      "World volume references: complete occurrences, instances, mixed optics and recovery pass");
  return 0;
} catch (const std::exception &e) {
  std::fprintf(stderr, "FAIL: %s\n", e.what());
  return 1;
}
