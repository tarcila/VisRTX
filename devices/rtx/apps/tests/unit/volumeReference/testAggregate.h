// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "gpu/aggregateVolume.h"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace aggregate_test {
using namespace visrtx;
namespace av = aggregate_volume;
namespace ref = optical_reference;
constexpr int CASES = 25;

struct Blackbody
{
  VISRTX_HOST_DEVICE vec3 operator()(float) const;
};
VISRTX_HOST_DEVICE vec3 Blackbody::operator()(float) const
{
  return vec3(0.f);
}

struct Fixture
{
  PrincipledVolumeGPUData pv[2];
  TransferFunctionReferenceGPUData tf;
  StoredFieldReferenceGPUData field;
  float values[8];
  vec4 colors[2];
  VolumeOccurrenceGPUData occurrences[2];
  AggregateVolumeSceneGPUData scene;
  visrtx::Ray input{vec3(.5f, .5f, -1.f), vec3(0, 0, 1), {0, 4}};
  bool valid{true};
  VISRTX_HOST_DEVICE explicit Fixture(int id);
};

VISRTX_HOST_DEVICE Fixture::Fixture(int id)
{
  scene = {occurrences, 2, true};
  for (int i = 0; i < 8; ++i)
    values[i] = id == 19 ? 2.f : .5f;
  field.data = values;
  field.indexMax = ivec3(1);
  for (int i = 0; i < 2; ++i) {
    pv[i].bindings.bounds = {
        vec3(0, 0, i ? .5f : 0.f), vec3(1, 1, i ? 1.5f : 1.f)};
    pv[i].bindings.density.scale = float(i + 1);
    pv[i].bindings.scatteringColor.scale = vec3(0);
    pv[i].bindings.emission.scale = i ? vec3(0, 4, 0) : vec3(2, 0, 0);
    occurrences[i].placement = size_t(i);
    occurrences[i].groupSlot = 0;
    for (int c = 0; c < 3; ++c)
      occurrences[i].worldToObject[c][c] = 1.;
  }
  switch (id) {
  case 1:
    pv[1].bindings.bounds = {vec3(0, 0, 1), vec3(1, 1, 2)};
    break;
  case 2:
    pv[0].bindings.bounds = {vec3(0), vec3(1, 1, 3)};
    pv[1].bindings.bounds = {vec3(0, 0, 1), vec3(1, 1, 2)};
    break;
  case 3:
    input.org.z = .5f;
    break;
  case 4:
    input.org.x = 2.f;
    break;
  case 5:
    input.dir.z = 2.f;
    input.t.upper = 2.f;
    break;
  case 6:
    scene.count = 1;
    occurrences[0].worldToObject[2][2] = .5;
    break;
  case 7:
    scene.count = 1;
    input.org.y = .25f;
    occurrences[0].worldToObject[0][0] = .5;
    occurrences[0].worldToObject[1][1] = 2.;
    occurrences[0].worldToObject[2][2] = 1. / 3.;
    break;
  case 8:
    scene.count = 1;
    input.org.x = .25f;
    occurrences[0].worldToObject[0][2] = -.5;
    break;
  case 9:
    input.t = {1.25f, 2.25f};
    break;
  case 10:
    scene.count = 1;
    pv[0].bindings.density.scale = 0;
    pv[0].bindings.emission.scale = vec3(2, 3, 5);
    break;
  case 12:
    pv[1].bindings.bounds = {vec3(0), vec3(1)};
    pv[0].bindings.scatteringColor.scale = vec3(.5f);
    pv[1].bindings.scatteringColor.scale = vec3(.75f);
    pv[0].bindings.anisotropy.scale = .2f;
    pv[0].bindings.secondaryAnisotropy.scale = -.3f;
    pv[0].bindings.secondaryWeight.scale = .4f;
    pv[1].bindings.anisotropy.scale = -.5f;
    pv[1].bindings.secondaryAnisotropy.scale = .6f;
    pv[1].bindings.secondaryWeight.scale = .25f;
    break;
  case 13:
    input.org.z = 3.f;
    input.dir.z = -1.f;
    break;
  case 14:
    input.org.x = 0;
    break;
  case 15:
    scene.count = 1;
    break;
  case 17:
    scene.count = 1;
    input.org.z = 0;
    break;
  case 18:
    scene.count = 1;
    input.org.z = 0;
    pv[0].bindings.bounds = {vec3(0, 0, 1), vec3(1, 1, 1.f + 0x1p-23f)};
    break;
  case 23:
    scene.count = 1;
    input.org.z = -3.f;
    occurrences[0].worldToObject[2][2] = -.5;
    break;
  case 24:
    input.org.x = 1.f + 0x1p-23f;
    break;
  case 22:
    input.t.upper = std::numeric_limits<float>::infinity();
    break;
  case 21:
    pv[1].bindings.bounds = {vec3(0, 0, 2), vec3(1, 1, 3)};
    break;
  case 19:
    scene.count = 1;
    pv[0].fields = &field;
    pv[0].fieldCount = 1;
    pv[0].bindings.density.field = 0;
    pv[0].bindings.density.domain = {vec3(0), vec3(1, 1, .5f)};
    break;
  default:
    break;
  }
  for (int i = 0; i < 2; ++i) {
    const auto &b = pv[i].bindings;
    const float c = b.transmissionColor.scale.x;
    ref::EffectiveRanges ranges{
        {b.density.scale, b.density.scale}, {{c, c}, {c, c}, {c, c}}, {1, 1}};
    if (id == 19 && i == 0)
      ranges.density = {0, 2};
    valid &= ref::buildBounds(ranges, pv[i].opticalBounds);
    occurrences[i].volume = {VolumeType::PRINCIPLED, b.bounds, &pv[i], nullptr};
  }
  if (id == 11 || id == 15) {
    colors[0] = colors[1] = vec4(0, 2, 0, .5f);
    tf.field = field;
    tf.colors = colors;
    tf.colorCount = 2;
    tf.valueRange = {0, 1};
    tf.unitDistance = 2;
    valid &= ref::buildTransferBounds({.5f, .5f}, {2, 2}, tf.opticalBounds);
    occurrences[id == 11 ? 1 : 0].volume = {VolumeType::TF1D,
        id == 11 ? box3(vec3(0), vec3(1)) : box3(vec3(2), vec3(1)),
        nullptr,
        &tf};
  }
  if (id == 20) {
    const auto swap = occurrences[0];
    occurrences[0] = occurrences[1];
    occurrences[1] = swap;
  }
  if (id == 16) {
    occurrences[1] = occurrences[0];
    occurrences[1].groupSlot = 1; // same payload, distinct authored occurrence
  }
}

struct Result
{
  av::SegmentIntegral integral;
  double lower[8]{}, upper[8]{};
  size_t active[8]{};
  double kernel[3]{};
  int spans{0};
  int failure{0};
};

VISRTX_HOST_DEVICE Result execute(int id)
{
  Fixture f(id);
  Result result;
  if (!f.valid) {
    result.failure = 1;
    return result;
  }
  const auto ray = av::worldDistanceRay(f.input);
  double cursor = ray.lower;
  av::Interval span;
  Blackbody blackbody;
  while (av::nextSpan(f.scene, ray, cursor, span)) {
    if (result.spans == 8 || !(span.lower < span.upper)) {
      result.failure = 2;
      return result;
    }
    const int slot = result.spans++;
    result.lower[slot] = span.lower;
    result.upper[slot] = span.upper;
    const double middle = ref::add(
        span.lower, ref::multiply(.5, ref::subtract(span.upper, span.lower)));
    const auto sample =
        av::evaluate(f.scene, ray, span, middle, .25f, blackbody);
    result.active[slot] = sample.activeCount;
    const auto metadataBounds = av::boundsForSpan(f.scene, ray, span);
    for (int c = 0; c < 3; ++c) {
      if (metadataBounds.absorption[c] != sample.bounds.absorption[c]
          || metadataBounds.scattering[c] != sample.bounds.scattering[c])
        result.failure = 4;
      if (ref::extinction(sample.coefficients, c)
          > ref::majorant(sample.bounds))
        result.failure = 3;
      result.kernel[c] = sample.phaseKernel[c];
    }
    av::integrateSegment(sample.coefficients,
        sample.coefficients.emission,
        ref::subtract(span.upper, span.lower),
        result.integral);
  }
  return result;
}

#ifndef AGGREGATE_DEVICE_ONLY
// Oracles compile with the host C++ compiler, never as device long double.
std::vector<Result> deviceResults();

inline void require(bool ok, const char *message)
{
  if (!ok)
    throw std::runtime_error(message);
}
inline bool close(double a, long double b, double tolerance = 2e-10)
{
  return std::isfinite(a)
      && std::abs(static_cast<long double>(a) - b)
      <= tolerance * (1 + std::abs(b));
}
inline long double hg(long double g, long double cosine)
{
  const long double pi = acosl(-1.L);
  return (1 - g * g) / (4 * pi * powl(1 + g * g - 2 * g * cosine, 1.5L));
}

inline void check(int id, const Result &r)
{
  require(!r.failure, "execution/bound invariant failed");
  const long double a = -logl(static_cast<long double>(.36787944117144233f)),
                    b = 2 * a;
  long double color[3]{}, transmission = 1;
  int expectedSpans = 0;
  auto segment = [&](long double sigma,
                     long double red,
                     long double green,
                     long double blue,
                     long double length,
                     size_t active) {
    require(expectedSpans < r.spans && r.active[expectedSpans] == active,
        "wrong active occurrence count");
    require(close(r.upper[expectedSpans] - r.lower[expectedSpans], length),
        "wrong world-distance span");
    ++expectedSpans;
    const long double e = expl(-sigma * length);
    const long double f = sigma > 0 ? -expm1l(-sigma * length) / sigma : length;
    color[0] += transmission * red * f;
    color[1] += transmission * green * f;
    color[2] += transmission * blue * f;
    transmission *= e;
  };
  switch (id) {
  case 0:
  case 5:
  case 14:
  case 20:
  case 22:
    segment(a, 2, 0, 0, .5, 1);
    segment(a + b, 2, 4, 0, .5, 2);
    segment(b, 0, 4, 0, .5, 1);
    break;
  case 1:
  case 21:
    segment(a, 2, 0, 0, 1, 1);
    segment(b, 0, 4, 0, 1, 1);
    break;
  case 2:
    segment(a, 2, 0, 0, 1, 1);
    segment(a + b, 2, 4, 0, 1, 2);
    segment(a, 2, 0, 0, 1, 1);
    break;
  case 3:
    segment(a + b, 2, 4, 0, .5, 2);
    segment(b, 0, 4, 0, .5, 1);
    break;
  case 4:
  case 15:
  case 24:
    break;
  case 6:
  case 23:
    segment(a, 2, 0, 0, 2, 1);
    break;
  case 7:
    segment(a, 2, 0, 0, 3, 1);
    break;
  case 8:
    segment(a, 2, 0, 0, .5, 1);
    break;
  case 9:
    segment(a, 2, 0, 0, .25, 1);
    segment(a + b, 2, 4, 0, .5, 2);
    segment(b, 0, 4, 0, .25, 1);
    break;
  case 10:
    segment(0, 2, 3, 5, 1, 1);
    break;
  case 11:
    segment(a + logl(2.L) / 2, 2, logl(2.L), 0, 1, 2);
    break;
  case 12: {
    segment(a + b, 2, 4, 0, 1, 2);
    const long double g0 = .2f, g1 = -.3f, w = .4f;
    const long double p = (1 - w) * hg(g0, .25L) + w * hg(g1, .25L);
    const long double q = .75L * hg(-.5f, .25L) + .25L * hg(.6f, .25L);
    for (int c = 0; c < 3; ++c)
      require(close(r.kernel[c], .5L * a * p + .75L * b * q, 2e-6),
          "phase kernels not additive");
    break;
  }
  case 13:
    segment(b, 0, 4, 0, .5, 1);
    segment(a + b, 2, 4, 0, .5, 2);
    segment(a, 2, 0, 0, .5, 1);
    break;
  case 16:
    segment(2 * a, 4, 0, 0, 1, 2);
    break;
  case 17:
    segment(a, 2, 0, 0, 1, 1);
    break;
  case 18:
    segment(a, 2, 0, 0, 0x1p-23L, 1);
    break;
  case 19:
    segment(2 * a, 2, 0, 0, .5, 1);
    segment(0, 2, 0, 0, .5, 1);
    break;
  }
  require(r.spans == expectedSpans, "wrong span count");
  for (int i = 1; i < r.spans; ++i)
    require(id == 21 ? r.lower[i] == r.upper[i - 1] + 1.
                     : r.lower[i] == r.upper[i - 1],
        "gap/double integration between spans");
  for (int c = 0; c < 3; ++c) {
    require(close(r.integral.transmittance[c], transmission), "Beer oracle");
    require(close(r.integral.radiance[c], color[c]), "source integral oracle");
  }
  if (id == 0 || id == 1 || id == 2 || id == 5 || id == 6 || id == 7 || id == 8
      || id == 11 || id == 12 || id == 14 || id == 16 || id == 18 || id == 19
      || id == 20 || id == 21 || id == 22 || id == 23)
    require(r.lower[0] == 1., "entry world distance");
  if (id == 17 || id == 3)
    require(r.lower[0] == 0., "inside/entry clipping");
}

inline int run()
try {
  for (int i = 0; i < CASES; ++i)
    check(i, execute(i));
  // Tiny positive optical depth must keep the source limit instead of 0/0 or
  // loss/t after product underflow. This is a numerical primitive test.
  ref::Coefficients c;
  c.absorption[0] = std::numeric_limits<double>::min();
  const double source[3]{1, 0, 0};
  av::SegmentIntegral integral;
  av::integrateSegment(c, source, 0x1p-100, integral);
  require(integral.radiance[0] == 0x1p-100, "thin-source numerical limit");
#ifdef AGGREGATE_CUDA
  const auto host = deviceResults();
  for (int i = 0; i < CASES; ++i)
    check(i, host[i]);
#endif
  std::printf("Aggregate traversal: %d analytic cases pass\n", CASES);
  return 0;
} catch (const std::exception &e) {
  std::fprintf(stderr, "FAIL: %s\n", e.what());
  return 1;
}
#endif // AGGREGATE_DEVICE_ONLY
} // namespace aggregate_test
