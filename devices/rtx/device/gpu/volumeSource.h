// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "aggregateVolumeTracking.h"
#include "sourceImportance.h"

namespace visrtx {
namespace volume_source {
namespace av = aggregate_volume;
namespace ref = optical_reference;

VISRTX_HOST_DEVICE bool eligible(const VolumeOccurrenceGPUData &o)
{
  return o.volume.mightEmit && ref::nonemptyDomain(o.volume.domain);
}
VISRTX_HOST_DEVICE bool available(const AggregateVolumeSceneGPUData &scene)
{
  return scene.complete && scene.sourceSamplingReady && scene.sourceCount > 0;
}

// Exact discrete uniform selection without a float CDF or an occurrence cap.
VISRTX_HOST_DEVICE size_t selectIndex(PCGState &state, size_t count)
{
  const uint64_t n = count, threshold = (uint64_t(0) - n) % n;
  uint64_t bits;
  do {
    const uint64_t high = detail::pcg_advance(&state);
    bits = (high << 32) | detail::pcg_advance(&state);
  } while (bits < threshold);
  return size_t(bits % n);
}
struct Sample
{
  bool valid{false};
  size_t occurrence{0};
  av::Ray ray; // receiver -> selected source point; unit WORLD direction
  double weight[3]{}; // j_i / (p_i * q_omega * q_distance), plus any
                      // resampling correction; never add another r^2
};

VISRTX_HOST_DEVICE glm::dvec3 direction(
    const glm::dvec3 &axis, double cosine, double sine, double azimuth)
{
  const glm::dvec3 up =
      std::abs(axis.z) < .9 ? glm::dvec3(0, 0, 1) : glm::dvec3(0, 1, 0);
  const auto u = glm::normalize(glm::cross(up, axis));
  const auto v = glm::cross(axis, u);
  return cosine * axis + sine * (cos(azimuth) * u + sin(azimuth) * v);
}

// Labeled occurrence estimator. Full-sphere support remains in every outside
// proposal; a cone is an importance heuristic, never a source-support proof.
// Very narrow numerically unresolved cones use the full sphere instead, not a
// clipped/omitted source. Misses and zero-source samples are never redrawn.
template <typename Blackbody>
VISRTX_HOST_DEVICE Sample sampleOne(const AggregateVolumeSceneGPUData &scene,
    const glm::dvec3 &receiver,
    PCGState &state,
    Blackbody &blackbody)
{
  Sample result;
  if (!available(scene))
    return result;
  const size_t selected = selectIndex(state, scene.sourceCount);
  size_t ordinal = 0;
  for (size_t i = 0; i < scene.count; ++i)
    if (eligible(scene.occurrences[i])) {
      if (ordinal++ == selected) {
        result.occurrence = i;
        break;
      }
    }
  const auto &o = scene.occurrences[result.occurrence];
  const auto offset = o.sourceCenter - receiver;
  const double distance = glm::length(offset);
  double width = 0.;
  if (distance > o.sourceRadius) {
    const double ratio = o.sourceRadius / distance;
    width = (ratio * ratio) / (1. + sqrt(1. - ratio * ratio));
  }
  // This changes proposal choice, not optical/source values or their support.
  const bool cone = width >= 0x1p-24;
  const glm::dvec3 axis = cone ? offset / distance : glm::dvec3(0, 0, 1);
  volume_tracking::Random random{&state};
  const bool guided = hasImportance(o);
  const bool fromGrid = guided && random.next() >= .1;
  constexpr double PI = 3.1415926535897932384626433832795;
  glm::dvec3 omega;
  double t = 0.;
  if (fromGrid) {
    const auto &grid = o.volume.sourceImportance;
    const uint32_t n = grid.resolution,
                   cell = importanceCell(grid, random.next());
    const uint32_t xyz[3]{cell % n, (cell / n) % n, cell / (n * n)};
    glm::dvec3 object;
    for (int c = 0; c < 3; ++c)
      object[c] = ref::add(ref::promote(o.volume.domain.lower[c]),
          ref::multiply(
              ref::divide(ref::add(double(xyz[c]), random.next()), double(n)),
              ref::subtract(ref::promote(o.volume.domain.upper[c]),
                  ref::promote(o.volume.domain.lower[c]))));
    glm::dvec3 world;
    for (int c = 0; c < 3; ++c) {
      world[c] = o.sourceObjectToWorld[c][3];
      for (int k = 0; k < 3; ++k)
        world[c] = ref::add(
            world[c], ref::multiply(o.sourceObjectToWorld[c][k], object[k]));
    }
    const auto displacement = world - receiver;
    t = glm::length(displacement);
    // Exact coincidence is a zero-measure geometric outcome, not a distance
    // floor or a request to resample. Near-source weights use the full mixture.
    if (t == 0.)
      return result;
    omega = displacement / t;
  } else {
    const double branch = random.next(), polar = random.next(),
                 azimuth = random.next();
    if (cone && branch >= .125) {
      const double loss = polar * width;
      omega = direction(
          axis, 1. - loss, sqrt(loss * (2. - loss)), 2. * PI * azimuth);
    } else {
      const double z = 1. - 2. * polar;
      omega = direction(
          glm::dvec3(0, 0, 1), z, sqrt((1. - z) * (1. + z)), 2. * PI * azimuth);
    }
  }
  const double q = cone
      ? (.125 / (4. * PI)
            + (glm::dot(axis, omega) >= 1. - width ? .875 / (2. * PI * width)
                                                   : 0.))
      : 1. / (4. * PI);
  for (int c = 0; c < 3; ++c) {
    result.ray.origin[c] = receiver[c];
    result.ray.direction[c] = omega[c];
  }
  result.ray.lower = 0.;
  result.ray.upper = std::numeric_limits<double>::infinity();
  av::Interval interval;
  if (!av::intersect(o, result.ray, interval))
    return result;
  const double length = ref::subtract(interval.upper, interval.lower);
  if (!fromGrid)
    t = ref::add(interval.lower, ref::multiply(random.next(), length));
  const auto localRay = av::objectRay(o, result.ray);
  const vec3 p = av::objectPosition(localRay, t);
  ref::Coefficients coefficients;
  if (o.volume.type == VolumeType::PRINCIPLED) {
    const auto &v = *o.volume.principled;
    ReferenceFieldSampler sampler{v.fields};
    auto &&thermal = bindVolumeBlackbody(blackbody, v.qualityCold);
    coefficients = ref::evaluateBindings(v.bindings,
        p,
        sampler,
        thermal,
        v.hasConstantTransmissionLog ? v.constantTransmissionLog : nullptr);
  } else if (o.volume.type == VolumeType::TF1D
      && ref::contains(o.volume.domain, p)) {
    const auto &v = *o.volume.transfer;
    ReferenceFieldSampler sampler{&v.field};
    coefficients = ref::evaluateTransfer(
        av::classify(v, sampler.sample<float>(0, p)), v.unitDistance);
  }
  double factor =
      ref::divide(ref::multiply(length, double(scene.sourceCount)), q);
  if (guided) {
    glm::dvec3 objectPoint;
    for (int c = 0; c < 3; ++c)
      objectPoint[c] =
          ref::add(localRay.origin[c], ref::multiply(localRay.direction[c], t));
    // Both components are densities in dOmega dr. The spatial proposal's
    // Jacobian is r^2; evaluate the WHOLE mixture regardless of the draw
    // origin.
    const double spatial = importanceDensity(o, objectPoint);
    const double joint = ref::add(ref::multiply(.1, ref::divide(q, length)),
        ref::multiply(.9, ref::multiply(spatial, ref::multiply(t, t))));
    factor = ref::divide(double(scene.sourceCount), joint);
  }
  for (int c = 0; c < 3; ++c)
    result.weight[c] = ref::multiply(coefficients.emission[c], factor);
  result.ray.upper = t;
  result.valid = true;
  return result;
}
// Two independent candidates, one visibility query. Conditional expectation
// of the selected weight is exactly (a.weight + b.weight)/2. The selection
// floor protects support even when normalized scores underflow.
template <typename Blackbody>
VISRTX_HOST_DEVICE VISRTX_NOINLINE Sample sampleResampled(
    const AggregateVolumeSceneGPUData &scene,
    const glm::dvec3 &receiver,
    PCGState &state,
    Blackbody &blackbody)
{
  const auto a = sampleOne(scene, receiver, state, blackbody);
  const auto b = sampleOne(scene, receiver, state, blackbody);
  double sa = 0., sb = 0.;
  for (int c = 0; c < 3; ++c) {
    if (a.valid)
      sa = sa > std::abs(a.weight[c]) ? sa : std::abs(a.weight[c]);
    if (b.valid)
      sb = sb > std::abs(b.weight[c]) ? sb : std::abs(b.weight[c]);
  }
  const double scale = sa > sb ? sa : sb;
  if (scale == 0.)
    return {};
  const double ma = sa / scale, mb = sb / scale;
  const double probability = .125 + .75 * ma / (ma + mb);
  volume_tracking::Random random{&state};
  const bool chooseA = random.next() < probability;
  auto result = chooseA ? a : b;
  const double correction = 2. * (chooseA ? probability : 1. - probability);
  for (int c = 0; c < 3; ++c)
    result.weight[c] /= correction;
  return result;
}
template <typename Blackbody>
VISRTX_HOST_DEVICE Sample sample(const AggregateVolumeSceneGPUData &scene,
    const glm::dvec3 &receiver,
    PCGState &state,
    Blackbody &blackbody,
    bool resampleCandidates = false)
{
  return resampleCandidates ? sampleResampled(scene, receiver, state, blackbody)
                            : sampleOne(scene, receiver, state, blackbody);
}
} // namespace volume_source
} // namespace visrtx
