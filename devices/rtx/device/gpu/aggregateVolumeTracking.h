// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "aggregateVolume.h"
#include "pcg.h"
#include "volumeTracking.h"

#include <cassert>

namespace visrtx {
namespace volume_tracking {
namespace av = aggregate_volume;

// Two PCG words provide 52 random bits. Half-bin centering is exactly
// representable in binary64, excluding both log singularities without clamping.
// This does not change pcg_uniform's existing [0,1) float convention/draw
// order.
VISRTX_HOST_DEVICE double openUnit(uint32_t high, uint32_t low)
{
  const uint64_t bits = (uint64_t(high >> 6) << 26) | uint64_t(low >> 6);
  return ref::multiply(ref::add(double(bits), .5), 0x1p-52);
}

struct Random
{
  PCGState *state;
  VISRTX_HOST_DEVICE double next();
};

VISRTX_HOST_DEVICE double Random::next()
{
  const uint32_t high = detail::pcg_advance(state);
  const uint32_t low = detail::pcg_advance(state);
  return openUnit(high, low);
}

// A strict World reference publication and matching finite reference bounds
// are preparation preconditions. Do not pass Interactive's permissive eligible
// subset merely because its scene.complete flag is true.
// The world-distance ray must resolve its intersections and candidate
// positions; this connector does not claim to fix the existing
// extreme-coordinate limits. Stream all occurrences/field cuts with constant
// storage and no work/overlap cap. Callback returns false only when its
// estimator has reached a terminal event.
template <typename RNG, typename Visitor>
VISRTX_HOST_DEVICE void candidates(const AggregateVolumeSceneGPUData &scene,
    const av::Ray &ray,
    double sourceProposal,
    RNG &random,
    Visitor &visitor)
{
  assert(scene.complete);
  double cursor = ray.lower;
  av::Interval span;
  if (!av::nextSpan(scene, ray, cursor, span))
    return;
  // nextSpan established membership; reuse the invariant single-occurrence
  // transform and global bound without changing candidates or random draws.
  av::ObjectRay singleRay;
  const av::ObjectRay *single = nullptr;
  double singleRate = 0.;
  if (scene.count == 1) {
    singleRay = av::objectRay(scene.occurrences[0], ray);
    single = &singleRay;
    singleRate =
        radianceRate(av::boundsForSpan(scene, ray, span), sourceProposal);
  }
  double tau = opticalDepth(random.next());
  do {
    const double rate = single
        ? singleRate
        : radianceRate(av::boundsForSpan(scene, ray, span), sourceProposal);
    const double length = ref::subtract(span.upper, span.lower);
    double offset = 0.;
    while (offset < length) {
      const auto step = advance(ref::subtract(length, offset), rate, tau);
      if (!step.collision)
        break;
      offset = ref::add(offset, step.distance);
      const double distance = ref::add(span.lower, offset);
      if (!visitor(scene, ray, span, distance, rate, single))
        return;
      tau = opticalDepth(random.next());
    }
  } while (av::nextSpan(scene, ray, cursor, span));
}

enum class Endpoint
{
  Boundary,
  Absorbed,
  Scattered
};
struct FlightResult
{
  Endpoint endpoint{Endpoint::Boundary};
  double distance{0.};
  av::Interval span;
  ref::Coefficients coefficients;
};

// The transport candidate consumes coefficients, not the preview phase kernel
// or a second reduction of bounds. Keep the exact reference coefficient graph.
template <typename Blackbody>
VISRTX_HOST_DEVICE ref::Coefficients sourceCoefficientsAt(
    const AggregateVolumeSceneGPUData &scene,
    const av::Ray &ray,
    av::Interval span,
    double distance,
    Blackbody &blackbody,
    const av::ObjectRay *single = nullptr)
{
  ref::Coefficients result;
  for (size_t i = 0; i < scene.count; ++i) {
    const auto &o = scene.occurrences[i];
    const auto local = single ? *single : av::objectRay(o, ray);
    if (!single) {
      av::Interval interval;
      if (!av::intersectDomain(
              o.volume.domain, local, ray.lower, ray.upper, interval)
          || !(interval.lower <= span.lower && span.lower < interval.upper))
        continue;
    }
    const vec3 p = av::objectPosition(local, distance);
    ref::Coefficients value;
    if (o.volume.type == VolumeType::PRINCIPLED) {
      const auto &v = *o.volume.principled;
      ReferenceFieldSampler sampler{v.fields};
      auto &&thermal = bindVolumeBlackbody(blackbody, v.qualityCold);
      value = ref::evaluateBindings(v.bindings,
          p,
          sampler,
          thermal,
          v.hasConstantTransmissionLog ? v.constantTransmissionLog : nullptr);
    } else if (o.volume.type == VolumeType::TF1D
        && ref::contains(o.volume.domain, p)) {
      const auto &v = *o.volume.transfer;
      ReferenceFieldSampler sampler{&v.field};
      value = ref::evaluateTransfer(
          av::classify(v, sampler.sample<float>(0, p)), v.unitDistance);
    }
    result = ref::combine(result, value);
  }
  return result;
}
VISRTX_HOST_DEVICE ref::Coefficients extinctionAt(
    const AggregateVolumeSceneGPUData &scene,
    const av::Ray &ray,
    av::Interval span,
    double distance,
    const av::ObjectRay *single = nullptr);

template <typename RNG, typename Blackbody>
struct RadianceVisitor
{
  RNG &random;
  Blackbody &blackbody;
  PathState &path;
  FlightResult result;
  VISRTX_HOST_DEVICE bool operator()(const AggregateVolumeSceneGPUData &scene,
      const av::Ray &ray,
      av::Interval span,
      double distance,
      double rate,
      const av::ObjectRay *single);
};

template <typename RNG, typename Blackbody>
VISRTX_HOST_DEVICE bool RadianceVisitor<RNG, Blackbody>::operator()(
    const AggregateVolumeSceneGPUData &scene,
    const av::Ray &ray,
    av::Interval span,
    double distance,
    double rate,
    const av::ObjectRay *single)
{
  const auto coefficients = path.sourceOwned
      ? sourceCoefficientsAt(scene, ray, span, distance, blackbody, single)
      : extinctionAt(scene, ray, span, distance, single);
  const auto event = collide(coefficients, rate, random.next(), path);
  if (event.event == Event::Null)
    return true;
  result = {
      event.event == Event::Absorb ? Endpoint::Absorbed : Endpoint::Scattered,
      distance,
      span,
      coefficients};
  return false;
}

// Caller preserves path state across nulls/intervals and supplies a separately
// seeded stream for shadow connections. No physical Beer factor on escape.
// path.sourceOwned is preserved through nulls/domain boundaries; callers change
// it only after a true surface/medium vertex, never according to a NEE outcome.
template <typename RNG, typename Blackbody>
VISRTX_HOST_DEVICE FlightResult trace(const AggregateVolumeSceneGPUData &scene,
    const av::Ray &ray,
    double sourceProposal,
    RNG &random,
    Blackbody &blackbody,
    PathState &path)
{
  assert(sourceProposal > 0.);
  RadianceVisitor<RNG, Blackbody> visitor{random, blackbody, path, {}};
  visitor.result.distance = ray.upper;
  // A true vertex with explicit source ownership has no continuation source
  // term left to estimate. Its flight needs only the certified optical rate;
  // a positive vacuum-source floor would generate pure no-op null events.
  // Camera, transmission/delta and unqualified closures retain the full floor.
  candidates(
      scene, ray, path.sourceOwned ? sourceProposal : 0., random, visitor);
  return visitor.result;
}

// Same reference optical graph as aggregate evaluation, without source or phase
// queries. Density support is not used to skip radiance intervals.
VISRTX_HOST_DEVICE ref::Coefficients extinctionAt(
    const AggregateVolumeSceneGPUData &scene,
    const av::Ray &ray,
    av::Interval span,
    double distance,
    const av::ObjectRay *single)
{
  ref::Coefficients result;
  for (size_t i = 0; i < scene.count; ++i) {
    const auto &occurrence = scene.occurrences[i];
    const auto local = single ? *single : av::objectRay(occurrence, ray);
    if (!single) {
      av::Interval interval;
      if (!av::intersectDomain(
              occurrence.volume.domain, local, ray.lower, ray.upper, interval)
          || !(interval.lower <= span.lower && span.lower < interval.upper))
        continue;
    }
    const vec3 p = av::objectPosition(local, distance);
    ref::Coefficients value;
    if (occurrence.volume.type == VolumeType::PRINCIPLED) {
      const auto &volume = *occurrence.volume.principled;
      const auto &b = volume.bindings;
      if (!ref::contains(b.bounds, p))
        continue;
      ReferenceFieldSampler sampler{volume.fields};
      PrincipledVolumeValues v;
      v.density = ref::evaluateBinding(b.density, p, sampler);
      v.scatteringColor = ref::evaluateBinding(b.scatteringColor, p, sampler);
      v.transmissionColor =
          ref::evaluateBinding(b.transmissionColor, p, sampler);
      v.transmissionDistance =
          ref::evaluateBinding(b.transmissionDistance, p, sampler);
      value = ref::evaluateOptical(v,
          volume.hasConstantTransmissionLog ? volume.constantTransmissionLog
                                            : nullptr);
    } else if (occurrence.volume.type == VolumeType::TF1D
        && ref::contains(occurrence.volume.domain, p)) {
      const auto &volume = *occurrence.volume.transfer;
      ReferenceFieldSampler sampler{&volume.field};
      const float alpha = av::classify(volume, sampler.sample<float>(0, p)).w;
      value = ref::evaluateTransfer(vec4(0, 0, 0, alpha), volume.unitDistance);
    }
    result = ref::combine(result, value);
  }
  return result;
}

struct ShadowVisitor
{
  double (&transmission)[3];
  VISRTX_HOST_DEVICE bool operator()(const AggregateVolumeSceneGPUData &scene,
      const av::Ray &ray,
      av::Interval span,
      double distance,
      double rate,
      const av::ObjectRay *single);
};
VISRTX_HOST_DEVICE bool ShadowVisitor::operator()(
    const AggregateVolumeSceneGPUData &scene,
    const av::Ray &ray,
    av::Interval span,
    double distance,
    double rate,
    const av::ObjectRay *single)
{
  shadowCandidate(
      extinctionAt(scene, ray, span, distance, single), rate, transmission);
  return transmission[0] != 0. || transmission[1] != 0.
      || transmission[2] != 0.;
}

template <typename RNG>
VISRTX_HOST_DEVICE void shadow(const AggregateVolumeSceneGPUData &scene,
    const av::Ray &ray,
    RNG &random,
    double (&transmission)[3])
{
  ShadowVisitor visitor{transmission};
  // No source proposal on shadows: exactly transparent spans consume no
  // optical depth, and their source/temperature fields are never evaluated.
  candidates(scene, ray, 0., random, visitor);
}

} // namespace volume_tracking
} // namespace visrtx
