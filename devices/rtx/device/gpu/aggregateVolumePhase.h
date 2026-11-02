// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "aggregateVolumeTracking.h"

namespace visrtx {
namespace volume_tracking {

struct PhaseComponent
{
  double scattering[3]{};
  VolumePhaseFunction phase;
};

VISRTX_HOST_DEVICE PhaseComponent phaseComponent(
    const VolumeOccurrenceGPUData &occurrence,
    const av::Ray &ray,
    av::Interval span,
    double distance)
{
  PhaseComponent result;
  av::Interval interval;
  if (occurrence.volume.type != VolumeType::PRINCIPLED
      || !av::intersect(occurrence, ray, interval)
      || !(interval.lower <= span.lower && span.lower < interval.upper))
    return result;
  const auto &volume = *occurrence.volume.principled;
  const auto &b = volume.bindings;
  const vec3 p = av::objectPosition(av::objectRay(occurrence, ray), distance);
  ReferenceFieldSampler sampler{volume.fields};
  PrincipledVolumeValues v;
  v.density = ref::evaluateBinding(b.density, p, sampler);
  v.transmissionColor = ref::evaluateBinding(b.transmissionColor, p, sampler);
  v.transmissionDistance =
      ref::evaluateBinding(b.transmissionDistance, p, sampler);
  v.scatteringColor = ref::evaluateBinding(b.scatteringColor, p, sampler);
  const auto coefficients = ref::evaluateOptical(v,
      volume.hasConstantTransmissionLog ? volume.constantTransmissionLog
                                        : nullptr);
  for (int c = 0; c < 3; ++c)
    result.scattering[c] = coefficients.scattering[c];
  result.phase = {ref::evaluateBinding(b.anisotropy, p, sampler),
      ref::evaluateBinding(b.secondaryAnisotropy, p, sampler),
      ref::evaluateBinding(b.secondaryWeight, p, sampler)};
  return result;
}

VISRTX_HOST_DEVICE double phaseMass(
    const PhaseComponent &component, double scale)
{
  double mass = 0.;
  for (int c = 0; c < 3; ++c)
    mass = ref::add(mass, ref::divide(component.scattering[c], scale));
  return mass;
}

struct PhaseSelection
{
  double scale{0.};
  double total{0.};
  size_t count{0};
};

// Scaling before the RGB sum avoids overflowing proposal normalization.
// This is conditional on a real scattering event, hence at least one S>0.
VISRTX_HOST_DEVICE PhaseSelection phaseSelection(
    const AggregateVolumeSceneGPUData &scene,
    const av::Ray &ray,
    av::Interval span,
    double distance)
{
  PhaseSelection result;
  for (size_t i = 0; i < scene.count; ++i) {
    const auto component =
        phaseComponent(scene.occurrences[i], ray, span, distance);
    for (int c = 0; c < 3; ++c)
      result.scale = result.scale > component.scattering[c]
          ? result.scale
          : component.scattering[c];
  }
  assert(result.scale > 0.);
  for (size_t i = 0; i < scene.count; ++i) {
    const double mass =
        phaseMass(phaseComponent(scene.occurrences[i], ray, span, distance),
            result.scale);
    if (mass > 0.) {
      result.total = ref::add(result.total, mass);
      ++result.count;
    }
  }
  return result;
}

struct PhaseEntry
{
  size_t index{0};
  double mass{0.};
  PhaseComponent component;
};

// Reference baseline: replay to enumerate positive components small-first.
// O(N^2) work, constant storage, no overlap cap. Ordering prevents a dim
// positive component from being appended to an already near-one CDF and
// disappearing. A faster prepared distribution may replace this only with
// matching PDFs.
VISRTX_HOST_DEVICE PhaseEntry nextPhaseEntry(
    const AggregateVolumeSceneGPUData &scene,
    const av::Ray &ray,
    av::Interval span,
    double distance,
    const PhaseSelection &selection,
    const PhaseEntry &previous,
    bool first)
{
  PhaseEntry next;
  bool found = false;
  for (size_t i = 0; i < scene.count; ++i) {
    const auto component =
        phaseComponent(scene.occurrences[i], ray, span, distance);
    const double mass = phaseMass(component, selection.scale);
    if (!(mass > 0.)
        || (!first
            && !(mass > previous.mass
                || (mass == previous.mass && i > previous.index))))
      continue;
    if (!found || mass < next.mass) {
      next = {i, mass, component};
      found = true;
    }
  }
  assert(found);
  return next;
}

VISRTX_HOST_DEVICE double phaseCdfUpper(
    double prefix, const PhaseSelection &selection, size_t ordinal)
{
  return ordinal + 1 == selection.count ? 1.
                                        : ref::divide(prefix, selection.total);
}

VISRTX_HOST_DEVICE PhaseMixture phaseAt(
    const AggregateVolumeSceneGPUData &scene,
    const av::Ray &ray,
    av::Interval span,
    double distance,
    const PhaseSelection &selection,
    double cosine)
{
  PhaseMixture mixture;
  PhaseEntry entry;
  double prefix = 0., lower = 0.;
  for (size_t ordinal = 0; ordinal < selection.count; ++ordinal) {
    entry = nextPhaseEntry(
        scene, ray, span, distance, selection, entry, ordinal == 0);
    prefix = ref::add(prefix, entry.mass);
    const double upper = phaseCdfUpper(prefix, selection, ordinal);
    accumulatePhase(entry.component.phase,
        entry.component.scattering,
        ref::subtract(upper, lower),
        cosine,
        mixture);
    lower = upper;
  }
  return mixture;
}

struct DirectionSample
{
  glm::dvec3 direction;
  PhaseMixture mixture;
  double cosine;
  size_t selectedOccurrence;
};

// Rotation stays in world space: instance transforms affect optical lengths,
// not the HG propagation convention. Float conversion for OptiX is
// caller-owned.
VISRTX_HOST_DEVICE glm::dvec3 worldDirection(
    const av::Ray &ray, double cosine, double azimuth)
{
  const glm::dvec3 w(ray.direction[0], ray.direction[1], ray.direction[2]);
  const glm::dvec3 up =
      std::abs(w.z) < .9 ? glm::dvec3(0, 0, 1) : glm::dvec3(0, 1, 0);
  const glm::dvec3 u = glm::normalize(glm::cross(up, w));
  const glm::dvec3 v = glm::cross(w, u);
  const double sine =
      sqrt(ref::multiply(ref::subtract(1., cosine), ref::add(1., cosine)));
  return w * cosine + (u * cos(azimuth) + v * sin(azimuth)) * sine;
}

template <typename RNG>
VISRTX_HOST_DEVICE DirectionSample sampleDirection(
    const AggregateVolumeSceneGPUData &scene,
    const av::Ray &ray,
    av::Interval span,
    double distance,
    RNG &random)
{
  const auto selection = phaseSelection(scene, ray, span, distance);
  const double medium = random.next();
  PhaseEntry entry;
  double prefix = 0.;
  for (size_t ordinal = 0; ordinal < selection.count; ++ordinal) {
    entry = nextPhaseEntry(
        scene, ray, span, distance, selection, entry, ordinal == 0);
    prefix = ref::add(prefix, entry.mass);
    if (medium < phaseCdfUpper(prefix, selection, ordinal))
      break;
  }
  const double lobe = random.next(), polar = random.next(),
               azimuth = random.next();
  const auto sample = samplePhase(entry.component.phase, lobe, polar, azimuth);
  return {worldDirection(ray, sample.cosine, sample.azimuth),
      phaseAt(scene, ray, span, distance, selection, sample.cosine),
      sample.cosine,
      entry.index};
}

} // namespace volume_tracking
} // namespace visrtx
