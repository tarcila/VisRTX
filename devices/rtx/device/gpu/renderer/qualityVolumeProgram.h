// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "gpu/blackbody/blackbody.h"
#include "gpu/renderer/qualityVolume.h"

namespace visrtx {

struct QualityCoverageVisitor
{
  quality_volume::Request &query;
  VISRTX_DEVICE bool operator()(const AggregateVolumeSceneGPUData &scene,
      const aggregate_volume::Ray &ray,
      aggregate_volume::Interval span,
      double distance,
      double rate,
      const aggregate_volume::ObjectRay *single);
};
VISRTX_DEVICE bool QualityCoverageVisitor::operator()(
    const AggregateVolumeSceneGPUData &scene,
    const aggregate_volume::Ray &ray,
    aggregate_volume::Interval span,
    double distance,
    double rate,
    const aggregate_volume::ObjectRay *single)
{
  namespace ref = optical_reference;
  const auto coefficients =
      volume_tracking::extinctionAt(scene, ray, span, distance, single);
  const double luminance[3]{.2126, .7152, .0722};
  double weight = 0.;
  for (int c = 0; c < 3; ++c)
    weight = ref::add(weight,
        ref::multiply(luminance[c],
            ref::multiply(query.transmission[c],
                ref::divide(ref::extinction(coefficients, c), rate))));
  for (int c = 0; c < 3; ++c) {
    const double t = ref::extinction(coefficients, c);
    if (t > 0.)
      query.coverageAlbedo[c] = ref::add(query.coverageAlbedo[c],
          ref::multiply(weight, ref::divide(coefficients.scattering[c], t)));
  }
  volume_tracking::shadowCandidate(coefficients, rate, query.transmission);
  return query.transmission[0] != 0. || query.transmission[1] != 0.
      || query.transmission[2] != 0.;
}

// Quality raygen -> this DC -> scalar/RGB field DC (maximum direct depth 2).
// Keep thermal/aggregate expansion out of the raygen's surface/light graph.
VISRTX_CALLABLE void __direct_callable__qualityVolume(
    quality_volume::Request *request)
{
  namespace tr = volume_tracking;
  namespace qv = quality_volume;
  auto &q = *request;
  tr::Random random{&q.random};
  switch (q.operation) {
  case qv::Operation::Source: {
    QualityBlackbody blackbody;
    q.source = volume_source::sample(*q.scene,
        glm::dvec3(q.ray.origin[0], q.ray.origin[1], q.ray.origin[2]),
        q.random,
        blackbody);
    break;
  }
  case qv::Operation::Flight: {
    QualityBlackbody blackbody;
    q.flight =
        tr::trace(*q.scene, q.ray, q.sourceProposal, random, blackbody, q.path);
    q.hasSinglePhase = q.flight.endpoint == tr::Endpoint::Scattered
        && q.scene->count == 1
        && q.scene->occurrences[0].volume.type == VolumeType::PRINCIPLED;
    if (q.hasSinglePhase) {
      const auto &o = q.scene->occurrences[0];
      const auto &v = *o.volume.principled;
      const auto p = aggregate_volume::objectPosition(
          aggregate_volume::objectRay(o, q.ray), q.flight.distance);
      ReferenceFieldSampler sampler{v.fields};
      q.singlePhase = {
          optical_reference::evaluateBinding(v.bindings.anisotropy, p, sampler),
          optical_reference::evaluateBinding(
              v.bindings.secondaryAnisotropy, p, sampler),
          optical_reference::evaluateBinding(
              v.bindings.secondaryWeight, p, sampler)};
    }
    break;
  }
  case qv::Operation::Coverage: {
    QualityCoverageVisitor visitor{q};
    tr::candidates(*q.scene, q.ray, 0., random, visitor);
    break;
  }
  case qv::Operation::Shadow:
    tr::shadow(*q.scene, q.ray, random, q.transmission);
    break;
  case qv::Operation::Direction:
    if (q.hasSinglePhase) {
      (void)random
          .next(); // preserve the medium-selection draw and subsequent streams
      const double lobe = random.next(), polar = random.next(),
                   azimuth = random.next();
      const auto sampled = tr::samplePhase(q.singlePhase, lobe, polar, azimuth);
      tr::PhaseMixture mixture;
      tr::accumulatePhase(q.singlePhase,
          q.flight.coefficients.scattering,
          1.,
          sampled.cosine,
          mixture);
      q.direction = {tr::worldDirection(q.ray, sampled.cosine, sampled.azimuth),
          mixture,
          sampled.cosine,
          0};
    } else
      q.direction = tr::sampleDirection(
          *q.scene, q.ray, q.flight.span, q.flight.distance, random);
    tr::applyDirection(q.direction.mixture.kernel,
        q.flight.coefficients.scattering,
        q.direction.mixture.pdf,
        q.path);
    break;
  case qv::Operation::Phase: {
    if (q.hasSinglePhase) {
      q.phase = {};
      tr::accumulatePhase(q.singlePhase,
          q.flight.coefficients.scattering,
          1.,
          q.cosine,
          q.phase);
      break;
    }
    const auto selection =
        tr::phaseSelection(*q.scene, q.ray, q.flight.span, q.flight.distance);
    q.phase = tr::phaseAt(
        *q.scene, q.ray, q.flight.span, q.flight.distance, selection, q.cosine);
    break;
  }
  }
}
} // namespace visrtx
