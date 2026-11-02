// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "volumeOpticsReference.h"

namespace visrtx {
namespace volume_tracking {
namespace ref = optical_reference;

// Reference stochastic transport primitives, not a preview-majorant adapter.
// Preconditions: valid reference coefficients, a finite positive proposal rate
// enclosing reference A+S, and representable weights/probabilities. Random
// variates are in (0,1). Geometry/progress and renderer integration are
// separate. Host arithmetic has the same requirements as
// volumeOpticsReference.h.
enum class Event
{
  Absorb = 0,
  Scatter = 1,
  Null = 2
};

struct Distribution
{
  Event order[3];
  double upper[3];
};

VISRTX_HOST_DEVICE double eventMass(
    const ref::Coefficients &v, double rate, Event event, int channel)
{
  const double value = event == Event::Absorb ? v.absorption[channel]
      : event == Event::Scatter
      ? v.scattering[channel]
      : ref::subtract(rate, ref::extinction(v, channel));
  return ref::divide(value, rate);
}

VISRTX_HOST_DEVICE bool achromaticEvents(const ref::Coefficients &v)
{
  return v.absorption[0] == v.absorption[1]
      && v.absorption[0] == v.absorption[2]
      && v.scattering[0] == v.scattering[1]
      && v.scattering[0] == v.scattering[2];
}

VISRTX_HOST_DEVICE Distribution eventDistribution(
    const ref::Coefficients &v, double rate)
{
  double mass[3]{};
  Distribution result{{Event::Absorb, Event::Scatter, Event::Null}, {}};
  const bool gray = achromaticEvents(v);
  for (int e = 0; e < 3; ++e) {
    if (gray) {
      // Reuse the division, not a reassociated 3*x or a scalar CDF. Retain
      // exactly the original three rounded additions and represented widths.
      const double value = eventMass(v, rate, Event(e), 0);
      for (int c = 0; c < 3; ++c)
        mass[e] = ref::add(mass[e], value);
    } else
      for (int c = 0; c < 3; ++c)
        mass[e] = ref::add(mass[e], eventMass(v, rate, Event(e), c));
  }
  // Normalize scaled masses, avoiding an overflowing sum of physical rates.
  // Small intervals precede large ones so a tiny nonzero branch is not erased
  // by appending it to a near-one CDF. Use represented interval widths below,
  // rather than assuming separately rounded probabilities sum to one.
  for (int i = 0; i < 2; ++i)
    for (int j = i + 1; j < 3; ++j)
      if (mass[j] < mass[i]) {
        const double m = mass[i];
        mass[i] = mass[j];
        mass[j] = m;
        const Event e = result.order[i];
        result.order[i] = result.order[j];
        result.order[j] = e;
      }
  const double total = ref::add(ref::add(mass[0], mass[1]), mass[2]);
  result.upper[0] = ref::divide(mass[0], total);
  result.upper[1] = ref::add(result.upper[0], ref::divide(mass[1], total));
  result.upper[2] = 1.;
  return result;
}

struct Candidate
{
  Event event;
  double probability;
};

VISRTX_HOST_DEVICE Candidate choose(const Distribution &proposal, double u)
{
  double lower = 0.;
  for (int i = 0; i < 3; ++i) {
    if (u < proposal.upper[i])
      return {proposal.order[i], ref::subtract(proposal.upper[i], lower)};
    lower = proposal.upper[i];
  }
  // Unreachable under the open-unit random contract.
  return {proposal.order[2], ref::subtract(1., proposal.upper[1])};
}

struct PathState
{
  double throughput[3]{1., 1., 1.};
  double radiance[3]{};
  bool sourceOwned{
      true}; // camera/delta/unsupported paths retain all-candidate j
};

// Source belongs to every candidate, INCLUDING absorption and null outcomes.
// Scattering returns the S/(M*qS) weight. Direction sampling must subsequently
// apply K/(S*qDirection); an external-light connection uses its own light PDF.
VISRTX_HOST_DEVICE Candidate collide(
    const ref::Coefficients &v, double rate, double u, PathState &path)
{
  const Candidate selected = choose(eventDistribution(v, rate), u);
  const bool gray = achromaticEvents(v);
  const double sharedWeight = gray && selected.event != Event::Absorb
      ? ref::divide(eventMass(v, rate, selected.event, 0), selected.probability)
      : 0.;
  for (int c = 0; c < 3; ++c) {
    if (path.sourceOwned)
      path.radiance[c] = ref::add(path.radiance[c],
          ref::multiply(path.throughput[c], ref::divide(v.emission[c], rate)));
    path.throughput[c] = selected.event == Event::Absorb
        ? 0.
        : ref::multiply(path.throughput[c],
              gray ? sharedWeight
                   : ref::divide(eventMass(v, rate, selected.event, c),
                         selected.probability));
  }
  return selected;
}

VISRTX_HOST_DEVICE void applyDirection(const double (&kernel)[3],
    const double (&scattering)[3],
    double directionPdf,
    PathState &path)
{
  for (int c = 0; c < 3; ++c)
    path.throughput[c] = scattering[c] == 0.
        ? 0.
        : ref::multiply(path.throughput[c],
              ref::divide(ref::divide(kernel[c], scattering[c]), directionPdf));
}

VISRTX_HOST_DEVICE void shadowCandidate(
    const ref::Coefficients &v, double rate, double (&transmission)[3])
{
  const double extinction[3]{
      ref::extinction(v, 0), ref::extinction(v, 1), ref::extinction(v, 2)};
  const bool gray =
      extinction[0] == extinction[1] && extinction[0] == extinction[2];
  const double sharedMass =
      gray ? ref::divide(ref::subtract(rate, extinction[0]), rate) : 0.;
  for (int c = 0; c < 3; ++c)
    transmission[c] = ref::multiply(transmission[c],
        gray ? sharedMass : eventMass(v, rate, Event::Null, c));
}

// A positive source proposal is NOT physical extinction. Call this on every
// active medium interval, even when its extinction bound is zero. Shadow-only
// queries can instead use the extinction majorant directly (and skip rate=0).
VISRTX_HOST_DEVICE double radianceRate(
    const ref::Bounds &bounds, double sourceProposal)
{
  double rate = sourceProposal;
  for (int c = 0; c < 3; ++c) {
    const double upper =
        ref::addUpper(bounds.absorption[c], bounds.scattering[c]);
    rate = rate > upper ? rate : upper;
  }
  return rate;
}

VISRTX_HOST_DEVICE double opticalDepth(double u)
{
  return -log1p(-u);
}

struct FlightStep
{
  bool collision;
  double distance;
};

// Carry exponential optical depth across deterministic interval boundaries.
// No-candidate escape adds no Beer factor; nulls/boundaries are not bounces.
// Caller owns geometric progress: do not replace a collapsed position with an
// epsilon jump or silently discard its source. This seam uses local lengths.
VISRTX_HOST_DEVICE FlightStep advance(
    double length, double rate, double &remainingOpticalDepth)
{
  if (rate == 0.)
    return {false, length};
  const double capacity = ref::multiply(length, rate);
  if (remainingOpticalDepth >= capacity) {
    remainingOpticalDepth = ref::subtract(remainingOpticalDepth, capacity);
    return {false, length};
  }
  return {true, ref::divide(remainingOpticalDepth, rate)};
}

constexpr double INV_FOUR_PI = 0.079577471545947667884441881686257181;

// |g|<1, -1<=mu<=1, with the propagation-direction convention (forward=+1).
VISRTX_HOST_DEVICE double henyeyGreenstein(double g, double mu)
{
  const double a = g < 0. ? -g : g;
  const double m = g < 0. ? -mu : mu;
  const double oneMinus = ref::subtract(1., a);
  const double denominator = ref::add(ref::multiply(oneMinus, oneMinus),
      ref::multiply(ref::multiply(2., a), ref::subtract(1., m)));
  return ref::divide(
      ref::multiply(INV_FOUR_PI, ref::multiply(oneMinus, ref::add(1., a))),
      ref::multiply(denominator, sqrt(denominator)));
}

VISRTX_HOST_DEVICE double phasePdf(const VolumePhaseFunction &phase, double mu)
{
  const double w = ref::promote(phase.secondaryWeight);
  return ref::add(ref::multiply(ref::subtract(1., w),
                      henyeyGreenstein(ref::promote(phase.anisotropy), mu)),
      ref::multiply(
          w, henyeyGreenstein(ref::promote(phase.secondaryAnisotropy), mu)));
}

// Algebraic inverse HG CDF without division by g or cancellation at small g.
// Reflection handles negative g; no small-anisotropy snap or cosine clamp.
VISRTX_HOST_DEVICE double sampleCosine(double g, double u)
{
  if (g == 0.)
    return ref::subtract(ref::multiply(2., u), 1.);
  const double a = g < 0. ? -g : g;
  const double polar = g < 0. ? ref::subtract(1., u) : u;
  const double oneMinus = ref::subtract(1., a);
  const double d =
      ref::add(oneMinus, ref::multiply(ref::multiply(2., a), polar));
  const double ratio = ref::divide(oneMinus, d);
  const double factor = ref::add(1., ref::multiply(a, polar));
  const double mu = ref::subtract(1.,
      ref::multiply(ref::multiply(ratio, ratio),
          ref::multiply(ref::multiply(2., ref::subtract(1., polar)), factor)));
  return g < 0. ? -mu : mu;
}

struct PhaseMixture
{
  double kernel[3]{};
  double pdf{0.};
};

// Replay all scattering occurrences using the SAME medium-selection masses
// used by the direction proposal. Keep each medium's two lobes: neither the
// selected component PDF nor an averaged anisotropy is the marginal density.
VISRTX_HOST_DEVICE void accumulatePhase(const VolumePhaseFunction &phase,
    const double (&scattering)[3],
    double selectionProbability,
    double cosine,
    PhaseMixture &mixture)
{
  const double p = phasePdf(phase, cosine);
  mixture.pdf = ref::add(mixture.pdf, ref::multiply(selectionProbability, p));
  for (int c = 0; c < 3; ++c)
    mixture.kernel[c] =
        ref::add(mixture.kernel[c], ref::multiply(scattering[c], p));
}

struct PhaseSample
{
  double cosine;
  double azimuth;
  double pdf;
};

VISRTX_HOST_DEVICE PhaseSample samplePhase(
    const VolumePhaseFunction &phase, double lobe, double polar, double azimuth)
{
  const double g = ref::promote(lobe < ref::promote(phase.secondaryWeight)
          ? phase.secondaryAnisotropy
          : phase.anisotropy);
  const double mu = sampleCosine(g, polar);
  return {mu,
      ref::divide(azimuth, ref::multiply(2., INV_FOUR_PI)),
      phasePdf(phase, mu)};
}

} // namespace volume_tracking
} // namespace visrtx
