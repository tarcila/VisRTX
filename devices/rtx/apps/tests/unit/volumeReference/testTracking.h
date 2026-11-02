// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cstdio>
#include <stdexcept>
#include <vector>
#include "gpu/volumeTracking.h"

namespace tracking_test {
using namespace visrtx;
namespace tr = volume_tracking;
namespace ref = optical_reference;
constexpr int SAMPLES = 131072;
struct Random
{
  uint64_t state;
  VISRTX_HOST_DEVICE double next();
};
VISRTX_HOST_DEVICE double Random::next()
{
  uint64_t z = (state += UINT64_C(0x9e3779b97f4a7c15));
  z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
  z ^= z >> 31;
  return (double(z >> 12) + .5) * 0x1p-52;
}
struct Result
{
  double value[9]{};
};

VISRTX_HOST_DEVICE ref::Coefficients coefficients(int id, int segment)
{
  ref::Coefficients v;
  for (int c = 0; c < 3; ++c) {
    v.absorption[c] = id == 0 ? 0. : .25 * double(c + 1);
    v.emission[c] = double(c + 1);
    if (id == 2 && segment == 1) {
      v.absorption[c] = c == 2 ? 0. : .5 * double(c + 1);
      v.emission[c] = double(3 - c);
    }
    if (id == 3) {
      v.scattering[c] = .5 * double(3 - c);
      v.emission[c] = v.absorption[c];
    }
  }
  return v;
}

// Frozen pre-reuse probability graph. This checks the optimization's exact
// CDF/weight equivalence, in addition to the independent analytic experiments.
VISRTX_HOST_DEVICE tr::Distribution originalDistribution(
    const ref::Coefficients &v, double rate)
{
  double mass[3]{};
  tr::Distribution result{
      {tr::Event::Absorb, tr::Event::Scatter, tr::Event::Null}, {}};
  for (int e = 0; e < 3; ++e)
    for (int c = 0; c < 3; ++c)
      mass[e] = ref::add(mass[e], tr::eventMass(v, rate, tr::Event(e), c));
  for (int i = 0; i < 2; ++i)
    for (int j = i + 1; j < 3; ++j)
      if (mass[j] < mass[i]) {
        const double m = mass[i];
        mass[i] = mass[j];
        mass[j] = m;
        const auto e = result.order[i];
        result.order[i] = result.order[j];
        result.order[j] = e;
      }
  const double total = ref::add(ref::add(mass[0], mass[1]), mass[2]);
  result.upper[0] = ref::divide(mass[0], total);
  result.upper[1] = ref::add(result.upper[0], ref::divide(mass[1], total));
  result.upper[2] = 1.;
  return result;
}
VISRTX_HOST_DEVICE bool reusedProbabilities(int index, double rate)
{
  ref::Coefficients v;
  for (int c = 0; c < 3; ++c) {
    v.absorption[c] = rate * .25;
    v.scattering[c] = rate * .125;
    v.emission[c] = rate * double(c + 1);
  }
  // Also exercise equal extinction with unequal partitions and generic RGB.
  if (index % 3 == 1) {
    v.absorption[1] = rate * .125;
    v.scattering[1] = rate * .25;
  }
  if (index % 3 == 2)
    v.absorption[2] = rate * .375;
  const auto expected = originalDistribution(v, rate),
             actual = tr::eventDistribution(v, rate);
  for (int i = 0; i < 3; ++i)
    if (actual.order[i] != expected.order[i]
        || actual.upper[i] != expected.upper[i])
      return false;
  const double choices[]{.05, .3, .95};
  const double u = choices[(index / 3) % 3];
  const auto selected = tr::choose(expected, u);
  tr::PathState path;
  path.sourceOwned = index % 2 == 0;
  for (int c = 0; c < 3; ++c)
    path.throughput[c] = double(c + 1);
  const auto got = tr::collide(v, rate, u, path);
  if (got.event != selected.event || got.probability != selected.probability)
    return false;
  double shadow[3]{1., 2., 3.};
  tr::shadowCandidate(v, rate, shadow);
  for (int c = 0; c < 3; ++c) {
    const double weight = selected.event == tr::Event::Absorb
        ? 0.
        : ref::multiply(double(c + 1),
              ref::divide(tr::eventMass(v, rate, selected.event, c),
                  selected.probability));
    const double source = path.sourceOwned
        ? ref::multiply(double(c + 1), ref::divide(v.emission[c], rate))
        : 0.;
    if (path.throughput[c] != weight || path.radiance[c] != source
        || shadow[c]
            != ref::multiply(
                double(c + 1), tr::eventMass(v, rate, tr::Event::Null, c)))
      return false;
  }
  return true;
}

VISRTX_HOST_DEVICE Result arithmeticTrial(int index)
{
  Result out;
  const double scales[3]{1e-150, 1., 1e150};
  const double scale = scales[index % 3];
  auto v = coefficients(3, 0);
  for (int c = 0; c < 3; ++c) {
    v.absorption[c] *= scale;
    v.scattering[c] *= scale;
    v.emission[c] *= scale;
  }
  const double rate = 4. * scale;
  const auto distribution = tr::eventDistribution(v, rate);
  double lower = 0.;
  for (int i = 0; i < 3; ++i) {
    const double probability = distribution.upper[i] - lower;
    if (probability > 0.) {
      tr::PathState path;
      tr::collide(v, rate, lower + .5 * probability, path);
      for (int c = 0; c < 3; ++c) {
        out.value[c] += probability * path.throughput[c];
        out.value[c + 3] += probability * path.radiance[c];
      }
    }
    lower = distribution.upper[i];
  }
  for (int c = 0; c < 3; ++c) {
    out.value[c] -= 1. - v.absorption[c] / rate;
    out.value[c + 3] -= v.emission[c] / rate;
  }
  ref::Coefficients tiny;
  tiny.scattering[0] = 1e-100;
  const auto d = tr::eventDistribution(tiny, 1.);
  out.value[6] = d.upper[1] > 0.
      && tr::choose(d, .5 * d.upper[1]).event == tr::Event::Scatter;
  double tau = .75;
  bool success = !tr::advance(10., 0., tau).collision && tau == .75;
  success = success && !tr::advance(.25, 2., tau).collision && tau == .25;
  const auto step = tr::advance(1., 4., tau);
  out.value[7] = success && step.collision && step.distance == .0625;
  ref::Bounds zero;
  out.value[8] = tr::radianceRate(zero, .5) == .5;
  PrincipledVolumeValues optical;
  optical.density = 3.f;
  optical.transmissionColor = vec3(.25f);
  optical.transmissionDistance = .7f;
  optical.scatteringColor = vec3(0, .7f, 1);
  ref::EffectiveRanges ranges{
      {0, 3}, {{.25f, .25f}, {.25f, .25f}, {.25f, .25f}}, {.7f, .7f}};
  ref::Bounds loose, tight;
  bool bounded = ref::buildBounds(ranges, loose)
      && ref::buildBounds(ranges, tight, &optical.scatteringColor);
  for (float density : {0.f, .1f, 1.f, 3.f}) {
    optical.density = density;
    const auto actual = ref::evaluate(optical);
    for (int c = 0; c < 3; ++c)
      bounded &= tight.absorption[c] >= actual.absorption[c]
          && tight.scattering[c] >= actual.scattering[c]
          && ref::majorant(tight) >= ref::extinction(actual, c);
  }
  bounded &= tight.scattering[0] == 0. && tight.absorption[2] == 0.
      && ref::majorant(tight) < .51 * ref::majorant(loose);
  // Sweep metadata/point combinations through the existing deterministic
  // trials, including subnormals and the rounded 1-a partition near endpoints.
  const float albedos[]{0.f, 0x1p-149f, 0x1p-26f, .7f, 0x1.fffffep-1f, 1.f};
  const float magnitudes[]{0x1p-149f, .7f, 0x1.fffffep127f};
  const float colors[]{0x1p-149f, .25f, 1.f};
  optical.scatteringColor = vec3(albedos[index % 6]);
  optical.density = magnitudes[(index / 6) % 3];
  optical.transmissionDistance = magnitudes[(index / 18) % 3];
  optical.transmissionColor = vec3(colors[(index / 54) % 3]);
  ranges.density = {0.f, optical.density};
  const float distance = optical.transmissionDistance;
  ranges.distance = {distance, distance};
  const float color = optical.transmissionColor.x;
  for (int c = 0; c < 3; ++c)
    ranges.transmission[c] = {color, color};
  bounded &= ref::buildBounds(ranges, tight, &optical.scatteringColor);
  const auto actual = ref::evaluate(optical);
  for (int c = 0; c < 3; ++c)
    bounded &= tight.absorption[c] >= actual.absorption[c]
        && tight.scattering[c] >= actual.scattering[c]
        && ref::majorant(tight) >= ref::extinction(actual, c);
  out.value[8] = out.value[8] == 1. && bounded
      && reusedProbabilities(index, scales[(index / 27) % 3]);
  return out;
}

VISRTX_HOST_DEVICE Result trial(int id, int index)
{
  if (id == 5)
    return arithmeticTrial(index);
  Random rng{UINT64_C(0x123456789ab) + uint64_t(index) * UINT64_C(0x100000001b3)
      + uint64_t(id) * UINT64_C(0x123987654321)};
  Result out;
  if (id == 4) {
    const VolumePhaseFunction p[2]{{.6f, -.3f, .2f}, {-.4f, .2f, .7f}};
    const double s[2][3]{{1., .5, 0.}, {0., 1., 3.}};
    const int selected = rng.next() < .25 ? 0 : 1;
    const double lobe = rng.next(), polar = rng.next(), azimuth = rng.next();
    const auto sample = tr::samplePhase(p[selected], lobe, polar, azimuth);
    tr::PhaseMixture mixture;
    tr::accumulatePhase(p[0], s[0], .25, sample.cosine, mixture);
    tr::accumulatePhase(p[1], s[1], .75, sample.cosine, mixture);
    double scattering[3];
    for (int c = 0; c < 3; ++c)
      scattering[c] = s[0][c] + s[1][c];
    tr::PathState path;
    tr::applyDirection(mixture.kernel, scattering, mixture.pdf, path);
    for (int c = 0; c < 3; ++c) {
      out.value[c] = path.throughput[c];
      out.value[c + 3] = path.throughput[c] * sample.cosine;
    }
    out.value[6] = sample.pdf / tr::phasePdf(p[selected], sample.cosine);
    out.value[7] = cos(sample.azimuth);
    out.value[8] = sin(sample.azimuth);
    return out;
  }
  tr::PathState path;
  double tau = tr::opticalDepth(rng.next());
  if (id == 3) {
    // Infinite-order isotropic scattering in a unit slab. Boundary radiance=1
    // in every direction, j=A: exact equilibrium radiance is RGB(1), even with
    // chromatic extinction/albedo. No depth or null-candidate cap.
    double z = .5, mu = 1.;
    const auto v = coefficients(id, 0);
    while (true) {
      const double distance = mu > 0. ? (1. - z) / mu : -z / mu;
      const auto step = tr::advance(distance, 3., tau);
      if (!step.collision) {
        for (int c = 0; c < 3; ++c)
          path.radiance[c] += path.throughput[c];
        break;
      }
      z += mu * step.distance;
      const auto event = tr::collide(v, 3., rng.next(), path);
      if (event.event == tr::Event::Absorb)
        break;
      if (event.event == tr::Event::Scatter) {
        mu = tr::sampleCosine(0., rng.next());
        out.value[6] += 1.;
      }
      tau = tr::opticalDepth(rng.next());
    }
    for (int c = 0; c < 3; ++c)
      out.value[c] = path.radiance[c];
    return out;
  }
  bool alive = true;
  for (int segment = 0; segment < 2 && alive; ++segment) {
    const auto v = coefficients(id, segment);
    const double rate = segment ? 3. : 2.;
    double position = 0.;
    while (true) {
      const auto step = tr::advance(1. - position, rate, tau);
      if (!step.collision)
        break;
      position += step.distance;
      if (tr::collide(v, rate, rng.next(), path).event == tr::Event::Absorb) {
        alive = false;
        break;
      }
      tau = tr::opticalDepth(rng.next());
    }
  }
  for (int c = 0; c < 3; ++c) {
    out.value[c] = path.radiance[c];
    out.value[c + 3] = path.throughput[c];
  }
  double transmission[3]{1., 1., 1.};
  tau = tr::opticalDepth(rng.next());
  for (int segment = 0; segment < 2; ++segment) {
    const auto v = coefficients(id, segment);
    const double rate = segment ? 3. : 2.;
    double position = 0.;
    while (true) {
      const auto step = tr::advance(1. - position, rate, tau);
      if (!step.collision)
        break;
      position += step.distance;
      tr::shadowCandidate(v, rate, transmission);
      tau = tr::opticalDepth(rng.next());
    }
  }
  for (int c = 0; c < 3; ++c)
    out.value[c + 6] = transmission[c];
  return out;
}

inline void require(bool condition, const char *message)
{
  if (!condition)
    throw std::runtime_error(message);
}

inline void deterministic()
{
  // Exact enumeration of event proposal expectations, including zero branches,
  // very small nonzero branches, and huge physical rates with finite ratios.
  for (double scale : {1e-150, 1., 1e150}) {
    auto v = coefficients(3, 0);
    for (int c = 0; c < 3; ++c) {
      v.absorption[c] *= scale;
      v.scattering[c] *= scale;
      v.emission[c] *= scale;
    }
    const double rate = 4. * scale;
    const auto d = tr::eventDistribution(v, rate);
    double lower = 0., expect[3]{}, emission[3]{};
    for (int i = 0; i < 3; ++i) {
      const double probability = d.upper[i] - lower;
      if (probability > 0.) {
        tr::PathState path;
        const auto selected =
            tr::collide(v, rate, lower + .5 * probability, path);
        require(selected.event == d.order[i], "event interval selection");
        for (int c = 0; c < 3; ++c) {
          expect[c] += probability * path.throughput[c];
          emission[c] += probability * path.radiance[c];
        }
      }
      lower = d.upper[i];
    }
    for (int c = 0; c < 3; ++c) {
      require(std::abs(expect[c] - (1. - v.absorption[c] / rate)) < 2e-14,
          "RGB event expectation");
      require(std::abs(emission[c] - v.emission[c] / rate) < 2e-14,
          "all-candidate emission expectation");
    }
  }
  ref::Coefficients tiny;
  tiny.scattering[0] = 1e-100;
  const auto d = tr::eventDistribution(tiny, 1.);
  require(d.upper[1] > 0.
          && tr::choose(d, .5 * d.upper[1]).event == tr::Event::Scatter,
      "tiny scattering support");
  double tau = .75;
  require(!tr::advance(10., 0., tau).collision && tau == .75,
      "vacuum gap optical depth");
  require(!tr::advance(.25, 2., tau).collision && tau == .25,
      "span residual optical depth");
  const auto step = tr::advance(1., 4., tau);
  require(step.collision && step.distance == .0625, "piecewise free flight");
  ref::Bounds zero;
  require(tr::radianceRate(zero, .5) == .5, "vacuum emission proposal support");
  for (double g :
      {-.9999999403953552, -.7, -1e-30, 0., 1e-30, .7, .9999999403953552})
    for (double u : {0x1p-53, .001, .25, .5, .9, 1. - 0x1p-53}) {
      const double mu = tr::sampleCosine(g, u);
      require(
          mu >= -1. && mu <= 1. && std::isfinite(tr::henyeyGreenstein(g, mu)),
          "HG finite support");
      if (std::abs(g) < 1e-20)
        require(std::abs(mu - (2. * u - 1.)) < 2e-14, "HG small-g limit");
      else {
        const long double gg = g, uu = u;
        const long double t = (1 - gg * gg) / (1 - gg + 2 * gg * uu);
        const double expected = double((1 + gg * gg - t * t) / (2 * gg));
        require(std::abs(mu - expected) < 2e-12, "HG independent inverse CDF");
      }
    }
}

inline void check(int id, const std::vector<Result> &values)
{
  if (id == 5) {
    for (const auto &v : values) {
      for (int c = 0; c < 6; ++c)
        require(std::abs(v.value[c]) < 2e-14,
            "executed event arithmetic expectation");
      for (int c = 6; c < 9; ++c)
        require(
            v.value[c] == 1., "executed branch/flight/source support control");
    }
    return;
  }
  double expected[9]{};
  int channels = 9;
  if (id == 3) {
    expected[0] = expected[1] = expected[2] = 1.;
    channels = 3;
    size_t multiple = 0;
    for (auto &v : values)
      multiple += v.value[6] >= 3.;
    require(multiple > 100, "three-or-more-scatter paths not exercised");
  } else if (id == 4) {
    channels = 9;
    const VolumePhaseFunction p[2]{{.6f, -.3f, .2f}, {-.4f, .2f, .7f}};
    const double s[2][3]{{1., .5, 0.}, {0., 1., 3.}};
    double moment[2];
    for (int i = 0; i < 2; ++i)
      moment[i] = (1. - double(p[i].secondaryWeight)) * p[i].anisotropy
          + double(p[i].secondaryWeight) * p[i].secondaryAnisotropy;
    for (int c = 0; c < 3; ++c) {
      expected[c] = 1.;
      expected[c + 3] =
          (s[0][c] * moment[0] + s[1][c] * moment[1]) / (s[0][c] + s[1][c]);
    }
    expected[6] = 1.;
  } else {
    for (int c = 0; c < 3; ++c) {
      double t = 1.;
      for (int segment = 0; segment < 2; ++segment) {
        const auto v = coefficients(id, segment);
        const double a = v.absorption[c];
        expected[c] += t * v.emission[c] * (a == 0. ? 1. : -std::expm1(-a) / a);
        t *= std::exp(-a);
      }
      expected[c + 3] = expected[c + 6] = t;
    }
  }
  for (int c = 0; c < channels; ++c) {
    long double sum = 0., squares = 0.;
    for (auto &v : values) {
      require(std::isfinite(v.value[c]), "nonfinite stochastic result");
      sum += v.value[c];
      squares += v.value[c] * v.value[c];
    }
    const double mean = double(sum / values.size());
    const double variance = std::max(0.,
        double((squares - sum * sum / values.size()) / (values.size() - 1)));
    // Predeclared statistical gate, separate from deterministic arithmetic
    // tolerances: six estimated standard errors plus 5e-5 absolute.
    const double tolerance = 6. * std::sqrt(variance / values.size()) + 5e-5;
    if (std::abs(mean - expected[c]) > tolerance) {
      std::fprintf(stderr,
          "case %d channel %d: %.12g expected %.12g tolerance %.12g\n",
          id,
          c,
          mean,
          expected[c],
          tolerance);
      throw std::runtime_error("transport expectation");
    }
  }
}

#ifdef TRACKING_CUDA
inline void checked(cudaError_t e)
{
  if (e != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(e));
}
VISRTX_GLOBAL void runTrials(int id, Result *output)
{
  const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < SAMPLES)
    output[i] = trial(id, i);
}
#endif
inline int run()
{
  deterministic();
  std::vector<Result> output(SAMPLES);
#ifdef TRACKING_CUDA
  Result *device = nullptr;
  checked(cudaMalloc(&device, output.size() * sizeof(Result)));
#endif
  for (int id = 0; id < 6; ++id) {
#ifdef TRACKING_CUDA
    runTrials<<<(SAMPLES + 127) / 128, 128>>>(id, device);
    checked(cudaGetLastError());
    checked(cudaDeviceSynchronize());
    checked(cudaMemcpy(output.data(),
        device,
        output.size() * sizeof(Result),
        cudaMemcpyDeviceToHost));
#else
    for (int i = 0; i < SAMPLES; ++i)
      output[i] = trial(id, i);
#endif
    check(id, output);
  }
#ifdef TRACKING_CUDA
  checked(cudaFree(device));
#endif
  std::puts(
      "Quality tracking: deterministic controls and five 131072-path analytic experiments passed");
  return 0;
}
} // namespace tracking_test
