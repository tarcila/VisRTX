// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#include <array>
#include <cstdio>
#include <stdexcept>
#include <vector>
#include "nativeReflection/fixtures.h"
#include "nativeReflection/oracle.h"
#include "shader_expressions.h"

namespace native_test {
std::vector<NextRay> deviceSamples(int id);
using Bin = std::array<double, 4>;
using Integral = std::array<Bin, BINS>;
static void require(bool ok, const char *message)
{
  if (!ok)
    throw std::runtime_error(message);
}
static int bin(const vec3 &d)
{
  double phi = std::atan2(double(d.y), double(d.x));
  if (phi < 0)
    phi += 2 * oracle::PI;
  return std::min(3, int(phi / (.5 * oracle::PI))) + 4 * (d.z >= .5f);
}
static Integral integrate(int id, int resolution)
{
  Integral out{};
  const auto f = fixture(id);
  const double weight = 2 * oracle::PI / (resolution * (2 * resolution));
  for (int y = 0; y < resolution; ++y)
    for (int x = 0; x < 2 * resolution; ++x) {
      const double z = (y + .5) / resolution,
                   phi = 2 * oracle::PI * (x + .5) / (2 * resolution),
                   r = std::sqrt(1 - z * z);
      const oracle::D l(r * std::cos(phi), r * std::sin(phi), z);
      const auto k = oracle::kernel(f, l);
      const double q = oracle::pdf(f, l);
      const int b = x / (resolution / 2) + 4 * (y >= resolution / 2);
      if (k != oracle::D(0))
        out[b][0] += weight * q;
      for (int c = 0; c < 3; ++c)
        out[b][c + 1] += weight * k[c];
    }
  return out;
}
static void qualify(int id, const std::vector<NextRay> &samples)
{
  const auto f = fixture(id);
  const auto coarse = integrate(id, 256), fine = integrate(id, 512);
  for (int b = 0; b < BINS; ++b)
    for (int c = 0; c < 4; ++c)
      require(std::abs(fine[b][c] - coarse[b][c]) <= 2e-4,
          "independent bin reference unresolved");
  Integral sum{}, square{};
  size_t boundary = 0, transmissions = 0;
  for (const auto &sample : samples) {
    if (isDead(sample)) {
      require(sample.direction == vec3(0)
              && sample.contributionWeight == vec3(0) && sample.pdf == 0,
          "dead initialization");
      continue;
    }
    if (sample.component == NextRayComponent::NativeTransmission) {
      ++transmissions;
      require(std::isinf(sample.pdf)
              && sample.flags == NEXT_RAY_CONTINUES_THROUGH_SURFACE,
          "transmitted ownership sentinel");
      require(sample.event
              == (f.state.eta == 1.f ? NextRayEvent::Singular
                                     : NextRayEvent::Continuous),
          "transmitted measure");
      if (f.state.eta == 1.f)
        require(sample.direction == -f.view, "eta1 atom direction");
      continue;
    }
    require(sample.component == NextRayComponent::NativeReflection
            && sample.event == NextRayEvent::Continuous
            && sample.direction.z > 0,
        "native reflection support/event");
    require(std::isfinite(sample.pdf) && sample.pdf > 0
            && std::abs(glm::length(sample.direction) - 1.f) < 2e-6,
        "direction/pdf");
    const int b = bin(sample.direction);
    sum[b][0] += 1;
    square[b][0] += 1;
    for (int c = 0; c < 3; ++c) {
      require(std::isfinite(sample.contributionWeight[c]), "finite weight");
      sum[b][c + 1] += sample.contributionWeight[c];
      square[b][c + 1] +=
          double(sample.contributionWeight[c]) * sample.contributionWeight[c];
    }
    const double nl = glm::dot(
        oracle::D(f.state.viewAdjustedNormal), oracle::D(sample.direction));
    const double cl = glm::dot(oracle::D(f.state.viewAdjustedClearcoatNormal),
        oracle::D(sample.direction));
    if (std::abs(nl) < 1e-5 || std::abs(cl) < 1e-5) {
      ++boundary;
      continue;
    }
    const double pdf = oracle::pdf(f, oracle::D(sample.direction));
    const auto kernel = oracle::kernel(f, oracle::D(sample.direction));
    if (std::abs(sample.pdf - pdf) > 2e-6 + 5e-4 * std::abs(pdf)) {
      std::fprintf(
          stderr, "PDF fixture %d: %.9g expected %.9g\n", id, sample.pdf, pdf);
      throw std::runtime_error("independent point PDF");
    }
    for (int c = 0; c < 3; ++c) {
      const double target = kernel[c] / pdf;
      if (std::abs(sample.contributionWeight[c] - target)
          > 2e-6 + 5e-4 * std::abs(target)) {
        std::fprintf(stderr,
            "K/q fixture %d channel %d: %.9g expected %.9g direction %.9g %.9g %.9g\n",
            id,
            c,
            sample.contributionWeight[c],
            target,
            sample.direction.x,
            sample.direction.y,
            sample.direction.z);
        throw std::runtime_error("independent point kernel/weight");
      }
    }
  }
  for (int b = 0; b < BINS; ++b)
    for (int c = 0; c < 4; ++c) {
      const double mean = sum[b][c] / samples.size();
      const double variance = std::max(0.,
          (square[b][c] - sum[b][c] * sum[b][c] / samples.size())
              / (samples.size() - 1));
      const double tolerance = 6 * std::sqrt(variance / samples.size()) + 5e-4;
      if (std::abs(mean - fine[b][c]) > tolerance) {
        std::fprintf(stderr,
            "measure fixture %d bin %d channel %d: %.9g expected %.9g tolerance %.9g\n",
            id,
            b,
            c,
            mean,
            fine[b][c],
            tolerance);
        throw std::runtime_error("directional measure");
      }
    }
  if (id == 7)
    require(std::abs(double(transmissions) / samples.size() - .875) < .004,
        "eta1 atom mass");
  std::printf(
      "native fixture %d passed; %zu near-support point comparisons separated\n",
      id,
      boundary);
}
static void transmissionControl()
{
  for (int id : {6, 7, 8})
    for (int i = 0; i < 4096; ++i) {
      const auto f = fixture(id);
      Ray ray{};
      ray.dir = -f.view;
      auto a = random(id, i), b = a;
      int cursor = 0;
      const auto old =
          sampleNativeProposal(&f.state, &ray, &a, nullptr, cursor);
      const auto now = sampleNativeReflection(&f.state, &ray, &b, nullptr, .5f);
      require(a.state == b.state, "outer old branch changed conditional draws");
      if (old.component == NextRayComponent::NativeTransmission) {
        require(
            now.direction == old.direction && now.component == old.component,
            "transmission proposal changed");
        for (int c = 0; c < 3; ++c)
          require(std::abs(now.contributionWeight[c]
                      - old.contributionWeight[c] / .875f)
                  < 1e-6,
              "outer compensation");
      }
    }
}
static void boundaries()
{
  auto f = fixture(2);
  // Target cosine/Smith floors, exact support edges, and the roughness floor.
  for (float v : {0.f, 1e-7f, 1e-6f, 1e-5f, 1e-4f, .5f, 1.f})
    for (float l : {0.f, 1e-7f, 1e-6f, 1e-5f, 1e-4f, .5f, 1.f}) {
      f.view = vec3(std::sqrt(1.f - v * v), 0, v);
      const vec3 direction(0, std::sqrt(1.f - l * l), l);
      const auto got = pbrReflectionKernel(&f.state, f.view, direction);
      const auto target = oracle::kernel(f, oracle::D(direction));
      const double pdf = pbrBsdfPdf(&f.state, f.view, direction),
                   reference = oracle::pdf(f, oracle::D(direction));
      require(std::abs(pdf - reference) < 2e-6 + 5e-4 * std::abs(reference),
          "boundary PDF");
      for (int c = 0; c < 3; ++c)
        require(
            std::abs(got[c] - target[c]) < 2e-6 + 5e-4 * std::abs(target[c]),
            "boundary target");
    }
  f = fixture(1);
  f.state.roughness = 0.f;
  const auto rough = pbrReflectionKernel(&f.state, f.view, f.view);
  const auto roughReference = oracle::kernel(f, oracle::D(f.view));
  for (int c = 0; c < 3; ++c)
    require(std::abs(rough[c] - roughReference[c]) < 5e-4 * roughReference[c],
        "roughness floor target");
  // Force tiny-probability old strata; do not wait for a random rare event.
  for (int variant = 0; variant < 3; ++variant) {
    f = fixture(0);
    if (variant == 0)
      f.state.clearcoat = 1e-10f;
    if (variant == 1)
      f.state.specular = 1e-10f;
    if (variant == 2)
      f = fixture(5);
    Ray ray{};
    ray.dir = -f.view;
    auto rng = random(0, 0);
    NextRayQmc q{
        {variant == 2 ? .25f : 0.f, .25f, .125f, 0.f}, variant == 2 ? 2 : 3};
    const auto next = sampleNativeReflection(
        &f.state, &ray, &rng, &q, variant == 2 ? .01f : .5f);
    require(
        !isDead(next) && next.component == NextRayComponent::NativeReflection,
        "forced reflection stratum");
    const auto k = oracle::kernel(f, oracle::D(next.direction));
    const auto p = oracle::pdf(f, oracle::D(next.direction));
    for (int c = 0; c < 3; ++c)
      require(std::abs(next.contributionWeight[c] - k[c] / p)
              < 2e-6 + 5e-4 * std::abs(k[c] / p),
          "forced stratum K/q");
  }
  for (float eta : {std::nextafter(1.f, 0.f), 1.f, std::nextafter(1.f, 2.f)}) {
    f = fixture(7);
    f.state.eta = eta;
    Ray ray{};
    ray.dir = -f.view;
    auto rng = random(7, 1);
    NextRayQmc q{{0.f, .25f, .125f, 0.f}, 3};
    const auto next = sampleNativeReflection(&f.state, &ray, &rng, &q, .5f);
    require(next.component == NextRayComponent::NativeTransmission,
        "eta-one neighborhood transmission");
    require(next.event
            == (eta == 1.f ? NextRayEvent::Singular : NextRayEvent::Continuous),
        "no epsilon delta switch");
  }
}
int run()
{
  boundaries();
  transmissionControl();
  for (int id = 0; id < CASES; ++id) {
#ifdef NATIVE_CUDA
    const auto samples = deviceSamples(id);
#else
    std::vector<NextRay> samples(SAMPLES);
    const auto f = fixture(id);
    Ray ray{};
    ray.dir = -f.view;
    for (int i = 0; i < SAMPLES; ++i) {
      auto rng = random(id, i);
      samples[i] = __direct_callable__nextRay(&f.state, &ray, &rng, nullptr);
    }
#endif
    qualify(id, samples);
  }
  return 0;
}
} // namespace native_test
int main()
try {
  return native_test::run();
} catch (const std::exception &e) {
  std::fprintf(stderr, "FAIL: %s\n", e.what());
  return 1;
}
