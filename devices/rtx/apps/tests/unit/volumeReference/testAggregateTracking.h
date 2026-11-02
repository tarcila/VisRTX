// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "gpu/aggregateVolumePhase.h"
#include "gpu/blackbody/blackbody.h"
#include "testAggregate.h"

namespace aggregate_tracking_test {
using namespace visrtx;
namespace tr = volume_tracking;
namespace av = aggregate_volume;
namespace ref = optical_reference;
constexpr int SAMPLES = 32768;
struct Result
{
  double value[12]{};
  int failure{0};
};

VISRTX_HOST_DEVICE Result trial(int id, int index)
{
  aggregate_test::Fixture f(id >= 200 ? 12 : id == 100 ? 10 : id);
  if (id == 100) {
    auto &b = f.pv[0].bindings;
    b.density.scale = 1.f;
    b.emission.scale = vec3(0);
    b.temperature.scale = 6500.f;
    b.blackbodyIntensity.scale = .5f;
    const float c = b.transmissionColor.scale.x;
    f.valid &= ref::buildBounds(
        {{1, 1}, {{c, c}, {c, c}, {c, c}}, {1, 1}}, f.pv[0].opticalBounds);
  }
  Result out;
  if (!f.valid) {
    out.failure = 1;
    return out;
  }
  PCGState camera, connection;
  pcg_init(&camera,
      detail::pcg_mix64(uint64_t(index) + 123),
      detail::pcg_mix64(uint64_t(id) + 63));
  pcg_init(&connection,
      detail::pcg_mix64(uint64_t(index) + 973),
      detail::pcg_mix64(uint64_t(id) + 837));
  tr::Random random{&camera}, shadowRandom{&connection};
  const auto ray = av::worldDistanceRay(f.input);
  CalibratedBlackbody blackbody;
  tr::PathState path;
  if (id >= 200) {
    f.pv[0].bindings.scatteringColor.scale = vec3(1, .5, 0);
    f.pv[1].bindings.scatteringColor.scale = vec3(0, .25, 1);
    for (int i = 0; i < 2; ++i) {
      f.pv[i].hasConstantTransmissionLog = true;
      for (int c = 0; c < 3; ++c)
        f.pv[i].constantTransmissionLog[c] =
            ref::negativeLog(f.pv[i].bindings.transmissionColor.scale[c]);
    }
    if (id == 201) {
      const auto swap = f.occurrences[0];
      f.occurrences[0] = f.occurrences[1];
      f.occurrences[1] = swap;
    }
    VolumeOccurrenceGPUData many[13];
    if (id == 202) {
      for (int i = 0; i < 13; ++i) {
        many[i] = f.occurrences[i % 2];
        many[i].groupSlot = size_t(i);
      }
      f.scene.occurrences = many;
      f.scene.count = 13;
    }
    const auto direction =
        tr::sampleDirection(f.scene, ray, {1., 2.}, 1.5, random);
    const auto optical =
        av::evaluate(f.scene, ray, {1., 2.}, 1.5, 0.f, blackbody);
    tr::applyDirection(direction.mixture.kernel,
        optical.coefficients.scattering,
        direction.mixture.pdf,
        path);
    for (int c = 0; c < 3; ++c) {
      out.value[c] = path.throughput[c];
      out.value[c + 3] = path.throughput[c] * direction.cosine;
    }
    out.value[6] = glm::length(direction.direction);
    out.value[7] = direction.direction.z - direction.cosine;
    out.value[8] =
        f.scene.occurrences[direction.selectedOccurrence].volume.principled
        == &f.pv[0];
    const glm::dvec3 axes[4]{
        {1., 2., 3.}, {0., 0., -1.}, {1., 0., 0.}, {.01, .02, 1.}};
    const auto axis = glm::normalize(axes[index % 4]);
    auto rotationRay = ray;
    for (int c = 0; c < 3; ++c)
      rotationRay.direction[c] = axis[c];
    const auto rotated = tr::worldDirection(rotationRay, direction.cosine, .37);
    out.value[9] = glm::length(rotated) - 1.;
    out.value[10] = glm::dot(rotated, axis) - direction.cosine;
    out.value[11] = direction.mixture.pdf > 0. ? 0. : 1.;
    return out;
  }
  const auto hit = tr::trace(f.scene, ray, 1., random, blackbody, path);
  if (hit.endpoint != tr::Endpoint::Boundary
      && !(hit.distance >= hit.span.lower && hit.distance < hit.span.upper))
    out.failure = 2;
  double transmission[3]{1., 1., 1.};
  tr::shadow(f.scene, ray, shadowRandom, transmission);
  for (int c = 0; c < 3; ++c) {
    out.value[c] = path.radiance[c];
    out.value[c + 3] =
        hit.endpoint == tr::Endpoint::Boundary ? path.throughput[c] : 0.;
    out.value[c + 6] = transmission[c];
    out.value[c + 9] =
        hit.endpoint == tr::Endpoint::Scattered ? path.throughput[c] : 0.;
  }
  return out;
}

inline void require(bool ok, const char *message)
{
  if (!ok)
    throw std::runtime_error(message);
}

inline Result expected(int id)
{
  if (id >= 200) {
    const double g0 =
        (1. - double(.4f)) * double(.2f) + double(.4f) * double(-.3f);
    const double g1 =
        (1. - double(.25f)) * double(-.5f) + double(.25f) * double(.6f);
    const double n0 = id == 202 ? 7. : 1., n1 = id == 202 ? 6. : 1.;
    Result out;
    for (int c = 0; c < 3; ++c)
      out.value[c] = 1.;
    out.value[3] = g0;
    out.value[4] = (n0 * g0 + n1 * g1) / (n0 + n1);
    out.value[5] = g1;
    out.value[6] = 1.;
    out.value[8] = 1.5 * n0 / (1.5 * n0 + 2.5 * n1);
    return out;
  }
  // Independent constant-segment analytic solutions, not aggregate traversal
  // or the production optical evaluator. The fixture's represented C is used.
  const long double a = -logl(static_cast<long double>(.36787944117144233f));
  long double t = 1., j[3]{};
  auto segment = [&](long double sigma,
                     long double length,
                     long double red,
                     long double green,
                     long double blue) {
    const long double factor =
        sigma == 0 ? length : -expm1l(-sigma * length) / sigma;
    j[0] += t * red * factor;
    j[1] += t * green * factor;
    j[2] += t * blue * factor;
    t *= expl(-sigma * length);
  };
  switch (id) {
  case 0:
  case 20:
    segment(a, .5, 2, 0, 0);
    segment(3 * a, .5, 2, 4, 0);
    segment(2 * a, .5, 0, 4, 0);
    break;
  case 1:
    segment(a, 1, 2, 0, 0);
    segment(2 * a, 1, 0, 4, 0);
    break;
  case 6:
    segment(a, 2, 2, 0, 0);
    break;
  case 10:
    segment(0, 1, 2, 3, 5);
    break;
  case 11: {
    const long double tf = logl(2.L) / 2;
    segment(a + tf, 1, 2, 2 * tf, 0);
    break;
  }
  case 12:
    segment(3 * a, 1, 2, 4, 0);
    break;
  case 16:
    segment(2 * a, 1, 4, 0, 0);
    break;
  case 19:
    segment(2 * a, .5, 2, 0, 0);
    segment(0, .5, 2, 0, 0);
    break;
  case 21:
    segment(a, 1, 2, 0, 0);
    segment(2 * a, 1, 0, 4, 0);
    break;
  case 24:
    break;
  case 100: {
    const auto b = CalibratedBlackbody{}(6500.f);
    segment(a, 1, .5L * a * b.x, .5L * a * b.y, .5L * a * b.z);
    break;
  }
  default:
    throw std::runtime_error("unknown analytic fixture");
  }
  Result out;
  for (int c = 0; c < 3; ++c) {
    out.value[c] = double(j[c]);
    out.value[c + 3] = out.value[c + 6] = double(t);
    if (id == 12)
      out.value[c + 9] = double((2.L / 3) * (1 - t));
  }
  return out;
}

inline void check(int id, const std::vector<Result> &results)
{
  const auto target = expected(id);
  for (const auto &r : results) {
    require(r.failure == 0, "candidate geometry/preparation invariant");
    if (id >= 200)
      require(std::abs(r.value[6] - 1.) < 2e-14 && std::abs(r.value[7]) < 2e-14
              && std::abs(r.value[9]) < 2e-14 && std::abs(r.value[10]) < 2e-14
              && r.value[11] == 0.,
          "world-space phase rotation/PDF");
  }
  for (int c = 0; c < 12; ++c) {
    long double sum = 0., squares = 0.;
    for (const auto &r : results) {
      require(std::isfinite(r.value[c]), "nonfinite aggregate tracking result");
      sum += r.value[c];
      squares += r.value[c] * r.value[c];
    }
    const double mean = double(sum / results.size());
    const double variance = std::max(0.,
        double((squares - sum * sum / results.size()) / (results.size() - 1)));
    const double tolerance = 6. * sqrt(variance / results.size()) + 5e-5;
    if (std::abs(mean - target.value[c]) > tolerance) {
      std::fprintf(stderr,
          "fixture %d channel %d: %.12g expected %.12g tolerance %.12g\n",
          id,
          c,
          mean,
          target.value[c],
          tolerance);
      throw std::runtime_error("aggregate transport expectation");
    }
  }
}

#ifdef AGGREGATE_TRACKING_CUDA
inline void checked(cudaError_t e)
{
  if (e != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(e));
}
VISRTX_GLOBAL void runTrials(int id, Result *out)
{
  const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < SAMPLES)
    out[i] = trial(id, i);
}
#endif
inline int run()
{
  require(tr::openUnit(0, 0) == 0x1p-53, "open RNG lower endpoint");
  require(tr::openUnit(UINT32_MAX, UINT32_MAX) == 1. - 0x1p-53,
      "open RNG upper endpoint");
  PCGState state;
  pcg_init(&state, 42, 29);
  auto copy = state;
  tr::Random random{&state};
  const uint32_t high = detail::pcg_advance(&copy),
                 low = detail::pcg_advance(&copy);
  require(random.next() == tr::openUnit(high, low) && state.state == copy.state,
      "PCG consumption/replay");
  std::vector<Result> output(SAMPLES);
#ifdef AGGREGATE_TRACKING_CUDA
  Result *device = nullptr;
  checked(cudaMalloc(&device, output.size() * sizeof(Result)));
#endif
  for (int id : {0, 1, 6, 10, 11, 12, 16, 19, 20, 21, 24, 100, 200, 201, 202}) {
#ifdef AGGREGATE_TRACKING_CUDA
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
#ifdef AGGREGATE_TRACKING_CUDA
  checked(cudaFree(device));
#endif
  std::puts(
      "Aggregate Quality tracking: fifteen 32768-path analytic cases passed");
  return 0;
}
} // namespace aggregate_tracking_test
