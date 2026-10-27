// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "gpu/volumeOpticsReference.h"

#ifdef VOLUME_REFERENCE_CUDA
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#endif

#include <cfenv>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace volume_reference_test {
namespace ref = visrtx::optical_reference;

struct Input
{
  visrtx::PrincipledVolumeValues values;
  ref::EffectiveRanges ranges;
  float alpha;
  float distance;
};

struct Result
{
  double logarithm;
  double logUpper;
  double logSingleton;
  ref::Coefficients coefficients;
  ref::Coefficients thermal;
  ref::Coefficients tf;
  ref::Bounds bounds;
  ref::Bounds singleton;
  ref::Bounds tfBounds;
  ref::Coefficients aggregate;
  ref::Bounds aggregateBounds;
  double majorant;
  int metadataChecks;
};

struct Blackbody
{
  VISRTX_HOST_DEVICE visrtx::vec3 operator()(float) const;
};

VISRTX_HOST_DEVICE visrtx::vec3 Blackbody::operator()(float) const
{
  return visrtx::vec3(1.f, 2.f, 3.f);
}

VISRTX_HOST_DEVICE Result evaluate(const Input &in)
{
  Result r{};
  const auto &v = in.values;
  r.logarithm = ref::negativeLog(v.transmissionColor.x);
  r.logUpper = ref::negativeLogUpper(in.ranges.transmission[0]);
  r.logSingleton =
      ref::negativeLogUpper({v.transmissionColor.x, v.transmissionColor.x});
  r.coefficients = ref::evaluate(v);
  Blackbody blackbody;
  r.thermal = ref::evaluate(v, blackbody);
  r.metadataChecks = ref::buildBounds(in.ranges, r.bounds) ? 1 : 0;
  ref::EffectiveRanges singleton{{v.density, v.density},
      {{v.transmissionColor.x, v.transmissionColor.x},
          {v.transmissionColor.y, v.transmissionColor.y},
          {v.transmissionColor.z, v.transmissionColor.z}},
      {v.transmissionDistance, v.transmissionDistance}};
  r.metadataChecks += ref::buildBounds(singleton, r.singleton) ? 1 : 0;
  r.tf =
      ref::evaluateTransfer(visrtx::vec4(2.f, 3.f, 5.f, in.alpha), in.distance);
  r.metadataChecks += ref::buildTransferBounds(
                          {-1.f, 2.f}, {in.distance, in.distance}, r.tfBounds)
      ? 1
      : 0;
  // Same accumulation graph, including repeated occurrences and zero terms.
  for (int i = 0; i < 17; ++i) {
    r.aggregate = ref::combine(r.aggregate, r.coefficients);
    r.aggregateBounds = ref::combineBounds(r.aggregateBounds, r.singleton);
    if ((i & 1) == 0) {
      r.aggregate = ref::combine(r.aggregate, r.tf);
      r.aggregateBounds = ref::combineBounds(r.aggregateBounds, r.tfBounds);
    }
  }
  r.majorant = ref::majorant(r.aggregateBounds);
  return r;
}

#ifdef VOLUME_REFERENCE_CUDA
VISRTX_GLOBAL void volumeReferenceKernel(
    const Input *in, Result *out, int count)
{
  const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < count)
    out[i] = evaluate(in[i]);
}
#endif

inline float fromBits(uint32_t bits)
{
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

inline void require(bool condition, const std::string &message)
{
  if (!condition)
    throw std::runtime_error(message);
}

inline bool close(double actual, long double expected)
{
  if (expected == 0.L)
    return actual == 0.;
  // Fixed BEFORE measurement: reference log contract is <2^-40 relative;
  // this budget additionally covers partition/basic arithmetic and fixture RN.
  return std::isfinite(actual)
      && std::abs(static_cast<long double>(actual) - expected)
      <= std::abs(expected) * 0x1p-38L;
}

inline void check(const Input &in, const Result &r, double oracle)
{
  require(r.metadataChecks == 3, "valid construction failed");
  require(close(r.logarithm, oracle), "log vs Decimal90");
  require(r.logUpper >= r.logarithm, "range log underbound");
  require(r.logSingleton >= r.logarithm, "singleton log underbound");
  require(std::isfinite(r.majorant), "nonfinite aggregate bound");
  const auto &v = in.values;
  for (int c = 0; c < 3; ++c) {
    const long double logValue = c == 0 ? oracle
        : c == 1                        ? 0.L
                 : -std::log(static_cast<long double>(v.transmissionColor[c]));
    const long double e = logValue * static_cast<long double>(v.density)
        / static_cast<long double>(v.transmissionDistance);
    const long double albedo = static_cast<long double>(v.scatteringColor[c]);
    require(close(r.coefficients.absorption[c], e * (1.L - albedo)),
        "absorption vs physical oracle");
    require(close(r.coefficients.scattering[c], e * albedo),
        "scattering vs physical oracle");
    require(r.coefficients.absorption[c] <= r.bounds.absorption[c]
            && r.coefficients.scattering[c] <= r.bounds.scattering[c]
            && r.coefficients.absorption[c] <= r.singleton.absorption[c]
            && r.coefficients.scattering[c] <= r.singleton.scattering[c],
        "sample exceeds range/singleton enclosure");
    require(
        ref::extinction(r.aggregate, c) <= r.majorant, "aggregate underbound");
    require(r.aggregate.absorption[c] <= r.aggregateBounds.absorption[c]
            && r.aggregate.scattering[c] <= r.aggregateBounds.scattering[c],
        "partition aggregate underbound");
    const long double thermal = static_cast<long double>(v.emission[c])
        + static_cast<long double>(r.coefficients.absorption[c])
            * static_cast<long double>(v.blackbodyTint[c])
            * static_cast<long double>(v.blackbodyIntensity) * (c + 1);
    require(close(r.thermal.emission[c], thermal),
        "thermal coupling narrowed or incorrect");
    require(r.coefficients.emission[c] == ref::promote(v.emission[c]),
        "independent source changed");
    // TF oracle preserves the legacy float alpha subtraction, but uses an
    // independent long-double logarithm, not the production polynomial.
    const float alpha = std::fmin(std::fmax(in.alpha, 0.f), 1.f - 0x1p-24f);
    const float transmission = 1.f - alpha;
    const long double tf = alpha <= 1e-7f
        ? 0.L
        : -std::log(static_cast<long double>(transmission))
            / static_cast<long double>(in.distance);
    require(close(r.tf.absorption[c], tf) && r.tf.scattering[c] == 0.,
        "TF optical oracle");
    require(close(r.tf.emission[c],
                tf
                    * (c == 0        ? 2
                            : c == 1 ? 3
                                     : 5)),
        "TF coupled source");
    require(r.tf.absorption[c] <= r.tfBounds.absorption[c], "TF underbound");
  }
}

inline int run(const char *oraclePath)
try {
  require(std::fegetround() == FE_TONEAREST, "host requires round-to-nearest");
  require(std::numeric_limits<double>::is_iec559
          && std::numeric_limits<double>::digits == 53,
      "binary64 required");
  const float tiny = fromBits(1);
  const float maxFloat = fromBits(0x7f7fffff);
  for (uint32_t bits : {0u,
           1u,
           0x7fffffu,
           0x800000u,
           0x3f7fffffu,
           0x7f7fffffu,
           0x80000000u,
           0x80000001u,
           0xff7fffffu}) {
    const float value = fromBits(bits);
    require(
        ref::promote(value) == static_cast<double>(value), "exact promotion");
    require(ref::floatBits(ref::narrow(ref::promote(value))) == bits,
        "conversion roundtrip");
  }
  std::ifstream stream(oraclePath);
  require(bool(stream), "missing Decimal log fixture");
  std::string comment;
  std::getline(stream, comment);
  std::vector<Input> inputs;
  std::vector<double> oracles;
  uint32_t bits;
  double oracle;
  const float densities[] = {0.f, tiny, 1.f, maxFloat};
  const float distances[] = {tiny, 1.f, maxFloat};
  const float albedos[] = {0.f, 1.f, .188f, tiny};
  const float alphas[] = {-1.f,
      0.f,
      fromBits(ref::floatBits(1e-7f) - 1),
      1e-7f,
      fromBits(ref::floatBits(1e-7f) + 1),
      .5f,
      1.f,
      2.f};
  while (stream >> std::hex >> bits >> std::dec >> oracle) {
    const size_t index = inputs.size();
    Input in{};
    auto &v = in.values;
    v.density = densities[index % 4];
    v.transmissionColor =
        visrtx::vec3(fromBits(bits), 1.f, fromBits(0x3f7fffff));
    v.transmissionDistance = distances[(index / 4) % 3];
    for (int c = 0; c < 3; ++c)
      v.scatteringColor[c] = albedos[(index + c) % 4];
    v.emission = visrtx::vec3(2.f, 3.f, 5.f);
    v.blackbodyIntensity = 2.f;
    v.blackbodyTint = visrtx::vec3(.25f, .5f, 1.f);
    // Cross adjacent exponent bins and subnormal/normal boundaries.
    const uint32_t lo = bits > 128 ? bits - 128 : 1;
    const uint32_t hi = bits < 0x3f800000u - 128 ? bits + 128 : 0x3f800000u;
    in.ranges = {{0.f, maxFloat},
        {{fromBits(lo), fromBits(hi)}, {1.f, 1.f}, {tiny, 1.f}},
        {tiny, maxFloat}};
    in.alpha = alphas[index % (sizeof(alphas) / sizeof(alphas[0]))];
    in.distance = distances[index % 3];
    inputs.push_back(in);
    oracles.push_back(oracle);
  }
  require(stream.eof() && inputs.size() > 600, "incomplete fixture");
  std::vector<Result> results;
  for (size_t i = 0; i < inputs.size(); ++i) {
    results.push_back(evaluate(inputs[i]));
    check(inputs[i], results.back(), oracles[i]);
  }
  // Invalid certificate metadata must not publish partial bounds.
  auto ranges = inputs[0].ranges;
  ref::Bounds untouched;
  untouched.absorption[0] = 42.;
  ranges.transmission[2].lower = 0.f;
  require(
      !ref::buildBounds(ranges, untouched) && untouched.absorption[0] == 42.,
      "unavailable certificate publication");
  ranges = inputs[0].ranges;
  ranges.distance.lower = 0.f;
  require(
      !ref::buildBounds(ranges, untouched), "zero lower distance certificate");
  ranges = inputs[0].ranges;
  ranges.density = {2.f, 1.f};
  require(!ref::buildBounds(ranges, untouched), "unordered range");
  ranges = inputs[0].ranges;
  ranges.transmission[0].upper = fromBits(0x7fc00000u);
  require(!ref::buildBounds(ranges, untouched), "nonfinite range metadata");
  ranges = inputs[0].ranges;
  ranges.density = {0.f, 0.f};
  ref::Bounds zero;
  require(ref::buildBounds(ranges, zero) && ref::majorant(zero) == 0.,
      "density-zero bound");
  for (auto &c : ranges.transmission)
    c = {1.f, 1.f};
  ranges.density = {tiny, maxFloat};
  require(ref::buildBounds(ranges, zero) && ref::majorant(zero) == 0.,
      "transparent bound");
  const auto invalidDistance = ref::evaluateTransfer(visrtx::vec4(1.f), 0.f);
  require(
      ref::extinction(invalidDistance, 0) == 0., "TF invalid-distance policy");
#ifdef VOLUME_REFERENCE_CUDA
  int deviceCount = 0;
  require(cudaGetDeviceCount(&deviceCount) == cudaSuccess && deviceCount > 0,
      "CUDA device required");
  thrust::device_vector<Input> deviceInputs(inputs);
  thrust::device_vector<Result> deviceResults(inputs.size());
  volumeReferenceKernel<<<(inputs.size() + 63) / 64, 64>>>(
      thrust::raw_pointer_cast(deviceInputs.data()),
      thrust::raw_pointer_cast(deviceResults.data()),
      int(inputs.size()));
  require(cudaGetLastError() == cudaSuccess, "CUDA launch failure");
  const auto error = cudaDeviceSynchronize();
  require(error == cudaSuccess, cudaGetErrorString(error));
  const thrust::host_vector<Result> gpu = deviceResults;
  for (size_t i = 0; i < inputs.size(); ++i) {
    check(inputs[i], gpu[i], oracles[i]);
    require(
        gpu[i].logarithm == results[i].logarithm, "CPU/CUDA RN log mismatch");
    for (int c = 0; c < 3; ++c) {
      require(gpu[i].coefficients.absorption[c]
                  == results[i].coefficients.absorption[c]
              && gpu[i].coefficients.scattering[c]
                  == results[i].coefficients.scattering[c]
              && gpu[i].thermal.emission[c] == results[i].thermal.emission[c]
              && gpu[i].tf.absorption[c] == results[i].tf.absorption[c],
          "CPU/CUDA RN optical mismatch");
    }
  }
#endif
  std::printf(
      "Reference optics: %zu cases, bounds, physical oracles and aggregate checks pass\n",
      inputs.size());
  return 0;
} catch (const std::exception &e) {
  std::fprintf(stderr, "FAIL: %s\n", e.what());
  return 1;
}

} // namespace volume_reference_test
