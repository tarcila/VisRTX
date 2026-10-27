// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include "gpu/blackbody/blackbody.h"
#include "gpu/volumeOpticsReference.h"
#ifdef BLACKBODY_CUDA
#include <cuda_runtime.h>
#endif

namespace blackbody_test {
struct Reference
{
  float temperature;
  double rgb[3], y;
};
struct Result
{
  double rgb[3], coupled[3], vacuum[3];
};
VISRTX_HOST_DEVICE Result evaluate(float temperature)
{
  visrtx::CalibratedBlackbody blackbody;
  const auto rgb = blackbody(temperature);
  visrtx::PrincipledVolumeValues v;
  v.temperature = temperature;
  v.density = 2.f;
  v.transmissionColor = visrtx::vec3(.5f, .25f, .125f);
  v.scatteringColor = visrtx::vec3(0.f, .5f, 1.f);
  v.blackbodyIntensity = temperature == 100.f ? 1e30f : 1.f;
  v.blackbodyTint = visrtx::vec3(v.blackbodyIntensity);
  const auto coupled = visrtx::optical_reference::evaluate(v, blackbody);
  v.density = 0.f;
  v.emission = visrtx::vec3(2.f, 4.f, 8.f);
  const auto vacuum = visrtx::optical_reference::evaluate(v, blackbody);
  Result result{};
  for (int c = 0; c < 3; ++c) {
    result.rgb[c] = rgb[c];
    result.coupled[c] = coupled.emission[c];
    result.vacuum[c] = vacuum.emission[c];
  }
  return result;
}
#ifdef BLACKBODY_CUDA
struct Buffers
{
  Reference *input{};
  Result *output{};
  ~Buffers();
};
inline Buffers::~Buffers()
{
  cudaFree(input);
  cudaFree(output);
}
VISRTX_GLOBAL void evaluateKernel(const Reference *input, Result *output, int n)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    output[i] = evaluate(input[i].temperature);
}
inline void checkCuda(cudaError_t result)
{
  if (result != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(result));
}
#endif
inline int run(const char *path)
{
  std::ifstream file(path);
  if (!file)
    throw std::runtime_error("missing spectral oracle");
  std::vector<Reference> input;
  std::string line;
  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#')
      continue;
    Reference r{};
    // Parse temperature as double: float extraction may reject subnormal input.
    double temperature;
    std::istringstream row(line);
    if (!(row >> temperature >> r.rgb[0] >> r.rgb[1] >> r.rgb[2] >> r.y))
      throw std::runtime_error("malformed spectral oracle");
    r.temperature = float(temperature);
    input.push_back(r);
  }
  if (input.size() < 1200)
    throw std::runtime_error("incomplete spectral controls");
  std::vector<Result> output(input.size());
#ifdef BLACKBODY_CUDA
  // Managed allocations are test-only buffers, released even on CUDA failure.
  Buffers buffers;
  checkCuda(
      cudaMallocManaged(&buffers.input, input.size() * sizeof(Reference)));
  checkCuda(cudaMallocManaged(&buffers.output, output.size() * sizeof(Result)));
  std::copy(input.begin(), input.end(), buffers.input);
  evaluateKernel<<<(input.size() + 127) / 128, 128>>>(
      buffers.input, buffers.output, int(input.size()));
  checkCuda(cudaGetLastError());
  checkCuda(cudaDeviceSynchronize());
  std::copy(buffers.output, buffers.output + output.size(), output.begin());
#else
  for (size_t i = 0; i < input.size(); ++i)
    output[i] = evaluate(input[i].temperature);
#endif
  // Independent inverse-matrix Y row; not the production float table.
  constexpr double weights[3] = {
      .21267284631836203, .715152167154881, .07217499957321263};
  double maxLut = 0., maxFallback = 0., maxY = 0.;
  int failures = 0;
  for (size_t i = 0; i < input.size(); ++i) {
    const auto &r = input[i];
    const auto &got = output[i];
    const bool lut = r.temperature >= 300.f && r.temperature <= 100000.f;
    const double tolerance = lut ? 2e-4 : 1e-3;
    double error = 0., y = 0., expectedCoupled[3];
    const double scale =
        r.temperature == 100.f ? double(1e30f) * double(1e30f) : 1.;
    double coupledScale = 0.;
    for (int c = 0; c < 3; ++c) {
      error = std::max(error, std::abs(got.rgb[c] - r.rgb[c]));
      y += weights[c] * got.rgb[c];
      expectedCoupled[c] = r.rgb[c] * 2. * (-std::log(std::ldexp(1., -c - 1)))
          * (1. - .5 * c) * scale;
      coupledScale = std::max(coupledScale, expectedCoupled[c]);
    }
    bool ok = true;
    for (int c = 0; c < 3; ++c) {
      ok &= std::isfinite(got.rgb[c]) && got.rgb[c] >= 0.;
      ok &= std::isfinite(got.coupled[c]) && got.coupled[c] >= 0.;
      ok &= std::abs(got.coupled[c] - expectedCoupled[c])
          <= tolerance * coupledScale;
      ok &= got.vacuum[c] == std::ldexp(2., c);
    }
    // Exact zero for underflow-to-zero references; otherwise use relative
    // tolerances even for cold tails far below the earlier float test floor.
    ok &= error <= tolerance * r.y && std::abs(y - r.y) <= tolerance * r.y;
    ok &=
        got.coupled[2] == 0.; // scattering albedo one suppresses thermal source
    if (r.y > 0.) {
      (lut ? maxLut : maxFallback) =
          std::max(lut ? maxLut : maxFallback, error / r.y);
      maxY = std::max(maxY, std::abs(y - r.y) / r.y);
    }
    if (!ok) {
      ++failures;
      std::fprintf(stderr,
          "T=%.9g RGB error/Y=%.9g coupled=(%.9g %.9g %.9g)\n",
          r.temperature,
          r.y > 0. ? error / r.y : error,
          got.coupled[0],
          got.coupled[1],
          got.coupled[2]);
    }
  }
  // Adjacent represented temperatures straddle each LUT/fallback seam.
  double maxSeam = 0.;
  for (float seam : {300.f, 100000.f}) {
    const auto center = std::find_if(input.begin(),
        input.end(),
        [=](const Reference &r) { return r.temperature == seam; });
    if (center == input.end() || center == input.begin()
        || center + 1 == input.end()
        || (center - 1)->temperature != std::nextafter(seam, 0.f)
        || (center + 1)->temperature != std::nextafter(seam, INFINITY))
      throw std::runtime_error("missing adjacent-float seam controls");
    const size_t i = size_t(center - input.begin());
    const double scale = *std::max_element(output[i].rgb, output[i].rgb + 3);
    for (size_t side : {i - 1, i + 1})
      for (int c = 0; c < 3; ++c)
        maxSeam = std::max(
            maxSeam, std::abs(output[side].rgb[c] - output[i].rgb[c]) / scale);
  }
  if (maxSeam > 5e-5)
    ++failures;
  std::printf(
      "Maximum adjacent-float seam jump / center max RGB: %.9g\n", maxSeam);
  std::printf(
      "Blackbody: %zu temperatures, LUT RGB/Y %.9g, fallback RGB/Y %.9g, relative Y %.9g; %d failures\n",
      input.size(),
      maxLut,
      maxFallback,
      maxY,
      failures);
  return failures ? 1 : 0;
}
} // namespace blackbody_test
