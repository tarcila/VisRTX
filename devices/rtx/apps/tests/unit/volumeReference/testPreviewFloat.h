// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>
#include "gpu/volumePreviewMath.h"
#include "volume/PreviewColdPreparation.h"
#ifdef PREVIEW_FLOAT_CUDA
#include <cuda_runtime.h>
#endif

namespace preview_float_test {
struct Input
{
  visrtx::PrincipledVolumeValues values;
  float length;
  float extinctionPerDensity[3];
};
struct Output
{
  visrtx::preview_math::Coefficients coefficients, prefactored;
  float color[3], transmission[3], cold[3];
};
VISRTX_HOST_DEVICE Output evaluate(
    const Input &input, const visrtx::PreviewColdKnot *table)
{
  Output result;
  result.coefficients =
      visrtx::preview_math::evaluateOptical(input.values, nullptr);
  result.prefactored = visrtx::preview_math::evaluateOptical(
      input.values, nullptr, input.extinctionPerDensity);
  visrtx::preview_math::integrateSegment(result.coefficients,
      result.coefficients.emission,
      input.length,
      result.color,
      result.transmission);
  visrtx::PrincipledVolumeValues cold;
  cold.density = 2.f;
  cold.scatteringColor = visrtx::vec3(0.f, .5f, 1.f);
  cold.transmissionColor = visrtx::vec3(.5f, .25f, .125f);
  cold.temperature = input.values.temperature;
  cold.blackbodyIntensity = 128.f;
  cold.blackbodyTint = visrtx::vec3(128.f);
  auto thermal = visrtx::preview_math::evaluateOptical(cold, nullptr);
  visrtx::preview_math::addThermal(thermal, cold, table);
  for (int c = 0; c < 3; ++c)
    result.cold[c] = thermal.emission[c];
  return result;
}
#ifdef PREVIEW_FLOAT_CUDA
struct Buffers
{
  Input *input{};
  Output *output{};
  visrtx::PreviewColdKnot *table{};
  ~Buffers();
};
inline Buffers::~Buffers()
{
  cudaFree(table);
  cudaFree(output);
  cudaFree(input);
}
VISRTX_GLOBAL void kernel(const Input *input,
    Output *output,
    int count,
    const visrtx::PreviewColdKnot *table)
{
  int i = int(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < count)
    output[i] = evaluate(input[i], table);
}
#endif
inline void require(bool condition, const char *what)
{
  if (!condition)
    throw std::runtime_error(what);
}
inline bool close(
    float actual, long double expected, double absolute, double relative)
{
  return std::isfinite(actual)
      && std::abs(static_cast<long double>(actual) - expected)
      <= absolute + relative * std::abs(expected);
}
inline int run()
try {
  const auto &table = visrtx::previewColdTable();
  double xyzBound = 0.;
  for (int c = 0; c < 3; ++c)
    xyzBound += std::exp(double(table[0].logXYZ[c]));
  require(8. * xyzBound * 0x1p30 < std::numeric_limits<float>::min(),
      "cold FTZ shortcut is not representability-safe");
  const float seams[] = {std::nextafter(100.f, 0.f),
      100.f,
      std::nextafter(100.f, INFINITY),
      std::nextafter(300.f, 0.f),
      300.f,
      std::nextafter(300.f, INFINITY)};
  std::vector<Input> input(4096 + 6);
  std::vector<Output> output(input.size());
  uint32_t state = 0x923ac317;
  auto draw = [&]() {
    state = 1664525u * state + 1013904223u;
    return float(state >> 8) * 0x1p-24f;
  };
  for (size_t i = 0; i < input.size(); ++i) {
    auto &v = input[i].values;
    v.density = i % 17 == 0 ? 0.f : 10.f * draw();
    v.transmissionDistance = .01f + 4.f * draw();
    for (int c = 0; c < 3; ++c) {
      v.transmissionColor[c] = i % 19 == 0 ? 1.f : expf(-10.f * draw());
      v.scatteringColor[c] = draw();
      v.emission[c] = 10.f * draw();
      input[i].extinctionPerDensity[c] =
          float(-std::log(double(v.transmissionColor[c])))
          / v.transmissionDistance;
    }
    input[i].length = powf(10.f, -8.f + 11.f * draw());
    v.temperature =
        i < 4096 ? 50.f + 250.f * float(i) / 4095.f : seams[i - 4096];
    output[i] = evaluate(input[i], table.data());
  }
#ifdef PREVIEW_FLOAT_CUDA
  Buffers buffers;
  auto &deviceInput = buffers.input;
  auto &deviceOutput = buffers.output;
  auto &deviceTable = buffers.table;
  require(cudaMalloc(&deviceTable, sizeof(table)) == cudaSuccess,
      "cold table allocation");
  require(cudaMemcpy(
              deviceTable, table.data(), sizeof(table), cudaMemcpyHostToDevice)
          == cudaSuccess,
      "cold table upload");
  require(cudaMalloc(&deviceInput, input.size() * sizeof(Input)) == cudaSuccess,
      "input allocation");
  require(
      cudaMalloc(&deviceOutput, output.size() * sizeof(Output)) == cudaSuccess,
      "output allocation");
  require(cudaMemcpy(deviceInput,
              input.data(),
              input.size() * sizeof(Input),
              cudaMemcpyHostToDevice)
          == cudaSuccess,
      "upload");
  kernel<<<(input.size() + 127) / 128, 128>>>(
      deviceInput, deviceOutput, int(input.size()), deviceTable);
  require(cudaGetLastError() == cudaSuccess
          && cudaDeviceSynchronize() == cudaSuccess,
      "kernel");
  require(cudaMemcpy(output.data(),
              deviceOutput,
              output.size() * sizeof(Output),
              cudaMemcpyDeviceToHost)
          == cudaSuccess,
      "download");
#endif
  for (size_t i = 0; i < input.size(); ++i) {
    const auto &v = input[i].values;
    const auto &got = output[i];
    const auto spectral = visrtx::CalibratedBlackbody{}(v.temperature);
    for (int c = 0; c < 3; ++c) {
      const double expected = spectral[c]
          * (2. * std::log(double(1u << (c + 1)))) * (1. - .5 * c) * 128.
          * 128.;
      if (!close(
              got.cold[c], expected, std::numeric_limits<float>::min(), 1e-3)) {
        std::fprintf(stderr,
            "cold T=%g c=%d actual=%.17g expected=%.17g\n",
            v.temperature,
            c,
            double(got.cold[c]),
            expected);
        throw std::runtime_error("cold preview thermal coupling error");
      }
    }
    for (int c = 0; c < 3; ++c) {
      const long double e = static_cast<long double>(v.density)
          * -std::log(static_cast<long double>(v.transmissionColor[c]))
          / v.transmissionDistance;
      const long double tau = e * input[i].length;
      const long double t = std::exp(-tau);
      const long double color =
          v.emission[c] * (tau == 0 ? input[i].length : -std::expm1(-tau) / e);
      require(close(got.coefficients.absorption[c],
                  e * (1.L - v.scatteringColor[c]),
                  2e-6,
                  2e-5),
          "FP32 absorption error");
      require(close(got.prefactored.absorption[c],
                  e * (1.L - v.scatteringColor[c]),
                  2e-6,
                  2e-5)
              && close(got.prefactored.scattering[c],
                  e * v.scatteringColor[c],
                  2e-6,
                  2e-5),
          "FP32 prepared coefficient error");
      require(close(got.coefficients.scattering[c],
                  e * v.scatteringColor[c],
                  2e-6,
                  2e-5),
          "FP32 scattering error");
      require(
          close(got.transmission[c], t, 2e-5, 2e-4), "FP32 attenuation error");
      require(close(got.color[c], color, 2e-5, 2e-4), "FP32 source error");
      if (e == 0)
        require(got.coefficients.absorption[c] == 0.f
                && got.coefficients.scattering[c] == 0.f,
            "exact zero optics");
    }
  }
  double seamScale = 0.;
  for (int c = 0; c < 3; ++c)
    seamScale = std::max(seamScale, double(output[4100].cold[c]));
  for (int i = 4099; i <= 4101; ++i)
    for (int c = 0; c < 3; ++c)
      require(std::abs(double(output[i].cold[c]) - output[4100].cold[c])
              <= 5e-5 * seamScale,
          "cold/warm LUT seam jump");
  std::printf(
      "FP32 preview: %zu independent coefficient/segment controls pass\n",
      input.size());
  return 0;
} catch (const std::exception &e) {
  std::fprintf(stderr, "FAIL: %s\n", e.what());
  return 1;
}
} // namespace preview_float_test
