// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cstdio>
#include <stdexcept>
#include <vector>
#include "gpu/blackbody/volumeBlackbody.h"
#include "volume/QualityColdPreparation.h"

namespace quality_cold_test {
using namespace visrtx;
inline void require(bool ok, const char *message)
{
  if (!ok)
    throw std::runtime_error(message);
}
#ifdef QUALITY_COLD_CUDA
inline void checked(cudaError_t e)
{
  if (e != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(e));
}
VISRTX_GLOBAL void evaluate(const PreviewColdKnot *table,
    const float *temperature,
    glm::dvec3 *out,
    size_t count)
{
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count)
    out[i] = QualityBlackbody{table}(temperature[i]);
}
#endif
inline int run()
{
  double minimum = INFINITY;
  for (const auto &row : blackbody_data::spectral)
    minimum = std::min(minimum, double(row[0]));
  require(minimum > 16000.,
      "underflow identity invalid for changed spectral table");
  std::vector<float> temperatures{0.f, 1.f, 10.f, 19.999f, 20.f, 21.f, 24.f};
  for (int i = 0; i <= 2048; ++i)
    temperatures.push_back(float(25. * std::exp(std::log(12.) * i / 2048.)));
  for (float seam : {25.f, 300.f}) {
    temperatures.push_back(std::nextafter(seam, 0.f));
    temperatures.push_back(seam);
    temperatures.push_back(std::nextafter(seam, INFINITY));
  }
  const auto &table = qualityColdTable();
  std::vector<glm::dvec3> values(temperatures.size());
#ifdef QUALITY_COLD_CUDA
  PreviewColdKnot *deviceTable = nullptr;
  float *deviceTemperatures = nullptr;
  glm::dvec3 *deviceValues = nullptr;
  checked(cudaMalloc(&deviceTable, sizeof(table)));
  checked(cudaMalloc(&deviceTemperatures, temperatures.size() * sizeof(float)));
  checked(cudaMalloc(&deviceValues, values.size() * sizeof(glm::dvec3)));
  checked(cudaMemcpy(
      deviceTable, table.data(), sizeof(table), cudaMemcpyHostToDevice));
  checked(cudaMemcpy(deviceTemperatures,
      temperatures.data(),
      temperatures.size() * sizeof(float),
      cudaMemcpyHostToDevice));
  evaluate<<<unsigned((values.size() + 127) / 128), 128>>>(
      deviceTable, deviceTemperatures, deviceValues, values.size());
  checked(cudaGetLastError());
  checked(cudaDeviceSynchronize());
  checked(cudaMemcpy(values.data(),
      deviceValues,
      values.size() * sizeof(glm::dvec3),
      cudaMemcpyDeviceToHost));
  checked(cudaFree(deviceTable));
  checked(cudaFree(deviceTemperatures));
  checked(cudaFree(deviceValues));
#else
  for (size_t i = 0; i < values.size(); ++i)
    values[i] = QualityBlackbody{table.data()}(temperatures[i]);
#endif
  double maximum = 0.;
  for (size_t i = 0; i < values.size(); ++i) {
    const auto reference = CalibratedBlackbody{}(temperatures[i]);
    double norm = reference.x + reference.y + reference.z, difference = 0.;
    for (int c = 0; c < 3; ++c) {
      require(std::isfinite(values[i][c]) && values[i][c] >= 0.,
          "invalid cold source");
      difference += std::abs(values[i][c] - reference[c]);
      const double amplified = values[i][c] * 0x1p500,
                   expected = reference[c] * 0x1p500;
      require(std::isfinite(amplified)
              && std::abs(amplified - expected)
                  <= 1e-3 * norm * 0x1p500 + 1e-300,
          "amplified cold source error");
    }
    if (norm > 0.) {
      maximum = std::max(maximum, difference / norm);
      require(difference / norm <= 1e-3, "cold spectral error budget");
    } else
      require(values[i] == glm::dvec3(0.), "underflow branch changed zero");
    if (temperatures[i] <= 20.f)
      require(
          reference == glm::dvec3(0.), "spectral underflow proof regression");
  }
  for (size_t base = values.size() - 6; base < values.size(); base += 3) {
    const auto lo = CalibratedBlackbody{}(temperatures[base]),
               hi = CalibratedBlackbody{}(temperatures[base + 2]);
    double residual = 0., norm = 0.;
    for (int c = 0; c < 3; ++c) {
      residual +=
          std::abs((values[base + 2][c] - values[base][c]) - (hi[c] - lo[c]));
      norm += hi[c];
    }
    require(residual <= 5e-5 * norm, "cold lookup seam residual");
  }
  std::printf("Quality cold source: %zu controls, max relative L1 %.9g\n",
      values.size(),
      maximum);
  return 0;
}
} // namespace quality_cold_test
