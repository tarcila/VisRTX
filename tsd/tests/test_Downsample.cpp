// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"

#ifdef TSD_ALGORITHMS_HAS_CUDA

// tsd
#include "tsd/algorithms/cuda/downsample.hpp"
// cuda
#include <cuda_runtime_api.h>
// std
#include <cmath>
#include <cstdint>
#include <vector>

namespace {

bool cudaAvailable()
{
  int n = 0;
  return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

// Uploads an RGBA32F image with the given per-pixel luminance values (stored
// as grayscale, so luminance(r,g,b) == value) and returns the SPD mean.
float gpuMeanLogLum(const std::vector<float> &lum, uint32_t w, uint32_t h)
{
  std::vector<float> rgba(size_t(w) * h * 4);
  for (size_t i = 0; i < lum.size(); i++) {
    rgba[i * 4 + 0] = lum[i];
    rgba[i * 4 + 1] = lum[i];
    rgba[i * 4 + 2] = lum[i];
    rgba[i * 4 + 3] = 1.f;
  }
  float *dev = nullptr;
  REQUIRE(cudaMalloc((void **)&dev, rgba.size() * sizeof(float)) == cudaSuccess);
  REQUIRE(cudaMemcpy(dev,
              rgba.data(),
              rgba.size() * sizeof(float),
              cudaMemcpyHostToDevice)
      == cudaSuccess);
  const float mean =
      tsd::algorithms::cuda::meanLogLuminance(cudaStream_t{0}, dev, w, h);
  cudaFree(dev);
  return mean;
}

} // namespace

SCENARIO("tsd::algorithms::cuda::meanLogLuminance", "[Downsample]")
{
  if (!cudaAvailable()) {
    SUCCEED("no CUDA device — skipping");
    return;
  }

  GIVEN("a constant-luminance power-of-two image")
  {
    const uint32_t w = 64, h = 64;
    std::vector<float> lum(size_t(w) * h, 0.5f);
    THEN("the mean log-luminance is exactly log2 of that constant")
    {
      REQUIRE(gpuMeanLogLum(lum, w, h) == Approx(-1.f).margin(1e-4));
    }
  }

  GIVEN("a half-bright / half-dark power-of-two image")
  {
    const uint32_t w = 128, h = 64;
    std::vector<float> lum(size_t(w) * h);
    for (uint32_t y = 0; y < h; y++) {
      for (uint32_t x = 0; x < w; x++)
        lum[size_t(y) * w + x] = x < w / 2 ? 0.25f : 4.f;
    }
    THEN("the mean log-luminance is the average of both halves")
    {
      // log2(0.25) = -2, log2(4) = 2 -> mean 0
      REQUIRE(gpuMeanLogLum(lum, w, h) == Approx(0.f).margin(1e-3));
    }
  }

  GIVEN("a wide image needing multiple downsample passes")
  {
    const uint32_t w = 5000, h = 117; // 13 levels: exercises the host loop
    std::vector<float> lum(size_t(w) * h, 2.f);
    THEN("a constant image still reduces to its exact log-luminance")
    {
      REQUIRE(gpuMeanLogLum(lum, w, h) == Approx(1.f).margin(1e-3));
    }
  }

  GIVEN("a non-power-of-two gradient image")
  {
    const uint32_t w = 37, h = 23;
    std::vector<float> lum(size_t(w) * h);
    float mn = 1e30f, mx = -1e30f;
    double exact = 0.0;
    for (size_t i = 0; i < lum.size(); i++) {
      lum[i] = 0.1f + 2.f * float(i) / float(lum.size());
      const float l = std::log2(lum[i]);
      exact += l;
      mn = std::min(mn, l);
      mx = std::max(mx, l);
    }
    exact /= double(lum.size());
    THEN("the SPD mean matches the exact mean (identity padding)")
    {
      const float mean = gpuMeanLogLum(lum, w, h);
      REQUIRE(mean >= mn);
      REQUIRE(mean <= mx);
      REQUIRE(mean == Approx(exact).margin(1e-3));
    }
  }
}

#else

SCENARIO("tsd::algorithms::cuda::meanLogLuminance", "[Downsample]")
{
  SUCCEED("built without CUDA — skipping");
}

#endif
