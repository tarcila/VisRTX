// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "gpu/owenSobol.h"

#ifdef SOBOL_CUDA
#include "cudaResources.h"

#include <cuda_runtime.h>
#endif

#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace sobol_test {

// Compile the unmodified pinned FastOwen excerpt as an independent host oracle.
// Reference bit reversal deliberately uses a simple bit-at-a-time construction.
namespace pinned {
static uint32_t ReverseBits32(uint32_t v)
{
  uint32_t result = 0;
  for (unsigned bit = 0; bit < 32; ++bit) {
    result = (result << 1) | (v & 1u);
    v >>= 1;
  }
  return result;
}
#define PBRT_CPU_GPU
#include "FastOwenScrambler.reference.txt"
#undef PBRT_CPU_GPU
} // namespace pinned

using Columns = std::array<std::array<uint32_t, 32>, 16>;

static void require(bool condition, const std::string &message)
{
  if (!condition)
    throw std::runtime_error(message);
}

static bool sameFloat(float a, float b)
{
  return std::memcmp(&a, &b, sizeof(float)) == 0;
}

static Columns referenceColumns()
{
  Columns result{};
  for (int i = 0; i < 32; ++i)
    result[0][i] = uint32_t(1) << (31 - i);
  std::ifstream input(SOBOL_PARAMETER_FILE);
  require(bool(input), "cannot open Joe-Kuo parameters");
  std::string line;
  // Attribution comments followed by the original column header.
  for (int i = 0; i < 4; ++i)
    std::getline(input, line);
  for (unsigned dimension = 1; dimension < 16; ++dimension) {
    unsigned d, s, a;
    require(bool(input >> d >> s >> a) && d == dimension + 1 && s <= 6,
        "invalid parameter row");
    auto &v = result[dimension];
    for (unsigned i = 0; i < s; ++i) {
      uint32_t numerator;
      require(bool(input >> numerator), "missing numerator");
      v[i] = numerator << (31 - i);
    }
    for (unsigned i = s; i < 32; ++i) {
      v[i] = v[i - s] ^ (v[i - s] >> s);
      for (unsigned k = 1; k < s; ++k)
        if (a & (1u << (s - 1 - k)))
          v[i] ^= v[i - k];
    }
  }
  return result;
}

struct Input
{
  uint32_t index, dimension, seed;
  uint32_t x = 0, y = 0, scrambleSeed = 0;
  int bounce = 0, transparency = 0;
  bool pixelSeedInput = false;
};

struct Output
{
  uint32_t pixelKey, vertexKey, raw, reversed, scrambled;
  float sample, converted;
  ::float4 camera;
  ::float2 hemi;
};

// Both executables call exactly these production seams. The independent
// expectations below come from pinned parameters and offline reference vectors.
VISRTX_HOST_DEVICE Output evaluate(Input in)
{
  const auto pixelKey = in.pixelSeedInput
      ? visrtx::owenPixelSeed(in.x, in.y, in.scrambleSeed)
      : in.seed;
  const auto vertexKey =
      visrtx::owenBounceSeed(pixelKey, in.bounce, in.transparency);
  const auto raw = visrtx::sobolUInt(in.index, in.dimension);
  const auto dimSeed = vertexKey ^ (0x9e3779b9u * (in.dimension + 1u));
  return {pixelKey,
      vertexKey,
      raw,
      visrtx::reverseBits32(raw),
      visrtx::owenScramble(raw, dimSeed),
      visrtx::owenSobol(in.index, in.dimension, vertexKey),
      visrtx::sobolFloat(in.index),
      visrtx::owenSobolCamera(in.index, vertexKey),
      visrtx::owenSobolHemi(in.index, vertexKey)};
}

#ifdef SOBOL_CUDA
static void cudaCheck(cudaError_t error)
{
  require(error == cudaSuccess, cudaGetErrorString(error));
}

VISRTX_GLOBAL void sampleKernel(const Input *inputs, Output *outputs, size_t n)
{
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    outputs[i] = evaluate(inputs[i]);
}

static void checkDevice(const std::vector<Input> &inputs)
{
  int count = 0;
  cudaCheck(cudaGetDeviceCount(&count));
  require(count > 0, "CUDA device required; missing device is not a pass");
  cudaDeviceProp properties{};
  cudaCheck(cudaGetDeviceProperties(&properties, 0));
  int driver, runtime;
  cudaCheck(cudaDriverGetVersion(&driver));
  cudaCheck(cudaRuntimeGetVersion(&runtime));
  std::cout << "CUDA " << properties.name << ", driver " << driver
            << ", runtime " << runtime << '\n';
  CudaBuffer<Input> deviceInputs;
  CudaBuffer<Output> deviceOutputs;
  cudaCheck(deviceInputs.allocate(inputs.size()));
  cudaCheck(deviceOutputs.allocate(inputs.size()));
  cudaCheck(cudaMemcpy(deviceInputs.get(),
      inputs.data(),
      inputs.size() * sizeof(Input),
      cudaMemcpyHostToDevice));
  sampleKernel<<<(inputs.size() + 127) / 128, 128>>>(
      deviceInputs.get(), deviceOutputs.get(), inputs.size());
  cudaCheck(cudaGetLastError());
  cudaCheck(cudaDeviceSynchronize());
  std::vector<Output> outputs(inputs.size());
  cudaCheck(cudaMemcpy(outputs.data(),
      deviceOutputs.get(),
      outputs.size() * sizeof(Output),
      cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto a = evaluate(inputs[i]);
    const auto b = outputs[i];
    require(a.pixelKey == b.pixelKey && a.vertexKey == b.vertexKey
            && a.raw == b.raw && a.reversed == b.reversed
            && a.scrambled == b.scrambled && sameFloat(a.sample, b.sample)
            && sameFloat(a.converted, b.converted)
            && sameFloat(a.camera.x, b.camera.x)
            && sameFloat(a.camera.y, b.camera.y)
            && sameFloat(a.camera.z, b.camera.z)
            && sameFloat(a.camera.w, b.camera.w)
            && sameFloat(a.hemi.x, b.hemi.x) && sameFloat(a.hemi.y, b.hemi.y),
        "host/device mismatch at vector " + std::to_string(i));
  }
  std::cout << inputs.size() << " exact host/CUDA vector comparisons\n";
}
#endif

static void checkPixelKeys()
{
  // Worked uint32 finalizer vectors, independent of production evaluation.
  struct Pixel
  {
    uint32_t x, y;
    std::array<uint32_t, 4> keys;
  };
  const Pixel pixels[] = {
      {0, 0, {0xf1a85082u, 0x9921c042u, 0x044f4c14u, 0x96c0d2c8u}},
      {1, 2, {0x339b28e2u, 0x5b12b822u, 0xc67c3474u, 0x54f3aaa8u}},
      {1920, 1080, {0xeebd993au, 0x863409fau, 0x1b5a85acu, 0x89d51b70u}},
      {0xffffffffu,
          0xffffffffu,
          {0x4ab06c79u, 0x2239fcb9u, 0xbf5770efu, 0x2dd8ee33u}}};
  const uint32_t seeds[] = {0, 1, 0x12345678u, 0xffffffffu};
  for (const auto &pixel : pixels) {
    require(visrtx::owenPixelSeed(pixel.x, pixel.y) == pixel.keys[0],
        "default pixel key preserves legacy vector");
    for (unsigned s = 0; s < 4; ++s) {
      require(
          visrtx::owenPixelSeed(pixel.x, pixel.y, seeds[s]) == pixel.keys[s],
          "explicit scramble seed pixel-key vector");
      require(
          visrtx::owenPixelSeed(pixel.x, pixel.y, seeds[s]) == pixel.keys[s],
          "repeated scramble seed reproduces pixel key");
    }
  }
}

static void checkSeedVectors(std::vector<Input> &inputs)
{
  std::ifstream vectors(SOBOL_SEED_VECTOR_FILE);
  require(bool(vectors), "cannot open pixel/depth vectors");
  std::string line;
  std::getline(vectors, line);
  std::getline(vectors, line);
  Input in{};
  in.pixelSeedInput = true;
  uint32_t pixel, vertex, grid;
  unsigned count = 0;
  while (vectors >> std::hex >> in.x >> in.y >> in.scrambleSeed >> in.bounce
      >> in.transparency >> in.index >> in.dimension >> pixel >> vertex
      >> grid) {
    const auto actual = evaluate(in);
    require(actual.pixelKey == pixel && actual.vertexKey == vertex,
        "fixed pixel/bounce/coverage key vector " + std::to_string(count));
    require(sameFloat(actual.sample, std::ldexp(float(grid), -24)),
        "fixed controlled-scramble sample vector " + std::to_string(count));
    require(sameFloat(evaluate(in).sample, actual.sample),
        "repeated controlled scramble reproduces sample");
    inputs.push_back(in);
    ++count;
  }
  require(vectors.eof() && count == 768, "incomplete pixel/depth vectors");
  // Only selected vectors must change, not every float or arbitrary depth
  // tuple.
  for (uint32_t seed : {1u, 0x12345678u, 0xffffffffu}) {
    const auto base = visrtx::owenSobolCamera(16, visrtx::owenPixelSeed(1, 2));
    const auto alternate =
        visrtx::owenSobolCamera(16, visrtx::owenPixelSeed(1, 2, seed));
    require(!sameFloat(base.x, alternate.x) || !sameFloat(base.y, alternate.y)
            || !sameFloat(base.z, alternate.z)
            || !sameFloat(base.w, alternate.w),
        "selected alternate scramble changes camera vector");
  }
  std::cout << count << " fixed pixel/depth sampler vectors pass\n";
}

inline int run()
try {
  checkPixelKeys();
  // Direct endpoints exercise production conversion, not accidental samples.
  require(visrtx::sobolFloat(0) == 0.f && visrtx::sobolFloat(0xffu) == 0.f
          && visrtx::sobolFloat(0x100u) == 0x1p-24f
          && visrtx::sobolFloat(0xffffffffu) == 0x1.fffffep-1f,
      "24-bit conversion endpoints");
  require(visrtx::reverseBits32(0) == 0
          && visrtx::reverseBits32(1) == 0x80000000u
          && visrtx::reverseBits32(0x01234567) == 0xe6a2c480u
          && visrtx::reverseBits32(0xffffffff) == 0xffffffff,
      "known-answer bit reversal");
  const auto reference = referenceColumns();
  std::vector<Input> inputs;
  unsigned failures = 0;
  for (unsigned d = 0; d < 16; ++d) {
    for (unsigned bit = 0; bit < 32; ++bit) {
      const uint32_t index = uint32_t(1) << bit;
      const auto actual = visrtx::sobolUInt(index, d);
      if (actual != reference[d][bit]) {
        std::cerr << "dimension " << d << ", column " << bit << ": actual "
                  << std::hex << actual << ", expected " << reference[d][bit]
                  << std::dec << '\n';
        ++failures;
      }
      inputs.push_back({index, d, 0});
    }
  }
  require(failures == 0, "direction columns disagree with independent oracle");
  require(visrtx::sobolUInt(16, 7) == 0x88000000u,
      "dimension 7 column 4 regression");

  std::ifstream vectors(SOBOL_VECTOR_FILE);
  require(bool(vectors), "cannot open known-answer vectors");
  std::string line;
  std::getline(vectors, line);
  std::getline(vectors, line);
  Input in{};
  uint32_t raw, reversed, scrambled, grid;
  unsigned vectorCount = 0;
  while (vectors >> std::hex >> in.index >> in.dimension >> in.seed >> raw
      >> reversed >> scrambled >> grid) {
    const auto actual = evaluate(in);
    require(actual.raw == raw, "binary-index integer vector");
    require(actual.reversed == reversed, "bit reversal vector");
    require(pinned::FastOwenScrambler(
                in.seed ^ (0x9e3779b9u * (in.dimension + 1u)))(raw)
            == scrambled,
        "offline fixture disagrees with pinned FastOwen operations");
    require(actual.scrambled == scrambled, "FastOwen integer vector");
    // All integers <= 2^24-1 are exactly representable as float; scaling by
    // a power of two is exact. Float expectations are separate from scramble.
    require(sameFloat(actual.sample, std::ldexp(float(grid), -24)),
        "float grid vector");
    require(actual.sample >= 0 && actual.sample < 1
            && std::ldexp(actual.sample, 24)
                == std::floor(std::ldexp(actual.sample, 24)),
        "sample outside exact half-open grid");
    const auto camera = actual.camera;
    require(camera.x == visrtx::owenSobol(in.index, 0, in.seed)
            && camera.y == visrtx::owenSobol(in.index, 1, in.seed)
            && camera.z == visrtx::owenSobol(in.index, 2, in.seed)
            && camera.w == visrtx::owenSobol(in.index, 3, in.seed)
            && actual.hemi.x == visrtx::owenSobol(in.index, 4, in.seed)
            && actual.hemi.y == visrtx::owenSobol(in.index, 5, in.seed),
        "camera/hemisphere component mapping");
    inputs.push_back(in);
    ++vectorCount;
  }
  require(
      vectors.eof() && vectorCount == 480, "incomplete known-answer fixture");
  for (uint32_t seed : {0u, 0x12345678u, 0xffffffffu}) {
    for (unsigned d = 0; d < 16; ++d) {
      for (unsigned m = 1; m <= 10; ++m) {
        const unsigned n = 1u << m;
        std::vector<bool> occupied(n, false);
        for (unsigned i = 0; i < n; ++i) {
          const auto sample = visrtx::owenSobol(i, d, seed);
          const unsigned bin = unsigned(sample * n);
          require(bin < n && !occupied[bin], "dyadic 1D stratum repeated");
          occupied[bin] = true;
        }
      }
    }
  }
  checkSeedVectors(inputs);
  for (uint32_t endpoint : {0u, 0xffu, 0x100u, 0xffffffffu})
    inputs.push_back({endpoint, 0, 0});
#ifdef SOBOL_CUDA
  checkDevice(inputs);
#endif
  std::cout << "512 columns, 480 vectors, endpoints and all 1D strata pass\n";
  return 0;
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}

} // namespace sobol_test
