// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// Distinct test-link symbol for nvcc's callable registration stub; body is
// still extracted verbatim from the production shader.
#define __direct_callable__nextRay nativeDeviceNextRay
#include <stdexcept>
#include <vector>
#include "nativeReflection/fixtures.h"
#include "shader_expressions.h"
namespace native_test {
VISRTX_GLOBAL void sampleNative(int id, NextRay *out)
{
  const int index = int(blockIdx.x * blockDim.x + threadIdx.x);
  if (index >= SAMPLES)
    return;
  const auto f = fixture(id);
  Ray ray{};
  ray.dir = -f.view;
  auto rng = random(id, index);
  out[index] = __direct_callable__nextRay(&f.state, &ray, &rng, nullptr);
}
static void checked(cudaError_t e)
{
  if (e != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(e));
}
std::vector<NextRay> deviceSamples(int id)
{
  std::vector<NextRay> out(SAMPLES);
  NextRay *device = nullptr;
  checked(cudaMalloc(&device, out.size() * sizeof(NextRay)));
  sampleNative<<<(SAMPLES + 127) / 128, 128>>>(id, device);
  checked(cudaGetLastError());
  checked(cudaDeviceSynchronize());
  checked(cudaMemcpy(out.data(),
      device,
      out.size() * sizeof(NextRay),
      cudaMemcpyDeviceToHost));
  checked(cudaFree(device));
  return out;
}
} // namespace native_test
