/*
 * Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "volumeOpticsTestCases.h"

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cstdio>

VISRTX_GLOBAL void checkVolumeOptics(int *results)
{
  const int i = int(threadIdx.x);
  if (i < visrtx::test::VOLUME_OPTICS_CASES)
    results[i] = visrtx::test::volumeOpticsTestCase(i);
}

int main()
{
  int deviceCount = 0;
  if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
    std::puts("SKIP: CUDA device unavailable");
    return 77;
  }
  thrust::device_vector<int> results(visrtx::test::VOLUME_OPTICS_CASES, -1);
  checkVolumeOptics<<<1, 32>>>(thrust::raw_pointer_cast(results.data()));
  const auto launchError = cudaGetLastError();
  const auto completionError = cudaDeviceSynchronize();
  if (launchError != cudaSuccess || completionError != cudaSuccess) {
    std::printf("CUDA failure: %s / %s\n",
        cudaGetErrorString(launchError),
        cudaGetErrorString(completionError));
    return 1;
  }
  const thrust::host_vector<int> host = results;
  int failures = 0;
  for (int i = 0; i < visrtx::test::VOLUME_OPTICS_CASES; ++i) {
    if (host[i]) {
      std::printf(
          "FAIL GPU case %d at volumeOpticsTestCases.h:%d\n", i, host[i]);
      ++failures;
    }
  }
  std::printf("GPU volume optics: %d cases, %d failures\n",
      visrtx::test::VOLUME_OPTICS_CASES,
      failures);
  return failures ? 1 : 0;
}
