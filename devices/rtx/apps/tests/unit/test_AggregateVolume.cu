// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#define AGGREGATE_DEVICE_ONLY
#include "volumeReference/testAggregate.h"

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace aggregate_test {
VISRTX_GLOBAL void aggregateCases(Result *out)
{
  const int i = int(threadIdx.x);
  if (i < CASES) {
    out[i] = execute(i);
    if (i == 0) {
      ref::Coefficients c;
      c.absorption[0] = std::numeric_limits<double>::min();
      const double source[3]{1, 0, 0};
      av::SegmentIntegral integral;
      av::integrateSegment(c, source, 0x1p-100, integral);
      if (integral.radiance[0] != 0x1p-100)
        out[i].failure = 5;
    }
  }
}

std::vector<Result> deviceResults()
{
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0)
    throw std::runtime_error("CUDA required");
  thrust::device_vector<Result> output(CASES);
  aggregateCases<<<1, 32>>>(thrust::raw_pointer_cast(output.data()));
  if (cudaGetLastError() != cudaSuccess
      || cudaDeviceSynchronize() != cudaSuccess)
    throw std::runtime_error("CUDA aggregate execution");
  const thrust::host_vector<Result> host = output;
  return std::vector<Result>(host.begin(), host.end());
}
} // namespace aggregate_test
