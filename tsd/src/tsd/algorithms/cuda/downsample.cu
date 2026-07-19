// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/algorithms/cuda/downsample.hpp"

#include "../math/color.h"
#include "detail/SinglePassDownsampler.h"

#include <cstdio>
#include <vector>

namespace tsd::algorithms::cuda {

static constexpr float MIN_LUMINANCE = 1e-4f;

namespace {

struct LogLuminanceLoader
{
  const float *hdr;
  uint32_t width;

  __device__ float operator()(uint32_t x, uint32_t y) const
  {
    const size_t idx = (size_t(y) * width + x) * 4;
    const float lum =
        max(math::luminance(hdr[idx + 0], hdr[idx + 1], hdr[idx + 2]),
            MIN_LUMINANCE);
    return log2f(lum);
  }
};

struct Sum4
{
  __device__ float operator()(float a, float b, float c, float d) const
  {
    return a + b + c + d;
  }
};

} // namespace

float meanLogLuminance(cudaStream_t stream,
    const float *hdrColor,
    uint32_t width,
    uint32_t height)
{
  if (width == 0 || height == 0)
    return 0.f;

  spd::MipChainView<float> chain;
  spd::spdBuildDims(uint2{width, height}, chain);
  if (chain.count == 0) { // 1x1 source: nothing to reduce
    float4 texel;
    cudaMemcpyAsync(
        &texel, hdrColor, sizeof(texel), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    return log2f(
        max(math::luminance(texel.x, texel.y, texel.z), MIN_LUMINANCE));
  }

  // One allocation backs every level plus the tile counter.
  size_t texels = 0;
  for (int i = 0; i < chain.count; i++)
    texels += size_t(chain.dims[i].x) * chain.dims[i].y;
  float *storage = nullptr;
  const size_t bytes = texels * sizeof(float) + sizeof(uint32_t);
  if (cudaMallocAsync(&storage, bytes, stream) != cudaSuccess || !storage) {
    fprintf(stderr,
        "[tsd] meanLogLuminance: %zu-byte scratch allocation failed\n",
        bytes);
    return 0.f;
  }
  auto *counter = reinterpret_cast<uint32_t *>(storage + texels);
  cudaMemsetAsync(counter, 0, sizeof(uint32_t), stream);

  float *cursor = storage;
  for (int i = 0; i < chain.count; i++) {
    chain.level[i] = cursor;
    cursor += size_t(chain.dims[i].x) * chain.dims[i].y;
  }

  // Identity (zero) padding + a sum reduction counts every texel exactly
  // once regardless of dimensions; the division below yields the exact mean.
  spd::singlePassDownsample(stream,
      LogLuminanceLoader{hdrColor, width},
      uint2{width, height},
      chain,
      Sum4{},
      counter,
      spd::PadMode::Identity,
      0.f);

  // 16 chain levels fold at most 65536 per axis; a larger source leaves a
  // >1x1 top level whose partial sums still cover every texel exactly once —
  // finish the reduction on the host.
  const uint2 topDims = chain.dims[chain.count - 1];
  std::vector<float> top(size_t(topDims.x) * topDims.y);
  cudaMemcpyAsync(top.data(),
      chain.level[chain.count - 1],
      top.size() * sizeof(float),
      cudaMemcpyDeviceToHost,
      stream);
  cudaStreamSynchronize(stream);
  cudaFreeAsync(storage, stream);
  float sum = 0.f;
  for (float v : top)
    sum += v;
  return sum / (float(width) * float(height));
}

} // namespace tsd::algorithms::cuda
