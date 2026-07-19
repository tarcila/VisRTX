// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime_api.h>
#include <cstdint>

namespace tsd::algorithms::cuda {

// Exact full-image mean of log2(luminance) over an interleaved RGBA32F
// device buffer, reduced through an SPD-style single-pass downsampler
// (detail/SinglePassDownsampler.h) with identity (zero) padding, so the sum
// counts every texel exactly once regardless of dimensions. Synchronizes
// `stream` to return the value.
float meanLogLuminance(cudaStream_t stream,
    const float *hdrColor,
    uint32_t width,
    uint32_t height);

} // namespace tsd::algorithms::cuda
