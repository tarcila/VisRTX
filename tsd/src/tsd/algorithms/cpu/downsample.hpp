// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace tsd::algorithms::cpu {

// Mean of log2(luminance) over an interleaved RGBA float host buffer. Same
// API as tsd::algorithms::cuda::meanLogLuminance; the CUDA path reduces every
// texel exactly through an SPD-style downsampler, while this host path strides
// to a fixed sample budget (a full-image scan is too slow per frame on the
// CPU). The strided mean tracks the exact mean to within ~0.02 stops.
float meanLogLuminance(
    const float *hdrColor, uint32_t width, uint32_t height);

} // namespace tsd::algorithms::cpu
