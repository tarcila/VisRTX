/*
 * Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <cuda_runtime.h>
#include "gpu/gpu_math.h"

namespace visrtx {

// Expand packed float3 host/device input to a float4 CUDA array. XYZ remain raw
// linear values; the padding component is not an opacity input. Caller owns
// array.
cudaError_t uploadRGBFieldArray(
    cudaArray_t &array, const void *data, const uvec3 &dims);

// CUDA textures have no float64 element channel. Canonicalize supported double
// storage to the field sampler's float32 value type instead of creating an
// invalid 64-bit texture descriptor.
cudaError_t uploadDoubleFieldArray(
    cudaArray_t &array, const void *data, const uvec3 &dims);

} // namespace visrtx
