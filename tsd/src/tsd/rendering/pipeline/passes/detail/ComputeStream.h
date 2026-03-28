// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if defined(ENABLE_CUDA)
#include <cuda_runtime_api.h>
#elif defined(ENABLE_METAL)
namespace MTL {
class CommandQueue;
}
#endif

namespace tsd::rendering::detail {

#if defined(ENABLE_CUDA)
using ComputeStream = cudaStream_t;
#elif defined(ENABLE_METAL)
using ComputeStream = MTL::CommandQueue *;
#else
using ComputeStream = void *;
#endif

} // namespace tsd::rendering::detail
