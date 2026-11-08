// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "gpu_decl.h"

namespace visrtx {

// Frame-owned, reset before each Launch and copied back in stream order.
// The first failing evaluation owns the identity fields. Host readers only
// inspect the record after GPU completion, never concurrently with writers.
enum class RenderErrorCategory : uint32_t
{
  NONE = 0,
  INVALID_COEFFICIENT,
  MAJORANT_VIOLATION,
  INVALID_FIELD_VALUE,
  INVALID_SCENE_DATA,
  UNSUPPORTED_MEDIUM,
  NUMERICAL_FAILURE
};

struct RenderError
{
  uint32_t category{0};
  uint32_t volume{~0u};
  uint32_t instance{~0u};
};

#ifdef __CUDACC__
VISRTX_DEVICE void recordRenderError(RenderError *error,
    RenderErrorCategory category,
    uint32_t volume,
    uint32_t instance)
{
  if (error && atomicCAS(&error->category, 0u, uint32_t(category)) == 0u) {
    error->volume = volume;
    error->instance = instance;
  }
}
#endif

} // namespace visrtx
