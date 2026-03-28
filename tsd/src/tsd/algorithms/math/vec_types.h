// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifdef __METAL_VERSION__
using float3 = metal::float3;
using float4 = metal::float4;
#else
#include "tsd/core/TSDMath.hpp"
namespace tsd::algorithms::math {
using float3 = tsd::math::float3;
using float4 = tsd::math::float4;
} // namespace tsd::algorithms::math
#endif
