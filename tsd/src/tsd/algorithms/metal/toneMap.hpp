// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <Metal/Metal.hpp>
#include <cstdint>
#include "tsd/algorithms/cpu/toneMap.hpp"

namespace tsd::algorithms::metal {

using tsd::algorithms::ToneMapOperator;

void toneMap(MTL::CommandBuffer *cmdBuf,
    MTL::Texture *hdrColor,
    uint32_t numPixels,
    float exposureScale,
    ToneMapOperator op);
void toneMap(MTL::Texture *hdrColor,
    uint32_t numPixels,
    float exposureScale,
    ToneMapOperator op);

} // namespace tsd::algorithms::metal
