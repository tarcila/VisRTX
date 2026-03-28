// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <Metal/Metal.hpp>
#include <cstdint>

namespace tsd::algorithms::metal {

float sumLogLuminance(MTL::CommandBuffer *cmdBuf,
    MTL::Texture *hdrColor,
    uint32_t numSamples,
    uint32_t stride);
float sumLogLuminance(
    MTL::Texture *hdrColor, uint32_t numSamples, uint32_t stride);

} // namespace tsd::algorithms::metal
