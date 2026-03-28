// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <Metal/Metal.hpp>
#include <cstdint>

namespace tsd::algorithms::metal {

void outputTransform(MTL::CommandBuffer *cmdBuf,
    MTL::Texture *hdrColor,
    MTL::Texture *colorIn,
    MTL::Texture *colorOut,
    uint32_t totalPixels,
    float invGamma,
    uint32_t colorFormat);
void outputTransform(MTL::Texture *hdrColor,
    MTL::Texture *colorIn,
    MTL::Texture *colorOut,
    uint32_t totalPixels,
    float invGamma,
    uint32_t colorFormat);

} // namespace tsd::algorithms::metal
