// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <Metal/Metal.hpp>
#include <cstdint>

namespace tsd::algorithms::metal {

void outline(MTL::CommandBuffer *cmdBuf,
    MTL::Texture *objectId,
    MTL::Texture *color,
    uint32_t outlineId,
    uint32_t w,
    uint32_t h);
void outline(MTL::Texture *objectId,
    MTL::Texture *color,
    uint32_t outlineId,
    uint32_t w,
    uint32_t h);

} // namespace tsd::algorithms::metal
