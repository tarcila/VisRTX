// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <Metal/Metal.hpp>
#include <cstddef>

namespace tsd::algorithms::metal {

void convertFloatToUint8(MTL::CommandBuffer *cmdBuf,
    MTL::Texture *input,
    MTL::Buffer *output,
    size_t totalSize);
void convertFloatToUint8(
    MTL::Texture *input, MTL::Buffer *output, size_t totalSize);

void convertFloatToBGRA8(
    MTL::CommandBuffer *cmdBuf, MTL::Texture *input, MTL::Texture *output);
void convertFloatToBGRA8(MTL::Texture *input, MTL::Texture *output);

} // namespace tsd::algorithms::metal
