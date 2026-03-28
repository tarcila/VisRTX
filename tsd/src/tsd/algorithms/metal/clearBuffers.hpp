// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <Metal/Metal.hpp>
#include <cstdint>

namespace tsd::algorithms::metal {

void fill(MTL::CommandBuffer *cmdBuf,
    MTL::Buffer *buf,
    uint32_t count,
    uint32_t value);
void fill(MTL::Buffer *buf, uint32_t count, uint32_t value);

void fill(
    MTL::CommandBuffer *cmdBuf, MTL::Buffer *buf, uint32_t count, float value);
void fill(MTL::Buffer *buf, uint32_t count, float value);

} // namespace tsd::algorithms::metal
