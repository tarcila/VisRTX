// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <Metal/Metal.hpp>
#include <cstdint>

namespace tsd::algorithms::metal {

void visualizeObjectId(MTL::CommandBuffer *cmdBuf,
    MTL::Texture *objectId,
    MTL::Texture *color,
    uint32_t w,
    uint32_t h);
void visualizeObjectId(
    MTL::Texture *objectId, MTL::Texture *color, uint32_t w, uint32_t h);

void visualizePrimitiveId(MTL::CommandBuffer *cmdBuf,
    MTL::Texture *primitiveId,
    MTL::Texture *color,
    uint32_t w,
    uint32_t h);
void visualizePrimitiveId(
    MTL::Texture *primitiveId, MTL::Texture *color, uint32_t w, uint32_t h);

void visualizeInstanceId(MTL::CommandBuffer *cmdBuf,
    MTL::Texture *instanceId,
    MTL::Texture *color,
    uint32_t w,
    uint32_t h);
void visualizeInstanceId(
    MTL::Texture *instanceId, MTL::Texture *color, uint32_t w, uint32_t h);

void visualizeDepth(MTL::CommandBuffer *cmdBuf,
    MTL::Texture *depth,
    MTL::Texture *color,
    float minDepth,
    float maxDepth,
    uint32_t w,
    uint32_t h);
void visualizeDepth(MTL::Texture *depth,
    MTL::Texture *color,
    float minDepth,
    float maxDepth,
    uint32_t w,
    uint32_t h);

void visualizeAlbedo(MTL::CommandBuffer *cmdBuf,
    MTL::Texture *albedo,
    MTL::Texture *color,
    uint32_t w,
    uint32_t h);
void visualizeAlbedo(
    MTL::Texture *albedo, MTL::Texture *color, uint32_t w, uint32_t h);

void visualizeNormal(MTL::CommandBuffer *cmdBuf,
    MTL::Texture *normal,
    MTL::Texture *color,
    uint32_t w,
    uint32_t h);
void visualizeNormal(
    MTL::Texture *normal, MTL::Texture *color, uint32_t w, uint32_t h);

void visualizeEdges(MTL::CommandBuffer *cmdBuf,
    MTL::Texture *objectId,
    MTL::Texture *color,
    bool invert,
    uint32_t w,
    uint32_t h);
void visualizeEdges(MTL::Texture *objectId,
    MTL::Texture *color,
    bool invert,
    uint32_t w,
    uint32_t h);

} // namespace tsd::algorithms::metal
