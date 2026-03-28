// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

namespace MTL {
class Texture;
} // namespace MTL

namespace tsd::algorithms::metal {

// Opaque display surface for zero-copy Metal → SDL path.
// Backed by IOSurface + Metal texture + CVPixelBuffer.
struct DisplaySurface;

DisplaySurface *createDisplaySurface(uint32_t width, uint32_t height);
void destroyDisplaySurface(DisplaySurface *surface);

// Returns the CVPixelBufferRef for SDL texture creation (cast to void*).
void *displaySurfacePixelBuffer(DisplaySurface *surface);

// Route all tsd_algorithms GPU work through the given queue.
// Must be called with the ANARI device's command queue to ensure
// proper serialization with the device's render commands.
void setSharedQueue(void *commandQueue);

// GPU blit: RGBA32Float texture → BGRA8 display surface. Fully async.
void blitToDisplaySurface(MTL::Texture *input, DisplaySurface *surface);

// GPU depth-composite: where overlay is closer, replace main pixel.
// Modifies mainColor and mainDepth in place.
void compositeByDepth(MTL::Texture *overlayColor,
    MTL::Texture *overlayDepth,
    MTL::Texture *mainColor,
    MTL::Texture *mainDepth);

} // namespace tsd::algorithms::metal
