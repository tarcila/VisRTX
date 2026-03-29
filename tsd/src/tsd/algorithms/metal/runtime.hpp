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

// --- Shared-memory buffer management ---

// Allocate a MTL::Buffer with StorageModeShared; returns opaque handle.
void *newSharedBuffer(size_t bytes);
void releaseBuffer(void *buffer);
// Returns the CPU-accessible pointer (MTL::Buffer::contents()).
void *bufferContents(void *buffer);

// Allocate a MTL::Buffer with StorageModePrivate; returns opaque handle.
// GPU-only: bufferContents() returns nullptr for these buffers.
void *newPrivateBuffer(size_t bytes);

// GPU blit: copy |bytes| from |src| to |dst| via Metal blit encoder.
// Synchronous: commits and waits for completion before returning.
void blitToBuffer(void *src, void *dst, size_t bytes);

// --- Generic compute dispatch ---

// Compile MSL source into a MTL::Library; returns opaque handle.
// Caller must call releaseLibrary() when done.
void *compileShaderSource(const char *source);
void releaseLibrary(void *library);

// Dispatch a 1-D compute kernel.
// |buffers| is an array of opaque MTL::Buffer* handles bound at indices 0..N-1.
// |constants| is copied into a constant-address-space argument at index N.
void dispatchKernel(void *library,
    const char *kernelName,
    void *const *buffers,
    uint32_t numBuffers,
    const void *constants,
    uint32_t constantsSize,
    uint32_t threadCount);

} // namespace tsd::algorithms::metal
