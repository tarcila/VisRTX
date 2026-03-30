// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace tsd::metal {

// Returns true if the ANARI Metal array extension is loaded.
bool isAvailable();

// Wraps anariNewArray{1D,2D,3D}MetalBuffer (resolved via dlsym).
// Returns ANARIArray handle as void*, or nullptr if extension unavailable.
void *newArray1D(void *device, void *metalBuffer, int type, uint64_t n);
void *newArray2D(
    void *device, void *metalBuffer, int type, uint64_t n1, uint64_t n2);
void *newArray3D(void *device,
    void *metalBuffer,
    int type,
    uint64_t n1,
    uint64_t n2,
    uint64_t n3);

// Notify the ANARI device that a Metal-backed array's buffer contents were
// modified externally (e.g. by a compute kernel).
void notifyArrayChanged(void *device, void *anariArray);

} // namespace tsd::metal
