// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/metal/interop.hpp"

#include <dlfcn.h>

namespace tsd::metal {

using NewArray1DFn = void *(*)(void *, void *, int, uint64_t);
using NewArray2DFn = void *(*)(void *, void *, int, uint64_t, uint64_t);
using NewArray3DFn =
    void *(*)(void *, void *, int, uint64_t, uint64_t, uint64_t);
using NotifyFn = void (*)(void *, void *);

static NewArray1DFn resolveNewArray1D()
{
  return (NewArray1DFn)dlsym(RTLD_DEFAULT, "anariNewArray1DMetalBuffer");
}

bool isAvailable()
{
  static auto *fn = resolveNewArray1D();
  return fn != nullptr;
}

void *newArray1D(void *device, void *metalBuffer, int type, uint64_t n)
{
  static auto *fn = resolveNewArray1D();
  return fn ? fn(device, metalBuffer, type, n) : nullptr;
}

void *newArray2D(
    void *device, void *metalBuffer, int type, uint64_t n1, uint64_t n2)
{
  static auto *fn =
      (NewArray2DFn)dlsym(RTLD_DEFAULT, "anariNewArray2DMetalBuffer");
  return fn ? fn(device, metalBuffer, type, n1, n2) : nullptr;
}

void *newArray3D(void *device,
    void *metalBuffer,
    int type,
    uint64_t n1,
    uint64_t n2,
    uint64_t n3)
{
  static auto *fn =
      (NewArray3DFn)dlsym(RTLD_DEFAULT, "anariNewArray3DMetalBuffer");
  return fn ? fn(device, metalBuffer, type, n1, n2, n3) : nullptr;
}

void notifyArrayChanged(void *device, void *anariArray)
{
  static auto *fn =
      (NotifyFn)dlsym(RTLD_DEFAULT, "anariNotifyArrayChangedMetal");
  if (fn)
    fn(device, anariArray);
}

} // namespace tsd::metal
