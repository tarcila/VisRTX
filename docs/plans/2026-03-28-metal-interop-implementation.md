# Metal Interop & Algorithms Library Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close the GPU-resident post-processing pipeline on macOS by adding Metal ANARI extensions to anari-mtl and a Metal compute backend to `tsd_algorithms`.

**Architecture:** Two ANARI extensions (`ANARI_MTL_ARRAY_METAL`, `ANARI_MTL_FRAME_BUFFERS_METAL`) let anari-mtl expose GPU textures/buffers to TSD. The algorithms library gains a `metal/` backend with MSL compute shaders that operate directly on those textures. Pipeline integration in `tsd_rendering` detects the extensions and dispatches to Metal algorithms when available.

**Tech Stack:** Metal-cpp, MSL (Metal Shading Language), CMake custom commands (xcrun metal/metallib), C++17

**Design doc:** `docs/plans/2026-03-27-metal-interop-and-algorithms-design.md`

---

## Task 1: Cross-platform Math Abstraction

This is the hard gate — modifies headers consumed by all backends.

**Files:**
- Create: `tsd/src/tsd/algorithms/math/vec_types.h`
- Create: `tsd/src/tsd/algorithms/math/math_compat.h`
- Modify: `tsd/src/tsd/algorithms/math/device_macros.h`
- Modify: `tsd/src/tsd/algorithms/math/tonemap_curves.h`
- Modify: `tsd/src/tsd/algorithms/math/color.h`

**Step 1: Create `vec_types.h`**

```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifdef __METAL_VERSION__
using float3 = metal::float3;
using float4 = metal::float4;
#else
#include "tsd/core/TSDMath.hpp"
namespace tsd::algorithms::math {
using float3 = tsd::math::float3;
using float4 = tsd::math::float4;
} // namespace tsd::algorithms::math
#endif
```

**Step 2: Create `math_compat.h`**

```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifdef __METAL_VERSION__
using metal::max;
using metal::min;
using metal::clamp;
using metal::pow;
using metal::log2;
using metal::exp2;
#else
#include <algorithm>
#include <cmath>
namespace tsd::algorithms::math {
using std::max;
using std::min;
using std::clamp;
using std::pow;
using std::log2;
using std::exp2;
} // namespace tsd::algorithms::math
#endif
```

**Step 3: Extend `device_macros.h`**

Add `__METAL_VERSION__` guard. Current file at `tsd/src/tsd/algorithms/math/device_macros.h`:

```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if defined(__CUDACC__)
#define TSD_DEVICE_FCN __device__
#define TSD_DEVICE_FCN_INLINE __forceinline__ __device__
#define TSD_HOST_DEVICE_FCN __host__ __device__
#elif defined(__METAL_VERSION__)
#define TSD_DEVICE_FCN
#define TSD_DEVICE_FCN_INLINE inline
#define TSD_HOST_DEVICE_FCN
#else
#define TSD_DEVICE_FCN
#define TSD_DEVICE_FCN_INLINE inline
#define TSD_HOST_DEVICE_FCN
#endif
```

**Step 4: Update `tonemap_curves.h`**

Replace `#include "tsd/core/TSDMath.hpp"` and `#include <algorithm>` / `#include <cmath>` with the new headers. Replace all `tsd::math::float3` with `math::float3` (the namespace alias from `vec_types.h`). Replace `std::max`, `std::min`, `std::clamp`, `std::log2` with unqualified calls (resolved by `math_compat.h`).

Changes at `tsd/src/tsd/algorithms/math/tonemap_curves.h`:
- Line 7: `#include "tsd/core/TSDMath.hpp"` → `#include "vec_types.h"`
- Lines 8-10: `#include <algorithm>` / `#include <cmath>` → `#include "math_compat.h"`
- All function signatures and bodies: `tsd::math::float3` → `float3` (within the `tsd::algorithms::math` namespace, `float3` resolves to the alias from `vec_types.h`)
- All `std::max`, `std::min`, `std::clamp`, `std::log2` → `max`, `min`, `clamp`, `log2`

**Step 5: Update `color.h`**

Same pattern as tonemap_curves.h. Changes at `tsd/src/tsd/algorithms/math/color.h`:
- Line 7: `#include "tsd/core/TSDMath.hpp"` → `#include "vec_types.h"`
- Lines 8-10: `#include <algorithm>` / `#include <cmath>` → `#include "math_compat.h"`
- All `tsd::math::float3` → `float3`
- All `std::pow`, `std::clamp` → `pow`, `clamp`

**Step 6: Build and test CPU + CUDA backends**

Run:
```bash
cd build && cmake --build . --parallel
ctest -C Release --output-on-failure
```

Expected: All existing tests pass. No regressions.

**Step 7: Commit**

```bash
jj describe -m "refactor: cross-platform math abstraction for Metal shader sharing

Introduce vec_types.h and math_compat.h so tonemap_curves.h and color.h
compile under C++, CUDA, and MSL. Extend device_macros.h with
__METAL_VERSION__ guard. Mechanical change — no behavioral differences."
jj new
```

---

## Task 2: ANARI Extensions in anari-mtl — Metal Array Types

**Files:**
- Create: `/Users/tarcila/Code/ANARI/mtl/interop/src/array/MetalArray1D.h`
- Create: `/Users/tarcila/Code/ANARI/mtl/interop/src/array/MetalArray2D.h`
- Create: `/Users/tarcila/Code/ANARI/mtl/interop/src/array/MetalArray3D.h`
- Modify: `/Users/tarcila/Code/ANARI/mtl/interop/src/MtlDevice.h` (lines 9-77)
- Modify: `/Users/tarcila/Code/ANARI/mtl/interop/src/MtlDevice.cpp` (lines 30-91, 251-269)

**Step 1: Create `MetalArray1D.h`**

```cpp
#pragma once

#include <helium/array/Array1D.h>
#include <Metal/Metal.hpp>

namespace mtl {

struct MetalArray1D : public helium::Array1D
{
    MetalArray1D(helium::BaseGlobalDeviceState *s,
        MTL::Buffer *buffer,
        ANARIDataType type,
        uint64_t numItems);

    MTL::Buffer *metalBuffer() const;
    const void *data() const override;

  private:
    MTL::Buffer *m_metalBuffer{nullptr};
};

} // namespace mtl
```

Note: `data()` override returns `buffer->contents()` so the host fallback path works transparently. The `MetalArray1D` constructor creates a `helium::Array1DMemoryDescriptor` with `appMemory = buffer->contents()` and passes it to the base class.

**Step 2: Create `MetalArray2D.h` and `MetalArray3D.h`**

Same pattern, inheriting from `helium::Array2D` and `helium::Array3D` respectively. Each holds `MTL::Buffer *m_metalBuffer` and overrides `data()`.

**Step 3: Add implementations**

Create `MetalArray1D.cpp`, `MetalArray2D.cpp`, `MetalArray3D.cpp` in `/Users/tarcila/Code/ANARI/mtl/interop/src/array/`. Each constructor:

```cpp
MetalArray1D::MetalArray1D(helium::BaseGlobalDeviceState *s,
    MTL::Buffer *buffer,
    ANARIDataType type,
    uint64_t numItems)
    : helium::Array1D(s, Array1DMemoryDescriptor{
          buffer->contents(), nullptr, nullptr, type, numItems})
    , m_metalBuffer(buffer)
{}

MTL::Buffer *MetalArray1D::metalBuffer() const { return m_metalBuffer; }
const void *MetalArray1D::data() const { return m_metalBuffer->contents(); }
```

**Step 4: Add extension factory methods to `MtlDevice`**

In `MtlDevice.h`, add after line 44 (after `newWorld()`):

```cpp
    // ANARI_MTL_ARRAY_METAL extension
    ANARIArray1D newArray1DMetalBuffer(
        MTL::Buffer *buffer, ANARIDataType type, uint64_t numItems);
    ANARIArray2D newArray2DMetalBuffer(
        MTL::Buffer *buffer, ANARIDataType type,
        uint64_t numItems1, uint64_t numItems2);
    ANARIArray3D newArray3DMetalBuffer(
        MTL::Buffer *buffer, ANARIDataType type,
        uint64_t numItems1, uint64_t numItems2, uint64_t numItems3);
```

In `MtlDevice.cpp`, implement after line 91:

```cpp
ANARIArray1D MtlDevice::newArray1DMetalBuffer(
    MTL::Buffer *buffer, ANARIDataType type, uint64_t numItems)
{
    initDevice();
    return (ANARIArray1D) new MetalArray1D(deviceState(), buffer, type, numItems);
}
// ... similarly for 2D and 3D
```

**Step 5: Add C extension entry points**

Create `/Users/tarcila/Code/ANARI/mtl/interop/src/extensions/ANARI_MTL_ARRAY_METAL.h`:

```cpp
#pragma once

#include <anari/anari.h>
#include <Metal/Metal.hpp>

ANARI_BEGIN_DECLARATIONS

ANARIArray1D anariNewArray1DMetalBuffer(
    ANARIDevice, MTL::Buffer *, ANARIDataType, uint64_t numItems);
ANARIArray2D anariNewArray2DMetalBuffer(
    ANARIDevice, MTL::Buffer *, ANARIDataType,
    uint64_t numItems1, uint64_t numItems2);
ANARIArray3D anariNewArray3DMetalBuffer(
    ANARIDevice, MTL::Buffer *, ANARIDataType,
    uint64_t numItems1, uint64_t numItems2, uint64_t numItems3);

ANARI_END_DECLARATIONS
```

Create `/Users/tarcila/Code/ANARI/mtl/interop/src/extensions/ANARI_MTL_ARRAY_METAL.cpp` implementing these by casting `ANARIDevice` to `MtlDevice*` and forwarding.

**Step 6: Register sources in CMakeLists.txt**

Add the new `.cpp` files to the source list in `/Users/tarcila/Code/ANARI/mtl/interop/src/CMakeLists.txt`.

**Step 7: Build and verify**

```bash
cd build && cmake --build . --parallel
```

Expected: Compiles. No new tests yet (extension tested in Task 2b).

**Step 8: Commit**

```bash
jj describe -m "feat(anari-mtl): add ANARI_MTL_ARRAY_METAL extension

New MetalArray1D/2D/3D types wrapping MTL::Buffer* for GPU-resident
array input. C extension functions: anariNewArray{1D,2D,3D}MetalBuffer."
jj new
```

---

## Task 3: ANARI Extensions in anari-mtl — Frame Buffer Metal Channels

**Files:**
- Create: `/Users/tarcila/Code/ANARI/mtl/interop/src/extensions/ANARI_MTL_FRAME_BUFFERS_METAL.h`
- Modify: `/Users/tarcila/Code/ANARI/mtl/interop/src/frame/Frame.cpp` (map function, ~lines 631-733)
- Modify: `/Users/tarcila/Code/ANARI/mtl/interop/src/MtlDevice.cpp` (lines 251-269, deviceGetProperty)
- Modify: `/Users/tarcila/Code/ANARI/mtl/interop/src/MtlDefinitions.json` (lines 6-47)

**Step 1: Add Metal channel branches to `Frame::map()`**

In `Frame.cpp`'s `map()` function, add before the existing `"channel.color"` branch (before ~line 631):

```cpp
// ANARI_MTL_FRAME_BUFFERS_METAL: return textures directly
if (channel == "channel.colorMTL") {
    *width = m_size.x;
    *height = m_size.y;
    *pixelType = ANARI_FLOAT32_VEC4;
    return (void *)m_colorTexture;
}
if (channel == "channel.depthMTL") {
    *width = m_size.x;
    *height = m_size.y;
    *pixelType = ANARI_FLOAT32;
    return (void *)m_depthTexture;
}
if (channel == "channel.normalMTL" && m_normalTexture) {
    *width = m_size.x;
    *height = m_size.y;
    *pixelType = ANARI_FLOAT32_VEC4;
    return (void *)m_normalTexture;
}
if (channel == "channel.albedoMTL" && m_albedoTexture) {
    *width = m_size.x;
    *height = m_size.y;
    *pixelType = ANARI_FLOAT32_VEC4;
    return (void *)m_albedoTexture;
}
if (channel == "channel.primitiveIdMTL" && m_primitiveIdTexture) {
    *width = m_size.x;
    *height = m_size.y;
    *pixelType = ANARI_UINT32;
    return (void *)m_primitiveIdTexture;
}
if (channel == "channel.objectIdMTL" && m_objectIdTexture) {
    *width = m_size.x;
    *height = m_size.y;
    *pixelType = ANARI_UINT32;
    return (void *)m_objectIdTexture;
}
if (channel == "channel.instanceIdMTL" && m_instanceIdTexture) {
    *width = m_size.x;
    *height = m_size.y;
    *pixelType = ANARI_UINT32;
    return (void *)m_instanceIdTexture;
}
```

**Step 2: Add `mtl.commandQueue` device property**

In `MtlDevice.cpp`, `deviceGetProperty()` (line 257), add before the `return 0` at line 268:

```cpp
else if (prop == "mtl.commandQueue" && type == ANARI_VOID_POINTER)
{
    helium::writeToVoidP(mem, (void *)deviceState()->metalContext.queue());
    return 1;
}
```

**Step 3: Register extensions in `MtlDefinitions.json`**

Add `"ANARI_MTL_ARRAY_METAL"` and `"ANARI_MTL_FRAME_BUFFERS_METAL"` to the device dependencies list at `/Users/tarcila/Code/ANARI/mtl/interop/src/MtlDefinitions.json` (within lines 6-47).

**Step 4: Create extension header**

Create `/Users/tarcila/Code/ANARI/mtl/interop/src/extensions/ANARI_MTL_FRAME_BUFFERS_METAL.h` documenting the channel names and semantics.

**Step 5: Build and verify**

```bash
cd build && cmake --build . --parallel
```

Expected: Compiles. Manual verification: create a frame, render, map `"channel.colorMTL"`, confirm non-null return.

**Step 6: Commit**

```bash
jj describe -m "feat(anari-mtl): add ANARI_MTL_FRAME_BUFFERS_METAL extension

Frame channels channel.*MTL return MTL::Texture* directly — zero-copy
GPU-resident frame buffer access. Adds mtl.commandQueue device property
for shared queue access."
jj new
```

---

## Task 4: Algorithms Library — CMake + MetalContext + clearBuffers

Minimal proof of concept: shader pipeline, embedded metallib, singleton, simplest kernel.

**Files:**
- Modify: `tsd/src/tsd/algorithms/CMakeLists.txt` (after line 54)
- Modify: `tsd/src/tsd/algorithms/tsd/algorithms/config.h`
- Create: `tsd/src/tsd/algorithms/metal/MetalContext.h`
- Create: `tsd/src/tsd/algorithms/metal/MetalContext.cpp`
- Create: `tsd/src/tsd/algorithms/tsd/algorithms/metal/clearBuffers.hpp`
- Create: `tsd/src/tsd/algorithms/metal/clearBuffers.cpp`
- Create: `tsd/src/tsd/algorithms/metal/shaders/clearBuffers.metal`

**Step 1: Create `MetalContext.h`**

At `tsd/src/tsd/algorithms/metal/MetalContext.h`:

```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <Metal/Metal.hpp>
#include <mutex>
#include <string>
#include <unordered_map>

namespace tsd::algorithms::metal {

struct MetalContext
{
    static MetalContext &instance();

    MTL::Device *device();
    MTL::CommandQueue *defaultQueue();
    void setQueue(MTL::CommandQueue *queue);
    MTL::ComputePipelineState *pipelineState(const char *kernelName);

  private:
    MetalContext();
    ~MetalContext();

    MTL::Device *m_device{nullptr};
    MTL::CommandQueue *m_queue{nullptr};
    MTL::CommandQueue *m_externalQueue{nullptr};
    MTL::Library *m_library{nullptr};
    std::unordered_map<std::string, MTL::ComputePipelineState *> m_pipelines;
    std::mutex m_mutex;
};

} // namespace tsd::algorithms::metal
```

**Step 2: Create `MetalContext.cpp`**

At `tsd/src/tsd/algorithms/metal/MetalContext.cpp`:

```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "MetalContext.h"

// Embedded metallib (generated by CMake build)
extern "C" {
extern unsigned char tsd_algorithms_metallib[];
extern unsigned int tsd_algorithms_metallib_len;
}

namespace tsd::algorithms::metal {

MetalContext &MetalContext::instance()
{
    static MetalContext ctx;
    return ctx;
}

MetalContext::MetalContext()
{
    m_device = MTL::CreateSystemDefaultDevice();
    m_queue = m_device->newCommandQueue();

    auto *data = dispatch_data_create(tsd_algorithms_metallib,
        tsd_algorithms_metallib_len,
        nullptr,
        DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    NS::Error *error = nullptr;
    m_library = m_device->newLibrary((dispatch_data_t)data, &error);
    if (!m_library) {
        // Handle error — log and abort or throw
    }
}

MetalContext::~MetalContext()
{
    for (auto &[_, pso] : m_pipelines)
        pso->release();
    m_library->release();
    m_queue->release();
    m_device->release();
}

MTL::Device *MetalContext::device() { return m_device; }

MTL::CommandQueue *MetalContext::defaultQueue()
{
    return m_externalQueue ? m_externalQueue : m_queue;
}

void MetalContext::setQueue(MTL::CommandQueue *queue)
{
    m_externalQueue = queue;
}

MTL::ComputePipelineState *MetalContext::pipelineState(const char *kernelName)
{
    std::lock_guard lock(m_mutex);
    auto it = m_pipelines.find(kernelName);
    if (it != m_pipelines.end())
        return it->second;

    auto *fn = m_library->newFunction(
        NS::String::string(kernelName, NS::ASCIIStringEncoding));
    NS::Error *error = nullptr;
    auto *pso = m_device->newComputePipelineState(fn, &error);
    fn->release();

    m_pipelines[kernelName] = pso;
    return pso;
}

} // namespace tsd::algorithms::metal
```

**Step 3: Create `clearBuffers.metal`**

At `tsd/src/tsd/algorithms/metal/shaders/clearBuffers.metal`:

```metal
#include <metal_stdlib>
using namespace metal;

kernel void fillUint32(
    device uint32_t *buf [[buffer(0)]],
    constant uint32_t &count [[buffer(1)]],
    constant uint32_t &value [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < count)
        buf[tid] = value;
}

kernel void fillFloat(
    device float *buf [[buffer(0)]],
    constant uint32_t &count [[buffer(1)]],
    constant float &value [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < count)
        buf[tid] = value;
}
```

Note: `clearBuffers` operates on `MTL::Buffer*` (not textures) because it fills raw pixel buffers. For the texture-based pipeline, a separate `clearTextures.metal` kernel may be needed for texture targets — but that depends on how `ImageBuffers` allocates Metal textures vs buffers (Task 8).

**Step 4: Create public header `metal/clearBuffers.hpp`**

At `tsd/src/tsd/algorithms/tsd/algorithms/metal/clearBuffers.hpp`:

```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <Metal/Metal.hpp>
#include <cstdint>

namespace tsd::algorithms::metal {

void fill(MTL::CommandBuffer *cmdBuf,
    MTL::Buffer *buf, uint32_t count, uint32_t value);
void fill(MTL::Buffer *buf, uint32_t count, uint32_t value);

void fill(MTL::CommandBuffer *cmdBuf,
    MTL::Buffer *buf, uint32_t count, float value);
void fill(MTL::Buffer *buf, uint32_t count, float value);

} // namespace tsd::algorithms::metal
```

**Step 5: Create `metal/clearBuffers.cpp`**

At `tsd/src/tsd/algorithms/metal/clearBuffers.cpp`:

```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/algorithms/metal/clearBuffers.hpp"
#include "MetalContext.h"

namespace tsd::algorithms::metal {

static void dispatchFill(MTL::CommandBuffer *cmdBuf,
    const char *kernelName,
    MTL::Buffer *buf,
    uint32_t count,
    const void *value,
    size_t valueSize)
{
    auto &ctx = MetalContext::instance();
    auto *pso = ctx.pipelineState(kernelName);

    auto *encoder = cmdBuf->computeCommandEncoder();
    encoder->setComputePipelineState(pso);
    encoder->setBuffer(buf, 0, 0);
    encoder->setBytes(&count, sizeof(count), 1);
    encoder->setBytes(value, valueSize, 2);

    auto tgSize = pso->maxTotalThreadsPerThreadgroup();
    if (tgSize > count) tgSize = count;
    encoder->dispatchThreads({count, 1, 1}, {(NS::UInteger)tgSize, 1, 1});
    encoder->endEncoding();
}

void fill(MTL::CommandBuffer *cmdBuf,
    MTL::Buffer *buf, uint32_t count, uint32_t value)
{
    dispatchFill(cmdBuf, "fillUint32", buf, count, &value, sizeof(value));
}

void fill(MTL::Buffer *buf, uint32_t count, uint32_t value)
{
    auto &ctx = MetalContext::instance();
    auto *cmdBuf = ctx.defaultQueue()->commandBuffer();
    fill(cmdBuf, buf, count, value);
    cmdBuf->commit();
    cmdBuf->waitUntilCompleted();
}

void fill(MTL::CommandBuffer *cmdBuf,
    MTL::Buffer *buf, uint32_t count, float value)
{
    dispatchFill(cmdBuf, "fillFloat", buf, count, &value, sizeof(value));
}

void fill(MTL::Buffer *buf, uint32_t count, float value)
{
    auto &ctx = MetalContext::instance();
    auto *cmdBuf = ctx.defaultQueue()->commandBuffer();
    fill(cmdBuf, buf, count, value);
    cmdBuf->commit();
    cmdBuf->waitUntilCompleted();
}

} // namespace tsd::algorithms::metal
```

**Step 6: Update `config.h`**

Add `TSD_ALGORITHMS_HAS_METAL` comment at `tsd/src/tsd/algorithms/tsd/algorithms/config.h` (line 8):

```cpp
//   TSD_ALGORITHMS_HAS_METAL — Metal backend compiled in
```

**Step 7: Add CMake Metal section**

Append to `tsd/src/tsd/algorithms/CMakeLists.txt` after line 54:

```cmake
# GPU backend: Metal
if (TSD_USE_METAL)
  if (NOT APPLE)
    message(FATAL_ERROR "TSD_USE_METAL requires macOS")
  endif()

  if (TSD_USE_CUDA)
    message(FATAL_ERROR "TSD_USE_CUDA and TSD_USE_METAL are mutually exclusive")
  endif()

  anari_sdk_fetch_project(
    NAME metal_cpp
    URL "https://developer.apple.com/metal/cpp/files/metal-cpp_macOS15_iOS18.zip"
  )
  project_include_directories(PUBLIC ${metal_cpp_LOCATION})
  project_compile_definitions(PUBLIC TSD_ALGORITHMS_HAS_METAL)

  # Shader compilation (.metal -> .air -> .metallib -> embedded C array)
  set(TSD_METAL_SHADERS
    metal/shaders/clearBuffers.metal
  )

  set(METAL_AIR_FILES)
  foreach(SHADER ${TSD_METAL_SHADERS})
    get_filename_component(SHADER_NAME ${SHADER} NAME_WE)
    set(AIR_FILE ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_NAME}.air)
    add_custom_command(
      OUTPUT ${AIR_FILE}
      COMMAND xcrun -sdk macosx metal
        -c ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER}
        -I ${CMAKE_CURRENT_SOURCE_DIR}/math
        -o ${AIR_FILE}
      DEPENDS ${SHADER}
    )
    list(APPEND METAL_AIR_FILES ${AIR_FILE})
  endforeach()

  set(METALLIB_FILE ${CMAKE_CURRENT_BINARY_DIR}/tsd_algorithms.metallib)
  add_custom_command(
    OUTPUT ${METALLIB_FILE}
    COMMAND xcrun -sdk macosx metallib ${METAL_AIR_FILES} -o ${METALLIB_FILE}
    DEPENDS ${METAL_AIR_FILES}
  )

  set(METALLIB_C ${CMAKE_CURRENT_BINARY_DIR}/tsd_algorithms_metallib.c)
  add_custom_command(
    OUTPUT ${METALLIB_C}
    COMMAND xxd -i tsd_algorithms.metallib ${METALLIB_C}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${METALLIB_FILE}
  )

  project_sources(PRIVATE
    metal/MetalContext.cpp
    metal/clearBuffers.cpp
    ${METALLIB_C}
  )

  find_library(METAL_FRAMEWORK Metal REQUIRED)
  find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
  project_link_libraries(PUBLIC ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})
endif()
```

**Step 8: Build**

```bash
cd build && cmake -DTSD_USE_METAL=ON .. && cmake --build . --parallel
```

Expected: Compiles. Metallib embedded. MetalContext initializes.

**Step 9: Commit**

```bash
jj describe -m "feat(algorithms): Metal backend scaffold — CMake, MetalContext, clearBuffers

Adds shader compilation pipeline (.metal -> .air -> .metallib -> xxd),
MetalContext singleton with pipeline state caching, and clearBuffers as
the first Metal compute kernel."
jj new
```

---

## Task 5: toneMap — First Algorithm Using Shared Math in MSL

Validates that cross-platform math headers compile under MSL and the full kernel dispatch pattern works.

**Files:**
- Create: `tsd/src/tsd/algorithms/tsd/algorithms/metal/toneMap.hpp`
- Create: `tsd/src/tsd/algorithms/metal/toneMap.cpp`
- Create: `tsd/src/tsd/algorithms/metal/shaders/toneMap.metal`
- Modify: `tsd/src/tsd/algorithms/CMakeLists.txt` (add to shader and source lists)

**Step 1: Create `toneMap.metal`**

At `tsd/src/tsd/algorithms/metal/shaders/toneMap.metal`:

```metal
#include <metal_stdlib>
using namespace metal;

#include "vec_types.h"
#include "math_compat.h"
#include "tonemap_curves.h"

using namespace tsd::algorithms::math;

constant constexpr uint32_t OP_NONE = 0;
constant constexpr uint32_t OP_REINHARD = 1;
constant constexpr uint32_t OP_ACES = 2;
constant constexpr uint32_t OP_HABLE = 3;
constant constexpr uint32_t OP_KHRONOS_PBR_NEUTRAL = 4;
constant constexpr uint32_t OP_AGX = 5;

kernel void toneMapKernel(
    texture2d<float, access::read_write> hdrColor [[texture(0)]],
    constant uint32_t &width [[buffer(0)]],
    constant float &exposureScale [[buffer(1)]],
    constant uint32_t &op [[buffer(2)]],
    uint2 tid [[thread_position_in_grid]])
{
    if (tid.x >= width || tid.y >= hdrColor.get_height())
        return;

    float4 pixel = hdrColor.read(tid);
    float3 c = pixel.xyz * exposureScale;

    switch (op) {
    case OP_NONE: break;
    case OP_REINHARD: c = tonemapReinhard(max0(c)); break;
    case OP_ACES: c = tonemapACES(max0(c)); break;
    case OP_HABLE: c = tonemapHable(max0(c)); break;
    case OP_KHRONOS_PBR_NEUTRAL: c = tonemapKhronosPbrNeutral(max0(c)); break;
    case OP_AGX: c = tonemapAgX(max0(c)); break;
    }

    hdrColor.write(float4(c, pixel.w), tid);
}
```

**Step 2: Create public header `metal/toneMap.hpp`**

At `tsd/src/tsd/algorithms/tsd/algorithms/metal/toneMap.hpp`:

```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/algorithms/cpu/toneMap.hpp" // ToneMapOperator enum
#include <Metal/Metal.hpp>

namespace tsd::algorithms::metal {

using tsd::algorithms::cpu::ToneMapOperator;

void toneMap(MTL::CommandBuffer *cmdBuf,
    MTL::Texture *hdrColor,
    uint32_t numPixels,
    float exposureScale,
    ToneMapOperator op);

void toneMap(MTL::Texture *hdrColor,
    uint32_t numPixels,
    float exposureScale,
    ToneMapOperator op);

} // namespace tsd::algorithms::metal
```

**Step 3: Create `metal/toneMap.cpp`**

At `tsd/src/tsd/algorithms/metal/toneMap.cpp`:

```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/algorithms/metal/toneMap.hpp"
#include "MetalContext.h"

namespace tsd::algorithms::metal {

void toneMap(MTL::CommandBuffer *cmdBuf,
    MTL::Texture *hdrColor,
    uint32_t numPixels,
    float exposureScale,
    ToneMapOperator op)
{
    auto &ctx = MetalContext::instance();
    auto *pso = ctx.pipelineState("toneMapKernel");

    uint32_t width = (uint32_t)hdrColor->width();
    uint32_t height = (uint32_t)hdrColor->height();
    uint32_t opVal = static_cast<uint32_t>(op);

    auto *encoder = cmdBuf->computeCommandEncoder();
    encoder->setComputePipelineState(pso);
    encoder->setTexture(hdrColor, 0);
    encoder->setBytes(&width, sizeof(width), 0);
    encoder->setBytes(&exposureScale, sizeof(exposureScale), 1);
    encoder->setBytes(&opVal, sizeof(opVal), 2);

    auto tgWidth = pso->threadExecutionWidth();
    auto tgHeight = pso->maxTotalThreadsPerThreadgroup() / tgWidth;
    encoder->dispatchThreads(
        {width, height, 1}, {tgWidth, (NS::UInteger)tgHeight, 1});
    encoder->endEncoding();
}

void toneMap(MTL::Texture *hdrColor,
    uint32_t numPixels,
    float exposureScale,
    ToneMapOperator op)
{
    auto &ctx = MetalContext::instance();
    auto *cmdBuf = ctx.defaultQueue()->commandBuffer();
    toneMap(cmdBuf, hdrColor, numPixels, exposureScale, op);
    cmdBuf->commit();
    cmdBuf->waitUntilCompleted();
}

} // namespace tsd::algorithms::metal
```

**Step 4: Update CMakeLists.txt**

Add `metal/shaders/toneMap.metal` to `TSD_METAL_SHADERS` list. Add `metal/toneMap.cpp` to `project_sources`.

**Step 5: Build**

```bash
cd build && cmake --build . --parallel
```

Expected: MSL compiles with shared math headers. `toneMapKernel` function found in metallib.

**Step 6: Commit**

```bash
jj describe -m "feat(algorithms): Metal toneMap kernel

First algorithm sharing math headers across C++/CUDA/MSL. Validates
cross-compilation pipeline end-to-end."
jj new
```

---

## Task 6: convertColorBuffer + outputTransform

**Files:**
- Create: `tsd/src/tsd/algorithms/tsd/algorithms/metal/convertColorBuffer.hpp`
- Create: `tsd/src/tsd/algorithms/metal/convertColorBuffer.cpp`
- Create: `tsd/src/tsd/algorithms/metal/shaders/convertColorBuffer.metal`
- Create: `tsd/src/tsd/algorithms/tsd/algorithms/metal/outputTransform.hpp`
- Create: `tsd/src/tsd/algorithms/metal/outputTransform.cpp`
- Create: `tsd/src/tsd/algorithms/metal/shaders/outputTransform.metal`
- Modify: `tsd/src/tsd/algorithms/CMakeLists.txt`

**Step 1: Create `convertColorBuffer.metal`**

Kernel reads float texture, writes packed RGBA8 to a buffer (or output texture):

```metal
#include <metal_stdlib>
using namespace metal;

kernel void convertFloatToUint8Kernel(
    texture2d<float, access::read> input [[texture(0)]],
    device uint8_t *output [[buffer(0)]],
    constant uint32_t &width [[buffer(1)]],
    uint2 tid [[thread_position_in_grid]])
{
    if (tid.x >= width || tid.y >= input.get_height())
        return;

    float4 pixel = input.read(tid);
    uint idx = (tid.y * width + tid.x) * 4;
    output[idx + 0] = uint8_t(saturate(pixel.x) * 255.0f);
    output[idx + 1] = uint8_t(saturate(pixel.y) * 255.0f);
    output[idx + 2] = uint8_t(saturate(pixel.z) * 255.0f);
    output[idx + 3] = uint8_t(saturate(pixel.w) * 255.0f);
}
```

**Step 2: Create `outputTransform.metal`**

Kernel applies gamma correction and writes packed RGBA8. ANARI data types passed as `uint32_t` constants:

```metal
#include <metal_stdlib>
using namespace metal;

#include "vec_types.h"
#include "math_compat.h"
#include "color.h"

using namespace tsd::algorithms::math;

// Replicated ANARI data type constants (MSL can't include ANARI headers)
constant constexpr uint32_t ANARI_UFIXED8_VEC4 = 0x00120004;
constant constexpr uint32_t ANARI_UFIXED8_RGBA_SRGB = 0x00140004;

kernel void outputTransformKernel(
    texture2d<float, access::read> hdrInput [[texture(0)]],
    texture2d<float, access::read_write> colorOutput [[texture(1)]],
    constant uint32_t &width [[buffer(0)]],
    constant float &invGamma [[buffer(1)]],
    constant uint32_t &colorFormat [[buffer(2)]],
    uint2 tid [[thread_position_in_grid]])
{
    if (tid.x >= width || tid.y >= hdrInput.get_height())
        return;

    float4 hdr = hdrInput.read(tid);
    float3 c = linearToGamma(hdr.xyz, invGamma);
    colorOutput.write(float4(c, hdr.w), tid);
}
```

**Step 3: Create public headers and `.cpp` implementations**

Follow the same pattern as toneMap: `MTL::CommandBuffer*` + `MTL::Texture*` signatures with convenience overloads.

Refer to the CUDA signatures for parameter lists:
- `convertFloatToUint8(stream, hdrColor, output, totalSize)` → Metal: texture input, buffer output
- `outputTransform(stream, hdrColor, colorIn, colorOut, totalPixels, invGamma, colorFormat)` → Metal: two textures + format uint

**Step 4: Update CMakeLists.txt, build, commit**

```bash
jj describe -m "feat(algorithms): Metal convertColorBuffer + outputTransform kernels"
jj new
```

---

## Task 7: autoExposure — Two-Pass Threadgroup Reduction

Most complex Metal-specific kernel.

**Files:**
- Create: `tsd/src/tsd/algorithms/tsd/algorithms/metal/autoExposure.hpp`
- Create: `tsd/src/tsd/algorithms/metal/autoExposure.cpp`
- Create: `tsd/src/tsd/algorithms/metal/shaders/autoExposure.metal`
- Modify: `tsd/src/tsd/algorithms/CMakeLists.txt`

**Step 1: Create `autoExposure.metal`**

Two-pass reduction: pass 1 computes partial sums per threadgroup, pass 2 reduces partials.

```metal
#include <metal_stdlib>
using namespace metal;

#include "vec_types.h"
#include "math_compat.h"
#include "color.h"

using namespace tsd::algorithms::math;

constant constexpr float MIN_LUMINANCE = 1e-4f;

kernel void sumLogLuminancePass1(
    texture2d<float, access::read> hdrColor [[texture(0)]],
    device float *partials [[buffer(0)]],
    constant uint32_t &width [[buffer(1)]],
    constant uint32_t &height [[buffer(2)]],
    constant uint32_t &stride [[buffer(3)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tgSize [[threads_per_threadgroup]],
    uint tgId [[threadgroup_position_in_grid]])
{
    threadgroup float shared_data[1024];

    uint totalPixels = width * height;
    uint sampleIdx = tid * stride;
    float val = 0.0f;

    if (sampleIdx < totalPixels) {
        uint2 coord = uint2(sampleIdx % width, sampleIdx / width);
        float4 pixel = hdrColor.read(coord);
        float lum = max(luminance(pixel.x, pixel.y, pixel.z), MIN_LUMINANCE);
        val = log2(lum);
    }

    shared_data[lid] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tgSize / 2; s > 0; s >>= 1) {
        if (lid < s)
            shared_data[lid] += shared_data[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0)
        partials[tgId] = shared_data[0];
}

kernel void sumLogLuminancePass2(
    device float *partials [[buffer(0)]],
    device float *result [[buffer(1)]],
    constant uint32_t &numPartials [[buffer(2)]],
    uint lid [[thread_position_in_threadgroup]],
    uint tgSize [[threads_per_threadgroup]])
{
    threadgroup float shared_data[1024];

    shared_data[lid] = (lid < numPartials) ? partials[lid] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tgSize / 2; s > 0; s >>= 1) {
        if (lid < s)
            shared_data[lid] += shared_data[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0)
        result[0] = shared_data[0];
}
```

**Step 2: Create public header**

```cpp
namespace tsd::algorithms::metal {

float sumLogLuminance(MTL::CommandBuffer *cmdBuf,
    MTL::Texture *hdrColor,
    uint32_t numSamples,
    uint32_t stride);

float sumLogLuminance(MTL::Texture *hdrColor,
    uint32_t numSamples,
    uint32_t stride);

} // namespace tsd::algorithms::metal
```

**Step 3: Create `autoExposure.cpp`**

Two-dispatch pattern: pass 1 fills partials buffer, pass 2 reduces to single float, read back from shared result buffer.

```cpp
float sumLogLuminance(MTL::CommandBuffer *cmdBuf,
    MTL::Texture *hdrColor,
    uint32_t numSamples,
    uint32_t stride)
{
    auto &ctx = MetalContext::instance();
    uint32_t width = (uint32_t)hdrColor->width();
    uint32_t height = (uint32_t)hdrColor->height();

    constexpr uint32_t TG_SIZE = 256;
    uint32_t numGroups = (numSamples + TG_SIZE - 1) / TG_SIZE;

    auto *partials = ctx.device()->newBuffer(
        numGroups * sizeof(float), MTL::ResourceStorageModeShared);
    auto *result = ctx.device()->newBuffer(
        sizeof(float), MTL::ResourceStorageModeShared);

    // Pass 1
    auto *pso1 = ctx.pipelineState("sumLogLuminancePass1");
    auto *enc1 = cmdBuf->computeCommandEncoder();
    enc1->setComputePipelineState(pso1);
    enc1->setTexture(hdrColor, 0);
    enc1->setBuffer(partials, 0, 0);
    enc1->setBytes(&width, sizeof(width), 1);
    enc1->setBytes(&height, sizeof(height), 2);
    enc1->setBytes(&stride, sizeof(stride), 3);
    enc1->dispatchThreads({numSamples, 1, 1}, {TG_SIZE, 1, 1});
    enc1->endEncoding();

    // Pass 2
    auto *pso2 = ctx.pipelineState("sumLogLuminancePass2");
    auto *enc2 = cmdBuf->computeCommandEncoder();
    enc2->setComputePipelineState(pso2);
    enc2->setBuffer(partials, 0, 0);
    enc2->setBuffer(result, 0, 1);
    enc2->setBytes(&numGroups, sizeof(numGroups), 2);
    uint32_t pass2Size = 1;
    while (pass2Size < numGroups) pass2Size <<= 1;
    if (pass2Size > 1024) pass2Size = 1024;
    enc2->dispatchThreads({pass2Size, 1, 1}, {pass2Size, 1, 1});
    enc2->endEncoding();

    cmdBuf->commit();
    cmdBuf->waitUntilCompleted();

    float sumLogLum = *(float *)result->contents();
    partials->release();
    result->release();
    return sumLogLum;
}
```

Note: `autoExposure` is the one algorithm where the convenience overload and the explicit overload both need to commit+wait, because the result must be read back to CPU. The explicit overload commits its own command buffer here (divergence from the general pattern — documented in design doc).

**Step 4: Update CMakeLists.txt, build, commit**

```bash
jj describe -m "feat(algorithms): Metal autoExposure with two-pass threadgroup reduction"
jj new
```

---

## Task 8: visualizeAOV + outline

**Files:**
- Create: `tsd/src/tsd/algorithms/tsd/algorithms/metal/visualizeAOV.hpp`
- Create: `tsd/src/tsd/algorithms/metal/visualizeAOV.cpp`
- Create: `tsd/src/tsd/algorithms/metal/shaders/visualizeAOV.metal`
- Create: `tsd/src/tsd/algorithms/tsd/algorithms/metal/outline.hpp`
- Create: `tsd/src/tsd/algorithms/metal/outline.cpp`
- Create: `tsd/src/tsd/algorithms/metal/shaders/outline.metal`
- Modify: `tsd/src/tsd/algorithms/CMakeLists.txt`

**Step 1: Create `visualizeAOV.metal`**

Multiple kernels mirroring the CUDA variants. Each reads from the appropriate texture (depth, albedo, normal, objectId, etc.) and writes to a color output texture. Uses `color.h` for `makeRandomColor` and `luminance`.

The CUDA version at `cuda/visualizeAOV.cu` has these variants — mirror each:
- `visualizeObjectId`, `visualizePrimitiveId`, `visualizeInstanceId` — read uint32 texture, write `makeRandomColor` to color texture
- `visualizeDepth` — read float depth texture, normalize to [0,1] range, write grayscale
- `visualizeAlbedo` — read float3 albedo texture, write directly
- `visualizeNormal` — read float3 normal texture, remap [-1,1] to [0,1]
- `visualizeEdges` — read uint32 objectId texture, detect neighbor differences, write edge overlay

**Step 2: Create `outline.metal`**

Reads objectId texture, scans 3x3 neighborhood for matching `outlineId`, writes colored outline to color texture. Neighborhood access is natural in MSL via `texture.read(uint2(...))`.

**Step 3: Create public headers, implementations, update CMake, build, commit**

Follow established patterns. Each function takes `MTL::CommandBuffer*` + `MTL::Texture*` parameters.

```bash
jj describe -m "feat(algorithms): Metal visualizeAOV + outline kernels"
jj new
```

---

## Task 9: Pipeline Integration — ComputeStream + ImageBuffers + Pass Dispatch

Wires everything together: TSD rendering pipeline detects Metal device, maps Metal textures, dispatches Metal algorithms.

**Files:**
- Modify: `tsd/src/tsd/rendering/pipeline/passes/detail/ComputeStream.h`
- Modify: `tsd/src/tsd/rendering/pipeline/passes/ImagePass.h` (ImageBuffers struct)
- Modify: `tsd/src/tsd/rendering/pipeline/passes/ImagePass.cpp` (allocate/free/memcpy)
- Modify: `tsd/src/tsd/rendering/pipeline/passes/AnariSceneRenderPass.cpp`
- Modify: `tsd/src/tsd/rendering/pipeline/passes/ToneMapPass.cpp`
- Modify: `tsd/src/tsd/rendering/pipeline/passes/AutoExposurePass.cpp`
- Modify: `tsd/src/tsd/rendering/pipeline/passes/ClearBuffersPass.cpp`
- Modify: `tsd/src/tsd/rendering/pipeline/passes/OutputTransformPass.cpp`
- Modify: `tsd/src/tsd/rendering/pipeline/passes/OutlineRenderPass.cpp`
- Modify: `tsd/src/tsd/rendering/pipeline/passes/VisualizeAOVPass.cpp`
- Modify: `tsd/src/tsd/rendering/CMakeLists.txt`

**Step 1: Update `ComputeStream.h`**

At `tsd/src/tsd/rendering/pipeline/passes/detail/ComputeStream.h`, replace content:

```cpp
// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifdef ENABLE_CUDA
#include <cuda_runtime_api.h>
#elif defined(ENABLE_METAL)
namespace MTL { class CommandQueue; }
#endif

namespace tsd::rendering::detail {

#if defined(ENABLE_CUDA)
using ComputeStream = cudaStream_t;
#elif defined(ENABLE_METAL)
using ComputeStream = MTL::CommandQueue *;
#else
using ComputeStream = void *;
#endif

} // namespace tsd::rendering::detail
```

**Step 2: Update `ImageBuffers` in `ImagePass.h`**

Add Metal texture fields after `stream` at line 34:

```cpp
struct ImageBuffers
{
  uint32_t *color{nullptr};
  float *hdrColor{nullptr};
  float exposure{0.f};
  float *depth{nullptr};
  uint32_t *objectId{nullptr};
  uint32_t *primitiveId{nullptr};
  uint32_t *instanceId{nullptr};
  tsd::math::float3 *albedo{nullptr};
  tsd::math::float3 *normal{nullptr};
  detail::ComputeStream stream{};

#if defined(TSD_ALGORITHMS_HAS_METAL)
  MTL::Texture *metalHdrColor{nullptr};
  MTL::Texture *metalDepth{nullptr};
  MTL::Texture *metalObjectId{nullptr};
  MTL::Texture *metalPrimitiveId{nullptr};
  MTL::Texture *metalInstanceId{nullptr};
  MTL::Texture *metalAlbedo{nullptr};
  MTL::Texture *metalNormal{nullptr};
#endif
};
```

Add forward declaration at top of file:
```cpp
#if defined(TSD_ALGORITHMS_HAS_METAL)
namespace MTL { class Texture; }
#endif
```

**Step 3: Add `supportsMetalFbData` to `AnariSceneRenderPass.cpp`**

After the existing `supportsCUDAFbData` function (~line 55), add:

```cpp
static bool supportsMetalFbData(anari::Device d)
{
#ifdef ENABLE_METAL
  auto list = (const char *const *)anariGetObjectInfo(
      d, ANARI_DEVICE, "default", "extension", ANARI_STRING_LIST);
  for (const char *const *i = list; *i != nullptr; ++i) {
    if (std::string(*i) == "ANARI_MTL_FRAME_BUFFERS_METAL")
      return true;
  }
#endif
  return false;
}
```

**Step 4: Update `AnariSceneRenderPass` constructor**

After line 67 (`m_deviceSupportsCUDAFrames = supportsCUDAFbData(d);`), add:

```cpp
  m_deviceSupportsMetalFrames = supportsMetalFbData(d);
  if (m_deviceSupportsMetalFrames) {
    tsd::core::logStatus("[ImagePipeline] using Metal-mapped fb channels");
    MTL::CommandQueue *queue = nullptr;
    anariGetProperty(d, d, "mtl.commandQueue",
        ANARI_VOID_POINTER, &queue, sizeof(queue), ANARI_WAIT);
    m_metalQueue = queue;
  }
```

Add member variables `bool m_deviceSupportsMetalFrames{false}` and `MTL::CommandQueue *m_metalQueue{nullptr}` to `AnariSceneRenderPass.h`.

**Step 5: Update `copyFrameData` for Metal channels**

In `AnariSceneRenderPass.cpp`, before the existing channel selection (~line 325), add Metal path:

```cpp
#ifdef ENABLE_METAL
  if (m_deviceSupportsMetalFrames) {
    m_buffers.stream = m_metalQueue;

    auto color = anari::map<void>(m_device, m_frame, "channel.colorMTL");
    m_buffers.metalHdrColor = reinterpret_cast<MTL::Texture *>(color.data);

    auto depth = anari::map<void>(m_device, m_frame, "channel.depthMTL");
    m_buffers.metalDepth = reinterpret_cast<MTL::Texture *>(depth.data);

    if (m_enableIDs) {
      auto oid = anari::map<void>(m_device, m_frame, "channel.objectIdMTL");
      m_buffers.metalObjectId = reinterpret_cast<MTL::Texture *>(oid.data);
    }
    if (m_enablePrimitiveId) {
      auto pid = anari::map<void>(m_device, m_frame, "channel.primitiveIdMTL");
      m_buffers.metalPrimitiveId = reinterpret_cast<MTL::Texture *>(pid.data);
    }
    if (m_enableInstanceId) {
      auto iid = anari::map<void>(m_device, m_frame, "channel.instanceIdMTL");
      m_buffers.metalInstanceId = reinterpret_cast<MTL::Texture *>(iid.data);
    }
    if (m_enableAlbedo) {
      auto alb = anari::map<void>(m_device, m_frame, "channel.albedoMTL");
      m_buffers.metalAlbedo = reinterpret_cast<MTL::Texture *>(alb.data);
    }
    if (m_enableNormals) {
      auto nrm = anari::map<void>(m_device, m_frame, "channel.normalMTL");
      m_buffers.metalNormal = reinterpret_cast<MTL::Texture *>(nrm.data);
    }

    // No copy needed — textures are passed through directly
    return;
  }
#endif
```

**Step 6: Update all pass dispatch files**

Each pass gains a Metal branch before the CUDA branch. Pattern for `ToneMapPass.cpp` (lines 55-63):

```cpp
#if TSD_ALGORITHMS_HAS_METAL
  if (b.metalHdrColor) {
    auto *cmdBuf = b.stream->commandBuffer();
    tsd::algorithms::metal::toneMap(
        cmdBuf, b.metalHdrColor, totalPixels, exposureScale, m_operator);
    cmdBuf->commit();
    return;
  }
#endif
#if TSD_ALGORITHMS_HAS_CUDA
  if (b.stream) {
    // ... existing CUDA path unchanged
```

Apply same pattern to:
- `AutoExposurePass.cpp:57-66` — call `metal::sumLogLuminance`
- `ClearBuffersPass.cpp:30-40` — Metal clearBuffers takes textures (or buffer fills depending on allocation strategy)
- `OutputTransformPass.cpp:42-47` — call `metal::outputTransform`
- `OutlineRenderPass.cpp:31-36` — call `metal::outline`
- `VisualizeAOVPass.cpp:43-76` — call `metal::visualize*` variants

Each pass includes the Metal header:
```cpp
#if TSD_ALGORITHMS_HAS_METAL
#include "tsd/algorithms/metal/toneMap.hpp"
#endif
```

**Step 7: Update rendering CMakeLists.txt**

At `tsd/src/tsd/rendering/CMakeLists.txt`, add after the CUDA block (after line 62):

```cmake
elseif(TSD_USE_METAL)
  find_library(METAL_FRAMEWORK Metal REQUIRED)
  find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
  project_link_libraries(PRIVATE ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})
  project_compile_definitions(PRIVATE ENABLE_METAL)
```

**Step 8: Build end-to-end**

```bash
cd build && cmake -DTSD_USE_METAL=ON .. && cmake --build . --parallel
```

Expected: Full pipeline compiles. Metal device detection works. Algorithm dispatch routes to Metal kernels.

**Step 9: Manual integration test**

Run `tsdViewer` or `tsdRender` against `anari-mtl`. Verify:
- Tone mapping applies on GPU (no CPU fallback log message)
- AOV visualization works
- Auto-exposure converges

**Step 10: Commit**

```bash
jj describe -m "feat(rendering): pipeline integration for Metal algorithms

ComputeStream = MTL::CommandQueue*, ImageBuffers gains Metal texture
fields. AnariSceneRenderPass detects ANARI_MTL_FRAME_BUFFERS_METAL and
maps textures directly. All passes dispatch to Metal backend when
available."
jj new
```

---

## Summary of Files

### New files (algorithms library)
```
tsd/src/tsd/algorithms/math/vec_types.h
tsd/src/tsd/algorithms/math/math_compat.h
tsd/src/tsd/algorithms/metal/MetalContext.h
tsd/src/tsd/algorithms/metal/MetalContext.cpp
tsd/src/tsd/algorithms/metal/clearBuffers.cpp
tsd/src/tsd/algorithms/metal/toneMap.cpp
tsd/src/tsd/algorithms/metal/autoExposure.cpp
tsd/src/tsd/algorithms/metal/outputTransform.cpp
tsd/src/tsd/algorithms/metal/visualizeAOV.cpp
tsd/src/tsd/algorithms/metal/outline.cpp
tsd/src/tsd/algorithms/metal/convertColorBuffer.cpp
tsd/src/tsd/algorithms/metal/shaders/clearBuffers.metal
tsd/src/tsd/algorithms/metal/shaders/toneMap.metal
tsd/src/tsd/algorithms/metal/shaders/autoExposure.metal
tsd/src/tsd/algorithms/metal/shaders/outputTransform.metal
tsd/src/tsd/algorithms/metal/shaders/visualizeAOV.metal
tsd/src/tsd/algorithms/metal/shaders/outline.metal
tsd/src/tsd/algorithms/metal/shaders/convertColorBuffer.metal
tsd/src/tsd/algorithms/tsd/algorithms/metal/clearBuffers.hpp
tsd/src/tsd/algorithms/tsd/algorithms/metal/toneMap.hpp
tsd/src/tsd/algorithms/tsd/algorithms/metal/autoExposure.hpp
tsd/src/tsd/algorithms/tsd/algorithms/metal/outputTransform.hpp
tsd/src/tsd/algorithms/tsd/algorithms/metal/visualizeAOV.hpp
tsd/src/tsd/algorithms/tsd/algorithms/metal/outline.hpp
tsd/src/tsd/algorithms/tsd/algorithms/metal/convertColorBuffer.hpp
```

### New files (anari-mtl)
```
/Users/tarcila/Code/ANARI/mtl/interop/src/array/MetalArray1D.h
/Users/tarcila/Code/ANARI/mtl/interop/src/array/MetalArray1D.cpp
/Users/tarcila/Code/ANARI/mtl/interop/src/array/MetalArray2D.h
/Users/tarcila/Code/ANARI/mtl/interop/src/array/MetalArray2D.cpp
/Users/tarcila/Code/ANARI/mtl/interop/src/array/MetalArray3D.h
/Users/tarcila/Code/ANARI/mtl/interop/src/array/MetalArray3D.cpp
/Users/tarcila/Code/ANARI/mtl/interop/src/extensions/ANARI_MTL_ARRAY_METAL.h
/Users/tarcila/Code/ANARI/mtl/interop/src/extensions/ANARI_MTL_ARRAY_METAL.cpp
/Users/tarcila/Code/ANARI/mtl/interop/src/extensions/ANARI_MTL_FRAME_BUFFERS_METAL.h
```

### Modified files
```
tsd/src/tsd/algorithms/math/device_macros.h
tsd/src/tsd/algorithms/math/tonemap_curves.h
tsd/src/tsd/algorithms/math/color.h
tsd/src/tsd/algorithms/tsd/algorithms/config.h
tsd/src/tsd/algorithms/CMakeLists.txt
tsd/src/tsd/rendering/CMakeLists.txt
tsd/src/tsd/rendering/pipeline/passes/detail/ComputeStream.h
tsd/src/tsd/rendering/pipeline/passes/ImagePass.h
tsd/src/tsd/rendering/pipeline/passes/ImagePass.cpp
tsd/src/tsd/rendering/pipeline/passes/AnariSceneRenderPass.h
tsd/src/tsd/rendering/pipeline/passes/AnariSceneRenderPass.cpp
tsd/src/tsd/rendering/pipeline/passes/ToneMapPass.cpp
tsd/src/tsd/rendering/pipeline/passes/AutoExposurePass.cpp
tsd/src/tsd/rendering/pipeline/passes/ClearBuffersPass.cpp
tsd/src/tsd/rendering/pipeline/passes/OutputTransformPass.cpp
tsd/src/tsd/rendering/pipeline/passes/OutlineRenderPass.cpp
tsd/src/tsd/rendering/pipeline/passes/VisualizeAOVPass.cpp
/Users/tarcila/Code/ANARI/mtl/interop/src/MtlDevice.h
/Users/tarcila/Code/ANARI/mtl/interop/src/MtlDevice.cpp
/Users/tarcila/Code/ANARI/mtl/interop/src/frame/Frame.cpp
/Users/tarcila/Code/ANARI/mtl/interop/src/MtlDefinitions.json
/Users/tarcila/Code/ANARI/mtl/interop/src/CMakeLists.txt
```
