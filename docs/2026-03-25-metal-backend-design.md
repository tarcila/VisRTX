# Metal Compute Backend for TSD Algorithms Library

## Context

The `tsd_algorithms` library currently supports CPU (serial/TBB) and CUDA backends for image-processing algorithms (tone mapping, auto-exposure, AOV visualization, etc.). The library was designed from the start for multiple GPU backends — the original design doc lists `metal/` and `vulkan/` as future directories, and the namespace convention (`tsd::algorithms::cpu::`, `tsd::algorithms::cuda::`, `tsd::algorithms::metal::`) is already established.

Adding a Metal backend enables GPU-accelerated post-processing on macOS/iOS, where CUDA is unavailable. This is a design document covering architecture, API, build integration, and trade-offs.

## Design Decisions

### 1. Metal-cpp for the host API

Apple's [Metal-cpp](https://developer.apple.com/metal/cpp/) headers provide C++ bindings for the Metal API. This means:

- Implementation files are plain `.cpp` (no Objective-C++ `.mm` needed)
- Public headers use Metal-cpp types directly: `MTL::CommandBuffer*`, `MTL::Buffer*`
- Analogous to how the CUDA backend uses `cudaStream_t` and raw device pointers

Metal-cpp is header-only. It can be fetched via `anari_sdk_fetch_project()` (archive URL) or found via `find_path()` if installed system-wide.

### 2. Public API convention

Following the CUDA pattern — backend-typed first parameter, plus a convenience overload:

```cpp
namespace tsd::algorithms::metal {

void toneMap(MTL::CommandBuffer *cmdBuf,
    MTL::Buffer *hdrColor,
    uint32_t numPixels,
    float exposureScale,
    ToneMapOperator op);

// Convenience: uses default command queue from MetalContext
void toneMap(MTL::Buffer *hdrColor,
    uint32_t numPixels,
    float exposureScale,
    ToneMapOperator op);

} // namespace tsd::algorithms::metal
```

The `MTL::Buffer*` carries the device pointer (no raw `float*` for Metal). Callers that interop with ANARI's Metal extensions will already have `MTL::Buffer` handles.

### 3. Cross-platform math abstraction

The `math/` headers (`tonemap_curves.h`, `color.h`) currently depend on `tsd::math::float3` (from ANARI SDK's `linalg.h`) and `std::` math functions. MSL cannot consume either.

**Approach:** Introduce thin abstraction headers so the same math source compiles for C++, CUDA, and MSL:

**`math/vec_types.h`** — conditional `float3` typedef:
```cpp
#ifdef __METAL_VERSION__
using float3 = metal::float3;
#else
#include "tsd/core/TSDMath.hpp"
using float3 = tsd::math::float3;
#endif
```

**`math/math_compat.h`** — wrap scalar/vector math functions:
```cpp
#ifdef __METAL_VERSION__
using metal::max; using metal::min; using metal::clamp;
using metal::pow; using metal::log2;
#else
using std::max; using std::min; using std::clamp;
using std::pow; using std::log2;
#endif
```

Update `tonemap_curves.h` and `color.h` to include these instead of `tsd/core/TSDMath.hpp` and `std::` directly. The `.metal` shaders then `#include` the same `tonemap_curves.h` and `color.h` via the Metal compiler's include path.

**Risk:** This modifies headers that CPU and CUDA backends consume. The change is mechanical (type aliases, `using` declarations) but must be validated against all existing backends.

### 4. Shader compilation: embedded metallib

`.metal` → `.air` → `.metallib` at CMake build time, then embedded as a C byte array in the binary via `xxd` (or `CMake file(READ ... HEX)`).

At runtime, `MetalContext` creates the library from the embedded data — no file path to manage.

### 5. MetalContext (internal singleton)

Metal requires explicit device, command queue, and pipeline state objects. A lightweight singleton manages these:

```cpp
// metal/MetalContext.h (internal, not in public headers)
struct MetalContext {
  static MetalContext &instance();

  MTL::Device *device();
  MTL::CommandQueue *defaultQueue();
  MTL::ComputePipelineState *pipelineState(const char *kernelName);

private:
  MetalContext();
  MTL::Device *m_device;
  MTL::CommandQueue *m_queue;
  MTL::Library *m_library;  // from embedded metallib
  std::unordered_map<std::string, MTL::ComputePipelineState *> m_pipelines;
  std::mutex m_mutex;
};
```

Pipeline states are cached after first creation. Thread-safe for the lookup (pipeline states are immutable once created).

### 6. Kernel dispatch pattern

Each `.cpp` implementation encodes a compute command:

```cpp
void toneMap(MTL::CommandBuffer *cmdBuf, MTL::Buffer *hdrColor,
    uint32_t numPixels, float exposureScale, ToneMapOperator op)
{
  auto &ctx = MetalContext::instance();
  auto pso = ctx.pipelineState("toneMapKernel");

  auto encoder = cmdBuf->computeCommandEncoder();
  encoder->setComputePipelineState(pso);
  encoder->setBuffer(hdrColor, 0, 0);
  encoder->setBytes(&numPixels, sizeof(numPixels), 1);
  encoder->setBytes(&exposureScale, sizeof(exposureScale), 2);
  uint32_t opVal = static_cast<uint32_t>(op);
  encoder->setBytes(&opVal, sizeof(opVal), 3);

  auto threadGroupSize = pso->maxTotalThreadsPerThreadgroup();
  if (threadGroupSize > numPixels) threadGroupSize = numPixels;
  encoder->dispatchThreads({numPixels, 1, 1}, {threadGroupSize, 1, 1});
  encoder->endEncoding();
}
```

The convenience overload creates a command buffer from the default queue, encodes, commits, and (optionally) waits.

### 7. Reductions (autoExposure)

`sumLogLuminance` needs a parallel reduction. Metal approach:

1. **Pass 1:** Each threadgroup computes a partial sum into shared (threadgroup) memory, writes one result per threadgroup to a temporary `MTL::Buffer`.
2. **Pass 2:** A single threadgroup reduces the partials.
3. The result is read back from a shared-storage buffer.

This is the most complex kernel and the main Metal-specific logic beyond the simple `for_each` pattern.

## Directory Layout

```
src/tsd/algorithms/
├── CMakeLists.txt                    # Extended: TSD_USE_METAL section
├── tsd/algorithms/                   # Public headers
│   ├── config.h                      # Add TSD_ALGORITHMS_HAS_METAL
│   ├── cpu/...                       # (unchanged)
│   ├── cuda/...                      # (unchanged)
│   └── metal/                        # NEW
│       ├── toneMap.hpp
│       ├── autoExposure.hpp
│       ├── outputTransform.hpp
│       ├── visualizeAOV.hpp
│       ├── outline.hpp
│       ├── clearBuffers.hpp
│       └── convertColorBuffer.hpp
├── math/
│   ├── device_macros.h               # Extend: __METAL_VERSION__ guard
│   ├── vec_types.h                   # NEW: cross-platform float3
│   ├── math_compat.h                 # NEW: cross-platform math functions
│   ├── tonemap_curves.h              # MODIFIED: use vec_types.h + math_compat.h
│   └── color.h                       # MODIFIED: use vec_types.h + math_compat.h
├── cpu/...                           # (unchanged)
├── cuda/...                          # (unchanged)
└── metal/                            # NEW
    ├── MetalContext.h                 # Internal singleton
    ├── MetalContext.cpp
    ├── toneMap.cpp
    ├── autoExposure.cpp
    ├── outputTransform.cpp
    ├── visualizeAOV.cpp
    ├── outline.cpp
    ├── clearBuffers.cpp
    ├── convertColorBuffer.cpp
    └── shaders/
        ├── toneMap.metal
        ├── autoExposure.metal
        ├── outputTransform.metal
        ├── visualizeAOV.metal
        ├── outline.metal
        ├── clearBuffers.metal
        └── convertColorBuffer.metal
```

## CMake Integration

```cmake
# GPU backend: Metal
if (TSD_USE_METAL)
  if (NOT APPLE)
    message(FATAL_ERROR "TSD_USE_METAL requires macOS or iOS")
  endif()

  # Metal-cpp headers (header-only)
  anari_sdk_fetch_project(
    NAME metal_cpp
    URL "https://developer.apple.com/metal/cpp/files/metal-cpp_macOS15_iOS18.zip"
    MD5 <hash>
  )
  project_include_directories(PUBLIC ${metal_cpp_LOCATION})

  project_compile_definitions(PUBLIC TSD_ALGORITHMS_HAS_METAL)

  # --- Shader compilation ---
  set(TSD_METAL_SHADERS
    metal/shaders/toneMap.metal
    metal/shaders/autoExposure.metal
    metal/shaders/outputTransform.metal
    metal/shaders/visualizeAOV.metal
    metal/shaders/outline.metal
    metal/shaders/clearBuffers.metal
    metal/shaders/convertColorBuffer.metal
  )

  # .metal -> .air -> .metallib -> embedded C array
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

  # Embed as C array
  set(METALLIB_C ${CMAKE_CURRENT_BINARY_DIR}/tsd_algorithms_metallib.c)
  add_custom_command(
    OUTPUT ${METALLIB_C}
    COMMAND xxd -i tsd_algorithms.metallib ${METALLIB_C}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${METALLIB_FILE}
  )

  # Sources
  set(TSD_ALGORITHMS_METAL_SOURCES
    metal/MetalContext.cpp
    metal/toneMap.cpp
    metal/autoExposure.cpp
    metal/outputTransform.cpp
    metal/visualizeAOV.cpp
    metal/outline.cpp
    metal/clearBuffers.cpp
    metal/convertColorBuffer.cpp
    ${METALLIB_C}
  )
  project_sources(PRIVATE ${TSD_ALGORITHMS_METAL_SOURCES})

  # Link Metal + Foundation frameworks
  find_library(METAL_FRAMEWORK Metal REQUIRED)
  find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
  project_link_libraries(PUBLIC ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})
endif()
```

The `-I ${CMAKE_CURRENT_SOURCE_DIR}/math` flag on the Metal compiler lets `.metal` files `#include "tonemap_curves.h"` etc. from the shared math directory.

## Rendering Pipeline Integration

### ComputeStream

CUDA and Metal are platform-exclusive (Linux/Windows vs. macOS), so `ComputeStream` extends cleanly:

```cpp
// detail/ComputeStream.h
#if defined(ENABLE_CUDA)
#include <cuda_runtime_api.h>
using ComputeStream = cudaStream_t;
#elif defined(ENABLE_METAL)
namespace MTL { class CommandBuffer; }
using ComputeStream = MTL::CommandBuffer *;
#else
using ComputeStream = void *;
#endif
```

### ImageBuffers

Metal operates on `MTL::Buffer*`, not raw pointers. Two options:

**Option A:** Add parallel `MTL::Buffer*` fields to `ImageBuffers` (mirrors each raw pointer). Callers populate both when Metal is active.

**Option B:** The Metal backend recovers `MTL::Buffer*` from raw pointers via a registry in `MetalContext` (pointers come from `buffer->contents()`). No changes to `ImageBuffers`.

Option A is more explicit and avoids the registry overhead. The additional fields are `#ifdef`-guarded.

### Pass dispatch

Same pattern as CUDA:

```cpp
#if TSD_ALGORITHMS_HAS_METAL
if (b.stream) {
  tsd::algorithms::metal::toneMap(
      b.stream, b.metalHdrColor, totalPixels, exposureScale, m_operator);
  return;
}
#endif
```

## Key Trade-offs

| Decision | Choice | Trade-off |
|---|---|---|
| Host API | Metal-cpp (pure C++) | Requires fetching metal-cpp headers; avoids Objective-C++ entirely |
| Math sharing | Cross-platform abstraction | Modifies 2 shared headers; single source of truth; future-proof for more backends |
| Metallib distribution | Embedded C array | Self-contained binary; slightly larger library size (~few KB) |
| Buffer model | Explicit `MTL::Buffer*` in `ImageBuffers` | More fields in the struct but no runtime registry lookup |
| No Thrust equivalent | Hand-written MSL kernels | More code per algorithm but cleaner, no template metaprogramming |
| Reductions | Two-pass threadgroup reduce | More complex than Thrust `transform_reduce` but standard Metal pattern |

## Concerns

- **Math refactor risk:** Changing `tonemap_curves.h` and `color.h` affects CPU and CUDA backends. Must validate all three after the refactor.
- **`anari::DataType` in shaders:** `outputTransform` branches on ANARI data types. MSL kernels can't include ANARI headers — pass as `uint32_t`, replicate needed enum values as constants.
- **`helium::cvt_color_*` in outline/outputTransform:** These bit-packing utilities need MSL equivalents (trivial: unpack uint32 to 4 bytes and back).
- **Threadgroup sizing:** Metal's optimal threadgroup size varies by GPU. Using `pso->maxTotalThreadsPerThreadgroup()` is the standard heuristic, but occupancy tuning may be needed later.
- **Managed vs. private storage:** Initial implementation should use `MTLStorageModeShared` (CPU+GPU visible). Optimize to `MTLStorageModePrivate` for GPU-only intermediates later.

## Implementation Order (for future executable plan)

1. **Math abstraction** — `vec_types.h`, `math_compat.h`, update existing headers. Validate CPU + CUDA still work.
2. **CMake + MetalContext + clearBuffers** — Minimal proof of concept: shader compilation pipeline, metallib embedding, singleton, simplest kernel.
3. **toneMap** — First algorithm using shared math in MSL. Validates the cross-compilation approach end-to-end.
4. **convertColorBuffer + outputTransform** — Two-buffer patterns, ANARI type bridging.
5. **autoExposure** — Reduction kernel (most complex Metal-specific work).
6. **visualizeAOV + outline** — Multiple variants, neighborhood access patterns.
7. **Rendering pipeline integration** — `ComputeStream`, `ImageBuffers` Metal fields, pass dispatch.
