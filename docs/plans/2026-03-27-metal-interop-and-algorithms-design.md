# Metal Interop Extensions and Algorithms Library Backend

## Context

TSD's `ImagePipeline` currently achieves GPU-resident post-processing only with CUDA devices, via `ANARI_NV_ARRAY_CUDA` and `ANARI_NV_FRAME_BUFFERS_CUDA`. When running against `anari-mtl` (the Metal-based ANARI ray tracer), every frame round-trips through host memory for tone mapping, compositing, and AOV visualization.

This document co-designs two ANARI extensions for the `anari-mtl` device and a Metal backend for the `tsd_algorithms` library, closing the GPU-resident pipeline end-to-end on macOS.

## Design Decisions

### 1. Extension scope: `ANARI_MTL_*`

Device-scoped extensions (`ANARI_MTL_ARRAY_METAL`, `ANARI_MTL_FRAME_BUFFERS_METAL`) rather than vendor-neutral Khronos extensions. anari-mtl is the only Metal ANARI device today; the extensions can be promoted to `KHR` later if other Metal devices appear.

### 2. Texture-centric frame buffers

Frame channels return `MTL::Texture*` (the render targets anari-mtl already produces) rather than `MTL::Buffer*`. This avoids an intermediate blit, and Metal compute kernels consume textures natively via `texture2d<float, access::read_write>`. Buffers are used only internally where needed (reductions, small parameter blocks).

### 3. Explicit array creation functions

Unlike the CUDA extension (which detects address space from raw pointers via `cudaPointerGetAttributes()`), Metal has no equivalent introspection. The extension provides explicit creation functions that accept `MTL::Buffer*` directly, for 1D, 2D, and 3D arrays.

### 4. Shared queue synchronization

The ANARI device exposes its `MTL::CommandQueue*` via a device property. The algorithms library and anari-mtl encode independent command buffers on this shared queue. Metal guarantees serial execution order within a single queue, so no explicit synchronization (events, fences) is needed between sequential operations. Multi-queue scenarios can add `MTL::SharedEvent` later.

### 5. Platform exclusivity

`TSD_USE_CUDA` and `TSD_USE_METAL` are mutually exclusive at the CMake level. This simplifies `ComputeStream`, `ImageBuffers`, and dispatch logic.

## ANARI Extensions

### `ANARI_MTL_ARRAY_METAL`

New C extension functions for creating ANARI arrays backed by `MTL::Buffer*`:

```cpp
ANARIArray1D anariNewArray1DMetalBuffer(
    ANARIDevice, MTL::Buffer *, ANARIDataType, uint64_t numItems);

ANARIArray2D anariNewArray2DMetalBuffer(
    ANARIDevice, MTL::Buffer *, ANARIDataType,
    uint64_t numItems1, uint64_t numItems2);

ANARIArray3D anariNewArray3DMetalBuffer(
    ANARIDevice, MTL::Buffer *, ANARIDataType,
    uint64_t numItems1, uint64_t numItems2, uint64_t numItems3);
```

**Ownership:** The application retains ownership of the `MTL::Buffer*`. anari-mtl reads from it but does not release it. The buffer must remain valid until the array is released or re-committed with different data.

**Storage modes:** `StorageModeShared` and `StorageModeManaged` are directly usable. `StorageModePrivate` also works (anari-mtl can encode GPU copies), but the application cannot populate it from the CPU side.

### `ANARI_MTL_FRAME_BUFFERS_METAL`

New frame channel names that return `MTL::Texture*` via `anariMapFrame()`:

| Channel | Texture format | Notes |
|---------|---------------|-------|
| `"channel.colorMTL"` | RGBA32Float | HDR linear color |
| `"channel.depthMTL"` | R32Float | |
| `"channel.normalMTL"` | RGBA32Float | |
| `"channel.albedoMTL"` | RGBA32Float | |
| `"channel.primitiveIdMTL"` | R32Uint | |
| `"channel.objectIdMTL"` | R32Uint | |
| `"channel.instanceIdMTL"` | R32Uint | |

`anariMapFrame()` returns the `MTL::Texture*` cast to `void*` in the mapped data pointer. No blit, no readback. The texture is valid until `anariUnmapFrame()`. The texture's storage mode is Private — the caller consumes it on the GPU via compute kernels.

`anariUnmapFrame()` for Metal channels is a no-op.

### Extension discovery

Both extensions are advertised via `anariGetObjectInfo(device, ANARI_DEVICE, "default", "extension", ANARI_STRING_LIST)`, matching the CUDA discovery pattern.

### Shared queue access

```cpp
MTL::CommandQueue *queue = nullptr;
anariGetProperty(device, device, "mtl.commandQueue",
    ANARI_VOID_POINTER, &queue, sizeof(queue), ANARI_WAIT);
```

## Algorithms Library Metal Backend

### Public API signatures

Each algorithm takes `MTL::CommandBuffer*` as the compute handle plus `MTL::Texture*` for framebuffer data. Each also has a convenience overload that creates its own command buffer from `MetalContext`'s queue.

```cpp
namespace tsd::algorithms::metal {

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

The same two-overload pattern applies to all 7 algorithms: `toneMap`, `autoExposure`, `outputTransform`, `visualizeAOV`, `outline`, `clearBuffers`, `convertColorBuffer`.

Notable Metal-specific details:

- **`autoExposure` (`sumLogLuminance`):** Reads from `MTL::Texture*`, reduces via two-pass threadgroup reduction into a small `MTL::Buffer`, reads result back. Returns `float` like CUDA.
- **`outline`:** Neighborhood access via direct texel sampling in MSL.
- **`outputTransform`:** ANARI data types passed as `uint32_t`; enum values replicated as constants in MSL (cannot include ANARI headers).

### Shader compilation

`.metal` → `.air` → `.metallib` at CMake build time, then embedded as a C byte array via `xxd`. At runtime, `MetalContext` loads the library from embedded data and caches `MTL::ComputePipelineState` per kernel name.

The Metal compiler gets `-I math/` so `.metal` files `#include "tonemap_curves.h"` and `"color.h"` via the cross-platform math abstraction.

### MetalContext (internal singleton)

Manages device, default queue, embedded metallib, and pipeline state cache. When TSD provides a queue (obtained from the ANARI device via `mtl.commandQueue`), the algorithms library uses that queue. Convenience overloads check for a configured queue before falling back to the internal default.

```cpp
struct MetalContext {
  static MetalContext &instance();

  MTL::Device *device();
  MTL::CommandQueue *defaultQueue();
  void setQueue(MTL::CommandQueue *); // override with ANARI device's queue
  MTL::ComputePipelineState *pipelineState(const char *kernelName);

private:
  MetalContext();
  MTL::Device *m_device;
  MTL::CommandQueue *m_queue;
  MTL::Library *m_library; // from embedded metallib
  std::unordered_map<std::string, MTL::ComputePipelineState *> m_pipelines;
  std::mutex m_mutex;
};
```

## Pipeline Integration

### ComputeStream

```cpp
// detail/ComputeStream.h
#if defined(ENABLE_CUDA)
#include <cuda_runtime_api.h>
using ComputeStream = cudaStream_t;
#elif defined(ENABLE_METAL)
namespace MTL { class CommandQueue; }
using ComputeStream = MTL::CommandQueue *;
#else
using ComputeStream = void *;
#endif
```

### ImageBuffers

Gains `#ifdef`-guarded `MTL::Texture*` fields alongside existing raw pointer fields:

```cpp
struct ImageBuffers
{
  // Existing fields (unchanged)
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

### AnariSceneRenderPass

Detects `ANARI_MTL_FRAME_BUFFERS_METAL` at construction time. When supported, maps Metal channels and populates `ImageBuffers` texture fields:

```cpp
auto mapped = anari::map<void>(m_device, m_frame, "channel.colorMTL");
b.metalHdrColor = reinterpret_cast<MTL::Texture *>(mapped.data);
```

Obtains the shared queue once at init:

```cpp
MTL::CommandQueue *queue = nullptr;
anariGetProperty(m_device, m_device, "mtl.commandQueue",
    ANARI_VOID_POINTER, &queue, sizeof(queue), ANARI_WAIT);
b.stream = queue;
```

### Algorithm dispatch in passes

Each pass adds a Metal branch, checked before CUDA and CPU:

```cpp
void ToneMapPass::render(ImageBuffers &b, int stageId)
{
  const float exposureScale = std::exp2(exposure);

#if TSD_ALGORITHMS_HAS_METAL
  if (b.metalHdrColor) {
    tsd::algorithms::metal::toneMap(
        b.stream, b.metalHdrColor, totalPixels, exposureScale, m_operator);
    return;
  }
#endif

#if TSD_ALGORITHMS_HAS_CUDA
  if (b.stream) {
    tsd::algorithms::cuda::toneMap(
        b.stream, b.hdrColor, totalPixels, exposureScale, m_operator);
    return;
  }
#endif

  tsd::algorithms::cpu::toneMap(
      b.hdrColor, totalPixels, exposureScale, m_operator);
}
```

## Cross-platform Math Abstraction

### New headers

**`math/vec_types.h`:**
```cpp
#ifdef __METAL_VERSION__
using float3 = metal::float3;
using float4 = metal::float4;
#else
#include "tsd/core/TSDMath.hpp"
using float3 = tsd::math::float3;
using float4 = tsd::math::float4;
#endif
```

**`math/math_compat.h`:**
```cpp
#ifdef __METAL_VERSION__
using metal::max; using metal::min; using metal::clamp;
using metal::pow; using metal::log2; using metal::exp2;
#else
using std::max; using std::min; using std::clamp;
using std::pow; using std::log2; using std::exp2;
#endif
```

### Changes to existing headers

`tonemap_curves.h` and `color.h` replace `#include "tsd/core/TSDMath.hpp"` with `#include "vec_types.h"` + `#include "math_compat.h"`, and drop explicit `std::` qualifiers on math functions (resolved by `math_compat.h`).

**`device_macros.h`** extends for Metal (no-op — MSL functions are implicitly device code):
```cpp
#if defined(__CUDA_ARCH__)
#define DEVICE_FCN __device__
#elif defined(__METAL_VERSION__)
#define DEVICE_FCN
#else
#define DEVICE_FCN
#endif
```

### Validation requirement

CPU and CUDA backends must be tested after this refactor. The changes are mechanical (type aliases and `using` declarations), but this is a hard gate before Metal-specific work proceeds.

## anari-mtl Device Changes

### Array extension

New `MetalArray1D`, `MetalArray2D`, `MetalArray3D` subclasses that hold a `MTL::Buffer*`:

```cpp
struct MetalArray1D : public helium::Array1D {
  MTL::Buffer *metalBuffer() const;
};
```

`MtlDevice` gains the three `anariNewArray*MetalBuffer` factory methods. When geometry or volumes commit, they check whether the array is a `MetalArray*` — if so, they bind the `MTL::Buffer*` directly instead of uploading from host memory.

### Frame extension

`Frame::map()` gains branches for `"channel.*MTL"` names. Returns the render-target `MTL::Texture*` cast to `void*`. No blit, no readback.

### Device property

`MtlDevice::getProperty()` handles `"mtl.commandQueue"` by returning `MetalContext::queue()`.

### Extension advertisement

`MtlDefinitions.json` lists `"ANARI_MTL_ARRAY_METAL"` and `"ANARI_MTL_FRAME_BUFFERS_METAL"` in the device's extension string list.

## Build Integration

### Algorithms library CMake

```cmake
if (TSD_USE_METAL)
  if (NOT APPLE)
    message(FATAL_ERROR "TSD_USE_METAL requires macOS")
  endif()

  anari_sdk_fetch_project(
    NAME metal_cpp
    URL "https://developer.apple.com/metal/cpp/files/metal-cpp_macOS15_iOS18.zip"
    MD5 <hash>
  )
  project_include_directories(PUBLIC ${metal_cpp_LOCATION})
  project_compile_definitions(PUBLIC TSD_ALGORITHMS_HAS_METAL)

  # Shader compilation (.metal -> .air -> .metallib -> embedded C array)
  set(TSD_METAL_SHADERS
    metal/shaders/toneMap.metal
    metal/shaders/autoExposure.metal
    metal/shaders/outputTransform.metal
    metal/shaders/visualizeAOV.metal
    metal/shaders/outline.metal
    metal/shaders/clearBuffers.metal
    metal/shaders/convertColorBuffer.metal
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
    metal/toneMap.cpp
    metal/autoExposure.cpp
    metal/outputTransform.cpp
    metal/visualizeAOV.cpp
    metal/outline.cpp
    metal/clearBuffers.cpp
    metal/convertColorBuffer.cpp
    ${METALLIB_C}
  )

  find_library(METAL_FRAMEWORK Metal REQUIRED)
  find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
  project_link_libraries(PUBLIC ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})
endif()
```

### Rendering library CMake

`ENABLE_METAL` compile definition gates `ComputeStream` typedef, `ImageBuffers` Metal fields, Metal branches in `AnariSceneRenderPass`, and pass dispatch code.

### Mutual exclusion

```cmake
if (TSD_USE_CUDA AND TSD_USE_METAL)
  message(FATAL_ERROR "TSD_USE_CUDA and TSD_USE_METAL are mutually exclusive")
endif()
```

## Key Trade-offs

| Decision | Choice | Trade-off |
|----------|--------|-----------|
| Extension scope | `ANARI_MTL_*` | Ships fast; promotable to KHR later |
| Frame buffer type | `MTL::Texture*` | Zero-copy from renderer; diverges from CUDA's pointer model |
| Array creation | Explicit `anariNewArray*MetalBuffer` | Clean API; requires new C functions (no address space detection) |
| Synchronization | Shared `MTL::CommandQueue*` | Implicit ordering; no events needed for single-queue use |
| Platform exclusivity | `TSD_USE_CUDA` xor `TSD_USE_METAL` | Simpler code; no simultaneous GPU backends |
| Math abstraction | Cross-platform headers | Modifies 2 shared headers; validated against all backends |

## Concerns

- **Math refactor risk:** Changing `tonemap_curves.h` and `color.h` affects CPU and CUDA backends. Must validate all three after the refactor.
- **`anari::DataType` in shaders:** `outputTransform` branches on ANARI data types. MSL kernels can't include ANARI headers — pass as `uint32_t`, replicate needed enum values as constants.
- **`helium::cvt_color_*` in outline/outputTransform:** Bit-packing utilities need MSL equivalents (unpack uint32 to 4 bytes and back).
- **Threadgroup sizing:** Metal's optimal threadgroup size varies by GPU. Using `pso->maxTotalThreadsPerThreadgroup()` is the standard heuristic.
- **Texture storage mode:** Frame textures are `StorageModePrivate`. Algorithms that need CPU readback (autoExposure result) use a small shared buffer internally.

## Implementation Order

1. **Math abstraction** — `vec_types.h`, `math_compat.h`, update existing headers. Validate CPU + CUDA. Hard gate.
2. **ANARI extensions in anari-mtl** — `MetalArray1D`/`2D`/`3D`, frame Metal channels, `mtl.commandQueue` property, extension advertisement.
3. **CMake + MetalContext + clearBuffers** — Shader compilation pipeline, metallib embedding, singleton, simplest kernel.
4. **toneMap** — First algorithm using shared math in MSL. Validates cross-compilation end-to-end.
5. **convertColorBuffer + outputTransform** — Two-texture patterns, ANARI type bridging.
6. **autoExposure** — Two-pass threadgroup reduction (most complex Metal-specific kernel).
7. **visualizeAOV + outline** — Multiple variants, neighborhood access.
8. **Pipeline integration** — `ComputeStream`, `ImageBuffers` Metal fields, `AnariSceneRenderPass` detection, pass dispatch.

Steps 2 and 3 are independent and can proceed in parallel.
