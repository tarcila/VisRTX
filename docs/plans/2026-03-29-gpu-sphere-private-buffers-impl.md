# GPU-Only Sphere Rendering with Private Metal Buffers — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the animatedParticlesMetal demo run fully GPU-resident with private Metal buffers, zero CPU readback from simulation to rendering.

**Architecture:** New Metal compute kernels in the mtl device handle sphere packing, AABB construction, index-gather, and bounds reduction on GPU. TSD's Metal runtime adds private buffer allocation and GPU blit. The demo switches from shared to private buffers with staging upload for initialization.

**Tech Stack:** C++17, Metal Shading Language, Metal-cpp, CMake, Jujutsu (jj) version control

**Design doc:** `docs/plans/2026-03-29-gpu-sphere-private-buffers-design.md`

**Two repos:**
- **mtl device:** `/Users/tarcila/Code/ANARI/mtl/interop` (Tasks 1-4)
- **TSD algorithms-library:** `/Users/tarcila/Code/ANARI/tsd/algorithms-library` (Tasks 5-8)

---

### Task 1: Sphere compute kernels in Conversion.metal

Add Metal compute kernels for GPU-side sphere finalize operations.

**Files:**
- Modify: `src/shaders/Conversion.metal` (mtl device, append at end ~line 272)

**Step 1: Add pack_spheres kernel**

Reads positions and optional radii/indices, writes packed float4(center.xyz, radius).
Uses function arguments to handle the four combinations: indexed/non-indexed × per-vertex-radii/global-radius.

```metal
// Pack sphere positions + radii into float4 (center.xyz, radius).
// indices: if non-null, idx = indices[tid]; otherwise idx = tid.
// radii: if non-null, r = radii[idx]; otherwise r = globalRadius.
kernel void pack_spheres(device const packed_float3 *positions [[buffer(0)]],
    device const float *radii [[buffer(1)]],
    device const uint *indices [[buffer(2)]],
    device float4 *dst [[buffer(3)]],
    constant uint &count [[buffer(4)]],
    constant float &globalRadius [[buffer(5)]],
    constant uint &hasRadii [[buffer(6)]],
    constant uint &hasIndices [[buffer(7)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= count)
        return;
    uint idx = hasIndices ? indices[tid] : tid;
    float3 c = float3(positions[idx]);
    float r = hasRadii ? radii[idx] : globalRadius;
    dst[tid] = float4(c, r);
}
```

**Step 2: Add compute_sphere_aabbs kernel**

```metal
// Compute per-sphere AABBs from packed float4 spheres.
// AABB layout: {minX, minY, minZ, maxX, maxY, maxZ} (6 floats per sphere).
kernel void compute_sphere_aabbs(device const float4 *spheres [[buffer(0)]],
    device float *aabbs [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= count)
        return;
    float4 s = spheres[tid];
    float3 c = s.xyz;
    float r = s.w;
    uint base = tid * 6;
    aabbs[base + 0] = c.x - r;
    aabbs[base + 1] = c.y - r;
    aabbs[base + 2] = c.z - r;
    aabbs[base + 3] = c.x + r;
    aabbs[base + 4] = c.y + r;
    aabbs[base + 5] = c.z + r;
}
```

**Step 3: Add gather_by_index kernel**

```metal
// Index-gather: dst[tid] = src[indices[tid]], copying strideWords uint32s per element.
kernel void gather_by_index(device const uint *src [[buffer(0)]],
    device const uint *indices [[buffer(1)]],
    device uint *dst [[buffer(2)]],
    constant uint &count [[buffer(3)]],
    constant uint &strideWords [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= count)
        return;
    uint srcBase = indices[tid] * strideWords;
    uint dstBase = tid * strideWords;
    for (uint w = 0; w < strideWords; w++)
        dst[dstBase + w] = src[srcBase + w];
}
```

**Step 4: Add compute_sphere_bounds kernel**

```metal
// Parallel bounding box reduction for packed float4 spheres (center.xyz, radius).
// Expands each sphere to center +/- radius before reducing.
// Result buffer: 8 uints [minX, minY, minZ, pad, maxX, maxY, maxZ, pad]
// Caller must initialize min slots to 0xFFFFFFFF and max slots to 0x00000000.
kernel void compute_sphere_bounds(device const float4 *spheres [[buffer(0)]],
    device atomic_uint *result [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tgSize [[threads_per_threadgroup]])
{
    threadgroup float3 localMin[256];
    threadgroup float3 localMax[256];

    float3 vMin = float3(FLT_MAX);
    float3 vMax = float3(-FLT_MAX);

    if (tid < count)
    {
        float4 s = spheres[tid];
        float3 c = s.xyz;
        float r = s.w;
        vMin = c - r;
        vMax = c + r;
    }

    localMin[lid] = vMin;
    localMax[lid] = vMax;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tgSize / 2; stride > 0; stride >>= 1)
    {
        if (lid < stride)
        {
            localMin[lid] = min(localMin[lid], localMin[lid + stride]);
            localMax[lid] = max(localMax[lid], localMax[lid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0)
    {
        for (int i = 0; i < 3; i++)
        {
            float minVal = localMin[0][i];
            float maxVal = localMax[0][i];
            uint minBits = as_type<uint>(minVal);
            uint maxBits = as_type<uint>(maxVal);
            uint minKey = (minBits >> 31) ? ~minBits : (minBits | 0x80000000u);
            uint maxKey = (maxBits >> 31) ? ~maxBits : (maxBits | 0x80000000u);
            atomic_fetch_min_explicit(&result[i], minKey, memory_order_relaxed);
            atomic_fetch_max_explicit(&result[i + 4], maxKey, memory_order_relaxed);
        }
    }
}
```

**Step 5: Build**

Run: `just build` (in mtl/interop)
Expected: compiles, links. Shader validation may fail if no Metal device (sandbox).

**Step 6: Commit**

```bash
jj describe -m "shaders: add sphere packing, AABB, gather, and bounds compute kernels"
jj new
```

---

### Task 2: MetalContext — sphere pipeline init and dispatch methods

Add pipeline states and dispatch methods for the sphere kernels.

**Files:**
- Modify: `src/gpu/MetalContext.h` (mtl device)
- Modify: `src/gpu/MetalContext.mm` (mtl device)

**Step 1: Add pipeline members to MetalContext.h**

After the existing remap pipeline members (around line 366 area, after
`m_remapQuadFixed16Vec3ToFloat3`), add:

```cpp
    MTL::ComputePipelineState *m_packSpheres{nullptr};
    MTL::ComputePipelineState *m_computeSphereAABBs{nullptr};
    MTL::ComputePipelineState *m_gatherByIndex{nullptr};
    MTL::ComputePipelineState *m_computeSphereBounds{nullptr};
```

**Step 2: Add dispatch method declarations to MetalContext.h**

After the existing dispatch methods (around line 207), add:

```cpp
    MTL::Buffer *dispatchPackSpheres(MTL::Buffer *positions,
        MTL::Buffer *radii,
        MTL::Buffer *indices,
        float globalRadius,
        uint32_t count);
    MTL::Buffer *dispatchComputeSphereAABBs(MTL::Buffer *packedSpheres, uint32_t count);
    MTL::Buffer *dispatchGatherByIndex(MTL::Buffer *src,
        MTL::Buffer *indices,
        uint32_t count,
        uint32_t strideWords);
    BoundsResult dispatchComputeSphereBounds(MTL::Buffer *packedSpheres, uint32_t count);
```

**Step 3: Initialize pipelines in MetalContext.mm**

In the `makePipeline` block (find where `m_computeBoundsPipeline` is
initialized), add after the last line:

```cpp
        m_packSpheres = makePipeline("pack_spheres");
        m_computeSphereAABBs = makePipeline("compute_sphere_aabbs");
        m_gatherByIndex = makePipeline("gather_by_index");
        m_computeSphereBounds = makePipeline("compute_sphere_bounds");
```

**Step 4: Release pipelines in destructor**

Find where existing conversion pipelines are released. Add:

```cpp
    if (m_packSpheres) m_packSpheres->release();
    if (m_computeSphereAABBs) m_computeSphereAABBs->release();
    if (m_gatherByIndex) m_gatherByIndex->release();
    if (m_computeSphereBounds) m_computeSphereBounds->release();
```

**Step 5: Implement dispatch methods in MetalContext.mm**

Add after the existing dispatch method implementations:

```cpp
MTL::Buffer *MetalContext::dispatchPackSpheres(MTL::Buffer *positions,
    MTL::Buffer *radii,
    MTL::Buffer *indices,
    float globalRadius,
    uint32_t count)
{
    auto *dst = m_device->newBuffer(
        size_t(count) * sizeof(float) * 4, MTL::ResourceStorageModeShared);
    auto *enc = m_conversionCmdBuf->computeCommandEncoder();
    enc->setComputePipelineState(m_packSpheres);
    enc->setBuffer(positions, 0, 0);
    if (radii)
        enc->setBuffer(radii, 0, 1);
    if (indices)
        enc->setBuffer(indices, 0, 2);
    enc->setBuffer(dst, 0, 3);
    enc->setBytes(&count, sizeof(uint32_t), 4);
    enc->setBytes(&globalRadius, sizeof(float), 5);
    uint32_t hasRadii = radii ? 1 : 0;
    uint32_t hasIndices = indices ? 1 : 0;
    enc->setBytes(&hasRadii, sizeof(uint32_t), 6);
    enc->setBytes(&hasIndices, sizeof(uint32_t), 7);
    auto tgSize = std::min(m_packSpheres->maxTotalThreadsPerThreadgroup(), NS::UInteger(256));
    enc->dispatchThreads(
        MTL::Size(count, 1, 1), MTL::Size(std::min(NS::UInteger(count), tgSize), 1, 1));
    enc->endEncoding();
    return dst;
}

MTL::Buffer *MetalContext::dispatchComputeSphereAABBs(MTL::Buffer *packedSpheres, uint32_t count)
{
    // 6 floats per AABB: minX, minY, minZ, maxX, maxY, maxZ
    auto *dst = m_device->newBuffer(
        size_t(count) * 6 * sizeof(float), MTL::ResourceStorageModeShared);
    auto *enc = m_conversionCmdBuf->computeCommandEncoder();
    enc->setComputePipelineState(m_computeSphereAABBs);
    enc->setBuffer(packedSpheres, 0, 0);
    enc->setBuffer(dst, 0, 1);
    enc->setBytes(&count, sizeof(uint32_t), 2);
    auto tgSize = std::min(m_computeSphereAABBs->maxTotalThreadsPerThreadgroup(), NS::UInteger(256));
    enc->dispatchThreads(
        MTL::Size(count, 1, 1), MTL::Size(std::min(NS::UInteger(count), tgSize), 1, 1));
    enc->endEncoding();
    return dst;
}

MTL::Buffer *MetalContext::dispatchGatherByIndex(MTL::Buffer *src,
    MTL::Buffer *indices,
    uint32_t count,
    uint32_t strideWords)
{
    auto *dst = m_device->newBuffer(
        size_t(count) * strideWords * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto *enc = m_conversionCmdBuf->computeCommandEncoder();
    enc->setComputePipelineState(m_gatherByIndex);
    enc->setBuffer(src, 0, 0);
    enc->setBuffer(indices, 0, 1);
    enc->setBuffer(dst, 0, 2);
    enc->setBytes(&count, sizeof(uint32_t), 3);
    enc->setBytes(&strideWords, sizeof(uint32_t), 4);
    auto tgSize = std::min(m_gatherByIndex->maxTotalThreadsPerThreadgroup(), NS::UInteger(256));
    enc->dispatchThreads(
        MTL::Size(count, 1, 1), MTL::Size(std::min(NS::UInteger(count), tgSize), 1, 1));
    enc->endEncoding();
    return dst;
}

MetalContext::BoundsResult MetalContext::dispatchComputeSphereBounds(
    MTL::Buffer *packedSpheres, uint32_t count)
{
    auto *result = m_device->newBuffer(8 * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto *data = static_cast<uint32_t *>(result->contents());
    for (int i = 0; i < 4; i++)
        data[i] = 0xFFFFFFFFu;
    for (int i = 4; i < 8; i++)
        data[i] = 0u;

    auto *enc = m_conversionCmdBuf->computeCommandEncoder();
    enc->setComputePipelineState(m_computeSphereBounds);
    enc->setBuffer(packedSpheres, 0, 0);
    enc->setBuffer(result, 0, 1);
    enc->setBytes(&count, sizeof(uint32_t), 2);
    auto tgSize = std::min(m_computeSphereBounds->maxTotalThreadsPerThreadgroup(), NS::UInteger(256));
    enc->dispatchThreads(
        MTL::Size(count, 1, 1), MTL::Size(std::min(NS::UInteger(count), tgSize), 1, 1));
    enc->endEncoding();

    return {result};
}
```

**Step 6: Build**

Run: `just build` (in mtl/interop)
Expected: compiles and links.

**Step 7: Commit**

```bash
jj describe -m "gpu: add sphere pack/AABB/gather/bounds dispatch methods"
jj new
```

---

### Task 3: Sphere::finalize() GPU path

Add the GPU-resident path to Sphere::finalize() that activates when
vertex.position is a private MetalArray.

**Files:**
- Modify: `src/geometry/Sphere.cpp` (mtl device)

**Step 1: Restructure finalize into CPU and GPU paths**

The key change: detect private MetalArray on positions, then branch. The GPU
path uses the dispatch methods from Task 2. The CPU path remains unchanged.

Replace the body of `Sphere::finalize()` (lines 57-250) with the two-path
structure. The full replacement is large, so here's the structure:

```cpp
void Sphere::finalize()
{
    if (!m_vertexPositions)
    {
        reportMessage(ANARI_SEVERITY_WARNING, "sphere: missing 'vertex.position'");
        return;
    }

    auto &ctx = deviceState()->metalContext;

    // Release old resources (same as current code, lines 67-84)
    // ...

    auto centerCount = m_vertexPositions->totalSize();
    uint32_t sphereCount = m_indices
        ? static_cast<uint32_t>(m_indices->totalSize())
        : static_cast<uint32_t>(centerCount);

    bool gpuPath = false;
    if (auto *mtl = dynamic_cast<MetalArray1D *>(&*m_vertexPositions))
        gpuPath = mtl->isPrivate();

    if (gpuPath)
    {
        // --- GPU path ---
        auto *posBuffer = dynamic_cast<MetalArray1D *>(&*m_vertexPositions)->metalBuffer();
        auto *radiiBuffer = m_vertexRadii
            ? getOrUploadBuffer(ctx, &*m_vertexRadii, m_vertexRadii->totalSize() * sizeof(float))
            : nullptr;
        MTL::Buffer *indexBuffer = nullptr;
        if (m_indices)
            indexBuffer = getOrUploadBuffer(ctx, &*m_indices, m_indices->totalSize() * sizeof(uint32_t));

        ctx.beginConversions();

        // 1. Pack spheres
        m_sphereBuffer = ctx.dispatchPackSpheres(
            posBuffer, radiiBuffer, indexBuffer, m_globalRadius, sphereCount);

        // 2. Compute AABBs
        auto *aabbBuffer = ctx.dispatchComputeSphereAABBs(m_sphereBuffer, sphereCount);

        // 3. Gather-by-index for colors/attributes (only if indexed)
        if (m_vertexColors && m_indices)
        {
            auto stride = anari::sizeOf(m_vertexColors->elementType());
            auto strideWords = static_cast<uint32_t>(stride / sizeof(uint32_t));
            auto *srcBuf = getOrUploadBuffer(ctx, &*m_vertexColors,
                m_vertexColors->totalSize() * stride);
            if (strideWords > 0 && stride % sizeof(uint32_t) == 0)
                m_colorBuffer = ctx.dispatchGatherByIndex(srcBuf, indexBuffer, sphereCount, strideWords);
        }
        else if (m_vertexColors)
        {
            auto stride = anari::sizeOf(m_vertexColors->elementType());
            m_colorBuffer = getOrUploadBuffer(ctx, &*m_vertexColors,
                m_vertexColors->totalSize() * stride);
        }

        for (int a = 0; a < 4; a++)
        {
            if (!m_vertexAttrs[a])
                continue;
            auto stride = anari::sizeOf(m_vertexAttrs[a]->elementType());
            auto strideWords = static_cast<uint32_t>(stride / sizeof(uint32_t));
            auto *srcBuf = getOrUploadBuffer(ctx, &*m_vertexAttrs[a],
                m_vertexAttrs[a]->totalSize() * stride);
            if (m_indices && strideWords > 0 && stride % sizeof(uint32_t) == 0)
                m_attrBuffers[a] = ctx.dispatchGatherByIndex(srcBuf, indexBuffer, sphereCount, strideWords);
            else
                m_attrBuffers[a] = srcBuf;
        }

        // 4. Bounds reduction
        auto boundsResult = ctx.dispatchComputeSphereBounds(m_sphereBuffer, sphereCount);

        ctx.endConversions(true);

        // 5. Build BLAS from GPU-computed AABBs
        m_blas = ctx.buildBoundingBoxBLAS(
            aabbBuffer->contents(), sphereCount * 6 * sizeof(float), sphereCount);
        ctx.deferRelease(aabbBuffer);

        // 6. Decode bounds
        m_bounds = boundsResult.decode();
        ctx.deferRelease(boundsResult.readbackBuffer);
    }
    else
    {
        // --- CPU path (existing code, lines 86-190, unchanged) ---
        auto *centers = m_vertexPositions->dataAs<float3>();
        auto *radii = m_vertexRadii ? m_vertexRadii->dataAs<float>() : nullptr;

        uint32_t localSphereCount = 0;
        const uint32_t *indices = nullptr;
        // ... existing index resolution, packing, AABB, color/attr upload, bounds ...
    }

    // Primitive colors/attrs (not indexed by vertex indices — uploaded directly)
    if (m_primitiveColors)
    {
        m_primitiveColorBuffer = getOrUploadBuffer(ctx, &*m_primitiveColors,
            m_primitiveColors->totalSize() * anari::sizeOf(m_primitiveColors->elementType()));
    }
    // ... primitive attr upload ...

    // Record setup (same for both paths, lines 192-232)
    m_record = {};
    m_record.vertexBufferAddress = ctx.bufferGPUAddress(m_sphereBuffer);
    // ... rest of record setup ...

    // Privatized array cleanup (same for both paths, lines 234-249)
}
```

**Important implementation notes:**
- The GPU path allocates `aabbBuffer` and `boundsResult.readbackBuffer` as
  shared storage (done by the dispatch methods). After `endConversions(true)`,
  their `contents()` is readable.
- `m_sphereBuffer` is shared storage (allocated by `dispatchPackSpheres`) —
  the GPU address is obtained via `bufferGPUAddress` as usual.
- The `getOrUploadBuffer` calls for colors/attrs/indices handle MetalArrays
  transparently (return the buffer directly if MetalArray, upload otherwise).
- Primitive colors/attrs are NOT indexed by vertex indices — they're per-sphere
  already. Use `getOrUploadBuffer` directly (works on both paths).

**Step 2: Build and test**

Run: `just build` (in mtl/interop)
Expected: compiles and links.

**Step 3: Commit**

```bash
jj describe -m "sphere: add GPU-resident finalize path for private Metal buffers"
jj new
```

---

### Task 4: Update mtl device documentation

**Files:**
- Modify: `IMPLEMENTATION.md` (mtl device, root)

**Step 1: Add note about Sphere GPU path**

In the "GPU Format Conversions" section, add after the existing paragraph:

```
Sphere geometry additionally performs GPU-side sphere packing (positions +
radii → float4), AABB construction, attribute index-gathering, and bounding
box reduction when vertex data arrives in private Metal buffers.
```

**Step 2: Commit**

```bash
jj describe -m "docs: note sphere GPU finalize path"
jj new
```

---

### Task 5: TSD runtime — private buffer and blit support

Add `newPrivateBuffer()` and `blitToBuffer()` to the TSD Metal runtime.

**Files:**
- Modify: `tsd/src/tsd/algorithms/metal/runtime.hpp` (TSD repo)
- Modify: `tsd/src/tsd/algorithms/metal/runtime.cpp` (TSD repo)

**Step 1: Add declarations to runtime.hpp**

After the existing `void *bufferContents(void *buffer);` line (around line 49),
add:

```cpp
// Allocate a MTL::Buffer with StorageModePrivate; returns opaque handle.
// GPU-only: bufferContents() returns nullptr for these buffers.
void *newPrivateBuffer(size_t bytes);

// GPU blit: copy |bytes| from |src| to |dst| via Metal blit encoder.
// Synchronous: commits and waits for completion before returning.
// Typical use: shared staging buffer → private buffer.
void blitToBuffer(void *src, void *dst, size_t bytes);
```

**Step 2: Implement in runtime.cpp**

After the existing `bufferContents()` implementation (around line 188), add:

```cpp
void *newPrivateBuffer(size_t bytes)
{
    auto &ctx = MetalContext::instance();
    auto *buf = ctx.device()->newBuffer(bytes, MTL::ResourceStorageModePrivate);
    return buf;
}

void blitToBuffer(void *src, void *dst, size_t bytes)
{
    if (!src || !dst || bytes == 0)
        return;
    auto &ctx = MetalContext::instance();
    auto *cmdBuf = ctx.defaultQueue()->commandBuffer();
    auto *blit = cmdBuf->blitCommandEncoder();
    blit->copyFromBuffer(
        static_cast<MTL::Buffer *>(src), 0,
        static_cast<MTL::Buffer *>(dst), 0,
        bytes);
    blit->endEncoding();
    cmdBuf->commit();
    cmdBuf->waitUntilCompleted();
}
```

**Step 3: Build**

Run: `just build` (in TSD algorithms-library)
Expected: compiles.

**Step 4: Commit**

```bash
jj describe -m "metal runtime: add newPrivateBuffer and synchronous blitToBuffer"
jj new
```

---

### Task 6: TSD Array — allow null metalData for private buffers

Relax the Metal Array constructor to accept null `metalData`.

**Files:**
- Modify: `tsd/src/tsd/scene/objects/Array.hpp` (TSD repo, line 28 comment)
- Modify: `tsd/src/tsd/scene/objects/Array.cpp` (TSD repo, lines 360-363)

**Step 1: Update the constructor validation**

In `Array.cpp`, the Metal constructor (around line 360) currently throws:

```cpp
if (!metalBuffer || !metalData)
    throw std::runtime_error("Metal array requires non-null buffer and data!");
```

Change to only require non-null buffer:

```cpp
if (!metalBuffer)
    throw std::runtime_error("Metal array requires non-null buffer!");
```

**Step 2: Update the MemoryKind::METAL comment**

In `Array.hpp`, line 28, change:

```cpp
METAL, // Backed by a caller-provided MTL::Buffer (shared memory)
```

to:

```cpp
METAL, // Backed by a caller-provided MTL::Buffer (any storage mode)
```

**Step 3: Build**

Run: `just build` (in TSD algorithms-library)

**Step 4: Commit**

```bash
jj describe -m "array: allow null metalData for private Metal buffers"
jj new
```

---

### Task 7: Demo — switch to private buffers

Rewire the animatedParticlesMetal demo to use private Metal buffers.

**Files:**
- Modify: `tsd/apps/interactive/demos/animatedParticlesMetal/SimulationControls.h`
- Modify: `tsd/apps/interactive/demos/animatedParticlesMetal/SimulationControls.cpp`

**Step 1: Simplify header — remove HOST mirror arrays**

In `SimulationControls.h`, remove the HOST mirror members and the GPU interop
toggle. The members to remove:

```cpp
tsd::scene::ObjectUsePtr<tsd::scene::Array> m_dataPoints;       // remove
tsd::scene::ObjectUsePtr<tsd::scene::Array> m_dataDistances;    // remove
tsd::scene::ObjectUsePtr<tsd::scene::Array> m_dataVelocities;   // remove
bool m_useGPUInterop{true};                                      // remove
```

Keep the Metal array members and rename them (drop the `Metal` suffix since
they're now the only arrays):

```cpp
tsd::scene::ObjectUsePtr<tsd::scene::Array> m_dataPoints;       // was m_dataPointsMetal
tsd::scene::ObjectUsePtr<tsd::scene::Array> m_dataDistances;    // was m_dataDistancesMetal
tsd::scene::ObjectUsePtr<tsd::scene::Array> m_dataVelocities;   // was m_dataVelocitiesMetal
```

**Step 2: Update remakeDataArrays()**

Replace the body with private buffer allocation:

```cpp
void SimulationControls::remakeDataArrays()
{
    namespace mtl = tsd::algorithms::metal;

    auto &scene = appContext()->tsd.scene;
    const int numParticles =
        m_particlesPerSide * m_particlesPerSide * m_particlesPerSide;
    releaseMetalBuffers();

    const size_t vec3Bytes = numParticles * sizeof(tsd::math::float3);
    const size_t floatBytes = numParticles * sizeof(float);

    m_posBuffer = mtl::newPrivateBuffer(vec3Bytes);
    m_velBuffer = mtl::newPrivateBuffer(vec3Bytes);
    m_distBuffer = mtl::newPrivateBuffer(floatBytes);

    m_dataPoints = scene.createArrayMetal(
        ANARI_FLOAT32_VEC3, numParticles, m_posBuffer, nullptr);
    m_dataDistances = scene.createArrayMetal(
        ANARI_FLOAT32, numParticles, m_distBuffer, nullptr);
    m_dataVelocities = scene.createArrayMetal(
        ANARI_FLOAT32_VEC3, numParticles, m_velBuffer, nullptr);

    m_dataBhPoints = scene.createArray(ANARI_FLOAT32_VEC3, 2);
}
```

**Step 3: Update resetSimulation()**

Use staging buffers for initialization:

```cpp
void SimulationControls::resetSimulation()
{
    namespace mtl = tsd::algorithms::metal;

    m_playing = false;
    m_angle = 0.f;

    remakeDataArrays();
    updateBhPoints();

    const int numParticles =
        m_particlesPerSide * m_particlesPerSide * m_particlesPerSide;
    const size_t vec3Bytes = numParticles * sizeof(tsd::math::float3);
    const size_t floatBytes = numParticles * sizeof(float);

    // Allocate staging buffers (shared, CPU-writable)
    auto *stagingPos = mtl::newSharedBuffer(vec3Bytes);
    auto *stagingVel = mtl::newSharedBuffer(vec3Bytes);
    auto *stagingDist = mtl::newSharedBuffer(floatBytes);

    auto *positions = static_cast<tsd::math::float3 *>(mtl::bufferContents(stagingPos));
    auto *velocities = static_cast<tsd::math::float3 *>(mtl::bufferContents(stagingVel));
    auto *distances = static_cast<float *>(mtl::bufferContents(stagingDist));

    // Initialize velocities
    if (m_randomizeInitialVelocities)
    {
        std::mt19937 rng;
        rng.seed(0);
        std::normal_distribution<float> dist(-0.1f, 0.1f);
        std::for_each((float *)velocities,
            (float *)(velocities + numParticles),
            [&](auto &v) { v = dist(rng) * 255; });
    }
    else
    {
        std::fill(velocities, velocities + numParticles, tsd::math::float3(0.f));
    }

    // Initialize positions on a grid
    const float d = 2.0f / m_particlesPerSide;
    size_t i = 0;
    for (int x = 0; x < m_particlesPerSide; x++)
    {
        for (int y = 0; y < m_particlesPerSide; y++)
        {
            for (int z = 0; z < m_particlesPerSide; z++)
            {
                auto p = tsd::math::float3(d * x - 1.f, d * y - 1.f, d * z - 1.f);
                positions[i] = p;
                distances[i] = tsd::math::length(p);
                i++;
            }
        }
    }

    // Blit staging → private
    mtl::blitToBuffer(stagingPos, m_posBuffer, vec3Bytes);
    mtl::blitToBuffer(stagingVel, m_velBuffer, vec3Bytes);
    mtl::blitToBuffer(stagingDist, m_distBuffer, floatBytes);

    // Release staging
    mtl::releaseBuffer(stagingPos);
    mtl::releaseBuffer(stagingVel);
    mtl::releaseBuffer(stagingDist);

    // Set geometry to use Metal arrays
    m_particleGeom->setParameterObject("vertex.position", *m_dataPoints);
    m_particleGeom->setParameterObject("vertex.attribute0", *m_dataDistances);

    updateColorMapScale();
}
```

**Step 4: Simplify iterateSimulation()**

Remove CPU readback and map/unmap:

```cpp
void SimulationControls::iterateSimulation()
{
    m_angle += m_rotationSpeed * 1e-4f;
    auto [bh1, bh2] = updateBhPoints();

    const int numParticles =
        m_particlesPerSide * m_particlesPerSide * m_particlesPerSide;

    tsd::demo::particlesComputeTimestepMetal(numParticles,
        m_posBuffer,
        m_velBuffer,
        m_distBuffer,
        tsd::math::float3(bh1.x, bh1.y, bh1.z),
        tsd::math::float3(bh2.x, bh2.y, bh2.z),
        m_params);
}
```

**Step 5: Update buildUI()**

Remove the "use GPU array interop" checkbox (the `if (ImGui::Checkbox("use
GPU array interop", ...))` block around lines 70-78). Also remove the
conditional in the reset block that checks `m_useGPUInterop`.

**Step 6: Build and test**

Run: `just build` (in TSD algorithms-library)
Then run the demo with the mtl device to verify particles render correctly.

**Step 7: Commit**

```bash
jj describe -m "demo: switch animatedParticlesMetal to private Metal buffers"
jj new
```

---

### Task 8: End-to-end validation

**Step 1: Build both repos**

```bash
cd /Users/tarcila/Code/ANARI/mtl/interop && just build
cd /Users/tarcila/Code/ANARI/tsd/algorithms-library && just build
```

**Step 2: Run the demo**

Launch the animatedParticlesMetal demo with the mtl device. Verify:
- Particles render correctly on screen
- Pressing "reset" re-initializes the grid
- Pressing play animates the simulation
- No crashes (which would indicate a CPU read of private buffer)

**Step 3: Performance comparison**

Compare frame times with private buffers vs. a temporary revert to shared
buffers. The private path should show measurably better performance since
`Sphere::finalize()` no longer reads positions on CPU.

**Step 4: Format**

```bash
cd /Users/tarcila/Code/ANARI/mtl/interop && nix fmt
cd /Users/tarcila/Code/ANARI/tsd/algorithms-library && nix fmt
```

---

## Execution Notes

- Tasks 1-4 are in the **mtl device** repo (`/Users/tarcila/Code/ANARI/mtl/interop`)
- Tasks 5-7 are in the **TSD** repo (`/Users/tarcila/Code/ANARI/tsd/algorithms-library`)
- Tasks 1-2 (kernels + MetalContext) must be done before Task 3 (Sphere finalize)
- Tasks 5-6 (runtime + Array) must be done before Task 7 (demo)
- Tasks 1-2 and 5-6 are independent and can be done in parallel across repos
- Task 8 requires all other tasks complete
