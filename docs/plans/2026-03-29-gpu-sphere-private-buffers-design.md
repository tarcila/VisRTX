# GPU-Only Sphere Rendering with Private Metal Buffers

## Goal

Make the animatedParticlesMetal demo run fully GPU-resident: private Metal
buffers from simulation to rendering with zero CPU readback. Requires GPU
compute kernels for Sphere geometry finalize in the mtl device, private buffer
support in TSD's Metal runtime, and demo rewiring.

## Decisions

| Question | Decision |
|----------|----------|
| Buffer storage | Private (GPU-only, crash on CPU read) |
| Init strategy | Staging upload (shared temp -> blit -> private) |
| Synchronization | Fully synchronous (no double buffering) |
| Sphere finalize | GPU compute path for private buffers, CPU path unchanged for host |
| TSD runtime additions | `newPrivateBuffer()`, `blitToBuffer()` (synchronous) |
| TSD Array | Allow null `metalData` for private Metal arrays |
| Demo CPU fallback | Remove or gate behind flag |

## Section 1: GPU Sphere finalize -- new compute kernels

New Metal compute kernels in `Conversion.metal` (mtl device):

- `pack_spheres(positions, radii, indices, dst, globalRadius, count)` -- reads
  positions[idx] and radii[idx], writes float4(center.xyz, radius). Handles
  optional indices (null = identity) and optional radii (null = global radius).
- `compute_sphere_aabbs(packed_float4, dst_aabbs, count)` -- center +/- radius
  per sphere.
- `gather_by_index(src, indices, dst, count, strideWords)` -- generic
  index-gather for re-indexing colors/attributes. When no index buffer, source
  buffer is used directly.
- Bounds variant for packed float4 spheres (xyz +/- w).

## Section 2: Sphere::finalize() GPU path

Two-path structure triggered by `isPrivate()` on vertex.position:

1. `ctx.beginConversions()`
2. Pack spheres on GPU (positions + radii -> float4)
3. Compute AABBs on GPU
4. Gather-by-index for colors/attributes (if indexed)
5. Bounds reduction on packed float4 spheres
6. `ctx.endConversions(true)` -- wait for AABBs and bounds
7. Build BLAS from GPU-computed AABBs (shared buffer, CPU-accessible after wait)
8. Decode bounds result

Shared/host arrays use existing CPU path unchanged.

## Section 3: TSD runtime -- private buffer support

Two new functions in `tsd::algorithms::metal` (runtime.hpp):

- `void *newPrivateBuffer(size_t bytes)` -- allocates StorageModePrivate buffer
- `void blitToBuffer(void *src, void *dst, size_t bytes)` -- GPU blit,
  synchronous (commit + wait)

Existing functions (`releaseBuffer`, `dispatchKernel`, etc.) already work with
any storage mode.

## Section 4: TSD Array -- private Metal buffer support

- Metal Array constructor: allow null `metalData` (GPU-only)
- `map()` / `dataAs<>()`: return null for private Metal arrays
- `makeANARIObject()`: works as-is (passes buffer handle regardless of storage)
- `Scene::createArrayMetal()`: allow null `metalData`

No new `MemoryKind`. `METAL` covers both shared and private -- the distinction
is the buffer itself.

## Section 5: Demo changes

### remakeDataArrays()
- `newPrivateBuffer()` instead of `newSharedBuffer()`
- `nullptr` as `metalData` in `createArrayMetal()`
- Drop HOST mirror arrays

### resetSimulation()
- Allocate temporary shared staging buffers
- Fill on CPU (existing grid init)
- `blitToBuffer(staging, private, bytes)` for each
- Release staging buffers

### iterateSimulation()
- Remove CPU memcpy-back branch
- Remove map/unmap on Metal arrays
- Pass private buffers directly to compute kernel

### UI
- Remove "use GPU array interop" checkbox (or repurpose for benchmarking)

## Section 6: What stays the same

- Particle simulation kernel (`particle_system.metal`) -- reads/writes private
  storage natively
- `ANARI_MTL_ARRAY_METAL` extension API
- `MetalArray1D/2D/3D` in mtl device -- already supports private buffers
- Other geometry types -- unaffected
- `makeANARIObject()` dlsym path -- already passes buffer handle
