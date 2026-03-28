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
    constant uint32_t &numSamples [[buffer(1)]],
    constant uint32_t &stride [[buffer(2)]],
    constant uint32_t &texWidth [[buffer(3)]],
    threadgroup float *shared [[threadgroup(0)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tgSize [[threads_per_threadgroup]],
    uint tgId [[threadgroup_position_in_grid]])
{
    float val = 0.0f;
    if (tid < numSamples) {
        uint32_t pixelIdx = tid * stride;
        uint32_t px = pixelIdx % texWidth;
        uint32_t py = pixelIdx / texWidth;
        float4 pixel = hdrColor.read(uint2(px, py));
        float lum = max(luminance(pixel.x, pixel.y, pixel.z), MIN_LUMINANCE);
        val = log2(lum);
    }

    shared[lid] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tgSize / 2; s > 0; s >>= 1) {
        if (lid < s)
            shared[lid] += shared[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0)
        partials[tgId] = shared[0];
}

kernel void sumLogLuminancePass2(
    device const float *partials [[buffer(0)]],
    device float *result [[buffer(1)]],
    constant uint32_t &numPartials [[buffer(2)]],
    threadgroup float *shared [[threadgroup(0)]],
    uint lid [[thread_position_in_threadgroup]],
    uint tgSize [[threads_per_threadgroup]])
{
    float val = (lid < numPartials) ? partials[lid] : 0.0f;
    shared[lid] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tgSize / 2; s > 0; s >>= 1) {
        if (lid < s)
            shared[lid] += shared[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0)
        result[0] = shared[0];
}
