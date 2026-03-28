#include <metal_stdlib>
using namespace metal;

#include "vec_types.h"
#include "math_compat.h"
#include "color.h"

using namespace tsd::algorithms::math;

// ANARI_FLOAT32_VEC4 = 13 (from anari_enums.h)
constant constexpr uint32_t ANARI_FLOAT32_VEC4_VALUE = 13;

kernel void outputTransformKernel(
    texture2d<float, access::read> hdrColor [[texture(0)]],
    texture2d<float, access::read> colorIn [[texture(1)]],
    texture2d<float, access::read_write> colorOut [[texture(2)]],
    constant float &invGamma [[buffer(0)]],
    constant uint32_t &colorFormat [[buffer(1)]],
    uint2 tid [[thread_position_in_grid]])
{
    if (tid.x >= colorOut.get_width() || tid.y >= colorOut.get_height())
        return;

    float4 c;

    if (colorFormat == ANARI_FLOAT32_VEC4_VALUE) {
        c = hdrColor.read(tid);
    } else {
        c = colorIn.read(tid);
    }

    float3 encoded = linearToGamma(float3(c.x, c.y, c.z), invGamma);
    colorOut.write(float4(encoded, c.w), tid);
}
