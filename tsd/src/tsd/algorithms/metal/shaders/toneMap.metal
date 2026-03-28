#include <metal_stdlib>
using namespace metal;

#include "vec_types.h"
#include "math_compat.h"
#include "tonemap_curves.h"

using namespace tsd::algorithms::math;

constant constexpr uint OP_NONE = 0;
constant constexpr uint OP_REINHARD = 1;
constant constexpr uint OP_ACES = 2;
constant constexpr uint OP_HABLE = 3;
constant constexpr uint OP_KHRONOS_PBR_NEUTRAL = 4;
constant constexpr uint OP_AGX = 5;

kernel void toneMapKernel(
    texture2d<float, access::read_write> hdrColor [[texture(0)]],
    constant float &exposureScale [[buffer(0)]],
    constant uint &op [[buffer(1)]],
    uint2 tid [[thread_position_in_grid]])
{
    if (tid.x >= hdrColor.get_width() || tid.y >= hdrColor.get_height())
        return;

    float4 pixel = hdrColor.read(tid);
    float3 c = pixel.xyz * exposureScale;

    switch (op) {
    case OP_NONE:
        break;
    case OP_REINHARD:
        c = tonemapReinhard(max0(c));
        break;
    case OP_ACES:
        c = tonemapACES(max0(c));
        break;
    case OP_HABLE:
        c = tonemapHable(max0(c));
        break;
    case OP_KHRONOS_PBR_NEUTRAL:
        c = tonemapKhronosPbrNeutral(max0(c));
        break;
    case OP_AGX:
        c = tonemapAgX(max0(c));
        break;
    }

    hdrColor.write(float4(c, pixel.w), tid);
}
