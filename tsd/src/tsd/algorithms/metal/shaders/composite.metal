#include <metal_stdlib>
using namespace metal;

kernel void compositeByDepthKernel(
    texture2d<float, access::read> overlayColor [[texture(0)]],
    texture2d<float, access::read> overlayDepth [[texture(1)]],
    texture2d<float, access::read_write> mainColor [[texture(2)]],
    texture2d<float, access::read_write> mainDepth [[texture(3)]],
    uint2 tid [[thread_position_in_grid]])
{
    if (tid.x >= mainColor.get_width() || tid.y >= mainColor.get_height())
        return;

    float oDepth = overlayDepth.read(tid).r;
    float mDepth = mainDepth.read(tid).r;

    if (oDepth < mDepth) {
        mainColor.write(overlayColor.read(tid), tid);
        mainDepth.write(float4(oDepth, 0, 0, 0), tid);
    }
}
