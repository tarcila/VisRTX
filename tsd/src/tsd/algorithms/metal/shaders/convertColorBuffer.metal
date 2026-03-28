#include <metal_stdlib>
using namespace metal;

kernel void convertFloatToUint8Kernel(
    texture2d<float, access::read> input [[texture(0)]],
    device uint32_t *output [[buffer(0)]],
    uint2 tid [[thread_position_in_grid]])
{
    if (tid.x >= input.get_width() || tid.y >= input.get_height())
        return;

    float4 c = saturate(input.read(tid));

    uint32_t r = uint32_t(c.r * 255.0f);
    uint32_t g = uint32_t(c.g * 255.0f);
    uint32_t b = uint32_t(c.b * 255.0f);
    uint32_t a = uint32_t(c.a * 255.0f);

    uint32_t idx = tid.y * input.get_width() + tid.x;
    output[idx] = r | (g << 8) | (b << 16) | (a << 24);
}

kernel void convertFloatToBGRA8Kernel(
    texture2d<float, access::read> input [[texture(0)]],
    texture2d<float, access::write> output [[texture(1)]],
    uint2 tid [[thread_position_in_grid]])
{
    if (tid.x >= input.get_width() || tid.y >= input.get_height())
        return;

    float4 c = saturate(input.read(tid));
    output.write(c, tid);
}
