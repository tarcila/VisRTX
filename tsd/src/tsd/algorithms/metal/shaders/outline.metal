#include <metal_stdlib>
using namespace metal;

kernel void outlineKernel(
    texture2d<uint, access::read> objectId [[texture(0)]],
    texture2d<float, access::read_write> color [[texture(1)]],
    constant uint32_t &outlineId [[buffer(0)]],
    uint2 tid [[thread_position_in_grid]])
{
    uint32_t width = color.get_width();
    uint32_t height = color.get_height();

    if (tid.x >= width || tid.y >= height)
        return;

    int cnt = 0;
    for (uint fy = max(0u, tid.y - 1); fy <= min(height - 1, tid.y + 1); fy++) {
        for (uint fx = max(0u, tid.x - 1); fx <= min(width - 1, tid.x + 1); fx++) {
            if (objectId.read(uint2(fx, fy)).r == outlineId)
                cnt++;
        }
    }

    if (cnt > 1 && cnt < 8) {
        float4 c_in = color.read(tid);
        float4 c_h = float4(1.0f, 0.5f, 0.0f, 1.0f);
        float4 c_out = mix(c_in, c_h, 0.8f);
        color.write(c_out, tid);
    }
}
