#include <metal_stdlib>
using namespace metal;

#include "vec_types.h"
#include "math_compat.h"
#include "color.h"

using namespace tsd::algorithms::math;

kernel void visualizeObjectIdKernel(
    texture2d<uint, access::read> objectId [[texture(0)]],
    texture2d<float, access::read_write> color [[texture(1)]],
    uint2 tid [[thread_position_in_grid]])
{
    if (tid.x >= color.get_width() || tid.y >= color.get_height())
        return;

    uint32_t id = objectId.read(tid).r;
    float3 c = makeRandomColor(id);
    color.write(float4(c, 1.0f), tid);
}

kernel void visualizePrimitiveIdKernel(
    texture2d<uint, access::read> primitiveId [[texture(0)]],
    texture2d<float, access::read_write> color [[texture(1)]],
    uint2 tid [[thread_position_in_grid]])
{
    if (tid.x >= color.get_width() || tid.y >= color.get_height())
        return;

    uint32_t id = primitiveId.read(tid).r;
    float3 c = makeRandomColor(id);
    color.write(float4(c, 1.0f), tid);
}

kernel void visualizeInstanceIdKernel(
    texture2d<uint, access::read> instanceId [[texture(0)]],
    texture2d<float, access::read_write> color [[texture(1)]],
    uint2 tid [[thread_position_in_grid]])
{
    if (tid.x >= color.get_width() || tid.y >= color.get_height())
        return;

    uint32_t id = instanceId.read(tid).r;
    float3 c = makeRandomColor(id);
    color.write(float4(c, 1.0f), tid);
}

kernel void visualizeDepthKernel(
    texture2d<float, access::read> depth [[texture(0)]],
    texture2d<float, access::read_write> color [[texture(1)]],
    constant float &minDepth [[buffer(0)]],
    constant float &maxDepth [[buffer(1)]],
    uint2 tid [[thread_position_in_grid]])
{
    if (tid.x >= color.get_width() || tid.y >= color.get_height())
        return;

    float d = depth.read(tid).r;
    float range = maxDepth - minDepth;
    float v = range > 0.0f ? saturate((d - minDepth) / range) : 0.0f;
    color.write(float4(v, v, v, 1.0f), tid);
}

kernel void visualizeAlbedoKernel(
    texture2d<float, access::read> albedo [[texture(0)]],
    texture2d<float, access::read_write> color [[texture(1)]],
    uint2 tid [[thread_position_in_grid]])
{
    if (tid.x >= color.get_width() || tid.y >= color.get_height())
        return;

    float4 a = albedo.read(tid);
    color.write(float4(a.xyz, 1.0f), tid);
}

kernel void visualizeNormalKernel(
    texture2d<float, access::read> normal [[texture(0)]],
    texture2d<float, access::read_write> color [[texture(1)]],
    uint2 tid [[thread_position_in_grid]])
{
    if (tid.x >= color.get_width() || tid.y >= color.get_height())
        return;

    float4 n = normal.read(tid);
    float3 visualNormal = (n.xyz + 1.0f) * 0.5f;
    color.write(float4(visualNormal, 1.0f), tid);
}

kernel void visualizeEdgesKernel(
    texture2d<uint, access::read> objectId [[texture(0)]],
    texture2d<float, access::read_write> color [[texture(1)]],
    constant uint32_t &invert [[buffer(0)]],
    uint2 tid [[thread_position_in_grid]])
{
    uint32_t width = color.get_width();
    uint32_t height = color.get_height();

    if (tid.x >= width || tid.y >= height)
        return;

    uint32_t centerID = objectId.read(tid).r;

    if (centerID == 0xFFFFFFFFu) {
        color.write(float4(0.0f, 0.0f, 0.0f, 1.0f), tid);
        return;
    }

    bool isEdge = false;
    for (int dy = -1; dy <= 1 && !isEdge; ++dy) {
        for (int dx = -1; dx <= 1 && !isEdge; ++dx) {
            if (dx == 0 && dy == 0)
                continue;

            int nx = int(tid.x) + dx;
            int ny = int(tid.y) + dy;

            if (nx >= 0 && nx < int(width) && ny >= 0 && ny < int(height)) {
                uint32_t neighborID = objectId.read(uint2(nx, ny)).r;
                if (centerID != neighborID)
                    isEdge = true;
            }
        }
    }

    float edgeValue = isEdge ? 1.0f : 0.0f;
    if (invert != 0)
        edgeValue = 1.0f - edgeValue;

    color.write(float4(edgeValue, edgeValue, edgeValue, 1.0f), tid);
}
