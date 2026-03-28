#include <metal_stdlib>
using namespace metal;

kernel void fillUint32(
    device uint32_t *buf [[buffer(0)]],
    constant uint32_t &count [[buffer(1)]],
    constant uint32_t &value [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < count)
        buf[tid] = value;
}

kernel void fillFloat(
    device float *buf [[buffer(0)]],
    constant uint32_t &count [[buffer(1)]],
    constant float &value [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < count)
        buf[tid] = value;
}
