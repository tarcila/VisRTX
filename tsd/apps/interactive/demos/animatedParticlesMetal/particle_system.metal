// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Reference copy of the MSL kernel. The authoritative source is the raw string
// literal embedded in particle_system.cpp (compiled at runtime).

#include <metal_stdlib>
using namespace metal;

struct ParticleParams {
    float gravity;
    float particleInvMass;
    float maxDistance;
    float deltaT;
    float3 bhPosition1;
    float3 bhPosition2;
};

kernel void particlesComputeTimestep(
    device float3 *positions  [[buffer(0)]],
    device float3 *velocities [[buffer(1)]],
    device float  *distances  [[buffer(2)]],
    constant ParticleParams &params [[buffer(3)]],
    uint idx [[thread_position_in_grid]])
{
    float3 p = positions[idx];
    float3 v = velocities[idx];

    const float3 d1 = params.bhPosition1 - p;
    const float3 d2 = params.bhPosition2 - p;
    const float dist1 = length(d1);
    const float dist2 = length(d2);
    const float3 f1 = (params.gravity / dist1) * normalize(d1);
    const float3 f2 = (params.gravity / dist2) * normalize(d2);
    const float3 force = f1 + f2;

    const float3 a = force * params.particleInvMass;
    p += v * params.deltaT + 0.5f * a * params.deltaT * params.deltaT;
    v += a * params.deltaT;

    if (dist1 > params.maxDistance || dist2 > params.maxDistance)
        p = float3(0.0f);

    positions[idx] = p;
    velocities[idx] = v;
    distances[idx] = length(p);
}
