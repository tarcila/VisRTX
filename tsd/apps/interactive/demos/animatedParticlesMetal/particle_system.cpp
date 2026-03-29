// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "particle_system.h"
// tsd_algorithms
#include <tsd/algorithms/metal/runtime.hpp>
// std
#include <cstring>
#include <stdexcept>

namespace tsd::demo {

static const char *kParticlesMSL = R"msl(
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
)msl";

// Packed to match MSL struct layout (float3 is 16-byte aligned in Metal)
struct ParticleParamsGPU
{
  float gravity;
  float particleInvMass;
  float maxDistance;
  float deltaT;
  float bhPosition1[4]; // float3 padded to float4 alignment
  float bhPosition2[4];
};

static void *g_library = nullptr;

void initMetalParticleSystem()
{
  if (g_library)
    return;
  g_library = tsd::algorithms::metal::compileShaderSource(kParticlesMSL);
  if (!g_library)
    throw std::runtime_error("Failed to compile Metal particle shader");
}

void shutdownMetalParticleSystem()
{
  if (g_library) {
    tsd::algorithms::metal::releaseLibrary(g_library);
    g_library = nullptr;
  }
}

void particlesComputeTimestepMetal(int numParticles,
    void *positionsBuffer,
    void *velocitiesBuffer,
    void *distancesBuffer,
    const tsd::math::float3 &bhPosition1,
    const tsd::math::float3 &bhPosition2,
    const ParticleSystemParameters &params)
{
  ParticleParamsGPU gpu;
  gpu.gravity = params.gravity;
  gpu.particleInvMass = 1.f / params.particleMass;
  gpu.maxDistance = params.maxDistance;
  gpu.deltaT = params.deltaT;

  std::memcpy(gpu.bhPosition1, &bhPosition1, sizeof(float) * 3);
  gpu.bhPosition1[3] = 0.f;
  std::memcpy(gpu.bhPosition2, &bhPosition2, sizeof(float) * 3);
  gpu.bhPosition2[3] = 0.f;

  void *buffers[] = {positionsBuffer, velocitiesBuffer, distancesBuffer};

  tsd::algorithms::metal::dispatchKernel(g_library,
      "particlesComputeTimestep",
      buffers,
      3,
      &gpu,
      sizeof(gpu),
      static_cast<uint32_t>(numParticles));
}

} // namespace tsd::demo
