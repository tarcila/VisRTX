// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tsd/core/TSDMath.hpp>

namespace tsd::demo {

struct ParticleSystemParameters
{
  float gravity{1000.f};
  float particleMass{0.1f};
  float maxDistance{45.f};
  float deltaT{5e-4f};
};

// Compile MSL kernel (call once at startup)
void initMetalParticleSystem();

// Release MSL kernel resources
void shutdownMetalParticleSystem();

// Dispatch Metal compute kernel on shared-memory buffers.
// All buffer arguments are opaque MTL::Buffer* handles.
void particlesComputeTimestepMetal(int numParticles,
    void *positionsBuffer,
    void *velocitiesBuffer,
    void *distancesBuffer,
    const tsd::math::float3 &bhPosition1,
    const tsd::math::float3 &bhPosition2,
    const ParticleSystemParameters &params);

} // namespace tsd::demo
