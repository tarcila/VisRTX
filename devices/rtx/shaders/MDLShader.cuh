#pragma once

#include "gpu/gpu_objects.h"
#include "gpu/sampleLight.h"
#include "gpu/shading_api.h"

#include <optix.h>
#include <cstdint>

namespace visrtx {

VISRTX_DEVICE vec4 shadeMDLSurface(const FrameGPUData &fd,
    const ScreenSample &ss,
    const MaterialGPUData::MDL &md,
    const Ray &ray,
    const SurfaceHit &hit,
    const LightSample &ls)
{
  // Call signature must match the actual implementation in MDLShader_ptx.cu
  return optixDirectCall<vec4>(
      // FIXME: MDL base implementation index, in the Sbt, need to be properly
      // computed and setup time and runtime shared with the compute code.
      static_cast<unsigned int>(MaterialType::MDL) + md.implementationId,
      &fd,
      &ss,
      &md,
      &ray,
      &hit,
      &ls);
}

} // namespace visrtx