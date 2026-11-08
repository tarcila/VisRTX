// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <mi/neuraylib/target_code_types.h>
#include <optix_device.h>
#include "gpu/gpu_objects.h"
#include "gpu/gpu_util.h"
#include "gpu/shadingState.h"

using namespace visrtx;
using Init = mi::neuraylib::Bsdf_init_function;
using Expression = mi::neuraylib::Material_function<vec3>::Type;
VISRTX_CALLABLE Init mdlVolumeInit;
VISRTX_CALLABLE Expression mdlVolumeAbsorption;
VISRTX_CALLABLE Expression mdlVolumeScattering;
VISRTX_CALLABLE Expression mdlVolumeEmission;

// This state is a medium sample, never a synthetic SurfaceHit. In particular,
// no geometry.normal evaluation or surface attribute/tangent setup runs here.
VISRTX_CALLABLE void __direct_callable__volumeCoefficients(
    const FrameGPUData *fd,
    const VolumeHit *hit,
    const vec3 *position,
    vec3 *absorption,
    vec3 *scattering,
    vec3 *emission)
{
  MDLShadingState state{};
  const auto &md =
      fd->registry.materials[hit->volume->data.mdl.material].materialData.mdl;
  state.state.position = bit_cast<float3>(*position);
  state.state.normal = make_float3(0.f, 0.f, 0.f);
  state.state.geom_normal = make_float3(0.f, 0.f, 0.f);
  state.state.animation_time = fd->mdlAnimationTime;
  state.state.meters_per_scene_unit = 1.f;
  state.state.object_id = bit_cast<int>(hit->volume->id);
  state.state.object_to_world =
      reinterpret_cast<const float4 *>(&hit->instance->objectToWorld);
  state.state.world_to_object =
      reinterpret_cast<const float4 *>(&hit->instance->worldToObject);
  state.state.text_results = reinterpret_cast<float4 *>(state.textureResults);
  state.textureHandler.fd = fd;
  state.textureHandler.isVolume = true;
  state.textureHandler.volumeID = hit->volume->id;
  state.textureHandler.instanceID = hit->instance->id;
  state.textureHandler.samplers = nullptr;
  state.textureHandler.numSamplers = 0;
  state.resData = {nullptr, &state.textureHandler};
  state.argBlock = md.volumeArgBlock;
  mdlVolumeInit(&state.state, &state.resData, state.argBlock);
  *absorption =
      mdlVolumeAbsorption(&state.state, &state.resData, state.argBlock);
  *scattering =
      mdlVolumeScattering(&state.state, &state.resData, state.argBlock);
  *emission = mdlVolumeEmission(&state.state, &state.resData, state.argBlock);
}
