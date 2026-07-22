/*
 * Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

// OptiX direct-callable entry points for the PhysicallyBased material. The BRDF
// math lives in gpu/physicallyBasedBsdf.h (shared verbatim with the wavefront
// renderer's plain-CUDA shade stage); these are thin wrappers so the
// interactive pipeline dispatches the identical evaluation.

#include "gpu/physicallyBasedBsdf.h"

using namespace visrtx;

VISRTX_CALLABLE void __direct_callable__init(
    PhysicallyBasedShadingState *shadingState,
    const FrameGPUData *fd,
    const SurfaceHit *hit,
    const MaterialGPUData::PhysicallyBased *md)
{
  pbrInitState(shadingState, fd, hit, md);
}

VISRTX_CALLABLE vec3 __direct_callable__evaluateTint(
    const PhysicallyBasedShadingState *shadingState)
{
  return shadingState->baseColor;
}

VISRTX_CALLABLE float __direct_callable__evaluateOpacity(
    const PhysicallyBasedShadingState *shadingState)
{
  return shadingState->opacity;
}

VISRTX_CALLABLE vec3 __direct_callable__evaluateEmission(
    const PhysicallyBasedShadingState *shadingState, const vec3 *outgoingDir)
{
  return shadingState->emission;
}

VISRTX_CALLABLE vec3 __direct_callable__evaluateTransmission(
    const PhysicallyBasedShadingState *shadingState)
{
  return computeTransmissionFilter(shadingState);
}

VISRTX_CALLABLE vec3 __direct_callable__evaluateNormal(
    const PhysicallyBasedShadingState *shadingState)
{
  return shadingState->normal;
}

VISRTX_CALLABLE vec3 __direct_callable__shadeSurface(
    const PhysicallyBasedShadingState *state,
    const SurfaceHit *hit,
    const LightSample *lightSample,
    const vec3 *outgoingDir)
{
  return pbrEvalNEE(state, hit, lightSample, outgoingDir);
}

VISRTX_CALLABLE NextRay __direct_callable__nextRay(
    const PhysicallyBasedShadingState *state, const Ray *ray, RandState *rs)
{
  return pbrSampleNextRay(state, ray, rs);
}

VISRTX_CALLABLE float __direct_callable__evaluatePdf(
    const PhysicallyBasedShadingState *state, const vec3 *wo, const vec3 *wi)
{
  return pbrBsdfPdf(state, *wo, *wi);
}
