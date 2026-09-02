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

#pragma once

// Shared geometry and radiometry leaves for the analytic AREA lights (rect and
// ring). Every quantity that both next-event estimation and the hit-side deposit
// need lives here exactly once.
//
// Why a separate header (ADR 0009): the hit-side pdf and the NEE pdf must be the
// SAME function, or the MIS balance heuristic is weighted against a density
// nothing actually sampled and the image is silently biased. Two copies that
// "look equivalent" is the failure mode this file exists to make impossible.
//
// This header is deliberately CUDA-FREE — glm and the GPU POD structs only, no
// CUB, no device atomics, no OptiX device intrinsics. sampleLight.h pulls all of
// those, which is why the math it used to own could not be unit tested. Keeping
// these leaves compilable on the host is what lets a unit test assert
// pdf(hit) == pdf(NEE) exactly instead of hoping a rendered image reveals a
// mismatch. Do not add a CUDA-only dependency here.
//
// Object-space area, deliberately: rectArea/ring oneOverArea measure the light in
// OBJECT space and apply no transform Jacobian, so a scaled instance's pdf is
// only approximate. That is pre-existing behavior (see the note in
// lightPickPower.h); the hit side reproduces it exactly rather than "fixing" one
// side and desyncing the pair. Correcting both together is separate work.

#include "gpu/gpu_math.h"
#include "gpu/gpu_objects.h"

namespace visrtx {

// Rect ///////////////////////////////////////////////////////////////////////

// The two derived quantities both sides need, from one cross product: the
// world-space unit normal and the object-space area.
struct RectFrame
{
  vec3 worldNormal; // normalized
  float area; // object-space |edge1 x edge2|
};

VISRTX_HOST_DEVICE RectFrame rectFrame(
    const RectLightGPUData &rect, const mat4 &xfm)
{
  RectFrame f;
  auto normal = cross(rect.edge1, rect.edge2);
  f.area = length(normal);
  f.worldNormal = normalize(xfmVec(xfm, normal));
  return f;
}

// The single `side` predicate. Resolves the light's front/back/both
// configuration against a direction and returns the SIGNED cosine: positive
// means the light emits toward `dirToLight`'s origin, non-positive means it does
// not. Both the NEE radiance/pdf gate and the hit-side cull consume this one
// function, so "which side is lit" cannot be answered two ways.
//
// dirToLight points FROM the shaded point TO the light, matching LightSample::dir.
VISRTX_HOST_DEVICE float rectEmissionCosTheta(const RectLightGPUData &rect,
    const vec3 &worldNormal,
    const vec3 &dirToLight)
{
  auto cosTheta = dot(worldNormal, -dirToLight);

  if (rect.side.back) {
    if (rect.side.front)
      cosTheta = fabsf(cosTheta); // Both sides: always positive
    else
      cosTheta = -cosTheta; // Back only: flip to back face
  }
  // Front only: use cosTheta as-is (positive for front face)

  return cosTheta;
}

// Lambertian: radiance is independent of distance and of viewing angle. The
// cosine is carried by the pdf, not by the radiance.
VISRTX_HOST_DEVICE vec3 rectRadiance(
    const RectLightGPUData &rect, const vec3 &color)
{
  return color * rect.intensity;
}

// Uniform-area sampling converted to solid angle: (1/area) * dist^2 / cosTheta.
// Caller must have established cosTheta > 0 via rectEmissionCosTheta.
//
// The operation order here is load-bearing for the "no behavior change"
// requirement of the extraction: it reproduces sampleRectLight's original
// `areaPdf * pow2(dist) / cosTheta` exactly, rather than the algebraically equal
// `pow2(dist) / (area * cosTheta)`, which rounds differently.
VISRTX_HOST_DEVICE float rectSolidAnglePdf(
    float area, float dist, float cosTheta)
{
  const float areaPdf = 1.0f / area;
  return areaPdf * pow2(dist) / cosTheta;
}

// Ring ///////////////////////////////////////////////////////////////////////

// Smoothstep cone falloff, shared so the visible disk shows the same attenuation
// the illumination has. cosTheta is measured against the ring's world-space axis.
VISRTX_HOST_DEVICE float ringSpotAttenuation(
    const RingLightGPUData &ring, float cosTheta)
{
  if (cosTheta < ring.cosOuterAngle) {
    // Outside cone: no illumination
    return 0.0f;
  } else if (cosTheta > ring.cosInnerAngle) {
    // Inside inner cone: full illumination
    return 1.0f;
  }
  // Falloff region: smooth interpolation using smoothstep function
  // smoothstep(t) = 3t^2 - 2t^3 provides C1 continuity
  float spot = (cosTheta - ring.cosOuterAngle)
      / (ring.cosInnerAngle - ring.cosOuterAngle);
  return spot * spot * (3.0f - 2.0f * spot);
}

VISRTX_HOST_DEVICE vec3 ringRadiance(
    const RingLightGPUData &ring, const vec3 &color, float spot)
{
  return color * ring.intensity * spot;
}

// Ring area is pi*(R^2 - r^2), precomputed host-side as oneOverArea.
// Same operation order as the original sampleRingLight, per rectSolidAnglePdf.
VISRTX_HOST_DEVICE float ringSolidAnglePdf(
    const RingLightGPUData &ring, float dist, float cosTheta)
{
  const float areaPdf = ring.oneOverArea; // This is 1 / ring_area
  return areaPdf * pow2(dist) / cosTheta;
}

} // namespace visrtx
