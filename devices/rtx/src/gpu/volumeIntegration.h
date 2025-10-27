/*
 * Copyright (c) 2019-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <texture_types.h>
#include "gpu/dda.h"
#include "gpu/gpu_debug.h"
#include "gpu/gpu_objects.h"
#include "gpu/gpu_util.h"
#include "gpu/sampleSpatialField.h"
#include "nanovdb/NanoVDB.h"

namespace visrtx {

namespace detail {

/**
 * Classify a scalar volume sample using the volume's transfer function.
 * 
 * @param v Volume GPU data containing transfer function information
 * @param s Scalar sample value to classify
 * @return vec4 containing RGB color and opacity (alpha) for the sample
 */
VISRTX_DEVICE vec4 classifySample(const VolumeGPUData &v, float s)
{
  vec4 retval(0.f);
  switch (v.type) {
  case VolumeType::TF1D: {
    if (v.data.tf1d.tfTex) {
      // Map scalar value to texture coordinate in [0,1] range
      float coord = position(s, v.data.tf1d.valueRange);
      retval = make_vec4(tex1D<::float4>(v.data.tf1d.tfTex, coord));
    } else
      // Use uniform color/opacity if no texture is provided
      retval = vec4(v.data.tf1d.uniformColor, v.data.tf1d.uniformOpacity);
    break;
  }
  default:
    break;
  }
  return retval;
}

/**
 * Ray marching implementation for volume rendering using the emission-absorption model.
 * Uses front-to-back compositing with early ray termination when opacity threshold is reached.
 * 
 * @tparam Sampler Type of spatial field sampler (e.g., texture sampler or NanoVDB sampler)
 * @param ss Screen sample containing random state and frame data
 * @param hit Volume hit information with ray and volume data
 * @param interval Ray interval [tmin, tmax] to march through
 * @param color Optional pointer to accumulate color contribution (nullptr to skip color)
 * @param opacity Output opacity accumulated during ray march
 * @param invSamplingRate Inverse of sampling rate multiplier (higher = fewer samples)
 */
template <typename Sampler>
VISRTX_DEVICE void _rayMarchVolume(ScreenSample &ss,
    const VolumeHit &hit,
    box1 interval,
    vec3 *color,
    float &opacity,
    float invSamplingRate)
{
  const auto &volume = *hit.volume;
  /////////////////////////////////////////////////////////////////////////////
  // TODO: need to generalize
  auto &svv = volume.data.tf1d;
  auto &field = getSpatialFieldData(*ss.frameData, svv.field);
  /////////////////////////////////////////////////////////////////////////////

  Sampler sampler(field);

  // Adjust step size based on sampling rate
  const float stepSize = volume.stepSize * invSamplingRate;
  // Precompute exponent for optical depth calculation
  const float exponent = stepSize * svv.oneOverUnitDistance;
  // Apply jitter to starting position to reduce banding artifacts
  interval.lower += stepSize * curand_uniform(&ss.rs);

  // Use logarithmic accumulation for better numerical stability
  float logTransmittance = 0.f;
  float transmittance = 1.f;
  constexpr float OPACITY_THRESHOLD = 0.99f;
  
  while (opacity < OPACITY_THRESHOLD && size(interval) > 0.f) {
    // Compute sample position along ray
    const vec3 p = hit.localRay.org + hit.localRay.dir * interval.lower;
    const float s = sampler(p);
    if (!glm::isnan(s)) {
      // Classify the sample using transfer function
      const vec4 co = detail::classifySample(volume, s);
      
      // Compute step alpha (opacity contribution from this sample)
      const float stepAlpha = 1.0f - glm::pow(1.0f - co.w, exponent);
      
      // Front-to-back compositing with proper alpha blending
      const float weight = (1.0f - opacity);
      if (color)
        *color += weight * stepAlpha * vec3(co);
      opacity += weight * stepAlpha;
      
      // Update transmittance for logarithmic tracking (optional, for consistency)
      logTransmittance += glm::log(1.0f - stepAlpha);
      transmittance = glm::exp(logTransmittance);
    }
    interval.lower += stepSize;
  }

}

VISRTX_DEVICE float rayMarchVolume(ScreenSample &ss,
    const VolumeHit &hit,
    vec3 *color,
    float &opacity,
    float invSamplingRate)
{
  const auto &volume = *hit.volume;
  /////////////////////////////////////////////////////////////////////////////
  // TODO: need to generalize
  auto &svv = volume.data.tf1d;
  auto &field = getSpatialFieldData(*ss.frameData, svv.field);
  /////////////////////////////////////////////////////////////////////////////
  const float stepSize = volume.stepSize;
  box1 interval = hit.localRay.t;
  const float depth = interval.lower;
  interval.lower += stepSize * curand_uniform(&ss.rs); // jitter

  switch (field.type) {
  case SpatialFieldType::STRUCTURED_REGULAR: {
    _rayMarchVolume<SpatialFieldSampler<cudaTextureObject_t>>(
        ss, hit, interval, color, opacity, invSamplingRate);
    break;
  }
  case SpatialFieldType::NANOVDB_REGULAR: {
    switch (field.data.nvdbRegular.gridType) {
    case nanovdb::GridType::Fp4: {
      _rayMarchVolume<NvdbSpatialFieldSampler<nanovdb::Fp4>>(
          ss, hit, interval, color, opacity, invSamplingRate);
      break;
    }
    case nanovdb::GridType::Fp8: {
      _rayMarchVolume<NvdbSpatialFieldSampler<nanovdb::Fp8>>(
          ss, hit, interval, color, opacity, invSamplingRate);
      break;
    }
    case nanovdb::GridType::Fp16: {
      _rayMarchVolume<NvdbSpatialFieldSampler<nanovdb::Fp16>>(
          ss, hit, interval, color, opacity, invSamplingRate);
      break;
    }
    case nanovdb::GridType::FpN: {
      _rayMarchVolume<NvdbSpatialFieldSampler<nanovdb::FpN>>(
          ss, hit, interval, color, opacity, invSamplingRate);
      break;
    }
    case nanovdb::GridType::Float: {
      _rayMarchVolume<NvdbSpatialFieldSampler<float>>(
          ss, hit, interval, color, opacity, invSamplingRate);
      break;
    }
    default:
      break;
    }
    break;
  }
  default:
    break;
  }

  return depth;
}

/**
 * Sample a scattering distance in a heterogeneous volume using Woodcock tracking.
 * 
 * Woodcock tracking (delta tracking) is an unbiased Monte Carlo method for sampling
 * free-flight distances in heterogeneous media. It uses:
 * 1. A majorant (upper bound) for extinction in each spatial region
 * 2. Random sampling with rejection to account for spatial variation
 * 3. No need to compute actual transmittance along the ray
 * 
 * The algorithm traverses the volume grid using DDA (Digital Differential Analyzer)
 * to efficiently access per-voxel majorants.
 * 
 * @tparam Sampler Type of spatial field sampler
 * @param ss Screen sample with random state
 * @param hit Volume hit information
 * @param albedo Output: scattering albedo (color) at sampled point
 * @param extinction Output: extinction coefficient at sampled point
 * @param tr Output: transmittance (0 if scattered, 1 if reached end)
 * @return Distance along ray to scattering event (or ray exit)
 */
template <typename Sampler>
VISRTX_DEVICE float _sampleDistance(ScreenSample &ss,
    const VolumeHit &hit,
    vec3 &albedo,
    float &extinction,
    float &tr)
{
  const auto &volume = *hit.volume;
  /////////////////////////////////////////////////////////////////////////////
  // TODO: need to generalize beyond TF1D volumes
  auto &svv = volume.data.tf1d;
  auto &field = getSpatialFieldData(*ss.frameData, svv.field);
  /////////////////////////////////////////////////////////////////////////////

  Sampler sampler(field);

  float t_out = hit.localRay.t.upper;
  tr = 1.f; // Transmittance: 1.0 = no interaction, 0.0 = scattered

  // Transform ray to object space for grid traversal
  Ray objRay = hit.localRay;
  objRay.org += hit.localRay.dir * hit.localRay.t.lower;
  objRay.t.lower -= hit.localRay.t.lower;
  objRay.t.upper -= hit.localRay.t.lower;

  /**
   * Woodcock tracking implementation per spatial partition (voxel).
   * 
   * For each voxel traversed by the ray:
   * 1. Use the voxel's majorant (max opacity) as the sampling rate
   * 2. Sample tentative collision distances: t = t - ln(1-ξ) / majorant
   * 3. At each collision, accept with probability: σ(x) / majorant
   * 4. If rejected, continue sampling (null collision)
   * 5. If accepted, return the collision point
   */
  auto woodcockFunc = [&](const int leafID, float t0, float t1) {
    // Get majorant (upper bound) for extinction in this voxel
    const float majorant = field.grid.maxOpacities[leafID];
    float t = t0;

    // Handle empty space optimization
    constexpr float EPSILON = 1e-7f;
    if (majorant <= EPSILON)
      return true; // Skip empty voxels

    while (t < t1) {
      // Sample tentative collision distance using exponential distribution
      // This is the standard Woodcock tracking step:
      // t_next = t_current + (-ln(1 - ξ) / σ_maj) * step_scale
      const float xi = curand_uniform(&ss.rs);
      t += -logf(1.f - xi) / (majorant * svv.oneOverUnitDistance);

      // Exit if we've left this voxel
      if (t >= t1)
        break;

      // Evaluate actual extinction at the tentative collision point
      const vec3 p = objRay.org + objRay.dir * t;
      const float s = sampler(p);
      
      if (!glm::isnan(s)) {
        const vec4 co = detail::classifySample(volume, s);
        const float actualExtinction = co.w * svv.oneOverUnitDistance;
        
        // Russian roulette: accept collision with probability σ(x) / σ_maj
        float u = curand_uniform(&ss.rs);
        if (actualExtinction >= u * majorant) {
          // Real collision - ray scattered
          albedo = vec3(co);
          extinction = actualExtinction;
          tr = 0.f;
          t_out = t + hit.localRay.t.lower; // Convert back to original ray space
          return false; // Stop DDA traversal
        }
        // Null collision - continue tracking
      }
    }

    return true; // Continue DDA traversal to next voxel
  };

  // Traverse volume grid using 3D DDA with Woodcock tracking per voxel
  // dda3(objRay, field.grid.dims, field.grid.worldBounds, woodcockFunc);
  amanatidesWoo3(objRay, field.grid.dims, field.grid.worldBounds, woodcockFunc);
  // If no scattering occurred, return the exit point
  return (tr > 0.5f) ? hit.localRay.t.lower : t_out;
}

} // namespace detail

/**
 * Ray march through a volume to compute accumulated opacity (no color).
 * 
 * @param ss Screen sample containing random state
 * @param hit Volume hit information
 * @param opacity Output accumulated opacity
 * @param invSamplingRate Inverse sampling rate (higher = fewer samples, faster but lower quality)
 * @return Distance to first volume intersection
 */
VISRTX_DEVICE float rayMarchVolume(ScreenSample &ss,
    const VolumeHit &hit,
    float &opacity,
    float invSamplingRate)
{
  return detail::rayMarchVolume(ss, hit, nullptr, opacity, invSamplingRate);
}

/**
 * Ray march through a volume to compute accumulated color and opacity.
 * 
 * @param ss Screen sample containing random state
 * @param hit Volume hit information
 * @param color Output accumulated color
 * @param opacity Output accumulated opacity
 * @param invSamplingRate Inverse sampling rate (higher = fewer samples, faster but lower quality)
 * @return Distance to first volume intersection
 */
VISRTX_DEVICE float rayMarchVolume(ScreenSample &ss,
    const VolumeHit &hit,
    vec3 &color,
    float &opacity,
    float invSamplingRate)
{
  return detail::rayMarchVolume(ss, hit, &color, opacity, invSamplingRate);
}

/**
 * Ray march through all volumes in the scene along a ray.
 * 
 * Handles multiple overlapping volumes by:
 * 1. Finding all volume intersections along the ray
 * 2. Ray marching through each volume segment
 * 3. Accumulating color and opacity with front-to-back compositing
 * 4. Early termination when opacity threshold is reached
 * 
 * @tparam RAY_TYPE Type of ray (primary, shadow, etc.)
 * @param ss Screen sample with random state
 * @param ray Ray to trace
 * @param type Ray type for intersection queries
 * @param tfar Maximum ray distance
 * @param invSamplingRate Inverse sampling rate for adaptive quality
 * @param color Output accumulated color
 * @param opacity Output accumulated opacity
 * @param objID Output object ID of closest volume
 * @param instID Output instance ID of closest volume
 * @return Distance to closest volume intersection
 */
template <typename RAY_TYPE>
VISRTX_DEVICE float rayMarchAllVolumes(ScreenSample &ss,
    Ray ray,
    RAY_TYPE type,
    float tfar,
    float invSamplingRate,
    vec3 &color,
    float &opacity,
    uint32_t &objID,
    uint32_t &instID)
{
  VolumeHit hit;
  ray.t.upper = tfar;
  float depth = tfar;
  bool firstHit = true;

  constexpr float OPACITY_THRESHOLD = 0.99f;
  constexpr float RAY_EPSILON = 1e-3f;

  do {
    hit.foundHit = false;
    intersectVolume(ss, ray, type, &hit);
    if (!hit.foundHit)
      break;
    
    // Record IDs from first volume hit
    if (firstHit) {
      objID = hit.volume->id;
      instID = hit.instance->id;
      firstHit = false;
    }
    
    // Track closest intersection depth
    depth = min(depth, hit.localRay.t.lower);
    
    // Clamp ray march interval to maximum distance
    hit.localRay.t.upper = glm::min(tfar, hit.localRay.t.upper);
    
    // Ray march through this volume segment
    detail::rayMarchVolume(ss, hit, &color, opacity, invSamplingRate);
    
    // Advance ray past this volume with small epsilon to avoid self-intersection
    ray.t.lower = hit.localRay.t.upper + RAY_EPSILON;
  } while (opacity < OPACITY_THRESHOLD);

  return depth;
}

VISRTX_DEVICE float sampleDistanceVolume(ScreenSample &ss,
    const VolumeHit &hit,
    vec3 &albedo,
    float &extinction,
    float &tr)
{
  const auto &volume = *hit.volume;
  /////////////////////////////////////////////////////////////////////////////
  // TODO: need to generalize
  auto &svv = volume.data.tf1d;
  auto &field = getSpatialFieldData(*ss.frameData, svv.field);
  /////////////////////////////////////////////////////////////////////////////

  switch (field.type) {
  case SpatialFieldType::STRUCTURED_REGULAR: {
    return detail::_sampleDistance<SpatialFieldSampler<cudaTextureObject_t>>(
        ss, hit, albedo, extinction, tr);
    break;
  }
  case SpatialFieldType::NANOVDB_REGULAR: {
    switch (field.data.nvdbRegular.gridType) {
    case nanovdb::GridType::Fp4: {
      return detail::_sampleDistance<NvdbSpatialFieldSampler<nanovdb::Fp4>>(
          ss, hit, albedo, extinction, tr);
      break;
    }
    case nanovdb::GridType::Fp8: {
      return detail::_sampleDistance<NvdbSpatialFieldSampler<nanovdb::Fp8>>(
          ss, hit, albedo, extinction, tr);
      break;
    }
    case nanovdb::GridType::Fp16: {
      return detail::_sampleDistance<NvdbSpatialFieldSampler<nanovdb::Fp16>>(
          ss, hit, albedo, extinction, tr);
      break;
    }
    case nanovdb::GridType::FpN: {
      return detail::_sampleDistance<NvdbSpatialFieldSampler<nanovdb::FpN>>(
          ss, hit, albedo, extinction, tr);
      break;
    }
    case nanovdb::GridType::Float: {
      return detail::_sampleDistance<NvdbSpatialFieldSampler<float>>(
          ss, hit, albedo, extinction, tr);
      break;
    }
    default:
      break;
    }
    break;
  }
  default:
    break;
  }

  return hit.localRay.t.upper;
}

/**
 * Sample scattering distance in all volumes along a ray using Woodcock tracking.
 * 
 * This function:
 * 1. Finds all volume intersections along the ray
 * 2. Applies Woodcock tracking in each volume segment
 * 3. Returns the closest scattering event across all volumes
 * 
 * Used for physically-based path tracing with heterogeneous volumes.
 * 
 * @tparam RAY_TYPE Type of ray for intersection queries
 * @param ss Screen sample with random state
 * @param ray Ray to trace
 * @param type Ray type
 * @param tfar Maximum ray distance
 * @param albedo Output scattering albedo at sampled point
 * @param extinction Output extinction coefficient at sampled point
 * @param transmittance Output transmittance (0 = scattered, 1 = no interaction)
 * @param objID Output object ID where scattering occurred
 * @param instID Output instance ID where scattering occurred
 * @return Distance to scattering event (or tfar if no scattering)
 */
template <typename RAY_TYPE>
VISRTX_DEVICE float sampleDistanceAllVolumes(ScreenSample &ss,
    Ray ray,
    RAY_TYPE type,
    float tfar,
    vec3 &albedo,
    float &extinction,
    float &transmittance,
    uint32_t &objID,
    uint32_t &instID)
{
  VolumeHit hit;
  ray.t.upper = tfar;
  float depth = tfar;
  transmittance = 1.f;

  constexpr float RAY_EPSILON = 1e-3f;

  while (true) {
    hit.foundHit = false;
    intersectVolume(ss, ray, type, &hit);
    if (!hit.foundHit)
      break;
    
    // Clamp interval to maximum distance
    hit.localRay.t.upper = glm::min(tfar, hit.localRay.t.upper);
    
    // Sample distance in this volume using Woodcock tracking
    vec3 alb(0.f);
    float ext = 0.f, tr = 0.f;
    float d = sampleDistanceVolume(ss, hit, alb, ext, tr);
    
    // Keep closest scattering event
    if (d < depth) {
      depth = d;
      albedo = alb;
      extinction = ext;
      transmittance = tr;
      objID = hit.volume->id;
      instID = hit.instance->id;
    }
    
    // Advance ray past this volume
    ray.t.lower = hit.localRay.t.upper + RAY_EPSILON;
  }

  return depth;
}

} // namespace visrtx
