/*
 * Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include "fieldBinding.h"

#include <cmath>

namespace visrtx {

// Coefficients are per world-space path length; emission is radiance per unit
// length. Summing these is valid for overlapping media, unlike alpha blending.
struct VolumeCoefficients
{
  vec3 absorption{0.f};
  vec3 scattering{0.f};
  vec3 emission{0.f};
};

struct VolumePhaseFunction
{
  float anisotropy{0.f};
  float secondaryAnisotropy{0.f};
  float secondaryWeight{0.f};
};

struct VolumeOpticalSample
{
  VolumeCoefficients coefficients;
  VolumePhaseFunction phase;
};

struct PrincipledVolumeValues
{
  float density{1.f};
  vec3 scatteringColor{1.f};
  vec3 transmissionColor{0.36787944117144233f};
  float transmissionDistance{1.f};
  VolumePhaseFunction phase;
  vec3 emission{0.f};
  float temperature{1.f};
  float blackbodyIntensity{0.f};
  vec3 blackbodyTint{1.f};
};

struct PrincipledVolumeBindingsGPUData
{
  box3 bounds;
  FieldBindingGPUData<float> density{1.f, 0.f};
  FieldBindingGPUData<vec3> scatteringColor;
  FieldBindingGPUData<vec3> transmissionColor{
      vec3(0.36787944117144233f), vec3(1.f)};
  FieldBindingGPUData<float> transmissionDistance;
  FieldBindingGPUData<float> anisotropy{0.f, 1.f};
  FieldBindingGPUData<float> secondaryAnisotropy{0.f, 1.f};
  FieldBindingGPUData<float> secondaryWeight{0.f, 1.f};
  FieldBindingGPUData<vec3> emission{vec3(0.f), vec3(0.f)};
  FieldBindingGPUData<float> temperature{1.f, 0.f};
  FieldBindingGPUData<float> blackbodyIntensity{0.f, 1.f};
  FieldBindingGPUData<vec3> blackbodyTint;
};

VISRTX_HOST_DEVICE vec3 volumeExtinction(const VolumeCoefficients &v)
{
  return v.absorption + v.scattering;
}

VISRTX_HOST_DEVICE VolumeCoefficients addVolumeCoefficients(
    const VolumeCoefficients &a, const VolumeCoefficients &b)
{
  return {a.absorption + b.absorption,
      a.scattering + b.scattering,
      a.emission + b.emission};
}

VISRTX_HOST_DEVICE vec3 volumeScatteringAlbedo(const VolumeCoefficients &v)
{
  const vec3 extinction = volumeExtinction(v);
  return {extinction.x > 0.f ? v.scattering.x / extinction.x : 0.f,
      extinction.y > 0.f ? v.scattering.y / extinction.y : 0.f,
      extinction.z > 0.f ? v.scattering.z / extinction.z : 0.f};
}

// cosTheta compares the incoming propagation direction to the outgoing
// propagation direction: +1 is forward. Valid g lies strictly between -1 and 1.
VISRTX_HOST_DEVICE float henyeyGreenstein(float g, float cosTheta)
{
  const float c = glm::clamp(cosTheta, -1.f, 1.f);
  const float oneMinusG = 1.f - g;
  const float onePlusG = 1.f + g;
  const float denom = g >= 0.f ? oneMinusG * oneMinusG + 2.f * g * (1.f - c)
                               : onePlusG * onePlusG - 2.f * g * (1.f + c);
  return (oneMinusG * onePlusG) / (4.f * kPi * denom * sqrtf(denom));
}

VISRTX_HOST_DEVICE float evaluateVolumePhase(
    const VolumePhaseFunction &phase, float cosTheta)
{
  const float primary = henyeyGreenstein(phase.anisotropy, cosTheta);
  const float secondary = henyeyGreenstein(phase.secondaryAnisotropy, cosTheta);
  return (1.f - phase.secondaryWeight) * primary
      + phase.secondaryWeight * secondary;
}

// Add these kernels for overlapping media. Do not average g or collapse the
// phase to that of whichever object supplies an ID; RGB weights can differ.
VISRTX_HOST_DEVICE vec3 volumeScatteringKernel(
    const VolumeOpticalSample &sample, float cosTheta)
{
  return sample.coefficients.scattering
      * evaluateVolumePhase(sample.phase, cosTheta);
}

// Keep the existing TF1D saturation/epsilon policy exactly: acceleration bounds
// use the same opacity conversion. This helper does not change renderer
// behavior.
VISRTX_HOST_DEVICE float transferFunctionExtinction(
    float opacity, float oneOverUnitDistance)
{
  constexpr float OPACITY_EPSILON = 1e-7f;
  constexpr float OPACITY_CEILING = 1.f - 0x1p-24f;
  const float alpha = glm::clamp(opacity, 0.f, OPACITY_CEILING);
  if (alpha <= OPACITY_EPSILON || !(oneOverUnitDistance > 0.f))
    return 0.f;
  return -logf(1.f - alpha) * oneOverUnitDistance;
}

VISRTX_HOST_DEVICE VolumeOpticalSample evaluateTransferFunctionOptics(
    const vec4 &colorOpacity, float oneOverUnitDistance)
{
  VolumeOpticalSample result;
  const float extinction =
      transferFunctionExtinction(colorOpacity.w, oneOverUnitDistance);
  result.coefficients.absorption = vec3(extinction);
  result.coefficients.emission = vec3(colorOpacity) * extinction;
  return result;
}

// The blackbody provider returns calibrated, nonnegative linear RGB radiance
// at T kelvin. Inject it so optical evaluation does not embed licensed tables
// or force uniform media to repeatedly evaluate spectra. No field-content
// checking or runtime invalid-data recovery is performed here.
template <typename Blackbody>
VISRTX_HOST_DEVICE VolumeOpticalSample evaluatePrincipledOptics(
    const PrincipledVolumeValues &v, Blackbody &blackbody)
{
  VolumeOpticalSample result;
  result.phase = v.phase;
  result.coefficients.emission = v.emission;
  if (v.density == 0.f)
    return result;

  vec3 extinction(0.f);
  for (int c = 0; c < 3; ++c) {
    // Exact transparent channels remain zero even for a very small distance.
    if (v.transmissionColor[c] != 1.f)
      extinction[c] =
          (-logf(v.transmissionColor[c]) * v.density) / v.transmissionDistance;
  }
  result.coefficients.scattering = extinction * v.scatteringColor;
  result.coefficients.absorption = extinction * (vec3(1.f) - v.scatteringColor);

  const vec3 thermalWeight =
      result.coefficients.absorption * v.blackbodyTint * v.blackbodyIntensity;
  if (v.temperature > 0.f
      && (thermalWeight.x > 0.f || thermalWeight.y > 0.f
          || thermalWeight.z > 0.f))
    result.coefficients.emission += thermalWeight * blackbody(v.temperature);
  return result;
}

template <typename Sampler, typename Blackbody>
VISRTX_HOST_DEVICE VolumeOpticalSample evaluatePrincipledOptics(
    const PrincipledVolumeBindingsGPUData &bindings,
    const vec3 &objectPosition,
    Sampler &sampler,
    Blackbody &blackbody)
{
  if (!containsFieldSample(bindings.bounds, objectPosition))
    return {};

  PrincipledVolumeValues v;
  v.density = evaluateFieldBinding(bindings.density, objectPosition, sampler);
  v.scatteringColor =
      evaluateFieldBinding(bindings.scatteringColor, objectPosition, sampler);
  v.transmissionColor =
      evaluateFieldBinding(bindings.transmissionColor, objectPosition, sampler);
  v.transmissionDistance = evaluateFieldBinding(
      bindings.transmissionDistance, objectPosition, sampler);
  v.phase.anisotropy =
      evaluateFieldBinding(bindings.anisotropy, objectPosition, sampler);
  v.phase.secondaryAnisotropy = evaluateFieldBinding(
      bindings.secondaryAnisotropy, objectPosition, sampler);
  v.phase.secondaryWeight =
      evaluateFieldBinding(bindings.secondaryWeight, objectPosition, sampler);
  v.emission = evaluateFieldBinding(bindings.emission, objectPosition, sampler);
  v.temperature =
      evaluateFieldBinding(bindings.temperature, objectPosition, sampler);
  v.blackbodyIntensity = evaluateFieldBinding(
      bindings.blackbodyIntensity, objectPosition, sampler);
  v.blackbodyTint =
      evaluateFieldBinding(bindings.blackbodyTint, objectPosition, sampler);
  return evaluatePrincipledOptics(v, blackbody);
}

} // namespace visrtx
