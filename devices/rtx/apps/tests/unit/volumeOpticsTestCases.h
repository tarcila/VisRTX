/*
 * Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include "gpu/volumeOptics.h"

#include <type_traits>

namespace visrtx::test {

inline constexpr int VOLUME_OPTICS_CASES = 17;

static_assert(fieldBindingAccepts<float>(FieldValueType::Scalar));
static_assert(fieldBindingAccepts<vec3>(FieldValueType::RGB));
static_assert(!fieldBindingAccepts<float>(FieldValueType::RGB));
static_assert(!fieldBindingAccepts<vec3>(FieldValueType::Scalar));
static_assert(std::is_trivially_copyable_v<FieldBindingGPUData<float>>);
static_assert(std::is_trivially_copyable_v<FieldBindingGPUData<vec3>>);
static_assert(std::is_trivially_copyable_v<PrincipledVolumeBindingsGPUData>);

struct Sampler
{
  int scalarCalls{0};
  int rgbCalls{0};
  int lastField{-1};

  template <typename T>
  VISRTX_HOST_DEVICE T sample(int32_t field, const vec3 &p);
};

struct Blackbody
{
  int calls{0};
  float lastTemperature{0.f};
  vec3 radiance{2.f, 3.f, 4.f};

  VISRTX_HOST_DEVICE vec3 operator()(float temperature);
};

template <typename T>
VISRTX_HOST_DEVICE T Sampler::sample(int32_t field, const vec3 &p)
{
  lastField = field;
  if constexpr (std::is_same_v<T, float>) {
    ++scalarCalls;
    return p.x + 1.f;
  } else {
    ++rgbCalls;
    return vec3(p.x + 1.f, p.y + 2.f, p.z + 3.f);
  }
}

VISRTX_HOST_DEVICE vec3 Blackbody::operator()(float temperature)
{
  ++calls;
  lastTemperature = temperature;
  return radiance;
}

VISRTX_HOST_DEVICE bool close(float a, float b, float tolerance = 2e-5f)
{
  return fabsf(a - b) <= tolerance * fmaxf(1.f, fabsf(b));
}

VISRTX_HOST_DEVICE bool close(const vec3 &a, const vec3 &b)
{
  return close(a.x, b.x) && close(a.y, b.y) && close(a.z, b.z);
}

#define OPTICS_REQUIRE(condition)                                              \
  do {                                                                         \
    if (!(condition))                                                          \
      return __LINE__;                                                         \
  } while (0)

// Return a source line rather than using assert(), so Release tests still run.
// The same cases execute on host and CUDA, including the fast-math build.
VISRTX_HOST_DEVICE int volumeOpticsTestCase(int id)
{
  Sampler sampler;
  Blackbody blackbody;
  const box3 box(vec3(0.f), vec3(2.f));
  switch (id) {
  case 0: {
    FieldBindingGPUData<float> scalar{2.f, 7.f};
    FieldBindingGPUData<vec3> rgb{vec3(3.f), vec3(9.f)};
    OPTICS_REQUIRE(evaluateFieldBinding(scalar, vec3(100.f), sampler) == 2.f);
    OPTICS_REQUIRE(
        close(evaluateFieldBinding(rgb, vec3(100.f), sampler), vec3(3.f)));
    OPTICS_REQUIRE(sampler.scalarCalls == 0 && sampler.rgbCalls == 0);
    break;
  }
  case 1: {
    FieldBindingGPUData<float> scalar{2.f, 7.f};
    scalar.field = 42;
    scalar.domain = box;
    OPTICS_REQUIRE(evaluateFieldBinding(scalar, vec3(1.f), sampler) == 4.f);
    OPTICS_REQUIRE(sampler.scalarCalls == 1 && sampler.lastField == 42);
    OPTICS_REQUIRE(evaluateFieldBinding(scalar, vec3(-1.f), sampler) == 14.f);
    OPTICS_REQUIRE(sampler.scalarCalls == 1);
    OPTICS_REQUIRE(evaluateFieldBinding(scalar, vec3(2.f), sampler) == 6.f);
    OPTICS_REQUIRE(sampler.scalarCalls == 2);
    break;
  }
  case 2: {
    FieldBindingGPUData<vec3> rgb{vec3(2.f, 3.f, 4.f), vec3(.5f, 2.f, 3.f)};
    rgb.field = 5;
    rgb.domain = box;
    OPTICS_REQUIRE(close(
        evaluateFieldBinding(rgb, vec3(1.f), sampler), vec3(4.f, 9.f, 16.f)));
    OPTICS_REQUIRE(close(
        evaluateFieldBinding(rgb, vec3(3.f), sampler), vec3(1.f, 6.f, 12.f)));
    OPTICS_REQUIRE(sampler.rgbCalls == 1 && sampler.scalarCalls == 0);
    break;
  }
  case 3: {
    const box3 roi(vec3(1.f), vec3(3.f));
    const box3 domain = fieldBindingDomain(box, roi);
    OPTICS_REQUIRE(
        close(domain.lower, vec3(1.f)) && close(domain.upper, vec3(2.f)));
    OPTICS_REQUIRE(containsFieldSample(domain, vec3(1.f)));
    OPTICS_REQUIRE(!containsFieldSample(domain, vec3(.5f)));
    OPTICS_REQUIRE(!containsFieldSample(
        fieldBindingDomain(box, box3(vec3(2.f), vec3(3.f))), vec3(2.f)));
    OPTICS_REQUIRE(!containsFieldSample(
        fieldBindingDomain(box, box3(vec3(3.f), vec3(4.f))), vec3(3.f)));
    FieldBindingGPUData<float> field{3.f, 2.f};
    field.field = 0; // Default domain is empty.
    OPTICS_REQUIRE(evaluateFieldBinding(field, vec3(0.f), sampler) == 6.f);
    OPTICS_REQUIRE(sampler.scalarCalls == 0);
    break;
  }
  case 4: {
    PrincipledVolumeValues v;
    const auto s = evaluatePrincipledOptics(v, blackbody);
    OPTICS_REQUIRE(close(volumeExtinction(s.coefficients), vec3(1.f)));
    OPTICS_REQUIRE(close(s.coefficients.scattering, vec3(1.f)));
    OPTICS_REQUIRE(close(s.coefficients.absorption, vec3(0.f)));
    OPTICS_REQUIRE(close(s.coefficients.emission, vec3(0.f)));
    OPTICS_REQUIRE(blackbody.calls == 0);
    break;
  }
  case 5: {
    PrincipledVolumeValues v;
    v.density = 2.f;
    v.transmissionColor = vec3(.25f, .5f, 1.f);
    v.transmissionDistance = 3.f;
    v.scatteringColor = vec3(.2f, .6f, .9f);
    const auto s = evaluatePrincipledOptics(v, blackbody);
    const vec3 ext = volumeExtinction(s.coefficients);
    OPTICS_REQUIRE(
        close(vec3(expf(-ext.x * 3.f), expf(-ext.y * 3.f), expf(-ext.z * 3.f)),
            vec3(.0625f, .25f, 1.f)));
    OPTICS_REQUIRE(
        close(volumeScatteringAlbedo(s.coefficients), vec3(.2f, .6f, 0.f)));
    break;
  }
  case 6: {
    PrincipledVolumeValues v;
    v.scatteringColor = vec3(.25f, .5f, 1.f);
    v.emission = vec3(.1f, .2f, .3f);
    v.blackbodyIntensity = 2.f;
    v.blackbodyTint = vec3(1.f, .5f, 3.f);
    v.temperature = 2500.f;
    const auto s = evaluatePrincipledOptics(v, blackbody);
    OPTICS_REQUIRE(close(s.coefficients.emission, vec3(3.1f, 1.7f, .3f)));
    OPTICS_REQUIRE(blackbody.calls == 1 && blackbody.lastTemperature == 2500.f);
    v.scatteringColor = vec3(1.f);
    OPTICS_REQUIRE(
        close(evaluatePrincipledOptics(v, blackbody).coefficients.emission,
            v.emission));
    OPTICS_REQUIRE(blackbody.calls == 1);
    break;
  }
  case 7: {
    PrincipledVolumeValues v;
    v.density = 0.f;
    v.emission = vec3(5.f);
    v.blackbodyIntensity = 1.f;
    OPTICS_REQUIRE(
        close(evaluatePrincipledOptics(v, blackbody).coefficients.emission,
            vec3(5.f)));
    OPTICS_REQUIRE(blackbody.calls == 0);
    v.density = 1.f;
    v.scatteringColor = vec3(0.f);
    v.temperature = 0.f;
    OPTICS_REQUIRE(
        close(evaluatePrincipledOptics(v, blackbody).coefficients.emission,
            vec3(5.f)));
    OPTICS_REQUIRE(blackbody.calls == 0);
    break;
  }
  case 8: {
    PrincipledVolumeBindingsGPUData b;
    b.bounds = box;
    b.density.field = 0;
    b.density.domain = box3(vec3(0.f), vec3(.5f));
    b.emission.field = 1;
    b.emission.scale = vec3(1.f);
    b.emission.domain = box;
    const auto s = evaluatePrincipledOptics(b, vec3(1.f), sampler, blackbody);
    OPTICS_REQUIRE(close(volumeExtinction(s.coefficients), vec3(0.f)));
    OPTICS_REQUIRE(close(s.coefficients.emission, vec3(2.f, 3.f, 4.f)));
    OPTICS_REQUIRE(sampler.scalarCalls == 0 && sampler.rgbCalls == 1);
    const auto outside =
        evaluatePrincipledOptics(b, vec3(3.f), sampler, blackbody);
    OPTICS_REQUIRE(close(outside.coefficients.emission, vec3(0.f)));
    OPTICS_REQUIRE(sampler.rgbCalls == 1 && blackbody.calls == 0);
    break;
  }
  case 9: {
    PrincipledVolumeBindingsGPUData b;
    b.bounds = box;
    b.scatteringColor.scale = vec3(.5f);
    b.density = FieldBindingGPUData<float>{2.f, 0.f};
    b.density.field = 7;
    b.density.domain = box;
    b.anisotropy.scale = .6f;
    b.secondaryAnisotropy.scale = -.2f;
    b.secondaryWeight.scale = .25f;
    auto s = evaluatePrincipledOptics(b, vec3(1.f), sampler, blackbody);
    OPTICS_REQUIRE(close(volumeExtinction(s.coefficients), vec3(4.f)));
    OPTICS_REQUIRE(close(s.coefficients.scattering, vec3(2.f)));
    OPTICS_REQUIRE(close(s.phase.anisotropy, .6f));
    OPTICS_REQUIRE(close(s.phase.secondaryAnisotropy, -.2f));
    OPTICS_REQUIRE(close(s.phase.secondaryWeight, .25f));
    break;
  }
  case 10: {
    const auto tf =
        evaluateTransferFunctionOptics(vec4(.2f, .4f, .8f, .5f), 2.f);
    const float ext = 2.f * logf(2.f);
    OPTICS_REQUIRE(close(tf.coefficients.absorption, vec3(ext)));
    OPTICS_REQUIRE(close(tf.coefficients.scattering, vec3(0.f)));
    OPTICS_REQUIRE(close(tf.coefficients.emission, vec3(.2f, .4f, .8f) * ext));
    OPTICS_REQUIRE(transferFunctionExtinction(0.f, 1.f) == 0.f);
    OPTICS_REQUIRE(transferFunctionExtinction(1e-8f, 1.f) == 0.f);
    OPTICS_REQUIRE(transferFunctionExtinction(.5f, 0.f) == 0.f);
    OPTICS_REQUIRE(
        close(transferFunctionExtinction(1.f, 1.f), 24.f * logf(2.f)));
    break;
  }
  case 11: {
    VolumeCoefficients a{vec3(1.f, 0.f, 2.f), vec3(3.f, 0.f, 0.f), vec3(2.f)};
    VolumeCoefficients b{vec3(0.f, 0.f, 1.f), vec3(1.f, 0.f, 0.f), vec3(4.f)};
    const auto sum = addVolumeCoefficients(a, b);
    OPTICS_REQUIRE(close(volumeExtinction(sum), vec3(5.f, 0.f, 3.f)));
    OPTICS_REQUIRE(close(sum.emission, vec3(6.f)));
    OPTICS_REQUIRE(close(volumeScatteringAlbedo(sum), vec3(.8f, 0.f, 0.f)));
    OPTICS_REQUIRE(close(
        volumeExtinction(addVolumeCoefficients(b, a)), volumeExtinction(sum)));
    break;
  }
  case 12: {
    OPTICS_REQUIRE(close(henyeyGreenstein(0.f, -.5f), 1.f / (4.f * kPi)));
    OPTICS_REQUIRE(henyeyGreenstein(.5f, 1.f) > henyeyGreenstein(.5f, -1.f));
    OPTICS_REQUIRE(
        close(henyeyGreenstein(.5f, .3f), henyeyGreenstein(-.5f, -.3f)));
    VolumePhaseFunction phase{.5f, -.3f, 0.f};
    OPTICS_REQUIRE(
        close(evaluateVolumePhase(phase, .2f), henyeyGreenstein(.5f, .2f)));
    phase.secondaryWeight = 1.f;
    OPTICS_REQUIRE(
        close(evaluateVolumePhase(phase, .2f), henyeyGreenstein(-.3f, .2f)));
    phase.secondaryWeight = .25f;
    OPTICS_REQUIRE(close(evaluateVolumePhase(phase, .2f),
        .75f * henyeyGreenstein(.5f, .2f)
            + .25f * henyeyGreenstein(-.3f, .2f)));
    break;
  }
  case 13: {
    VolumeOpticalSample a;
    a.coefficients.scattering = vec3(1.f, 0.f, 2.f);
    a.phase.anisotropy = .5f;
    VolumeOpticalSample b;
    b.coefficients.scattering = vec3(0.f, 3.f, 1.f);
    b.phase.anisotropy = -.5f;
    const vec3 kernel =
        volumeScatteringKernel(a, .25f) + volumeScatteringKernel(b, .25f);
    const float pa = henyeyGreenstein(.5f, .25f);
    const float pb = henyeyGreenstein(-.5f, .25f);
    OPTICS_REQUIRE(close(kernel, vec3(pa, 3.f * pb, 2.f * pa + pb)));
    break;
  }
  case 14: {
    // Numerically integrate the phase over the sphere. Do not only test a few
    // values or a formula rearranged identically to the implementation.
    VolumePhaseFunction phase{.6f, -.4f, .3f};
    float integral = 0.f;
    constexpr int N = 8192;
    for (int i = 0; i < N; ++i)
      integral += evaluateVolumePhase(phase, -1.f + (i + .5f) * (2.f / N));
    OPTICS_REQUIRE(close(integral * (4.f * kPi / N), 1.f, 2e-4f));
    break;
  }
  case 15: {
    // Exercise every field-driven input, not just the constant-value adapter.
    PrincipledVolumeBindingsGPUData b;
    b.bounds = box;
    b.density.field = 0;
    b.transmissionDistance.field = 1;
    b.anisotropy.field = 2;
    b.secondaryAnisotropy.field = 3;
    b.secondaryWeight.field = 4;
    b.temperature.field = 5;
    b.blackbodyIntensity.field = 6;
    b.scatteringColor.field = 7;
    b.transmissionColor.field = 8;
    b.emission.field = 9;
    b.blackbodyTint.field = 10;
    b.density.domain = b.transmissionDistance.domain = b.anisotropy.domain =
        box;
    b.secondaryAnisotropy.domain = b.secondaryWeight.domain = box;
    b.temperature.domain = b.blackbodyIntensity.domain = box;
    b.scatteringColor.domain = b.transmissionColor.domain = box;
    b.emission.domain = b.blackbodyTint.domain = box;
    b.scatteringColor.scale = vec3(.1f);
    b.transmissionColor.scale = vec3(.2f);
    b.anisotropy.scale = .2f;
    b.secondaryAnisotropy.scale = -.2f;
    b.secondaryWeight.scale = .4f;
    b.emission.scale = vec3(2.f);
    b.temperature.scale = 2000.f;
    b.blackbodyIntensity.scale = .8f;
    const auto s = evaluatePrincipledOptics(b, vec3(.25f), sampler, blackbody);
    const vec3 rgb(1.25f, 2.25f, 3.25f);
    const vec3 ext(-logf(.25f), -logf(.45f), -logf(.65f));
    const vec3 absorption = ext * (vec3(1.f) - rgb * .1f);
    OPTICS_REQUIRE(close(volumeExtinction(s.coefficients), ext));
    OPTICS_REQUIRE(close(s.coefficients.emission,
        rgb * 2.f + absorption * rgb * blackbody.radiance));
    OPTICS_REQUIRE(close(s.phase.anisotropy, .25f));
    OPTICS_REQUIRE(close(s.phase.secondaryAnisotropy, -.25f));
    OPTICS_REQUIRE(close(s.phase.secondaryWeight, .5f));
    OPTICS_REQUIRE(blackbody.calls == 1 && blackbody.lastTemperature == 2500.f);
    OPTICS_REQUIRE(sampler.scalarCalls == 7 && sampler.rgbCalls == 4);
    break;
  }
  case 16: {
    PrincipledVolumeValues v;
    v.transmissionColor = vec3(1.f);
    v.transmissionDistance = 1e-30f;
    v.density = 1e20f;
    v.blackbodyIntensity = 1.f;
    v.emission = vec3(2.f);
    const auto vacuum = evaluatePrincipledOptics(v, blackbody);
    OPTICS_REQUIRE(close(volumeExtinction(vacuum.coefficients), vec3(0.f)));
    OPTICS_REQUIRE(close(vacuum.coefficients.emission, v.emission));
    OPTICS_REQUIRE(blackbody.calls == 0);
    FieldBindingGPUData<float> f{2.f, 7.f};
    f.field = 0;
    f.domain = box;
    OPTICS_REQUIRE(evaluateFieldBinding(f, vec3(2.f), sampler) == 6.f);
    OPTICS_REQUIRE(
        evaluateFieldBinding(f, vec3(nextafterf(2.f, 3.f)), sampler) == 14.f);
    OPTICS_REQUIRE(sampler.scalarCalls == 1);
    break;
  }
  default:
    return -1;
  }
  return 0;
}

#undef OPTICS_REQUIRE

} // namespace visrtx::test
