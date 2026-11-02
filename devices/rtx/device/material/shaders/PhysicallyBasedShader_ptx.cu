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

#include "gpu/adjustNormalForView.h"
#include "gpu/gpu_decl.h"
#include "gpu/gpu_math.h"
#include "gpu/gpu_objects.h"
#include "gpu/sampleLight.h"
#include "gpu/shadingState.h"
#include "gpu/shading_api.h"

using namespace visrtx;

// Clearcoat is fixed to IOR 1.5 per glTF KHR_materials_clearcoat (F0 = 0.04).
constexpr float CLEARCOAT_F0 = 0.04f;

//-----------------------------------------------------------------------------
// Helpers
//-----------------------------------------------------------------------------

VISRTX_DEVICE vec3 applyNormalMap(
    const vec3 &tangentSpaceNormal, const SurfaceHit &hit, const vec3 &N)
{
  vec3 T = normalize(hit.tU);
  vec3 B = normalize(hit.tV);
  // Gram-Schmidt to build an orthonormal frame tied to N.
  T = normalize(T - dot(T, N) * N);
  B = normalize(B - dot(B, N) * N - dot(B, T) * T);
  return normalize(T * tangentSpaceNormal.x + B * tangentSpaceNormal.y
      + N * tangentSpaceNormal.z);
}

VISRTX_DEVICE vec3 sampleNormalMap(const FrameGPUData &fd,
    DeviceObjectIndex samplerIdx,
    const SurfaceHit &hit,
    const vec3 &fallback)
{
  if (samplerIdx == ~visrtx::DeviceObjectIndex{0})
    return fallback;
  // Per the ANARI spec the sampler already returns the tangent-space normal;
  // decoding texels (2 * texel - 1) is the app's job, via outTransform and
  // outOffset.
  const vec3 ts = normalize(vec3(evaluateSampler(fd, samplerIdx, hit)));
  const vec3 N = applyNormalMap(ts, hit, hit.Ns);
  // Negated comparison catches NaN (zero-valued samples, zero-summed tangents)
  // as well as zero-length results — fall back to the geometric normal so the
  // shading frame is always usable.
  return (dot(N, N) > 1e-12f) ? N : hit.Ng;
}

VISRTX_DEVICE vec3 computeVolumeTransmission(
    const PhysicallyBasedShadingState *state)
{
  if (!(state->thickness > 0.0f && state->attenuationDistance > 0.0f
          && isfinite(state->attenuationDistance)))
    return vec3(1.0f);

  const float k = state->thickness / state->attenuationDistance;
  return vec3(powf(fmaxf(state->attenuationColor.x, 1e-6f), k),
      powf(fmaxf(state->attenuationColor.y, 1e-6f), k),
      powf(fmaxf(state->attenuationColor.z, 1e-6f), k));
}

VISRTX_DEVICE vec3 computeTransmissionFilter(
    const PhysicallyBasedShadingState *state)
{
  const float transmission =
      fmaxf(0.0f, (1.0f - state->metallic) * state->transmission);
  return state->baseColor * transmission * computeVolumeTransmission(state);
}

// Smith Lambda for GGX (common subterm of G1 / G2).
VISRTX_DEVICE float smithLambdaGGX(float NdotX, float alpha2)
{
  const float NdotX2 = NdotX * NdotX;
  const float safe = fmaxf(NdotX2, 1e-8f);
  return 0.5f
      * (-1.0f + sqrtf(fmaxf(0.0f, 1.0f + alpha2 * (1.0f - safe) / safe)));
}

VISRTX_DEVICE float smithG2GGX(float NdotV, float NdotL, float alpha2)
{
  return 1.0f
      / (1.0f + smithLambdaGGX(NdotV, alpha2) + smithLambdaGGX(NdotL, alpha2));
}

VISRTX_DEVICE float smithG1GGX(float NdotV, float alpha2)
{
  return 1.0f / (1.0f + smithLambdaGGX(NdotV, alpha2));
}

VISRTX_DEVICE float ggxD(float NdotH, float alpha2)
{
  // The textbook denom `x·(α²−1) + 1` cancels catastrophically in fp32 once
  // α² is below eps(1) ≈ 1.19e-7 (our α² floor is 1e-8): `α²−1` rounds to
  // exactly −1, and at x=1 the whole denom collapses to 0. The algebraically
  // equivalent `α²·x + (1−x)` has no near-1 subtraction so it stays exact.
  // The fminf clamp keeps `1−x ≥ 0` against dot-product rounding above 1.
  const float NdotH2 = fminf(NdotH * NdotH, 1.0f);
  const float denom = alpha2 * NdotH2 + (1.0f - NdotH2);
  return alpha2 / (kPi * denom * denom);
}

// Heitz 2018 (https://jcgt.org/published/0007/04/01/) visible-normal sampling
// for GGX. Ve is the view direction in local tangent space (+z = normal).
VISRTX_DEVICE vec3 sampleGGXVNDF(
    const vec3 &Ve, float alpha, float u1, float u2)
{
  const vec3 Vh = normalize(vec3(alpha * Ve.x, alpha * Ve.y, Ve.z));
  const float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
  const vec3 T1 = lensq > 0.0f ? vec3(-Vh.y, Vh.x, 0.0f) * (1.0f / sqrtf(lensq))
                               : vec3(1.0f, 0.0f, 0.0f);
  const vec3 T2 = glm::cross(Vh, T1);
  const float r = sqrtf(u1);
  const float phi = kTwoPi * u2;
  const float t1 = r * cosf(phi);
  float t2 = r * sinf(phi);
  const float s = 0.5f * (1.0f + Vh.z);
  t2 = (1.0f - s) * sqrtf(fmaxf(0.0f, 1.0f - t1 * t1)) + s * t2;
  const vec3 Nh =
      t1 * T1 + t2 * T2 + sqrtf(fmaxf(0.0f, 1.0f - t1 * t1 - t2 * t2)) * Vh;
  return normalize(vec3(alpha * Nh.x, alpha * Nh.y, fmaxf(0.0f, Nh.z)));
}

// Charlie distribution (Estevez-Kulla 2017) for sheen.
VISRTX_DEVICE float charlieD(float NdotH, float alpha)
{
  const float invAlpha = 1.0f / fmaxf(alpha, 1e-4f);
  const float sin2 = fmaxf(0.0f, 1.0f - NdotH * NdotH);
  return (2.0f + invAlpha) * powf(sin2, 0.5f * invAlpha) / (kTwoPi);
}

// Ashikhmin visibility term (Neubelt-Pettineo variant) used with Charlie D.
VISRTX_DEVICE float charlieV(float NdotV, float NdotL)
{
  return 1.0f / (4.0f * (NdotV + NdotL - NdotV * NdotL) + 1e-6f);
}

// glTF KHR_materials_iridescence thin-film Fresnel (port of the reference
// implementation at github.com/KhronosGroup/glTF-Sample-Renderer). Returns a
// per-channel Fresnel reflectance for a thin film of thickness T sitting on a
// base with Schlick F0. See the spec's Appendix B for the math.
VISRTX_DEVICE vec3 fresnel0ToIor(vec3 F0)
{
  const vec3 s = glm::sqrt(glm::clamp(F0, vec3(0.0f), vec3(0.9999f)));
  return (vec3(1.0f) + s) / (vec3(1.0f) - s);
}

VISRTX_DEVICE vec3 iorToFresnel0(vec3 transmittedIor, float incidentIor)
{
  const vec3 t = (transmittedIor - vec3(incidentIor))
      / (transmittedIor + vec3(incidentIor));
  return t * t;
}

VISRTX_DEVICE float iorToFresnel0(float transmittedIor, float incidentIor)
{
  const float t =
      (transmittedIor - incidentIor) / (transmittedIor + incidentIor);
  return t * t;
}

VISRTX_DEVICE vec3 evalSensitivity(float opd, vec3 shift)
{
  // Approximate spectral sensitivity of the standard observer as three
  // Gaussians (Belcour & Barla 2017, simplified) so the result stays in RGB.
  const float phase = kTwoPi * opd * 1e-9f;
  const vec3 val = vec3(5.4856e-13f, 4.4201e-13f, 5.2481e-13f);
  const vec3 pos = vec3(1.6810e+06f, 1.7953e+06f, 2.2084e+06f);
  const vec3 var = vec3(4.3278e+09f, 9.3046e+09f, 6.6121e+09f);

  vec3 xyz = val * glm::sqrt(kTwoPi * var) * glm::cos(pos * phase + shift)
      * glm::exp(-var * phase * phase);
  xyz.x += 9.7470e-14f * sqrtf(kTwoPi * 4.5282e+09f)
      * cosf(2.2399e+06f * phase + shift.x)
      * expf(-4.5282e+09f * phase * phase);
  xyz /= 1.0685e-7f;

  // sRGB conversion (D65).
  return vec3(3.2404542f * xyz.x - 1.5371385f * xyz.y - 0.4985314f * xyz.z,
      -0.9692660f * xyz.x + 1.8760108f * xyz.y + 0.0415560f * xyz.z,
      0.0556434f * xyz.x - 0.2040259f * xyz.y + 1.0572252f * xyz.z);
}

VISRTX_DEVICE vec3 evalIridescence(float outsideIor,
    float iridescenceIor,
    float cosTheta1,
    float thickness,
    vec3 baseF0)
{
  // Handle the case where thin-film IOR is close to the outside IOR: return
  // the base Fresnel to avoid division by zero and phase artifacts.
  const float iridescenceIorSafe = fmaxf(iridescenceIor, outsideIor + 1e-4f);

  // Force iridescenceIor > outsideIor (otherwise Snell's law cannot refract).
  const float sinTheta2Sq =
      pow2(outsideIor / iridescenceIorSafe) * (1.0f - cosTheta1 * cosTheta1);
  const float cosTheta2Sq = 1.0f - sinTheta2Sq;
  if (cosTheta2Sq < 0.0f)
    return vec3(1.0f); // Total internal reflection.
  const float cosTheta2 = sqrtf(cosTheta2Sq);

  // First interface: Fresnel between outside and thin film.
  const float R0_12 = iorToFresnel0(iridescenceIorSafe, outsideIor);
  const float R12 = R0_12 + (1.0f - R0_12) * pow5(1.0f - cosTheta1);
  const float T121 = 1.0f - R12;
  const float phi12 = iridescenceIorSafe < outsideIor ? kPi : 0.0f;
  const float phi21 = kPi - phi12;

  // Second interface: film to base.
  const vec3 baseIor =
      fresnel0ToIor(glm::clamp(baseF0, vec3(0.f), vec3(0.9999f)));
  const vec3 R1 = iorToFresnel0(baseIor, iridescenceIorSafe);
  const vec3 R23 = R1 + (vec3(1.0f) - R1) * pow5(1.0f - cosTheta2);
  const vec3 phi23 = vec3(baseIor.x < iridescenceIorSafe ? kPi : 0.0f,
      baseIor.y < iridescenceIorSafe ? kPi : 0.0f,
      baseIor.z < iridescenceIorSafe ? kPi : 0.0f);

  const float opd = 2.0f * iridescenceIorSafe * thickness * cosTheta2;
  const vec3 phi = vec3(phi21) + phi23;

  const vec3 R123 = glm::clamp(R12 * R23, vec3(1e-5f), vec3(0.9999f));
  const vec3 r123 = glm::sqrt(R123);
  const vec3 Rs = pow2(T121) * R23 / (vec3(1.0f) - R123);

  // DC term.
  vec3 C0 = R12 + Rs;
  vec3 I = C0;

  // Higher-order terms.
  vec3 Cm = Rs - T121;
  for (int m = 1; m <= 2; ++m) {
    Cm *= r123;
    const vec3 Sm = 2.0f * evalSensitivity(float(m) * opd, float(m) * phi);
    I += Cm * Sm;
  }

  return glm::max(I, vec3(0.0f));
}

//-----------------------------------------------------------------------------
// Initialize shading state from material parameters
//-----------------------------------------------------------------------------

VISRTX_CALLABLE void __direct_callable__init(
    PhysicallyBasedShadingState *shadingState,
    const FrameGPUData *fd,
    const SurfaceHit *hit,
    const MaterialGPUData::PhysicallyBased *md)
{
  const vec4 color = getMaterialParameter(*fd, md->baseColor, *hit);
  const float opacity = getMaterialParameter(*fd, md->opacity, *hit).x;
  shadingState->baseColor = vec3(color);
  shadingState->geometricNormal = hit->Ng;

  const vec3 N = sampleNormalMap(*fd, md->normalSampler, *hit, hit->Ns);
  shadingState->shadingNormal = N;
  shadingState->viewAdjustedNormal = adjustNormalForView(hit->Ng, hit->V, N);

  shadingState->opacity =
      adjustedMaterialOpacity(color.w * opacity, md->alphaMode, md->cutoff);
  shadingState->eta = hit->isFrontFace ? 1.0f / md->ior : md->ior;
  shadingState->metallic = getMaterialParameter(*fd, md->metallic, *hit).x;
  shadingState->roughness = getMaterialParameter(*fd, md->roughness, *hit).x;
  shadingState->emission = vec3(getMaterialParameter(*fd, md->emissive, *hit));
  shadingState->transmission =
      getMaterialParameter(*fd, md->transmission, *hit).x;

  shadingState->occlusion =
      md->occlusionSampler == ~visrtx::DeviceObjectIndex{0}
      ? 1.0f
      : evaluateSampler(*fd, md->occlusionSampler, *hit).x;

  shadingState->specular = getMaterialParameter(*fd, md->specular, *hit).x;
  shadingState->specularColor =
      vec3(getMaterialParameter(*fd, md->specularColor, *hit));

  shadingState->clearcoat = getMaterialParameter(*fd, md->clearcoat, *hit).x;
  shadingState->clearcoatRoughness =
      getMaterialParameter(*fd, md->clearcoatRoughness, *hit).x;
  const vec3 Nc =
      sampleNormalMap(*fd, md->clearcoatNormalSampler, *hit, hit->Ns);
  shadingState->viewAdjustedClearcoatNormal =
      adjustNormalForView(hit->Ng, hit->V, Nc);

  shadingState->thickness = getMaterialParameter(*fd, md->thickness, *hit).x;
  shadingState->attenuationDistance = md->attenuationDistance;
  shadingState->attenuationColor = md->attenuationColor;

  shadingState->sheenColor =
      vec3(getMaterialParameter(*fd, md->sheenColor, *hit));
  shadingState->sheenRoughness =
      getMaterialParameter(*fd, md->sheenRoughness, *hit).x;

  shadingState->iridescence =
      getMaterialParameter(*fd, md->iridescence, *hit).x;
  shadingState->iridescenceIor = md->iridescenceIor;
  shadingState->iridescenceThickness =
      getMaterialParameter(*fd, md->iridescenceThickness, *hit).x;
}

//-----------------------------------------------------------------------------
// Simple accessors
//-----------------------------------------------------------------------------

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
  return shadingState->shadingNormal;
}

//-----------------------------------------------------------------------------
// NEE shading: base (diffuse + GGX specular) + clearcoat + sheen
//-----------------------------------------------------------------------------

VISRTX_DEVICE vec3 computeDielectricF0(const PhysicallyBasedShadingState *state)
{
  // KHR_materials_specular: `specular` scales the dielectric reflection, so the
  // ANARI default specular = 0 yields a pure-diffuse dielectric (no Fresnel).
  const float iorF0 = pow2((1.0f - state->eta) / (1.0f + state->eta));
  return glm::min(vec3(iorF0) * state->specularColor, vec3(1.0f))
      * state->specular;
}

VISRTX_DEVICE vec3 computeF0(const PhysicallyBasedShadingState *state)
{
  return glm::mix(
      computeDielectricF0(state), state->baseColor, state->metallic);
}

VISRTX_DEVICE vec3 computeF90(const PhysicallyBasedShadingState *state)
{
  return glm::mix(vec3(state->specular), vec3(1.0f), state->metallic);
}

VISRTX_DEVICE vec3 schlickFresnel(vec3 F0, vec3 F90, float VdotH)
{
  return F0 + (F90 - F0) * pow5(1.0f - fabsf(VdotH));
}

VISRTX_DEVICE vec3 evalFresnelWithIridescence(
    const PhysicallyBasedShadingState *state,
    const vec3 &F0,
    const vec3 &F90,
    float cosTheta)
{
  vec3 F = schlickFresnel(F0, F90, cosTheta);
  if (state->iridescence > 0.0f && state->iridescenceThickness > 0.0f) {
    const vec3 iridescent = evalIridescence(
        1.0f, state->iridescenceIor, cosTheta, state->iridescenceThickness, F0);
    F = glm::mix(F, iridescent, state->iridescence);
  }
  return F;
}

VISRTX_DEVICE float pbrCoatPick(
    const PhysicallyBasedShadingState *s, const vec3 &v)
{
  const float c = fmaxf(dot(s->viewAdjustedClearcoatNormal, v), 0.f);
  return glm::clamp(
      s->clearcoat * (CLEARCOAT_F0 + (1.f - CLEARCOAT_F0) * pow5(1.f - c)),
      0.f,
      1.f);
}
VISRTX_DEVICE float pbrCoatExit(
    const PhysicallyBasedShadingState *s, const vec3 &l)
{
  if (s->clearcoat == 0.f)
    return 1.f;
  const float c = fabsf(dot(s->viewAdjustedClearcoatNormal, l));
  return glm::clamp(1.f
          - s->clearcoat
              * (CLEARCOAT_F0 + (1.f - CLEARCOAT_F0) * pow5(1.f - c)),
      0.f,
      1.f);
}
VISRTX_DEVICE bool pbrTransmissionFold(
    const PhysicallyBasedShadingState *s, const vec3 &v, const vec3 &h)
{
  if (!(luminance(computeTransmissionFilter(s)) > 0.f))
    return false;
  const auto transmitted = glm::refract(-v, h, s->eta);
  // Retained compatibility fold: true TIR OR refraction on the macro wrong
  // side.
  return glm::length(transmitted) < 1e-6f
      || dot(transmitted, s->viewAdjustedNormal) >= 0.f;
}

// One regularized reflection target, shared by NEE and every reflected sample.
// Floors below belong to the accepted target model, NEVER proposal
// probabilities.
VISRTX_DEVICE vec3 pbrReflectionKernel(
    const PhysicallyBasedShadingState *s, const vec3 &v, const vec3 &l)
{
  if (!(dot(s->geometricNormal, v) > 0.f)
      || !(dot(s->geometricNormal, l) > 0.f))
    return vec3(0);
  const auto n = s->viewAdjustedNormal, nc = s->viewAdjustedClearcoatNormal;
  const float nv = dot(n, v), nl = dot(n, l), cv = dot(nc, v), cl = dot(nc, l);
  const bool base = nv > 0.f && nl > 0.f,
             coat = s->clearcoat > 0.f && cv > 0.f && cl > 0.f;
  if (!base && !coat)
    return vec3(0);
  const auto h = normalize(v + l);
  const float vh = fmaxf(dot(v, h), 0.f);
  vec3 result(0);
  if (base) {
    const float vs = fmaxf(nv, 1e-6f), ls = fmaxf(nl, 1e-6f),
                nh = fmaxf(dot(n, h), 0.f);
    const auto f0 = computeF0(s), f90 = computeF90(s);
    const auto fv = evalFresnelWithIridescence(s, f0, f90, vs);
    const auto er = pbrTransmissionFold(s, v, h)
        ? vec3(1.f)
        : evalFresnelWithIridescence(s, f0, f90, vh);
    const auto ed = glm::max(vec3(1.f) - fv, vec3(0)) * s->baseColor
        * (1.f - s->metallic) * (1.f - s->transmission) * s->occlusion;
    const float alpha = fmaxf(pow2(s->roughness), 1e-4f),
                alpha2 = alpha * alpha;
    result = ((ed * (nl * kInvPi))
                 + er
                     * (ggxD(nh, alpha2) * smithG2GGX(vs, ls, alpha2) * nl
                         / (4.f * vs * ls)))
        * ((1.f - pbrCoatPick(s, v)) * pbrCoatExit(s, l));
    if (glm::any(glm::greaterThan(s->sheenColor, vec3(0)))) {
      const float as = fmaxf(pow2(s->sheenRoughness), 1e-4f);
      result += s->sheenColor * (charlieD(nh, as) * charlieV(vs, ls) * nl);
    }
  }
  if (coat) {
    const float vs = fmaxf(cv, 1e-6f), ls = fmaxf(cl, 1e-6f);
    const float alpha = fmaxf(pow2(s->clearcoatRoughness), 1e-4f),
                alpha2 = alpha * alpha;
    const float fc = CLEARCOAT_F0 + (1.f - CLEARCOAT_F0) * pow5(1.f - vh);
    result += vec3(s->clearcoat * fc * ggxD(fmaxf(dot(nc, h), 0.f), alpha2)
        * smithG2GGX(vs, ls, alpha2) * cl / (4.f * vs * ls));
  }
  return result;
}
VISRTX_CALLABLE vec3 __direct_callable__evalBsdf(
    const PhysicallyBasedShadingState *state, const vec3 *wo, const vec3 *wi)
{
  return pbrReflectionKernel(state, *wo, *wi);
}

// Ideal visible-GGX reflection Jacobian, without the target Smith floor.
VISRTX_DEVICE float pbrVndfReflectionPdf(
    const vec3 &n, const vec3 &v, const vec3 &l, float alpha)
{
  const float nv = dot(n, v);
  if (!(nv > 0.f) || !(dot(n, l) > 0.f))
    return 0.f;
  const auto h = normalize(v + l);
  const float a2 = alpha * alpha;
  return ggxD(fmaxf(dot(n, h), 0.f), a2)
      / (2.f * (nv + sqrtf(a2 + (1.f - a2) * nv * nv)));
}
VISRTX_DEVICE float pbrOldReflectionPdf(
    const PhysicallyBasedShadingState *s, const vec3 &v, const vec3 &l)
{
  const auto n = s->viewAdjustedNormal, nc = s->viewAdjustedClearcoatNormal;
  const float nv = dot(n, v), nl = dot(n, l), pc = pbrCoatPick(s, v);
  float density = pc
      * pbrVndfReflectionPdf(
          nc, v, l, fmaxf(pow2(s->clearcoatRoughness), 1e-4f));
  if (nv > 0.f && nl > 0.f && pc < 1.f) {
    const auto f0 = computeF0(s), f90 = computeF90(s);
    const auto fv = evalFresnelWithIridescence(s, f0, f90, nv);
    const auto tf = computeTransmissionFilter(s);
    const auto complement = glm::max(vec3(1.f) - fv, vec3(0));
    const float spec =
        fmaxf(luminance(fv), 0.f) + fmaxf(luminance(complement * tf), 0.f);
    const float diff =
        fmaxf(luminance(complement * s->baseColor * (1.f - s->metallic)
                  * (1.f - s->transmission) * s->occlusion),
            0.f);
    if (spec + diff > 0.f) {
      const auto h = normalize(v + l);
      const auto f =
          evalFresnelWithIridescence(s, f0, f90, fmaxf(dot(v, h), 0.f));
      const bool fold = pbrTransmissionFold(s, v, h);
      const float r = fold ? 1.f : fmaxf(luminance(f), 0.f);
      const float t = fold
          ? 0.f
          : fmaxf(luminance(glm::max(vec3(1.f) - f, vec3(0)) * tf), 0.f);
      const float reflection = r + t > 0.f ? r / (r + t) : 1.f;
      density += (1.f - pc)
          * (diff / (spec + diff) * nl * kInvPi
              + spec / (spec + diff) * reflection
                  * pbrVndfReflectionPdf(
                      n, v, l, fmaxf(pow2(s->roughness), 1e-4f)));
    }
  }
  return density;
}
VISRTX_DEVICE float pbrBsdfPdf(
    const PhysicallyBasedShadingState *s, const vec3 &v, const vec3 &l)
{
  if (!(dot(s->geometricNormal, v) > 0.f)
      || !(dot(s->geometricNormal, l) > 0.f))
    return 0.f;
  return .875f * pbrOldReflectionPdf(s, v, l) + .125f / (2.f * kPi);
}

// Retained O proposal: identical conditional draw order and transmitted
// measure. Reflection results here are proposal records, not final K/q samples.
VISRTX_DEVICE NextRay sampleNativeProposal(
    const PhysicallyBasedShadingState *state,
    const Ray *ray,
    RandState *rs,
    const NextRayQmc *qmc,
    int &qmcCursor)
{
  const vec3 V = -ray->dir;
  auto u = [&]() { return nextRayUniform(rs, qmc, qmcCursor); };
  const vec3 Nc = state->viewAdjustedClearcoatNormal;
  const float clearcoatPick = pbrCoatPick(state, V);
  if (clearcoatPick > 0.f && u() < clearcoatPick) {
    const auto basis = computeOrthonormalBasis(Nc);
    const auto local = glm::transpose(basis) * V;
    if (local.z <= 0.f)
      return deadNextRay();
    const float alpha = fmaxf(pow2(state->clearcoatRoughness), 1e-4f);
    const float a = u(), b = u();
    const auto h = sampleGGXVNDF(local, alpha, a, b);
    const auto l = glm::reflect(-local, h);
    if (l.z <= 0.f)
      return deadNextRay();
    return {normalize(basis * l),
        vec3(0),
        0.f,
        NEXT_RAY_NONE,
        NextRayEvent::Continuous,
        0,
        NextRayComponent::NativeReflection};
  }
  const vec3 N = state->viewAdjustedNormal;
  const auto basis = computeOrthonormalBasis(N);
  const auto local = glm::transpose(basis) * V;
  if (local.z <= 0.f)
    return deadNextRay();
  const float alpha = fmaxf(pow2(state->roughness), 1e-4f),
              alpha2 = alpha * alpha;
  const auto f0 = computeF0(state), f90 = computeF90(state);
  const auto fv = evalFresnelWithIridescence(state, f0, f90, local.z);
  const auto tf = computeTransmissionFilter(state);
  const auto diff = glm::max(vec3(1.f) - fv, vec3(0)) * state->baseColor
      * (1.f - state->metallic) * (1.f - state->transmission)
      * state->occlusion;
  const float sw = fmaxf(luminance(fv), 0.f)
      + fmaxf(luminance(glm::max(vec3(1.f) - fv, vec3(0)) * tf), 0.f);
  const float dw = fmaxf(luminance(diff), 0.f);
  if (sw + dw <= 0.f)
    return deadNextRay();
  const float ps = sw / (sw + dw);
  if (u() >= ps) {
    const float a = u(), b = u();
    return {sampleHemisphere(a, b, N),
        vec3(0),
        0.f,
        NEXT_RAY_NONE,
        NextRayEvent::Continuous,
        0,
        NextRayComponent::NativeReflection};
  }
  const float a = u(), b = u();
  const auto h = sampleGGXVNDF(local, alpha, a, b);
  const auto f =
      evalFresnelWithIridescence(state, f0, f90, fmaxf(dot(local, h), 0.f));
  const auto reflected = glm::reflect(-local, h);
  const auto transmitted = glm::refract(-local, h, state->eta);
  const bool fold = luminance(tf) > 0.f
      && (glm::length(transmitted) < 1e-6f || transmitted.z >= 0.f);
  const auto er = fold ? vec3(1.f) : f;
  const auto et = fold ? vec3(0) : glm::max(vec3(1.f) - f, vec3(0)) * tf;
  const float rw = fmaxf(luminance(er), 0.f), tw = fmaxf(luminance(et), 0.f);
  const float r = rw + tw > 0.f ? rw / (rw + tw) : 1.f;
  if (u() < r) {
    if (reflected.z <= 0.f)
      return deadNextRay();
    return {normalize(basis * reflected),
        vec3(0),
        0.f,
        NEXT_RAY_NONE,
        NextRayEvent::Continuous,
        0,
        NextRayComponent::NativeReflection};
  }
  const auto direction =
      state->eta == 1.f ? -V : normalize(basis * transmitted);
  // Keep the old transmitted compensation floors and target approximation.
  const auto weight = et
      * (smithG2GGX(local.z, -transmitted.z, alpha2)
          / fmaxf(smithG1GGX(local.z, alpha2), 1e-8f))
      * pbrCoatExit(state, direction) / fmaxf(ps * (1.f - r), 1e-8f);
  return {direction,
      weight,
      INFINITY,
      NEXT_RAY_CONTINUES_THROUGH_SURFACE,
      state->eta == 1.f ? NextRayEvent::Singular : NextRayEvent::Continuous,
      0,
      NextRayComponent::NativeTransmission};
}
VISRTX_DEVICE NextRay sampleNativeReflection(
    const PhysicallyBasedShadingState *state,
    const Ray *ray,
    RandState *rs,
    const NextRayQmc *qmc,
    float outerChoice)
{
  const vec3 v = -ray->dir;
  if (!(dot(state->geometricNormal, v) > 0.f))
    return deadNextRay();
  int cursor = 0;
  NextRay result;
  if (outerChoice < .125f) {
    const float a = nextRayUniform(rs, qmc, cursor),
                b = nextRayUniform(rs, qmc, cursor);
    const float z = 1.f - a, r = sqrtf(1.f - z * z), phi = kTwoPi * b;
    result.direction = normalize(computeOrthonormalBasis(state->geometricNormal)
        * vec3(r * cosf(phi), r * sinf(phi), z));
    result.component = NextRayComponent::NativeReflection;
  } else {
    result = sampleNativeProposal(state, ray, rs, qmc, cursor);
    if (isDead(result))
      return result;
    if (result.component == NextRayComponent::NativeTransmission) {
      result.contributionWeight /= .875f;
      return result;
    }
  }
  const auto kernel = pbrReflectionKernel(state, v, result.direction);
  if (kernel == vec3(0))
    return deadNextRay();
  const float pdf = pbrBsdfPdf(state, v, result.direction);
  return {result.direction,
      kernel / pdf,
      pdf,
      NEXT_RAY_NONE,
      NextRayEvent::Continuous,
      0,
      NextRayComponent::NativeReflection};
}

//-----------------------------------------------------------------------------
// BSDF sampling pdf at (wo, wi) for environment MIS — the same closed-form
// density nextRay reports in NextRay.pdf, so NEE-side and escape-side weights
// agree. Both directions are world space.
//-----------------------------------------------------------------------------

VISRTX_CALLABLE NextRay __direct_callable__nextRay(
    const PhysicallyBasedShadingState *state,
    const Ray *ray,
    RandState *rs,
    const NextRayQmc *qmc)
{
  if (!(dot(state->geometricNormal, -ray->dir) > 0.f))
    return deadNextRay();
  // This categorical choice must be conditional-independent across bounces.
  // Re-scrambling the SAME Sobol dimension at each bounce correlates its 1/8
  // prefix intervals (e.g. multiple glass interfaces). Keep the old proposal's
  // overlay order and use the path's PCG stream for the outer coin only.
  const float choice = pcg_uniform(rs);
  return sampleNativeReflection(state, ray, rs, qmc, choice);
}

VISRTX_CALLABLE float __direct_callable__evaluatePdf(
    const PhysicallyBasedShadingState *state, const vec3 *wo, const vec3 *wi)
{
  return pbrBsdfPdf(state, *wo, *wi);
}
