/*
 * Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// Wavefront MDL shade shell (ticket 10). Compiled to RELOCATABLE PTX at build
// time; at MDL material commit the device nvJitLinks this shell against a
// compiled material's stitched PTX (mdlInit / mdlBsdf_* / mdlEmission_* + the
// texture runtime), producing one loadable cubin per material. Runs in plain
// CUDA — it calls the MDL entries directly as linked device symbols, never
// through the OptiX direct-callable path the interactive renderer uses.
//
// The shell mirrors the builtin wavefront shade-emit stage
// (WavefrontLaunch.cu): for each assigned pool slot it evaluates the surface,
// picks ONE light for next-event estimation, and writes the deferred shade
// record the shared shadow-trace + resolve stages consume. The only difference
// is the BSDF: MDL's bsdf_evaluate / EDF replace the Lambert term.

// Geometry-Light NEE uses the light's mean radiance here (the static path), so
// sampleLight()'s GEOMETRY branch stays out of ptxas codegen — an
// optixDirectCall would leave an unresolved symbol in this CUDA-only shell.
// Must precede EVERY include (sampleLight.h is pulled transitively under
// #pragma once; a later define would be too late).
#define VISRTX_STATIC_GEOMETRY_LIGHT_EMISSION

#include "gpu/evalMaterialParameters.h" // readAttributeValue
#include "gpu/gpu_decl.h" // VISRTX_CALLABLE, VISRTX_GLOBAL
#include "gpu/gpu_objects.h" // FrameGPUData, Wavefront* records, MaterialGPUData
#include "gpu/sampleLight.h" // sampleLight, LightSample, ScreenSample
#include "gpu/shadingState.h" // MDLShadingState, TextureHandler

#include <anari/anari_cpp/ext/linalg.h>
#include <mi/neuraylib/target_code_types.h>

using namespace visrtx;

namespace {

using ShadingStateMaterial = mi::neuraylib::Shading_state_material;
using ResourceData = mi::neuraylib::Resource_data;
using BsdfEvaluateData =
    mi::neuraylib::Bsdf_evaluate_data<mi::neuraylib::DF_HSM_NONE>;
using BsdfAuxiliaryData =
    mi::neuraylib::Bsdf_auxiliary_data<mi::neuraylib::DF_HSM_NONE>;
using BsdfSampleData = mi::neuraylib::Bsdf_sample_data;
using EdfEvaluateData =
    mi::neuraylib::Edf_evaluate_data<mi::neuraylib::DF_HSM_NONE>;

using BsdfInitFunc = mi::neuraylib::Bsdf_init_function;
using BsdfEvaluateFunc = mi::neuraylib::Bsdf_evaluate_function;
using BsdfSampleFunc = mi::neuraylib::Bsdf_sample_function;
using BsdfAuxiliaryFunc = mi::neuraylib::Bsdf_auxiliary_function;
using EdfEvaluateFunc = mi::neuraylib::Edf_evaluate_function;
using OpacityExprFunc = mi::neuraylib::Material_function<float>::Type;
using EmissionIntensityExprFunc = mi::neuraylib::Material_function<vec3>::Type;

} // namespace

// The compiled-material PTX defines these; nvJitLink resolves them per
// material. Declared exactly as in MDLShader_ptx.cu (VISRTX_CALLABLE = extern
// "C"
// __device__) so the mangled names match the stitched material symbols.
VISRTX_CALLABLE BsdfInitFunc mdlInit;
VISRTX_CALLABLE BsdfEvaluateFunc mdlBsdf_evaluate;
VISRTX_CALLABLE BsdfSampleFunc mdlBsdf_sample;
VISRTX_CALLABLE BsdfAuxiliaryFunc mdlBsdf_auxiliary;
VISRTX_CALLABLE EdfEvaluateFunc mdlEmission_evaluate;
VISRTX_CALLABLE OpacityExprFunc mdlOpacity;
VISRTX_CALLABLE EmissionIntensityExprFunc mdlEmissionIntensity;

namespace {

// Populate the MDL shading state from the hit + material data. Mirrors
// MDLShader_ptx.cu's __direct_callable__init (single source of the ABI contract
// between the State layout the backend generates and what we fill).
__device__ void initMdlState(MDLShadingState &s,
    const FrameGPUData &fd,
    const SurfaceHit &hit,
    const MaterialGPUData::MDL &md)
{
  s.textureCoords[0] = readAttributeValue(MaterialAttribute::ATTRIB_0, hit);
  s.textureCoords[1] = readAttributeValue(MaterialAttribute::ATTRIB_1, hit);
  s.textureCoords[2] = readAttributeValue(MaterialAttribute::ATTRIB_2, hit);
  s.textureCoords[3] = readAttributeValue(MaterialAttribute::ATTRIB_3, hit);

  for (int i = 0; i < 4; ++i) {
    s.textureTangentsU[i] = hit.tU;
    s.textureTangentsV[i] = hit.tV;
  }

  s.state.animation_time = 0.0f;
  s.state.geom_normal = bit_cast<float3>(hit.Ng);
  s.state.normal = bit_cast<float3>(hit.Ns);
  s.state.position = bit_cast<float3>(hit.hitpoint);
  s.state.meters_per_scene_unit = 1.0f;
  s.state.object_id = hit.objID;
  s.state.object_to_world =
      reinterpret_cast<const float4 *>(&hit.instance->objectToWorld);
  s.state.world_to_object =
      reinterpret_cast<const float4 *>(&hit.instance->worldToObject);
  s.state.ro_data_segment = nullptr;
  s.state.text_coords = reinterpret_cast<const float3 *>(s.textureCoords);
  s.state.text_results = reinterpret_cast<float4 *>(s.textureResults);
  s.state.tangent_u = reinterpret_cast<const float3 *>(s.textureTangentsU);
  s.state.tangent_v = reinterpret_cast<const float3 *>(s.textureTangentsV);

  s.textureHandler.vtable = nullptr;
  s.textureHandler.fd = &fd;
  s.textureHandler.samplers = md.samplers;
  s.textureHandler.numSamplers = md.numSamplers;
  s.resData = {nullptr, &s.textureHandler};

  s.isFrontFace = hit.isFrontFace;
  s.argBlock = md.argBlock;

  mdlInit(&s.state, &s.resData, s.argBlock);
}

// View-independent BSDF albedo (auxiliary) for the AOV / ambient term. Matches
// MDLShader_ptx.cu's __direct_callable__evaluateTint.
__device__ vec3 evalMdlAlbedo(const MDLShadingState &s)
{
  BsdfAuxiliaryData aux = {};
  aux.ior1 = make_float3(1.0f, 1.0f, 1.0f);
  aux.ior2.x = MI_NEURAYLIB_BSDF_USE_MATERIAL_IOR;
  aux.k1 = s.state.normal;
  mdlBsdf_auxiliary(&aux, &s.state, &s.resData, s.argBlock);
  return make_vec3(aux.albedo_diffuse) + make_vec3(aux.albedo_glossy);
}

// Emitted radiance L(wo) = edf(wo) * intensity. Matches
// __direct_callable__evaluateEmission (intensity is radiant exitance, edf is
// 1/PI for diffuse EDF; no cos/pdf factors — those are the sample-path
// quantity).
__device__ vec3 evalMdlEmission(const MDLShadingState &s, const vec3 &wo)
{
  EdfEvaluateData edf = {};
  edf.k1 = make_float3(wo);
  mdlEmission_evaluate(&edf, &s.state, &s.resData, s.argBlock);
  const vec3 intensity = mdlEmissionIntensity(&s.state, &s.resData, s.argBlock);
  return make_vec3(edf.edf) * intensity;
}

// Direct-lighting BSDF term toward a sampled light. Matches
// __direct_callable__shadeSurface: k1 = view dir, k2 = light dir, contribution
// is (diffuse + glossy); the caller weights by radiance/pdf. MDL folds the
// cosine into the returned lobe values.
__device__ vec3 evalMdlBsdf(
    const MDLShadingState &s, const vec3 &wo, const vec3 &wi, float &pdfOut)
{
  pdfOut = 0.f;
  if (dot(wo, normalize(make_vec3(s.state.normal))) <= 0.0f)
    return vec3(0.f);
  BsdfEvaluateData eval = {};
  if (s.isFrontFace) {
    eval.ior1 = make_float3(1.0f, 1.0f, 1.0f);
    eval.ior2.x = MI_NEURAYLIB_BSDF_USE_MATERIAL_IOR;
  } else {
    eval.ior1.x = MI_NEURAYLIB_BSDF_USE_MATERIAL_IOR;
    eval.ior2 = make_float3(1.0f, 1.0f, 1.0f);
  }
  eval.k1 = make_float3(normalize(wo));
  eval.k2 = make_float3(normalize(wi));
  mdlBsdf_evaluate(&eval, &s.state, &s.resData, s.argBlock);
  pdfOut = eval.pdf; // solid-angle sampling density toward wi, for env MIS
  return make_vec3(eval.bsdf_diffuse) + make_vec3(eval.bsdf_glossy);
}

// Importance-sample the MDL BSDF for the continuation ray. Mirrors
// MDLShader_ptx.cu's __direct_callable__nextRay: k1 = view dir, xi = 4
// uniforms; returns the sampled direction and bsdf-over-pdf throughput factor.
// An absorbed lobe yields a zero weight so the resolve stage kills the path.
// Writes the bounce into the shade record for the material-agnostic resolve
// stage.
__device__ void sampleMdlBounce(const MDLShadingState &s,
    const vec3 &wo,
    RandState &rng,
    WavefrontShadeRecord &sr)
{
  BsdfSampleData sd = {};
  if (s.isFrontFace) {
    sd.ior1 = make_float3(1.0f, 1.0f, 1.0f);
    sd.ior2.x = MI_NEURAYLIB_BSDF_USE_MATERIAL_IOR;
  } else {
    sd.ior1.x = MI_NEURAYLIB_BSDF_USE_MATERIAL_IOR;
    sd.ior2 = make_float3(1.0f, 1.0f, 1.0f);
  }
  sd.k1 = make_float3(normalize(wo));
  sd.xi = make_float4(pcg_uniform(&rng),
      pcg_uniform(&rng),
      pcg_uniform(&rng),
      pcg_uniform(&rng));

  mdlBsdf_sample(&sd, &s.state, &s.resData, s.argBlock);

  sr.hasSampledBounce = 1u;
  // A specular lobe reports pdf 0 but a valid bsdf_over_pdf, so gate only on
  // the absorb event, not pdf. bsdf_over_pdf already carries cos/pdf.
  if (sd.event_type == mi::neuraylib::BSDF_EVENT_ABSORB) {
    sr.bounceWeight = vec3(0.f);
    sr.bounceDir = vec3(0.f, 0.f, 1.f);
    sr.bouncePdf = 0.f;
  } else {
    sr.bounceDir = normalize(make_vec3(sd.k2));
    sr.bounceWeight = make_vec3(sd.bsdf_over_pdf);
    // Env MIS density at the next miss: a specular lobe reports pdf 0 (a delta
    // NEE can't reach) -> +inf so the escape owns the env (w_bsdf = 1).
    sr.bouncePdf = sd.pdf > 0.f ? sd.pdf : INFINITY;
  }
}

} // namespace

// Per-material MDL shade kernel. Launched once per registered compiled material
// over that material's compacted slot list (`packed[*offset ..
// *offset+*count)`, produced by the material-sorted compaction pass). Writes
// the same WavefrontShadeRecord contract as the builtin shade-emit stage —
// including the importance-sampled continuation bounce — so the shared
// shadow-trace + resolve stages need no MDL awareness. The builtin stage runs
// first and leaves a geometry-only placeholder for MDL hits, which this kernel
// overwrites. Only surface hits are handled here; misses stay on the builtin
// path.
// Register-heavy (~70 reg/thread -> 50% occupancy at 256/block, per ncu), but
// compute/memory-bound rather than occupancy-bound: raising occupancy
// (128/block or a __launch_bounds__ register cap) does not help — measured in
// 10i. See launchWavefrontMdlShade for the block-size finding.
extern "C" __global__ void wavefrontMdlShade(
    const FrameGPUData *fd, const uint32_t *packed, const uint32_t *count)
{
  const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= *count)
    return;
  const uint32_t i = packed[t];

  // The compaction pass already filtered to this material's live surface hits,
  // so the per-slot alive / hit / material guards are not repeated here.
  WavefrontPathState &path = fd->wavefrontPaths[i];
  const WavefrontHitRecord &rec = fd->wavefrontHits[i];
  WavefrontShadeRecord &sr = fd->wavefrontShade[i];

  MDLShadingState s;
  initMdlState(s, *fd, rec.hit, rec.hit.material->materialData.mdl);

  vec3 N = rec.hit.Ns;
  if (!(dot(N, N) > 1e-12f))
    N = rec.hit.Ng;

  const vec3 wo = -rec.rayDir; // toward the camera / previous vertex
  const float opacity =
      fminf(1.0f, fmaxf(0.0f, mdlOpacity(&s.state, &s.resData, s.argBlock)));

  RandState rng = path.rng;

  // Stochastic cutout: draw once against the MDL cutout opacity. On a
  // transparent draw the surface is treated as absent for this sample — no
  // shading, no coverage — and the path continues straight through so whatever
  // is behind shows through holes. Averaged over samples this yields the
  // authored alpha, unlike a deterministic opacity*shading dimming (which never
  // reveals the background). Reuses the continuation machinery
  // (bounceDir/Weight/Org).
  if (pcg_uniform(&rng) >= opacity) {
    sr.unshadowed = vec3(0.f);
    sr.directContrib = vec3(0.f);
    sr.albedo = vec3(0.f);
    sr.normal = N;
    sr.opacity = 0.f;
    sr.visibility = 1.f;
    sr.shadowDist = 0.f;
    sr.hasHit = 1u;
    sr.depth = rec.hit.t;
    sr.primID = rec.hit.primID;
    sr.objID = rec.hit.objID;
    sr.instID = rec.hit.instID;
    sr.hasSampledBounce = 1u;
    sr.bounceDir = rec.rayDir; // straight through, unchanged direction
    sr.bounceWeight = vec3(1.f); // no attenuation
    sr.bouncePdf = INFINITY; // delta pass-through: escape owns the env (no NEE)
    const float side = dot(rec.rayDir, rec.hit.Ng) >= 0.f ? 1.f : -1.f;
    sr.bounceOrg = rec.hit.hitpoint + rec.hit.Ng * (rec.hit.epsilon * side);
    path.rng = rng;
    return;
  }

  // Opaque this sample: full shading, full coverage — the cutout opacity is
  // already folded stochastically above, so no per-term opacity factor here.
  const vec3 albedo = evalMdlAlbedo(s);
  const vec3 emission = evalMdlEmission(s, wo);
  const vec3 ambient =
      albedo * fd->renderer.ambientIntensity * fd->renderer.ambientColor;
  sr.directContrib = vec3(0.f);
  sr.shadowDist = 0.f;
  sr.visibility = 1.f;
  sr.hasHit = 1u;
  sr.unshadowed = ambient + emission;
  sr.albedo = albedo;
  sr.normal = N;
  sr.opacity = 1.f;
  sr.depth = rec.hit.t;
  sr.primID = rec.hit.primID;
  sr.objID = rec.hit.objID;
  sr.instID = rec.hit.instID;
  sr.shadowOrg = rec.hit.hitpoint + rec.hit.Ng * rec.hit.epsilon;

  // Next-event estimation: pick one light and evaluate the MDL BSDF toward it.
  const uint32_t n = uint32_t(fd->world.numLightInstances);
  if (n > 0) {
    ScreenSample ss;
    ss.frameData = fd;
    ss.rs = rng;
    ss.shadowContribWeight = 1.0f;
    uint32_t k = uint32_t(pcg_uniform(&ss.rs) * float(n));
    if (k >= n)
      k = n - 1u;
    const InstanceLightGPUData &li = fd->world.lightInstances[k];
    const bool isEnv =
        fd->registry.lights[li.lightIndex].type == LightType::HDRI;
    const LightSample ls = sampleLight(
        ss, sr.shadowOrg, li.lightIndex, li.xfm, li.surfaceInstanceIndex);
    if (ls.pdf > 0.f && ls.dist > 0.f) {
      float pBsdf = 0.f;
      const vec3 f = evalMdlBsdf(s, wo, ls.dir, pBsdf);
      // Environment MIS: balance-heuristic weight the env NEE against the MDL
      // BSDF pdf toward it (ls.pdf is the env importance pdf == envPdf(ls.dir),
      // folded with the uniform 1/n env pick). Analytic lights keep w = 1.
      float wNee = 1.f;
      if (isEnv) {
        const float pLight = ls.pdf / float(n);
        wNee = pLight / (pLight + pBsdf);
      }
      // Uniform 1/n light pick -> reweight by n. MDL's f already carries cos.
      // Cutout opacity is folded stochastically at the top, not here.
      sr.directContrib = f * ls.radiance / ls.pdf * float(n) * wNee;
      sr.shadowDir = ls.dir;
      sr.shadowDist = ls.dist;
    }
    rng = ss.rs;
  }

  // Importance-sampled continuation bounce for the indirect path (MDL BSDF, not
  // the resolve stage's diffuse fallback). Always sampled, independent of NEE.
  sampleMdlBounce(s, wo, rng, sr);
  path.rng = rng;

  // Offset the continuation origin to the side the sampled ray leaves on: +Ng
  // for a reflection lobe, -Ng for a transmission lobe (dir points into the
  // surface). Using shadowOrg's fixed +Ng offset would self-intersect glass.
  const float side = dot(sr.bounceDir, rec.hit.Ng) >= 0.f ? 1.f : -1.f;
  sr.bounceOrg = rec.hit.hitpoint + rec.hit.Ng * (rec.hit.epsilon * side);
}
