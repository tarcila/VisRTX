// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#include <optix_device.h>
#include "probe.h"

using namespace visrtx;
using namespace mi::neuraylib;
extern "C" {
__constant__ mdl_target_test::Params params;
}
VISRTX_CALLABLE Bsdf_sample_function mdlBsdf_sample;
VISRTX_CALLABLE Bsdf_evaluate_function mdlBsdf_evaluate;
VISRTX_CALLABLE Bsdf_pdf_function mdlBsdf_pdf;

VISRTX_GLOBAL void __raygen__mdlSurfaceProbe()
{
  using namespace mdl_target_test;
  const auto index = optixGetLaunchIndex().x;
  GeometryGPUData geometry{};
  geometry.type = GeometryType::TRIANGLE;
  for (int c = 0; c < 5; ++c)
    geometry.attrUniform[c] = vec4(0.f, 0.f, 0.f, 1.f);
  InstanceSurfaceGPUData instance{};
  instance.objectToWorld = mat3x4(0.f);
  instance.worldToObject = mat3x4(0.f);
  for (int c = 0; c < 3; ++c) {
    instance.objectToWorld[c][c] = 1.f;
    instance.worldToObject[c][c] = 1.f;
  }
  SurfaceHit hit{};
  hit.Ng = hit.Ns = hit.V = vec3(0, 0, 1);
  hit.tU = vec3(1, 0, 0);
  hit.tV = vec3(0, 1, 0);
  hit.hitpoint = vec3(0);
  hit.isFrontFace = params.state != 1;
  hit.geometry = &geometry;
  hit.instance = &instance;
  MaterialGPUData::MDL material{};
  material.argBlock = params.argumentBlock;
  MDLShadingState state{};
  optixDirectCall<void>(0, &state, params.frame, &hit, &material);
  // Deliberate post-init state fixture: expose the SDK's geometric rejection,
  // independently of the production adapt_normal hook. This is test-only.
  if (params.state == 2) {
    const auto n = normalize(vec3(.98f, 0, .2f));
    state.state.normal = make_float3(n.x, n.y, n.z);
  }
  PCGState xiState;
  pcg_init(&xiState, detail::pcg_mix64(index + 73), 19);
  NextRayQmc q{};
  q.n = 4;
  for (int c = 0; c < 4; ++c)
    q.u[c] = pcg_uniform(&xiState);
  Bsdf_sample_data sample{};
  if (state.isFrontFace) {
    sample.ior1 = make_float3(1.f, 1.f, 1.f);
    sample.ior2.x = MI_NEURAYLIB_BSDF_USE_MATERIAL_IOR;
  } else {
    sample.ior1.x = MI_NEURAYLIB_BSDF_USE_MATERIAL_IOR;
    sample.ior2 = make_float3(1.f, 1.f, 1.f);
  }
  sample.k1 = make_float3(0, 0, 1);
  sample.xi = make_float4(q.u[0], q.u[1], q.u[2], q.u[3]);
  mdlBsdf_sample(&sample, &state.state, &state.resData, state.argBlock);
  Result result{};
  result.event = uint32_t(sample.event_type);
  result.pdf = sample.pdf;
  result.direction = vec3(sample.k2.x, sample.k2.y, sample.k2.z);
  result.weight = vec3(
      sample.bsdf_over_pdf.x, sample.bsdf_over_pdf.y, sample.bsdf_over_pdf.z);
  result.normal =
      vec3(state.state.normal.x, state.state.normal.y, state.state.normal.z);
  Ray ray{vec3(0), vec3(0, 0, -1)};
  PCGState wrapperRandom;
  pcg_init(&wrapperRandom, 42, 7);
  const auto before = wrapperRandom;
  result.consumed =
      optixDirectCall<NextRay>(1, &state, &ray, &wrapperRandom, &q);
  result.unchangedRandom =
      before.state == wrapperRandom.state && before.inc == wrapperRandom.inc;
  if (sample.event_type != BSDF_EVENT_ABSORB) {
    Bsdf_evaluate_data<DF_HSM_NONE> eval{};
    eval.ior1 = sample.ior1;
    eval.ior2 = sample.ior2;
    eval.k1 = sample.k1;
    eval.k2 = sample.k2;
    mdlBsdf_evaluate(&eval, &state.state, &state.resData, state.argBlock);
    result.evaluated = vec3(eval.bsdf_diffuse.x + eval.bsdf_glossy.x,
        eval.bsdf_diffuse.y + eval.bsdf_glossy.y,
        eval.bsdf_diffuse.z + eval.bsdf_glossy.z);
    Bsdf_pdf_data pdf{};
    pdf.ior1 = sample.ior1;
    pdf.ior2 = sample.ior2;
    pdf.k1 = sample.k1;
    pdf.k2 = sample.k2;
    mdlBsdf_pdf(&pdf, &state.state, &state.resData, state.argBlock);
    result.queriedPdf = pdf.pdf;
    const vec3 v(0, 0, 1);
    result.wrapperEvaluated =
        optixDirectCall<vec3>(2, &state, &v, &result.direction);
    result.wrapperPdf =
        optixDirectCall<float>(3, &state, &v, &result.direction);
  }
  params.output[index] = result;
}
