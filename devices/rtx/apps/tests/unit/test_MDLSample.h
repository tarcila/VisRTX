// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <array>
#include <cfenv>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include "material/shaders/MDLSample.h"

namespace mdl_sample_test {
using namespace visrtx;
using namespace mi::neuraylib;
constexpr int COUNT = 9;
struct Input
{
  Bsdf_sample_data sample{};
  vec3 normal{0, 0, 1};
};
inline void require(bool ok, const char *message)
{
  if (!ok)
    throw std::runtime_error(message);
}
// Test-only generated-target stand-in: rejection AFTER writing a live result.
// This is a consumer-order regression, not a real MDL target qualification.
VISRTX_HOST_DEVICE void rejectAfterWriting(Bsdf_sample_data &sample)
{
  sample.k2 = {0, 0, -1};
  sample.bsdf_over_pdf = {7, 11, 13};
  sample.pdf = 0;
  sample.event_type = BSDF_EVENT_ABSORB;
}
inline std::array<Input, COUNT> inputs()
{
  std::array<Input, COUNT> in{};
  for (auto &v : in) {
    v.sample.k2 = {0, 0, 1};
    v.sample.bsdf_over_pdf = {.25f, .5f, 1.f};
    v.sample.pdf = .375f;
  }
  rejectAfterWriting(in[0].sample);
  in[0].normal = vec3(0);
  in[1].sample.event_type = BSDF_EVENT_ABSORB;
  const uint32_t snan = 0x7f800001u;
  float poison;
  std::memcpy(&poison, &snan, sizeof(poison));
  in[1].sample.k2 = {poison, poison, poison};
  in[1].sample.bsdf_over_pdf = {poison, poison, poison};
  in[1].normal = vec3(0);
  const unsigned masks[7]{BSDF_EVENT_DIFFUSE | BSDF_EVENT_REFLECTION,
      BSDF_EVENT_GLOSSY | BSDF_EVENT_REFLECTION,
      BSDF_EVENT_SPECULAR | BSDF_EVENT_REFLECTION,
      BSDF_EVENT_SPECULAR | BSDF_EVENT_TRANSMISSION,
      BSDF_EVENT_DIFFUSE | BSDF_EVENT_TRANSMISSION,
      BSDF_EVENT_SPECULAR | BSDF_EVENT_GLOSSY | BSDF_EVENT_REFLECTION,
      BSDF_EVENT_DIFFUSE | BSDF_EVENT_REFLECTION};
  for (int i = 2; i < COUNT; ++i)
    in[i].sample.event_type = static_cast<Bsdf_event_type>(masks[i - 2]);
  in[4].sample.pdf = in[5].sample.pdf = 0.f; // live delta events must survive
  in[5].sample.k2 = in[6].sample.k2 = {0, 0, -1};
  in[8].sample.bsdf_over_pdf = {
      0, 0, 0}; // zero weight does not redefine SDK event
  return in;
}
inline void check(
    const std::array<Input, COUNT> &in, const std::array<NextRay, COUNT> &out)
{
  for (int i = 0; i < COUNT; ++i) {
    const auto &n = out[i];
    if (i < 2) {
      require(isDead(n) && n.direction == vec3(0)
              && n.contributionWeight == vec3(0) && n.pdf == 0.f
              && n.flags == NEXT_RAY_NONE && n.mdlEventMask == 0
              && n.component == NextRayComponent::None,
          "dead record not fully initialized");
      continue;
    }
    const auto &s = in[i].sample;
    const bool singular = i == 4 || i == 5 || i == 7,
               through = i == 5 || i == 6;
    require(!isDead(n)
            && n.event
                == (singular ? NextRayEvent::Singular
                             : NextRayEvent::Continuous),
        "MDL event classification");
    require(n.mdlEventMask == uint32_t(s.event_type)
            && n.component == NextRayComponent::MDL,
        "raw event bits/component lost");
    require(n.direction == vec3(s.k2.x, s.k2.y, s.k2.z)
            && n.contributionWeight
                == vec3(
                    s.bsdf_over_pdf.x, s.bsdf_over_pdf.y, s.bsdf_over_pdf.z),
        "live sample changed");
    require(n.flags
            == (through ? NEXT_RAY_CONTINUES_THROUGH_SURFACE : NEXT_RAY_NONE),
        "legacy side policy changed");
    require(singular || through ? std::isinf(n.pdf) : n.pdf == s.pdf,
        "legacy MIS pdf changed");
  }
}
#ifdef MDL_SAMPLE_CUDA
inline void checked(cudaError_t e)
{
  if (e != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(e));
}
VISRTX_GLOBAL void consume(const Input *in, NextRay *out)
{
  const int i = int(threadIdx.x);
  if (i < COUNT)
    out[i] = consumeMdlSample(in[i].sample, vec3(0, 0, -1), in[i].normal);
}
#endif
inline int run()
{
  const auto in = inputs();
  std::array<NextRay, COUNT> out{};
  std::feclearexcept(FE_ALL_EXCEPT);
  for (int i = 0; i < COUNT; ++i)
    out[i] = consumeMdlSample(in[i].sample, vec3(0, 0, -1), in[i].normal);
  require((std::fetestexcept(FE_INVALID | FE_DIVBYZERO)) == 0,
      "absorbed sample consumed poisoned data/normal");
  check(in, out);
#ifdef MDL_SAMPLE_CUDA
  Input *deviceIn = nullptr;
  NextRay *deviceOut = nullptr;
  checked(cudaMalloc(&deviceIn, sizeof(in)));
  checked(cudaMalloc(&deviceOut, sizeof(out)));
  checked(cudaMemcpy(deviceIn, in.data(), sizeof(in), cudaMemcpyHostToDevice));
  consume<<<1, 32>>>(deviceIn, deviceOut);
  checked(cudaGetLastError());
  checked(cudaDeviceSynchronize());
  checked(
      cudaMemcpy(out.data(), deviceOut, sizeof(out), cudaMemcpyDeviceToHost));
  checked(cudaFree(deviceIn));
  checked(cudaFree(deviceOut));
  check(in, out);
#endif
  std::puts(
      "MDL sample consumer: dead ordering, live delta/continuous events and raw bits pass");
  return 0;
}
} // namespace mdl_sample_test
