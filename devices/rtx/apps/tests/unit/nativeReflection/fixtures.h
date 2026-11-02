// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "gpu/shadingState.h"
namespace native_test {
using namespace visrtx;
constexpr int CASES = 10, SAMPLES = 262144, BINS = 8;
struct Fixture
{
  PhysicallyBasedShadingState state;
  vec3 view;
};
VISRTX_HOST_DEVICE Fixture fixture(int id)
{
  Fixture f{};
  auto &s = f.state;
  s.geometricNormal = s.viewAdjustedNormal = s.viewAdjustedClearcoatNormal =
      vec3(0, 0, 1);
  s.baseColor = vec3(.7f, .4f, .2f);
  s.roughness = .5f;
  s.clearcoatRoughness = .6f;
  s.eta = 1.f / 1.5f;
  s.occlusion = 1.f;
  s.specularColor = s.attenuationColor = vec3(1.f);
  s.sheenRoughness = .5f;
  f.view = vec3(0, 0, 1);
  switch (id) {
  case 1:
    s.metallic = 1.f;
    break;
  case 2:
    s.specular = 1.f;
    s.roughness = .65f;
    f.view = normalize(vec3(.7f, 0, .714f));
    break;
  case 3:
    s.baseColor = vec3(0);
    s.occlusion = 0;
    s.clearcoat = 1.f;
    s.viewAdjustedNormal = vec3(0, 0, -1);
    s.viewAdjustedClearcoatNormal = normalize(vec3(.6f, 0, .8f));
    break;
  case 4:
    s.specular = 1.f;
    s.clearcoat = .5f;
    s.roughness = .6f;
    s.viewAdjustedNormal = normalize(vec3(.3f, 0, .954f));
    s.viewAdjustedClearcoatNormal = normalize(vec3(-.3f, .4f, .866f));
    f.view = normalize(vec3(.2f, .4f, .89f));
    break;
  case 5:
    s.baseColor = vec3(0);
    s.sheenColor = vec3(.8f, .4f, .2f);
    break;
  case 6:
    s.specular = 1.f;
    s.transmission = 1.f;
    s.roughness = .55f;
    f.view = normalize(vec3(.6f, 0, .8f));
    break;
  case 7:
    s.transmission = 1.f;
    s.eta = 1.f;
    break;
  case 8:
    s.specular = 1.f;
    s.transmission = 1.f;
    s.eta = 1.5f;
    f.view = normalize(vec3(.9f, 0, .435f));
    break;
  case 9:
    s.viewAdjustedNormal = normalize(vec3(.8f, 0, .6f));
    s.roughness = .6f;
    break;
  }
  return f;
}
VISRTX_HOST_DEVICE PCGState random(int id, int index)
{
  PCGState state;
  pcg_init(&state,
      detail::pcg_mix64(uint64_t(index) + 93821),
      detail::pcg_mix64(uint64_t(id) + 739));
  return state;
}
} // namespace native_test
