// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <cstdio>
#include <cstdlib>
#include "shader_expressions.h"

namespace {
int failures = 0;
void check(bool ok, const char *what)
{
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
  }
}
void direction(const vec3 &actual, const vec3 &expected, const char *what)
{
  check(glm::length(actual - expected) < 1e-6f, what);
}
NextRay oldProposal(const PhysicallyBasedShadingState *s,
    const Ray *ray,
    RandState *rng,
    const NextRayQmc *q)
{
  int cursor = 0;
  return sampleNativeProposal(s, ray, rng, q, cursor);
}
PhysicallyBasedShadingState material()
{
  PhysicallyBasedShadingState s{};
  s.baseColor = vec3(1.f);
  s.geometricNormal = vec3(0, 0, 1);
  s.viewAdjustedNormal = s.viewAdjustedClearcoatNormal = vec3(0, 0, 1);
  s.roughness = s.clearcoatRoughness = .3f;
  s.eta = 1.f / 1.5f;
  s.occlusion = s.specular = 1.f;
  s.specularColor = s.attenuationColor = vec3(1.f);
  return s;
}
} // namespace

int main()
{
  auto s = material();
  Ray ray{};
  ray.dir = vec3(0, 0, -1);
  RandState rng;
  pcg_init(&rng, 42, 7);
  const auto before = rng;
  {
    NextRayQmc safe{{.25f, .125f, 0.f, 0.f}, 2};
    const auto result = sampleNativeReflection(&s, &ray, &rng, &safe, .01f);
    const float radius = std::sqrt(1.f - .75f * .75f);
    direction(result.direction,
        vec3(radius * std::cos(kPi / 4), radius * std::sin(kPi / 4), .75f),
        "safe proposal consumes polar/azimuth in order");
    check(rng.state == before.state,
        "forced safe proposal consumes exactly two overlay draws");
    NextRayQmc old{{.9f, .25f, .125f, 0.f}, 3};
    const auto reflected = sampleNativeReflection(&s, &ray, &rng, &old, .5f);
    direction(reflected.direction,
        vec3(.5f * std::cos(kPi / 4), .5f * std::sin(kPi / 4), std::sqrt(.75f)),
        "old proposal retains overlay draw order");
    check(rng.state == before.state,
        "forced old diffuse path consumes three overlay draws");
    auto expected = before;
    const float choice = pcg_uniform(&expected);
    const auto manual =
        sampleNativeReflection(&s, &ray, &expected, &old, choice);
    const auto sampled = __direct_callable__nextRay(&s, &ray, &rng, &old);
    direction(
        sampled.direction, manual.direction, "production outer coin is PCG");
    check(rng.state == expected.state,
        "production outer coin advances PCG independently of overlay");
    rng = before;
  }
  // No coat: base pick, then hemisphere u before v. Tagged values deliberately
  // differ, so swapping the arguments changes the observed outgoing direction.
  NextRayQmc q{{.9f, .25f, .125f, .75f}, 3};
  auto result = oldProposal(&s, &ray, &rng, &q);
  const float r = .5f;
  direction(result.direction,
      vec3(r * std::cos(kPi / 4), r * std::sin(kPi / 4), std::sqrt(.75f)),
      "no-coat diffuse assigns u before v");
  check(rng.state == before.state, "three supplied draws do not advance PCG");
  // Coat selected: selection, VNDF radius, then VNDF azimuth; no base draw.
  s.clearcoat = 1.f;
  q = {{.01f, .25f, .125f, .99f}, 3};
  result = oldProposal(&s, &ray, &rng, &q);
  const vec3 h = glm::normalize(vec3(.09f * .5f * std::cos(kPi / 4),
      .09f * .5f * std::sin(kPi / 4),
      std::sqrt(.75f)));
  const vec3 reflection(
      2.f * h.z * h.x, 2.f * h.z * h.y, 2.f * h.z * h.z - 1.f);
  direction(result.direction, reflection, "coat VNDF assigns u before v");
  check(
      rng.state == before.state, "selected coat does not request a fifth draw");
  // Rejected coat + diffuse consumes exactly four supplied values.
  q = {{.99f, .9f, .25f, .125f}, 4};
  result = oldProposal(&s, &ray, &rng, &q);
  direction(result.direction,
      vec3(r * std::cos(kPi / 4), r * std::sin(kPi / 4), std::sqrt(.75f)),
      "coat rejected diffuse keeps sequential roles");
  check(
      rng.state == before.state, "rejected coat diffuse does not advance PCG");

  // Rejected coat + specular: reflection selection is the fifth request.
  q = {{.99f, 0.f, .25f, .125f}, 4};
  auto expectedRng = before;
  (void)pcg_uniform(&expectedRng);
  result = oldProposal(&s, &ray, &rng, &q);
  direction(result.direction, reflection, "base VNDF assigns u before v");
  check(rng.state == expectedRng.state && rng.inc == expectedRng.inc,
      "fifth request takes exactly one PCG draw");
  q = {{.125f, .25f, .5f, .75f}, 4};
  rng = before;
  const auto xi = mdlXi(&rng, &q);
  check(xi.x == .125f && xi.y == .25f && xi.z == .5f && xi.w == .75f,
      "MDL xi x/y/z/w receive sequential supplied draws");
  check(rng.state == before.state, "MDL supplied xi does not advance PCG");
  rng = before;
  expectedRng = before;
  const float hemiU = pcg_uniform(&expectedRng);
  const float hemiV = pcg_uniform(&expectedRng);
  const auto hemisphere = sampleHemisphere(rng, vec3(0, 0, 1));
  const float hemiR = std::sqrt(hemiU);
  direction(hemisphere,
      vec3(hemiR * std::cos(kTwoPi * hemiV),
          hemiR * std::sin(kTwoPi * hemiV),
          std::sqrt(1.f - hemiU)),
      "PCG-only hemisphere assigns radius before azimuth");
  check(
      rng.state == expectedRng.state, "hemisphere takes exactly two PCG draws");
  rng = before;
  const auto uv = rectCoordinates(rng);
  check(uv.x == hemiU && uv.y == hemiV,
      "rectangle assigns first PCG draw to edge1, second to edge2");
  check(
      rng.state == expectedRng.state, "rectangle takes exactly two PCG draws");
  rng = before;
  expectedRng = before;
  const float row = pcg_uniform(&expectedRng);
  const float col = pcg_uniform(&expectedRng);
  const float jitterU = pcg_uniform(&expectedRng);
  const float jitterV = pcg_uniform(&expectedRng);
  const float cdf[] = {.5f, 1.f, .5f, 1.f, .5f, 1.f};
  LightGPUData ld{};
  ld.hdri.size = uvec2(2, 2);
  ld.hdri.marginalCDF = cdf;
  ld.hdri.conditionalCDF = cdf + 2;
  const auto hdriUV = hdriCoordinates(ld, rng);
  check(hdriUV.x == ((col > .5f ? 1.f : 0.f) + jitterU) / 2.f
          && hdriUV.y == ((row > .5f ? 1.f : 0.f) + jitterV) / 2.f,
      "HDRI draws row, column, then jitter u before v");
  check(rng.state == expectedRng.state, "HDRI takes exactly four PCG draws");
  // A no-coat specular path fits in the four-value overlay.
  s = material();
  rng = before;
  q = {{0.f, .25f, .125f, .5f}, 4};
  result = oldProposal(&s, &ray, &rng, &q);
  direction(
      result.direction, reflection, "no-coat specular uses draws 1/2 for VNDF");
  check(rng.state == before.state,
      "no-coat specular consumes four supplied draws");

  // The fifth PCG draw also drives transmission, not an eagerly moved pick.
  s.clearcoat = 1.f;
  s.transmission = 1.f;
  q = {{.99f, 0.f, .25f, .125f}, 4};
  expectedRng = before;
  const float reflectPick = pcg_uniform(&expectedRng);
  check(reflectPick > .1f,
      "controlled PCG selects transmission for this fixture");
  result = oldProposal(&s, &ray, &rng, &q);
  const float eta = s.eta;
  const float cosT = std::sqrt(1.f - eta * eta * (1.f - h.z * h.z));
  const vec3 refracted = glm::normalize(vec3((eta * h.z - cosT) * h.x,
      (eta * h.z - cosT) * h.y,
      -eta + (eta * h.z - cosT) * h.z));
  direction(result.direction, refracted, "fifth PCG draw selects transmission");
  check(result.flags == NEXT_RAY_CONTINUES_THROUGH_SURFACE
          && rng.state == expectedRng.state,
      "transmission retains its flag and single fallback draw");

  // Boundaries retain short-circuit consumption. n is precisely the expected
  // prefix length, so any eager draw past it would change controlled PCG state.
  auto dead = [&](const char *label, int count) {
    rng = before;
    q.n = count;
    const auto next = oldProposal(&s, &ray, &rng, &q);
    check(isDead(next) && next.pdf == 0.f
            && glm::length(next.contributionWeight) == 0.f,
        label);
    check(rng.state == before.state,
        "early exit consumes no draws past its prefix");
  };
  s = material();
  ray.dir = vec3(0, 0, 1);
  dead("invalid base local view exits before base selection", 0);
  s.clearcoat = 1.f;
  q.u[0] = 0.f;
  dead("invalid coat local view exits after coat selection", 1);
  ray.dir = vec3(0, 0, -1);
  s = material();
  s.baseColor = vec3(0.f);
  s.specular = 0.f;
  dead("zero base energy exits before base selection", 0);
  s.clearcoat = 1.f;
  q.u[0] = .99f;
  dead("rejected coat zero base energy consumes only coat selection", 1);
  s = material();
  s.clearcoat = 1.f;
  s.clearcoatRoughness = 1.f;
  q = {{0.f, .9f, .125f, .5f}, 3};
  dead("coat below-surface reflection exits after its pair", 3);
  s = material();
  s.roughness = 1.f;
  q = {{0.f, .9f, .125f, .5f}, 4};
  dead("base below-surface reflection exits after reflect selection", 4);

  // Interactive/null QMC and partial overlays preserve component order too.
  for (int count = 0; count <= 4; ++count) {
    q = {{.125f, .25f, .5f, .75f}, count};
    rng = expectedRng = before;
    float expectedXi[4];
    for (int i = 0; i < 4; ++i)
      expectedXi[i] = i < count ? q.u[i] : pcg_uniform(&expectedRng);
    const auto mixed = mdlXi(&rng, &q);
    check(mixed.x == expectedXi[0] && mixed.y == expectedXi[1]
            && mixed.z == expectedXi[2] && mixed.w == expectedXi[3],
        "MDL partial overlay maps all four components in sequence");
    check(
        rng.state == expectedRng.state, "MDL partial overlay PCG advancement");
  }
  rng = expectedRng = before;
  const float x = pcg_uniform(&expectedRng);
  const float y = pcg_uniform(&expectedRng);
  const float z = pcg_uniform(&expectedRng);
  const float w = pcg_uniform(&expectedRng);
  const auto nullXi = mdlXi(&rng, nullptr);
  check(nullXi.x == x && nullXi.y == y && nullXi.z == z && nullXi.w == w
          && rng.state == expectedRng.state,
      "MDL null overlay retains sequential PCG-only sampling");
  return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
