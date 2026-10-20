/*
 * Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

// Host unit tests for lightPickPower — the Pick Power estimate that drives the
// power-proportional Light Pick. These lock the *relative* ordering the pick
// depends on (brighter/larger/closer-to-both-sided lights outweigh dimmer ones,
// infinite lights grow with the scene), which the API additivity test cannot
// see because any unbiased pick converges to the same image. Absolute constants
// are deliberately not asserted: Pick Power only needs to be proportional.

#include "gpu/lightPickPower.h"

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>

using namespace visrtx;

static int g_failures = 0;

#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);              \
      ++g_failures;                                                            \
    }                                                                          \
  } while (0)

static bool near(float a, float b, float eps = 1e-5f)
{
  return std::fabs(a - b) <= eps * std::fmax(1.f, std::fabs(b));
}

static LightGPUData pointLight(float intensity, const vec3 &color = vec3(1.f))
{
  LightGPUData ld{};
  ld.type = LightType::POINT;
  ld.color = color;
  ld.point.intensity = intensity;
  return ld;
}

static LightGPUData rectLight(
    float intensity, float oneOverArea, bool front, bool back)
{
  LightGPUData ld{};
  ld.type = LightType::RECT;
  ld.color = vec3(1.f);
  ld.rect.intensity = intensity;
  ld.rect.oneOverArea = oneOverArea;
  ld.rect.side.front = front ? 1 : 0;
  ld.rect.side.back = back ? 1 : 0;
  return ld;
}

static LightGPUData directionalLight(float irradiance)
{
  LightGPUData ld{};
  ld.type = LightType::DIRECTIONAL;
  ld.color = vec3(1.f);
  ld.distant.irradiance = irradiance;
  return ld;
}

int main()
{
  const mat4 identity(1.f);
  const float radius = 3.f;

  // Monotonic in intensity, and proportional (2x intensity => 2x power).
  {
    const float p1 = lightPickPower(pointLight(1.f), identity, radius);
    const float p2 = lightPickPower(pointLight(2.f), identity, radius);
    CHECK(p1 > 0.f);
    CHECK(p2 > 2.f * p1 - 1e-3f && p2 < 2.f * p1 + 1e-3f);
  }

  // A dark light contributes no pick weight (positivity boundary).
  CHECK(lightPickPower(pointLight(0.f), identity, radius) == 0.f);

  // Luminance-weighted: a green light outweighs an equal-intensity blue one.
  {
    const float green =
        lightPickPower(pointLight(1.f, vec3(0.f, 1.f, 0.f)), identity, radius);
    const float blue =
        lightPickPower(pointLight(1.f, vec3(0.f, 0.f, 1.f)), identity, radius);
    CHECK(green > blue);
  }

  // A rectangle emitting from both sides has twice the power of one side.
  {
    const float oneSide =
        lightPickPower(rectLight(1.f, 0.25f, true, false), identity, radius);
    const float bothSides =
        lightPickPower(rectLight(1.f, 0.25f, true, true), identity, radius);
    CHECK(
        bothSides > 2.f * oneSide - 1e-3f && bothSides < 2.f * oneSide + 1e-3f);
    // Larger area (smaller oneOverArea) => more power.
    const float bigger =
        lightPickPower(rectLight(1.f, 0.1f, true, false), identity, radius);
    CHECK(bigger > oneSide);
  }

  // Texture means modulate power after multiplication by the light color.
  {
    auto rect = rectLight(1.f, .25f, true, false);
    const float white = lightPickPower(rect, identity, radius);
    rect.rect.colorMap =
        1; // nonzero marker; this CPU test does not sample CUDA
    rect.rect.colorMapMean = vec3(.25f);
    CHECK(std::abs(lightPickPower(rect, identity, radius) - .25f * white)
        < 1e-5f * white);
    rect.rect.colorMapMean = vec3(0.f);
    CHECK(lightPickPower(rect, identity, radius) == 0.f);
    rect.color = vec3(1.f, 0.f, 0.f);
    rect.rect.colorMapMean = vec3(0.f, 1.f, 0.f);
    CHECK(lightPickPower(rect, identity, radius) == 0.f);
  }

  // Non-uniform instance scale enlarges an area light's Pick Power (area
  // grows).
  {
    const auto rect = rectLight(1.f, 0.25f, true, false);
    const float base = lightPickPower(rect, identity, radius);
    const mat4 scaled = glm::scale(identity, vec3(2.f, 2.f, 2.f));
    CHECK(lightPickPower(rect, scaled, radius) > base);
  }

  // Infinite lights grow with the scene cross-section (~radius^2).
  // Quad cone/focus shaping scales Pick Power by the surviving Lambertian flux
  // fraction: hard cone c -> sin^2(c); black-tint focus f -> 2/(f+2).
  {
    const auto shaped = [](float coneDeg, float softness, float focus,
                            const vec3 &tint) {
      auto ld = rectLight(1.f, 0.25f, true, false);
      ld.rect.coneAngle = coneDeg * kPi / 180.f;
      ld.rect.innerAngle = ld.rect.coneAngle * (1.f - softness);
      ld.rect.cosOuterAngle = std::cos(ld.rect.coneAngle);
      ld.rect.cosInnerAngle = std::cos(ld.rect.innerAngle);
      ld.rect.focus = focus;
      ld.rect.focusTint = tint;
      return ld;
    };
    const float full =
        lightPickPower(rectLight(1.f, 0.25f, true, false), identity, radius);
    const vec3 black(0.f);
    CHECK(near(lightPickPower(shaped(90.f, 0.f, 0.f, black), identity, radius)
            / full, 1.f));
    CHECK(near(lightPickPower(shaped(180.f, 0.f, 0.f, black), identity, radius)
            / full, 1.f));
    const float s30 = std::sin(30.f * kPi / 180.f);
    CHECK(near(lightPickPower(shaped(30.f, 0.f, 0.f, black), identity, radius)
            / full, s30 * s30, 1e-3f));
    const float soft = lightPickPower(shaped(30.f, 0.5f, 0.f, black), identity, radius);
    CHECK(soft > 0.f && soft < s30 * s30 * full);
    CHECK(near(lightPickPower(shaped(90.f, 0.f, 2.f, black), identity, radius)
            / full, 0.5f, 1e-3f));
    CHECK(near(lightPickPower(shaped(90.f, 0.f, 20.f, black), identity, radius)
            / full, 2.f / 22.f, 1e-3f));
    CHECK(near(lightPickPower(shaped(90.f, 0.f, 20.f, vec3(1.f)), identity, radius)
            / full, 1.f));
    CHECK(lightPickPower(shaped(15.f, 0.f, 20.f, black), identity, radius) > 0.f);
    // Extreme focus: the lobe collapses onto mu = 1, where any fixed-sample
    // quadrature underflows to zero and World::appendLight would then drop the
    // light from the CDF even though it still emits along the axis. The closed
    // form must track 2/(f+2) all the way down.
    // near()'s max(1,|b|) floor degenerates to an absolute tolerance down here,
    // which a badly underestimated (but nonzero) fraction would still pass, so
    // compare relative error directly.
    for (const float f : {1e2f, 1e3f, 1e4f, 1e6f, 1e9f}) {
      const float got =
          lightPickPower(shaped(90.f, 0.f, f, black), identity, radius) / full;
      const float want = 2.f / (f + 2.f);
      CHECK(got > 0.f);
      CHECK(std::fabs(got - want) <= 1e-2f * want);
    }
    // Chromatic tint: each channel integrates separately, then luminance.
    {
      const vec3 tint(1.f, 0.f, 0.f); // red passes, green/blue focus-shaped
      const float survived = 2.f / (4.f + 2.f); // focus = 4
      const vec3 perChannel = tint + (vec3(1.f) - tint) * survived;
      const float expected =
          glm::dot(perChannel, vec3(0.2126f, 0.7152f, 0.0722f));
      CHECK(near(
          lightPickPower(shaped(90.f, 0.f, 4.f, tint), identity, radius) / full,
          expected,
          1e-3f));
    }
    // Narrow hard cone: the penumbra is empty, so sin^2 must stay exact where a
    // difference of nearly equal powers would cancel.
    for (const float deg : {5.f, 1.f, 0.1f, 0.01f, 0.001f}) {
      const float s = std::sin(deg * kPi / 180.f);
      const float got = lightPickPower(
          shaped(deg, 0.f, 0.f, black), identity, radius) / full;
      CHECK(got > 0.f);
      CHECK(std::fabs(got - s * s) <= 1e-4f * s * s);
    }
    // Independent angular-coordinate reference. Resolve the focus scale in
    // theta rather than using the production power-coordinate quadrature.
    // Beyond 12/sqrt(f+2), the black-tint focus tail is negligible (~exp(-72)).
    const auto focusedReference = [](const LightGPUData &ld) {
      const auto &r = ld.rect;
      const double end = std::min(double(r.coneAngle),
          std::min(1.5707963267948966, 12.0 / std::sqrt(double(r.focus) + 2.0)));
      constexpr int n = 1 << 16;
      const double step = end / n;
      double sum = 0.0;
      for (int i = 0; i < n; ++i) {
        const double theta = (i + 0.5) * step;
        const double t = std::clamp((double(r.coneAngle) - theta)
                / (double(r.coneAngle) - double(r.innerAngle)),
            0.0, 1.0);
        const double s = std::sin(theta * 0.5);
        const double focus = std::exp(double(r.focus) * std::log1p(-2.0 * s * s));
        sum += t * t * (3.0 - 2.0 * t) * focus * std::cos(theta) * std::sin(theta);
      }
      return 2.0 * step * sum;
    };
    for (const float deg : {0.01f, 45.f, 120.f}) {
      for (const float soft : {0.5f, 0.9999f, 1.f}) {
        for (const float focus : {1e4f, 1e6f, 1e9f}) {
          const auto ld = shaped(deg, soft, focus, black);
          const double want = focusedReference(ld);
          const double got = lightPickPower(ld, identity, radius) / full;
          CHECK(got > 0.0);
          CHECK(std::fabs(got - want) <= 2e-3 * want);
        }
      }
    }
    // Soft cone against a brute-force reference integral of the same leaves.
    {
      const auto reference = [&](const LightGPUData &ld) {
        const int n = 1 << 16;
        double sum = 0.0;
        for (int i = 0; i < n; ++i) {
          const float mu = (float(i) + 0.5f) / float(n);
          sum += double(rectConeAttenuation(ld.rect, mu)
              * rectFocusAttenuation(ld.rect, mu).x * mu);
        }
        return float(2.0 * sum / double(n));
      };
      for (const float soft : {0.25f, 0.5f, 1.f}) {
        const auto ld = shaped(45.f, soft, 0.f, black);
        CHECK(near(
            lightPickPower(ld, identity, radius) / full, reference(ld), 1e-2f));
      }
    }
    // Both sides shape identically: still exactly double.
    auto both = shaped(30.f, 0.f, 0.f, black);
    both.rect.side.back = 1;
    CHECK(near(lightPickPower(both, identity, radius),
        2.f * lightPickPower(shaped(30.f, 0.f, 0.f, black), identity, radius)));
  }

  {
    const auto sun = directionalLight(1.f);
    const float small = lightPickPower(sun, identity, 1.f);
    const float big = lightPickPower(sun, identity, 2.f);
    CHECK(big > 3.9f * small && big < 4.1f * small); // 2^2 = 4x
  }

  // Sphere (a point light with radius > 0): power grows with intensity and
  // area.
  {
    LightGPUData dim{}, bright{}, bigger{};
    dim.type = bright.type = bigger.type = LightType::SPHERE;
    dim.color = bright.color = bigger.color = vec3(1.f);
    dim.sphere.intensity = 1.f;
    dim.sphere.radius = 1.f;
    bright.sphere.intensity = 2.f;
    bright.sphere.radius = 1.f;
    bigger.sphere.intensity = 1.f;
    bigger.sphere.radius = 2.f;
    CHECK(lightPickPower(dim, identity, radius) > 0.f);
    CHECK(lightPickPower(bright, identity, radius)
        > lightPickPower(dim, identity, radius));
    CHECK(lightPickPower(bigger, identity, radius)
        > lightPickPower(dim, identity, radius));
  }

  // Spot: power grows with intensity and with a wider cone (smaller cosOuter).
  {
    LightGPUData narrow{}, wide{}, bright{};
    narrow.type = wide.type = bright.type = LightType::SPOT;
    narrow.color = wide.color = bright.color = vec3(1.f);
    narrow.spot.intensity = 1.f;
    narrow.spot.cosOuterAngle = 0.9f;
    wide.spot.intensity = 1.f;
    wide.spot.cosOuterAngle = 0.1f; // wider cone
    bright.spot.intensity = 3.f;
    bright.spot.cosOuterAngle = 0.9f;
    CHECK(lightPickPower(narrow, identity, radius) > 0.f);
    CHECK(lightPickPower(wide, identity, radius)
        > lightPickPower(narrow, identity, radius));
    CHECK(lightPickPower(bright, identity, radius)
        > lightPickPower(narrow, identity, radius));
  }

  // Ring: power grows with intensity and with area (smaller oneOverArea).
  {
    LightGPUData small{}, big{}, bright{};
    small.type = big.type = bright.type = LightType::RING;
    small.color = big.color = bright.color = vec3(1.f);
    small.ring.intensity = 1.f;
    small.ring.oneOverArea = 1.f;
    big.ring.intensity = 1.f;
    big.ring.oneOverArea = 0.25f; // 4x area
    bright.ring.intensity = 2.f;
    bright.ring.oneOverArea = 1.f;
    CHECK(lightPickPower(small, identity, radius) > 0.f);
    CHECK(lightPickPower(big, identity, radius)
        > lightPickPower(small, identity, radius));
    CHECK(lightPickPower(bright, identity, radius)
        > lightPickPower(small, identity, radius));
  }

  // HDRI: an infinite light — power grows with scale, with radius^2, and with
  // the map's mean luminance (recovered from pdfWeight; larger pdfWeight means
  // a DIMMER map, so power falls as pdfWeight grows). A zero pdfWeight (all-
  // black map) yields zero power so it is never picked.
  {
    LightGPUData env{};
    env.type = LightType::HDRI;
    env.color = vec3(1.f);
    env.hdri.scale = 1.f;
    env.hdri.pdfWeight = 1.f; // meanLuminance = 1/(4*pi)
    LightGPUData brighter = env;
    brighter.hdri.scale = 2.f;
    LightGPUData higherMean = env;
    higherMean.hdri.pdfWeight = 0.5f; // dimmer denominator => brighter map
    LightGPUData black = env;
    black.hdri.pdfWeight = 0.f;
    CHECK(lightPickPower(env, identity, radius) > 0.f);
    CHECK(lightPickPower(brighter, identity, radius)
        > lightPickPower(env, identity, radius));
    CHECK(lightPickPower(higherMean, identity, radius)
        > lightPickPower(env, identity, radius));
    CHECK(lightPickPower(black, identity, radius) == 0.f);
    CHECK(lightPickPower(env, identity, 2.f)
        > lightPickPower(env, identity, 1.f));
  }

  if (g_failures) {
    std::printf("%d lightPickPower checks failed\n", g_failures);
    return 1;
  }
  std::printf("lightPickPower unit tests passed\n");
  return 0;
}
