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

// Stage 1.75 Geometry Light equivalence for Cylinder and Cone. These analytic
// primitives have no analytic ANARI light counterpart, so each emissive
// primitive set is cross-checked against a FINELY TESSELLATED emissive triangle
// mesh of the same closed surface — the triangle Geometry Light path is already
// validated (TestEmissiveGeometryLight/InstanceScale). A matching floor pool
// validates the cylinder/cone total area (buildAreaData lateral+caps), the
// per-sub-surface point/normal generation (lateral, caps, cone slant), AND that
// the sampler pdf and hit-side MIS pdf are consistent (a bias would show as a
// mismatch against the unbiased triangle reference).
//
// Each set has TWO primitives with DISTINCT radii and MIXED caps (both / first-
// only) via a per-vertex vertex.cap array, so the run exercises: the
// multi-primitive area CDF pick, per-primitive radius (cylinder) / per-vertex
// radius (cone), single-cap enablement, and the vertex.cap override path. Each
// case is checked under an identity instance AND a NON-UNIFORM scale instance
// (applied to both the primitive and its reference mesh), exercising the affine
// area Jacobian for these primitives. The closed shells enclose their inward
// emission (matte-black base absorbs it), so the tessellated double-sided mesh
// and the single-sided primitive light the exterior floor alike. Emitter parked
// above frame; 'quality', linear float, firefly off.

// anari_cpp
#define ANARI_EXTENSION_UTILITY_IMPL
#include <anari/anari_cpp/ext/std.h>
#include <anari/anari_cpp.hpp>
// VisRTX
#include <anari/ext/visrtx/makeVisRTXDevice.h>
// std
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <vector>
#include "DeviceTestArray.h"

struct ResidencyCase
{
  const char *name;
  int deviceInputs;
};

static constexpr std::array<ResidencyCase, 6> TUBE_CASES = {
    {{"host-indexed", 0},
        {"device-positions", 1},
        {"device-radii", 2},
        {"device-vertex-caps", 4},
        {"device-indices", 8},
        {"all-device", 15}}};

using vec3 = std::array<float, 3>;
using vec4 = std::array<float, 4>;
using uvec3 = std::array<unsigned, 3>;
using mat4 = std::array<float, 16>;

static void statusFunc(const void *,
    ANARIDevice,
    ANARIObject source,
    ANARIDataType,
    ANARIStatusSeverity severity,
    ANARIStatusCode,
    const char *message)
{
  if (severity <= ANARI_SEVERITY_ERROR) {
    fprintf(stderr, "[ERROR][%p] %s\n", source, message);
    std::exit(1);
  }
}

static constexpr std::array<unsigned, 2> IMAGE_SIZE = {256, 256};
static constexpr int PIXEL_SAMPLES = 1024;
static constexpr float EMISSIVE_RADIANCE = 8.f;
static constexpr double TOLERANCE = 0.03;
static bool equivalent(double actual, double reference)
{
  return std::isfinite(actual) && std::isfinite(reference) && actual > 0.0
      && reference > 0.0
      && std::abs(actual - reference) / reference <= TOLERANCE;
}
static constexpr int TESS =
    128; // circumferential segments of the reference mesh

// One tapered tube (cylinder if r0==r1). Two parallel bars above the origin,
// axes along X, offset in Z, so their walls face the floor. Above the top of
// frame; only the cast pool is measured.
struct Tube
{
  vec3 p0;
  float r0;
  vec3 p1;
  float r1;
  bool cap0; // cap at the p0 end
  bool cap1; // cap at the p1 end
};

static const std::array<Tube, 2> CYLINDERS = {
    Tube{{-0.6f, 1.5f, -0.3f}, 0.28f, {0.6f, 1.5f, -0.3f}, 0.28f, true, true},
    Tube{{-0.6f, 1.5f, 0.3f}, 0.18f, {0.6f, 1.5f, 0.3f}, 0.18f, true, false}};
static const std::array<Tube, 2> CONES = {
    Tube{{-0.6f, 1.5f, -0.3f}, 0.35f, {0.6f, 1.5f, -0.3f}, 0.12f, true, true},
    Tube{{-0.6f, 1.5f, 0.3f}, 0.10f, {0.6f, 1.5f, 0.3f}, 0.28f, true, false}};

// Non-uniform scale exercising the affine area Jacobian (column-major).
static constexpr mat4 SCALE_XFM = {1.7f,
    0.f,
    0.f,
    0.f,
    0.f,
    1.f,
    0.f,
    0.f,
    0.f,
    0.f,
    0.6f,
    0.f,
    0.f,
    0.f,
    0.f,
    1.f};

static vec3 sub(const vec3 &a, const vec3 &b)
{
  return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}
static vec3 add(const vec3 &a, const vec3 &b)
{
  return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}
static vec3 mul(const vec3 &a, float s)
{
  return {a[0] * s, a[1] * s, a[2] * s};
}
static float dot(const vec3 &a, const vec3 &b)
{
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
static vec3 cross(const vec3 &a, const vec3 &b)
{
  return {a[1] * b[2] - a[2] * b[1],
      a[2] * b[0] - a[0] * b[2],
      a[0] * b[1] - a[1] * b[0]};
}
static vec3 norml(const vec3 &a)
{
  return mul(a, 1.f / std::sqrt(dot(a, a)));
}

static anari::Material makeEmissiveMaterial(
    ANARIDevice device, float emissionScale = 1.f)
{
  auto mat = anari::newObject<anari::Material>(device, "physicallyBased");
  anari::setParameter(device, mat, "baseColor", vec3{0.f, 0.f, 0.f});
  anari::setParameter(device, mat, "metallic", 0.f);
  anari::setParameter(device, mat, "roughness", 1.f);
  anari::setParameter(device,
      mat,
      "emissive",
      vec3{EMISSIVE_RADIANCE * emissionScale,
          EMISSIVE_RADIANCE * emissionScale,
          EMISSIVE_RADIANCE * emissionScale});
  anari::commitParameters(device, mat);
  return mat;
}

static anari::Surface makeFloor(ANARIDevice device)
{
  const std::array<vec3, 4> pos = {vec3{-6.f, 0.f, -6.f},
      vec3{6.f, 0.f, -6.f},
      vec3{6.f, 0.f, 6.f},
      vec3{-6.f, 0.f, 6.f}};
  const std::array<uvec3, 2> idx = {uvec3{0, 1, 2}, uvec3{0, 2, 3}};

  auto geom = anari::newObject<anari::Geometry>(device, "triangle");
  anari::setParameterArray1D(device, geom, "vertex.position", pos.data(), 4);
  anari::setParameterArray1D(device, geom, "primitive.index", idx.data(), 2);
  anari::commitParameters(device, geom);

  auto mat = anari::newObject<anari::Material>(device, "physicallyBased");
  anari::setParameter(device, mat, "baseColor", vec3{0.5f, 0.5f, 0.5f});
  anari::setParameter(device, mat, "roughness", 1.f);
  anari::commitParameters(device, mat);

  auto surface = anari::newObject<anari::Surface>(device);
  anari::setAndReleaseParameter(device, surface, "geometry", geom);
  anari::setAndReleaseParameter(device, surface, "material", mat);
  anari::commitParameters(device, surface);
  return surface;
}

// Capped, tessellated emissive triangle mesh of the given tubes: two rings of
// TESS points per tube plus a center vertex for each ENABLED cap. Winding does
// not matter (emission is double-sided); the closed shell absorbs its own
// inward emission so it lights the exterior floor like a single-sided
// primitive.
static anari::Surface makeEmissiveTubeMesh(
    ANARIDevice device, const std::array<Tube, 2> &tubes, float emissionScale)
{
  std::vector<vec3> pos;
  std::vector<uvec3> idx;

  for (const auto &t : tubes) {
    const vec3 axis = norml(sub(t.p1, t.p0));
    const vec3 ref = std::abs(axis[0]) < 0.9f ? vec3{1, 0, 0} : vec3{0, 1, 0};
    const vec3 e0 = norml(cross(axis, ref));
    const vec3 e1 = cross(axis, e0);
    const auto ring = [&](const vec3 &c, float r, int i) {
      const float a = 2.f * float(M_PI) * (float(i) / TESS);
      return add(c, add(mul(e0, r * std::cos(a)), mul(e1, r * std::sin(a))));
    };

    const unsigned base = unsigned(pos.size());
    for (int i = 0; i < TESS; ++i) {
      pos.push_back(ring(t.p0, t.r0, i)); // base + 2i
      pos.push_back(ring(t.p1, t.r1, i)); // base + 2i+1
    }
    for (int i = 0; i < TESS; ++i) {
      const unsigned a0 = base + 2u * i, b0 = base + 2u * ((i + 1) % TESS);
      idx.push_back({a0, b0, b0 + 1}); // wall
      idx.push_back({a0, b0 + 1, a0 + 1});
    }
    if (t.cap0) {
      const unsigned c = unsigned(pos.size());
      pos.push_back(t.p0);
      for (int i = 0; i < TESS; ++i)
        idx.push_back(
            {c, base + 2u * ((i + 1) % TESS), base + 2u * unsigned(i)});
    }
    if (t.cap1) {
      const unsigned c = unsigned(pos.size());
      pos.push_back(t.p1);
      for (int i = 0; i < TESS; ++i)
        idx.push_back(
            {c, base + 2u * unsigned(i) + 1, base + 2u * ((i + 1) % TESS) + 1});
    }
  }

  auto geom = anari::newObject<anari::Geometry>(device, "triangle");
  anari::setParameterArray1D(
      device, geom, "vertex.position", pos.data(), pos.size());
  anari::setParameterArray1D(
      device, geom, "primitive.index", idx.data(), idx.size());
  anari::commitParameters(device, geom);

  auto surface = anari::newObject<anari::Surface>(device);
  anari::setAndReleaseParameter(device, surface, "geometry", geom);
  anari::setAndReleaseParameter(
      device, surface, "material", makeEmissiveMaterial(device, emissionScale));
  anari::commitParameters(device, surface);
  return surface;
}

// Per-endpoint vertex.cap (0/1) for a soup tube set: endpoints of tube k are
// vertices (2k, 2k+1).
static std::vector<unsigned char> tubeVertexCaps(
    const std::array<Tube, 2> &tubes)
{
  std::vector<unsigned char> caps;
  for (const auto &t : tubes) {
    caps.push_back(t.cap0 ? 1 : 0);
    caps.push_back(t.cap1 ? 1 : 0);
  }
  return caps;
}

static anari::Surface makeEmissiveCylinderPrim(
    ANARIDevice device, int deviceInputs, bool indexed)
{
  std::vector<vec3> pos;
  std::vector<float> radii; // per primitive
  for (const auto &t : CYLINDERS) {
    pos.push_back(t.p0);
    pos.push_back(t.p1);
    radii.push_back(t.r0); // cylinder: single radius per primitive
  }
  const auto caps = tubeVertexCaps(CYLINDERS);

  auto geom = anari::newObject<anari::Geometry>(device, "cylinder");
  setTestArray(device, geom, "vertex.position", pos, deviceInputs & 1);
  setTestArray(device, geom, "primitive.radius", radii, deviceInputs & 2);
  setTestArray(device, geom, "vertex.cap", caps, deviceInputs & 4);
  if (indexed) {
    const std::array<std::array<unsigned, 2>, 2> indices = {{{0, 1}, {2, 3}}};
    setTestArray(device, geom, "primitive.index", indices, deviceInputs & 8);
  }
  anari::commitParameters(device, geom);

  auto surface = anari::newObject<anari::Surface>(device);
  anari::setAndReleaseParameter(device, surface, "geometry", geom);
  anari::setAndReleaseParameter(
      device, surface, "material", makeEmissiveMaterial(device));
  anari::commitParameters(device, surface);
  return surface;
}

static anari::Surface makeEmissiveConePrim(
    ANARIDevice device, int deviceInputs, bool indexed)
{
  std::vector<vec3> pos;
  std::vector<float> radii; // per vertex
  for (const auto &t : CONES) {
    pos.push_back(t.p0);
    pos.push_back(t.p1);
    radii.push_back(t.r0);
    radii.push_back(t.r1);
  }
  const auto caps = tubeVertexCaps(CONES);

  auto geom = anari::newObject<anari::Geometry>(device, "cone");
  setTestArray(device, geom, "vertex.position", pos, deviceInputs & 1);
  setTestArray(device, geom, "vertex.radius", radii, deviceInputs & 2);
  setTestArray(device, geom, "vertex.cap", caps, deviceInputs & 4);
  if (indexed) {
    const std::array<std::array<unsigned, 2>, 2> indices = {{{0, 1}, {2, 3}}};
    setTestArray(device, geom, "primitive.index", indices, deviceInputs & 8);
  }
  anari::commitParameters(device, geom);

  auto surface = anari::newObject<anari::Surface>(device);
  anari::setAndReleaseParameter(device, surface, "geometry", geom);
  anari::setAndReleaseParameter(
      device, surface, "material", makeEmissiveMaterial(device));
  anari::commitParameters(device, surface);
  return surface;
}

// Instance a single surface, optionally under a non-uniform scale transform.
static anari::Instance instanceOf(
    ANARIDevice device, anari::Surface surface, bool scaled)
{
  auto group = anari::newObject<anari::Group>(device);
  anari::setParameterArray1D(device, group, "surface", &surface, 1);
  anari::release(device, surface);
  anari::commitParameters(device, group);

  auto inst = anari::newObject<anari::Instance>(device, "transform");
  anari::setAndReleaseParameter(device, inst, "group", group);
  if (scaled)
    anari::setParameter(
        device, inst, "transform", ANARI_FLOAT32_MAT4, SCALE_XFM.data());
  anari::commitParameters(device, inst);
  return inst;
}

static std::vector<vec4> render(ANARIDevice device,
    const std::function<anari::Surface(ANARIDevice)> &emitter,
    bool scaled)
{
  const std::array<anari::Instance, 2> instances = {
      instanceOf(device, makeFloor(device), false),
      instanceOf(device, emitter(device), scaled)};

  auto world = anari::newObject<anari::World>(device);
  anari::setParameterArray1D(
      device, world, "instance", instances.data(), instances.size());
  for (auto i : instances)
    anari::release(device, i);
  anari::commitParameters(device, world);

  auto camera = anari::newObject<anari::Camera>(device, "perspective");
  anari::setParameter(device, camera, "position", vec3{0.f, 0.5f, -3.f});
  anari::setParameter(device, camera, "direction", vec3{0.f, -0.15f, 1.f});
  anari::setParameter(device, camera, "up", vec3{0.f, 1.f, 0.f});
  anari::setParameter(
      device, camera, "aspect", IMAGE_SIZE[0] / float(IMAGE_SIZE[1]));
  anari::commitParameters(device, camera);

  auto renderer = anari::newObject<anari::Renderer>(device, "quality");
  anari::setParameter(device, renderer, "background", vec4{0.f, 0.f, 0.f, 1.f});
  anari::setParameter(device, renderer, "ambientRadiance", 0.f);
  anari::setParameter(device, renderer, "pixelSamples", PIXEL_SAMPLES);
  anari::setParameter(device, renderer, "fireflyFilterMode", "none");
  anari::commitParameters(device, renderer);

  auto frame = anari::newObject<anari::Frame>(device);
  anari::setParameter(device, frame, "size", IMAGE_SIZE);
  anari::setParameter(device, frame, "channel.color", ANARI_FLOAT32_VEC4);
  anari::setAndReleaseParameter(device, frame, "world", world);
  anari::setAndReleaseParameter(device, frame, "camera", camera);
  anari::setAndReleaseParameter(device, frame, "renderer", renderer);
  anari::commitParameters(device, frame);

  anari::render(device, frame);
  anari::wait(device, frame);

  auto fb = anari::map<vec4>(device, frame, "channel.color");
  std::vector<vec4> out(fb.data, fb.data + size_t(fb.width) * fb.height);
  anari::unmap(device, frame, "channel.color");
  anari::release(device, frame);
  return out;
}

// Mean linear luminance over the lit floor region (lower-center of the frame,
// ANARI's bottom-left framebuffer origin).
static double poolMeanLuminance(const std::vector<vec4> &fb)
{
  double sum = 0.0;
  uint64_t n = 0;
  for (uint32_t y = IMAGE_SIZE[1] / 8; y < IMAGE_SIZE[1] / 2; ++y) {
    for (uint32_t x = 3 * IMAGE_SIZE[0] / 8; x < 5 * IMAGE_SIZE[0] / 8; ++x) {
      const vec4 &p = fb[y * IMAGE_SIZE[0] + x];
      sum += 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2];
      ++n;
    }
  }
  return n ? sum / double(n) : 0.0;
}

static bool checkFamily(ANARIDevice device,
    const char *name,
    const std::array<Tube, 2> &tubes,
    const std::function<anari::Surface(ANARIDevice, int, bool)> &prim,
    bool scaled)
{
  const char *transform = scaled ? "nonuniform-scale" : "identity";
  const auto reference = [&](const char *phase, float emissionScale) {
    fprintf(stderr, "[case] %s/%s/%s\n", name, transform, phase);
    return poolMeanLuminance(render(
        device,
        [&](ANARIDevice d) {
          return makeEmissiveTubeMesh(d, tubes, emissionScale);
        },
        scaled));
  };
  // References stay local to this family AND transform. All other scene,
  // camera, Quality renderer, 1024-spp accumulation and output settings are
  // fixed by render(). Fresh repeat and half-power scenes are not reused as
  // the full-power reference or shared with another family/transform.
  const double mesh = reference("reference", 1.f);
  const double repeat = reference("reference-repeat", 1.f);
  const double wrong = reference("reference-half-power", 0.5f);
  bool ok = equivalent(mesh, repeat) && std::isfinite(wrong) && wrong > 0.0
      && !equivalent(mesh, wrong);
  printf("%s/%s controls repeatRelErr=%g wrongRelErr=%g tolerance=%g %s\n",
      name,
      transform,
      std::abs(repeat - mesh) / mesh,
      std::abs(mesh - wrong) / wrong,
      TOLERANCE,
      ok ? "PASS" : "FAIL");

  const auto check = [&](const char *residency,
                         int deviceInputs,
                         bool indexed) {
    fprintf(stderr, "[case] %s/%s/%s\n", name, transform, residency);
    const double actual = poolMeanLuminance(render(
        device,
        [&](ANARIDevice d) { return prim(d, deviceInputs, indexed); },
        scaled));
    // Keep the original 3% MC/faceting allowance and use that same comparator
    // to reject the independently rendered half-power control for every case.
    const bool matches = equivalent(actual, mesh) && !equivalent(actual, wrong);
    printf("%s/%s/%s primitive=%f reference=%f relErr=%g tolerance=%g %s\n",
        name,
        transform,
        residency,
        actual,
        mesh,
        std::abs(actual - mesh) / mesh,
        TOLERANCE,
        matches ? "PASS" : "FAIL");
    ok = matches && ok;
  };
  check("host-implicit-indices", 0, false);
  for (const auto &residency : TUBE_CASES) {
    // Residency roles are isolated at identity; the all-device case also
    // witnesses their integration with the nonuniform area Jacobian.
    if (!scaled || residency.deviceInputs == 15)
      check(residency.name, residency.deviceInputs, true);
  }
  return ok;
}

int main()
{
  auto device = makeVisRTXDevice(statusFunc);
  bool ok = true;
  for (bool scaled : {false, true}) {
    ok = checkFamily(
             device, "cylinder", CYLINDERS, makeEmissiveCylinderPrim, scaled)
        && ok;
    ok = checkFamily(device, "cone", CONES, makeEmissiveConePrim, scaled) && ok;
  }
  anari::release(device, device);
  return ok ? 0 : 1;
}
