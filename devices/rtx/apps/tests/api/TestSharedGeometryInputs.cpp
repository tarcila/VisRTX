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

#include <anari/anari_cpp/ext/std.h>
#include <anari/ext/visrtx/makeVisRTXDevice.h>
#include <anari/anari_cpp.hpp>
#include <array>
#include <cmath>
#include <cstdio>
#include "DeviceTestArray.h"

using vec3 = std::array<float, 3>;
using vec4 = std::array<float, 4>;
static int errors = 0;
static void status(const void *,
    ANARIDevice,
    ANARIObject,
    ANARIDataType,
    ANARIStatusSeverity severity,
    ANARIStatusCode,
    const char *message)
{
  if (severity <= ANARI_SEVERITY_ERROR) {
    ++errors;
    fprintf(stderr, "%s\n", message);
  }
}
struct Scene
{
  anari::Geometry geometry;
  anari::World world;
  anari::Frame frame;
};
static Scene makeScene(ANARIDevice d, anari::Array1D positions)
{
  Scene s;
  s.geometry = anari::newObject<anari::Geometry>(d, "curve");
  anari::setParameter(d, s.geometry, "vertex.position", positions);
  anari::setParameter(d, s.geometry, "radius", 0.125f);
  anari::commitParameters(d, s.geometry);
  auto material = anari::newObject<anari::Material>(d, "matte");
  anari::setParameter(d, material, "color", vec3{1, 0, 0});
  anari::commitParameters(d, material);
  auto surface = anari::newObject<anari::Surface>(d);
  anari::setParameter(d, surface, "geometry", s.geometry);
  anari::setAndReleaseParameter(d, surface, "material", material);
  anari::commitParameters(d, surface);
  s.world = anari::newObject<anari::World>(d);
  anari::setParameterArray1D(d, s.world, "surface", &surface, 1);
  anari::release(d, surface);
  anari::commitParameters(d, s.world);
  auto camera = anari::newObject<anari::Camera>(d, "perspective");
  anari::setParameter(d, camera, "position", vec3{0, 0, -3});
  anari::setParameter(d, camera, "direction", vec3{0, 0, 1});
  anari::commitParameters(d, camera);
  auto renderer = anari::newObject<anari::Renderer>(d, "debug");
  anari::setParameter(d, renderer, "method", "baseColor");
  anari::setParameter(d, renderer, "background", vec4{0, 0, 0, 1});
  anari::commitParameters(d, renderer);
  s.frame = anari::newObject<anari::Frame>(d);
  anari::setParameter(d, s.frame, "size", std::array<unsigned, 2>{64, 64});
  anari::setParameter(d, s.frame, "channel.color", ANARI_FLOAT32_VEC4);
  anari::setParameter(d, s.frame, "world", s.world);
  anari::setAndReleaseParameter(d, s.frame, "camera", camera);
  anari::setAndReleaseParameter(d, s.frame, "renderer", renderer);
  anari::commitParameters(d, s.frame);
  return s;
}
static bool check(ANARIDevice d, Scene s, float y)
{
  anari::render(d, s.frame);
  anari::wait(d, s.frame);
  std::array<float, 6> bounds{};
  bool ok = anariGetProperty(d,
      s.world,
      "bounds",
      ANARI_FLOAT32_BOX3,
      bounds.data(),
      sizeof(bounds),
      ANARI_WAIT);
  ok = ok && std::abs(bounds[1] - (y - 0.125f)) < 1e-4f
      && std::abs(bounds[4] - (y + 0.125f)) < 1e-4f;
  auto fb = anari::map<vec4>(d, s.frame, "channel.color");
  const float red = fb.data[32 * 64 + 32][0];
  ok = ok && std::isfinite(red) && (y == 0 ? red > 0.5f : red < 0.01f);
  anari::unmap(d, s.frame, "channel.color");
  printf("shared-input y=%g bounds=[%g,%g] center=%g %s\n",
      y,
      bounds[1],
      bounds[4],
      red,
      ok ? "PASS" : "FAIL");
  return ok;
}
int main()
{
  auto d = makeVisRTXDevice(status);
  auto positions = anari::newArray1D(d, ANARI_FLOAT32_VEC3, 2);
  auto update = [&](float y) {
    auto *p = anari::map<vec3>(d, positions);
    p[0] = {-0.5f, y, 0};
    p[1] = {0.5f, y, 0};
    anari::unmap(d, positions);
  };
  update(0);
  const std::array<Scene, 2> scenes = {
      makeScene(d, positions), makeScene(d, positions)};
  bool ok = check(d, scenes[0], 0) && check(d, scenes[1], 0);
  update(1);
  ok = check(d, scenes[0], 1) && check(d, scenes[1], 1) && ok;
  const std::array<vec3, 2> replacement = {{{-0.5f, 0, 0}, {0.5f, 0, 0}}};
  setTestArray(d, scenes[0].geometry, "vertex.position", replacement, true);
  anari::commitParameters(d, scenes[0].geometry);
  ok = check(d, scenes[0], 0) && check(d, scenes[1], 1) && ok;
  for (auto s : scenes) {
    anari::release(d, s.frame);
    anari::release(d, s.world);
    anari::release(d, s.geometry);
  }
  anari::release(d, positions);
  anari::release(d, d);
  return ok && errors == 0 ? 0 : 1;
}
