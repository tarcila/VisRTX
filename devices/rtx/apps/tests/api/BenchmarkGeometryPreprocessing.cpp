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

// Manual evidence tool, deliberately not a wall-clock CTest assertion.
#include <anari/anari_cpp/ext/std.h>
#include <anari/ext/visrtx/makeVisRTXDevice.h>
#include <anari/anari_cpp.hpp>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
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
int main(int argc, char **argv)
{
  if (argc != 5 && argc != 6) {
    fprintf(stderr,
        "usage: %s triangle|sphere|cone|cylinder|quad|curve|sdf count device repeats [shared-surfaces]\n",
        argv[0]);
    return 2;
  }
  const std::string type = argv[1];
  const size_t n = std::stoull(argv[2]);
  const bool device = std::atoi(argv[3]) != 0;
  const int repeats = std::atoi(argv[4]);
  const int sharedSurfaces = argc == 6 ? std::atoi(argv[5]) : 1;
  if (!n || repeats < 2 || sharedSurfaces < 1)
    return 2;
  auto d = makeVisRTXDevice(status);
  auto geometry = anari::newObject<anari::Geometry>(d, type.c_str());
  std::vector<vec3> positions;
  std::vector<float> radii;
  std::vector<unsigned> sphereIndices;
  std::vector<std::array<unsigned, 2>> tubeIndices;
  std::vector<std::array<unsigned, 3>> triIndices;
  std::vector<std::array<unsigned, 4>> quadIndices;
  std::vector<uint8_t> sdf;
  for (size_t i = 0; i < n; ++i) {
    const float x = float(i % 256) * 0.04f, y = float(i / 256) * 0.04f;
    const unsigned base = unsigned(positions.size());
    if (type == "triangle" || type == "quad") {
      positions.push_back({x, y, 0});
      positions.push_back({x + 0.02f, y, 0});
      positions.push_back({x + 0.02f, y + 0.02f, 0});
      if (type == "quad") {
        positions.push_back({x, y + 0.02f, 0});
        quadIndices.push_back({base, base + 1, base + 2, base + 3});
      } else
        triIndices.push_back({base, base + 1, base + 2});
    } else if (type == "sphere") {
      positions.push_back({x, y, 0});
      radii.push_back(0.01f);
      sphereIndices.push_back(base);
    } else if (type == "sdf") {
      const size_t offset = sdf.size();
      sdf.resize(offset + 72);
      const vec3 center = {x, y, 0};
      const float radius = 0.01f;
      std::memcpy(sdf.data() + offset + 20, center.data(), sizeof(center));
      std::memcpy(sdf.data() + offset + 44, &radius, sizeof(radius));
      std::memcpy(sdf.data() + offset + 48, &radius, sizeof(radius));
    } else {
      positions.push_back({x, y, 0});
      positions.push_back({x + 0.02f, y, 0});
      tubeIndices.push_back({base, base + 1});
      radii.push_back(0.01f);
      if (type != "cylinder")
        radii.push_back(0.005f);
    }
  }
  if (type == "sdf")
    setTestArray(d, geometry, "primitive.sdf", sdf, device, 0);
  else
    setTestArray(d, geometry, "vertex.position", positions, device, 0);
  if (type == "triangle")
    setTestArray(d, geometry, "primitive.index", triIndices, device, 0);
  if (type == "quad")
    setTestArray(d, geometry, "primitive.index", quadIndices, device, 0);
  if (type == "sphere")
    setTestArray(d, geometry, "primitive.index", sphereIndices, device, 0);
  if (type == "cone" || type == "cylinder")
    setTestArray(d, geometry, "primitive.index", tubeIndices, device, 0);
  if (type != "curve" && !radii.empty())
    setTestArray(d,
        geometry,
        type == "cylinder" ? "primitive.radius" : "vertex.radius",
        radii,
        device,
        0);
  if (type == "curve")
    anari::setParameter(d, geometry, "radius", 0.01f);
  anari::commitParameters(d, geometry);
  auto material = anari::newObject<anari::Material>(d, "physicallyBased");
  anari::setParameter(d, material, "emissive", vec3{1, 1, 1});
  anari::commitParameters(d, material);
  std::vector<anari::Surface> surfaces;
  for (int i = 0; i < sharedSurfaces; ++i) {
    auto surface = anari::newObject<anari::Surface>(d);
    anari::setParameter(d, surface, "geometry", geometry);
    anari::setParameter(d, surface, "material", material);
    anari::commitParameters(d, surface);
    surfaces.push_back(surface);
  }
  anari::release(d, material);
  auto world = anari::newObject<anari::World>(d);
  anari::setParameterArray1D(
      d, world, "surface", surfaces.data(), surfaces.size());
  for (auto surface : surfaces)
    anari::release(d, surface);
  anari::commitParameters(d, world);
  // Query completes required preprocessing and AS builds, but performs no ray
  // tracing. Later iterations invalidate geometry without replacing its inputs.
  double warm = 0;
  for (int iteration = 0; iteration < repeats; ++iteration) {
    const auto start = std::chrono::steady_clock::now();
    anari::setParameter(
        d, geometry, "attribute0", vec4{float(iteration), 0, 0, 0});
    anari::commitParameters(d, geometry);
    std::array<float, 6> bounds{};
    const bool ok = anariGetProperty(d,
        world,
        "bounds",
        ANARI_FLOAT32_BOX3,
        bounds.data(),
        sizeof(bounds),
        ANARI_WAIT);
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start)
                          .count();
    if (!ok || errors)
      return 1;
    if (iteration == 0)
      printf("first_ms=%.6f\n", ms);
    else
      warm += ms;
  }
  printf("type=%s count=%zu device=%d warm_rebuild_ms=%.6f repeats=%d\n",
      type.c_str(),
      n,
      device,
      warm / (repeats - 1),
      repeats);
  anari::release(d, world);
  anari::release(d, geometry);
  anari::release(d, d);
  return 0;
}
