// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <climits>
#include <cstdio>
#include <vector>
#include "gpu/cameraCreateRay.h"
#include "gpu/sampleOrdinal.h"

using visrtx::sampleOrdinal;

int main()
{
  int failures = 0;
  auto check = [&](bool ok, const char *label) {
    if (!ok) {
      std::fprintf(stderr, "FAIL: %s\n", label);
      ++failures;
    }
  };
  // A literal prefix from the specification, independent of batch boundaries.
  const std::vector<uint32_t> prefix = {0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12,
      13,
      14,
      15,
      16,
      17,
      18,
      19,
      20,
      21,
      22,
      23};
  for (int spp : {1, 2, 3, 4, 8}) {
    for (int run = 0; run < 2; ++run) { // reset restarts at zero
      std::vector<uint32_t> trace;
      for (int completed = 0; completed < 24; completed += spp)
        for (int i = 0; i < spp; ++i)
          trace.push_back(sampleOrdinal(completed, i));
      check(trace == prefix, "contiguous prefix across batches and reset");
    }
  }
  check(sampleOrdinal(0, 0) == 0,
      "underlying ordinal zero, without camera override");
  // Four checkerboard tiles share the completed count; effective spp is one.
  std::vector<uint32_t> tiles;
  for (int cycle = 0; cycle < 3; ++cycle)
    for (int tile = 0; tile < 4; ++tile)
      tiles.push_back(sampleOrdinal(cycle, 0));
  check(tiles == std::vector<uint32_t>{0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2},
      "each tile receives cycle ordinal");
  const std::vector<uint32_t> nearLimit = {2147483639u,
      2147483640u,
      2147483641u,
      2147483642u,
      2147483643u,
      2147483644u,
      2147483645u,
      2147483646u};
  std::vector<uint32_t> trace;
  for (int i = 0; i < 8; ++i)
    trace.push_back(sampleOrdinal(INT_MAX - 8, i));
  check(trace == nearLimit, "last supported eight-sample batch");
  check((INT_MAX - 8) + 8 == INT_MAX, "frame update remains representable");
  // Exercise the real camera path independently of ordinal addressing.
  visrtx::FrameGPUData frame{};
  frame.fb.invSize = visrtx::vec2(1.f / 16.f);
  frame.camera.type = visrtx::CameraType::PERSPECTIVE;
  frame.camera.region = visrtx::vec4(0.f, 0.f, 1.f, 1.f);
  frame.camera.perspective.dir_00 = visrtx::vec3(-1.f, -1.f, -1.f);
  frame.camera.perspective.dir_du = visrtx::vec3(2.f, 0.f, 0.f);
  frame.camera.perspective.dir_dv = visrtx::vec3(0.f, 2.f, 0.f);
  frame.camera.perspective.aspect = 1.f;
  frame.camera.perspective.scaledAperture = 0.1f;
  visrtx::ScreenSample ss{};
  ss.frameData = &frame;
  ss.pixel = visrtx::uvec2(3, 5);
  const auto r = visrtx::owenSobolCamera(0, visrtx::owenPixelSeed(3, 5));
  const auto jittered = visrtx::makePrimaryRay(ss, 0, false);
  check(ss.screen == visrtx::vec2((3.f + r.x) / 16.f, (5.f + r.y) / 16.f),
      "ordinal zero normally uses Sobol jitter");
  const auto centered = visrtx::makePrimaryRay(ss, 0, true);
  check(ss.screen == visrtx::vec2(3.5f / 16.f, 5.5f / 16.f),
      "first accumulated ray overrides pixel jitter");
  check(jittered.dir != centered.dir,
      "center override is independently observable");
  check(jittered.org == centered.org && centered.org != frame.camera.pos,
      "center override preserves sampled aperture");
  frame.camera.type = visrtx::CameraType::RAY_BUFFER;
  frame.camera.rayBuffer = {}; // switch the active camera union member
  const visrtx::vec3 rayDirection(0.f, 0.f, -2.f);
  frame.camera.rayBuffer.valid = true;
  frame.camera.rayBuffer.size = visrtx::uvec2(1);
  frame.camera.rayBuffer.dir = &rayDirection;
  ss.pixel = visrtx::uvec2(0);
  const auto buffered = visrtx::makePrimaryRay(ss, 19, false);
  check(buffered.org == frame.camera.pos
          && buffered.dir == visrtx::vec3(0.f, 0.f, -1.f)
          && ss.screen == visrtx::vec2(0.5f / 16.f),
      "ray-buffer bypass ignores jitter and aperture");
  return failures ? 1 : 0;
}
