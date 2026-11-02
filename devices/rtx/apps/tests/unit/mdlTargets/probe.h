// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "gpu/shadingState.h"
namespace mdl_target_test {
using namespace visrtx;
constexpr unsigned SAMPLES = 65536;
struct Result
{
  vec3 direction, weight, evaluated, wrapperEvaluated, normal;
  float pdf, queriedPdf, wrapperPdf;
  uint32_t event, unchangedRandom;
  NextRay consumed;
};
struct Params
{
  FrameGPUData *frame;
  const char *argumentBlock;
  Result *output;
  unsigned state;
};
} // namespace mdl_target_test
