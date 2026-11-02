// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cstdint>
namespace visrtx {
// Realized normalized CDF, including endpoints 0 and 1. Zero-width cells are
// legal proposal holes, NOT proof that the physical source is zero there.
struct SourceImportanceGPUData
{
  const double *cdf{nullptr};
  uint32_t resolution{0};
};
} // namespace visrtx
