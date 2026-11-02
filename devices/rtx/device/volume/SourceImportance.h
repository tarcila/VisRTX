// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "gpu/aggregateVolumeData.h"
namespace visrtx {
// One volume owns one object-space proposal, shared by all its occurrences.
// Changes to source/bindings reset it; camera/light/instance changes do not.
class SourceImportance
{
 public:
  SourceImportance() = default;
  ~SourceImportance();
  SourceImportance(const SourceImportance &) = delete;
  SourceImportance &operator=(const SourceImportance &) = delete;
  SourceImportanceGPUData prepare(const ReferenceVolumeGPUData &volume);
  void reset();

 private:
  bool m_attempted{false};
  double *m_cdf{nullptr};
  uint32_t m_resolution{0};
};
} // namespace visrtx
