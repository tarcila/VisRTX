// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "sbt.h"
#include "shadingState.h"

namespace visrtx {
namespace volume_source {
enum class SurfacePlan
{
  ContinuationOnly,
  NativeReflection,
  DirectOnly
};
VISRTX_HOST_DEVICE SurfacePlan surfacePlan(
    bool proposalAvailable, uint32_t callable)
{
  if (!proposalAvailable)
    return SurfacePlan::ContinuationOnly;
  if (callable == uint32_t(SbtCallableEntryPoints::PBR))
    return SurfacePlan::NativeReflection;
  if (callable == uint32_t(SbtCallableEntryPoints::Matte))
    return SurfacePlan::DirectOnly;
  // In particular, no material-name allowlist or speculative MDL replacement.
  return SurfacePlan::ContinuationOnly;
}
VISRTX_HOST_DEVICE bool outgoingOwner(SurfacePlan plan, const NextRay &sample)
{
  return !(plan == SurfacePlan::NativeReflection
      && sample.event == NextRayEvent::Continuous
      && sample.component == NextRayComponent::NativeReflection);
}
} // namespace volume_source
} // namespace visrtx
