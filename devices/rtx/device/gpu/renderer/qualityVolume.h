// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "gpu/aggregateVolumePhase.h"
#include "gpu/sbt.h"
#include "gpu/volumeSource.h"
#if defined(__CUDA_ARCH__)
#include <optix_device.h>
#endif

namespace visrtx {
namespace quality_volume {

// Explicit request ABI at the Quality transport -> reference field boundary.
// No preview cache/texture or Frame output policy is implicit in this
// interface.
enum class Operation
{
  Flight,
  Shadow,
  Direction,
  Phase,
  Coverage,
  Source
};
struct Request
{
  Operation operation{Operation::Flight};
  const AggregateVolumeSceneGPUData *scene{nullptr};
  aggregate_volume::Ray ray;
  PCGState random{};
  double sourceProposal{1.};
  volume_tracking::PathState path;
  volume_tracking::FlightResult flight;
  double transmission[3]{1., 1., 1.};
  double coverageAlbedo[3]{};
  double cosine{0.};
  volume_tracking::DirectionSample direction;
  volume_tracking::PhaseMixture phase;
  bool hasSinglePhase{false};
  VolumePhaseFunction singlePhase;
  volume_source::Sample source;
};

static_assert(std::is_trivially_copyable_v<Request>);

#if defined(__CUDA_ARCH__)
VISRTX_DEVICE void execute(Request &request)
{
  optixDirectCall<void>(
      uint32_t(SbtCallableEntryPoints::QualityVolume), &request);
}
#endif

} // namespace quality_volume
} // namespace visrtx
