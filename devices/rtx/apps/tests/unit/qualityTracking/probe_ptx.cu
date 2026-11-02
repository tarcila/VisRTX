// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#include <optix_device.h>
#include "gpu/gpu_objects.h"
#include "gpu/renderer/qualityVolume.h"

namespace visrtx {
DECLARE_FRAME_DATA(frameData)
struct ProbeResult
{
  double value[9]{};
};

VISRTX_GLOBAL void __raygen__qualityVolumeProbe()
{
  namespace qv = quality_volume;
  namespace tr = volume_tracking;
  const uint32_t index = optixGetLaunchIndex().x;
  qv::Request q;
  q.scene = frameData.referenceVolumes;
  q.ray = aggregate_volume::worldDistanceRay(
      {vec3(.5f, .5f, -1.f), vec3(0, 0, 1), {0, 4}});
  pcg_init(&q.random, detail::pcg_mix64(uint64_t(index) + 37), 91);
  qv::execute(q);
  bool valid = true;
  if (q.flight.endpoint == tr::Endpoint::Scattered) {
    q.operation = qv::Operation::Direction;
    qv::execute(q);
    valid = std::abs(glm::length(q.direction.direction) - 1.) < 2e-12
        && q.direction.mixture.pdf > 0.;
    q.cosine = q.direction.cosine;
    q.operation = qv::Operation::Phase;
    qv::execute(q);
    valid &= q.phase.pdf == q.direction.mixture.pdf;
    for (int c = 0; c < 3; ++c)
      valid &= q.phase.kernel[c] == q.direction.mixture.kernel[c];
  }
  ProbeResult result;
  for (int c = 0; c < 3; ++c) {
    result.value[c] = q.path.radiance[c];
    result.value[c + 3] =
        q.flight.endpoint == tr::Endpoint::Boundary ? q.path.throughput[c] : 0.;
  }
  q.operation = qv::Operation::Shadow;
  pcg_init(&q.random, detail::pcg_mix64(uint64_t(index) + 93), 47);
  qv::execute(q);
  for (int c = 0; c < 3; ++c)
    result.value[c + 6] = q.transmission[c];
  // Test-only failure signal, not a production Frame status or validity check.
  if (!valid)
    result.value[0] = nan("");
  reinterpret_cast<ProbeResult *>(
      frameData.fb.buffers.colorAccumulation)[index] = result;
}
} // namespace visrtx
