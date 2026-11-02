// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <mi/neuraylib/target_code_types.h>
#include "gpu/shadingState.h"

namespace visrtx {

// Consume the actual generated-target record. ABSORB may leave stale k2 and
// bsdf_over_pdf from a subsequently rejected child: even zero initialization of
// the input record does not make those fields live. No normal/direction/weight
// arithmetic may precede this event check. No pdf/weight threshold replaces it.
VISRTX_HOST_DEVICE NextRay consumeMdlSample(
    const mi::neuraylib::Bsdf_sample_data &sample,
    const vec3 &incoming,
    const vec3 &normal)
{
  if (sample.event_type == mi::neuraylib::BSDF_EVENT_ABSORB)
    return deadNextRay();
  const vec3 direction(sample.k2.x, sample.k2.y, sample.k2.z);
  const vec3 n = normalize(normal);
  const uint32_t flags = dot(incoming, n) * dot(direction, n) > 0.f
      ? NEXT_RAY_CONTINUES_THROUGH_SURFACE
      : NEXT_RAY_NONE;
  const bool singular =
      (sample.event_type & mi::neuraylib::BSDF_EVENT_SPECULAR) != 0;
  // Retain the old environment-ownership and geometric side policy separately
  // from the actual SDK measure. In particular, continuous transmission has an
  // infinite legacy pdf but is NOT a singular event.
  const float pdf = singular || (flags & NEXT_RAY_CONTINUES_THROUGH_SURFACE)
      ? INFINITY
      : sample.pdf;
  return {direction,
      vec3(sample.bsdf_over_pdf.x,
          sample.bsdf_over_pdf.y,
          sample.bsdf_over_pdf.z),
      pdf,
      flags,
      singular ? NextRayEvent::Singular : NextRayEvent::Continuous,
      uint32_t(sample.event_type),
      NextRayComponent::MDL};
}
} // namespace visrtx
