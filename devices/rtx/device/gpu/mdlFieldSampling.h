// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "gpu/RenderError.h"
#include "gpu/shadingState.h"
#include "spatial_field/CustomFieldSamplerInline.h"
#include "spatial_field/NvdbRectilinearSamplerInline.h"
#include "spatial_field/NvdbRegularSamplerInline.h"
#include "spatial_field/StructuredRectilinearSamplerInline.h"
#include "spatial_field/StructuredRegularSamplerInline.h"

namespace visrtx {
// Support is geometric, never inferred from the scalar's numeric value. Keep
// finite NanoVDB background values as data, and diagnose NaN/Inf as failures.
VISRTX_DEVICE bool sampleMdlField(const FrameGPUData &fd,
    const VolumeHit &hit,
    const vec3 &position,
    float &value,
    bool &supported)
{
  supported = false;
  const auto &binding = hit.volume->data.mdl;
  if (binding.field < 0)
    return true;
  const auto &field = fd.registry.fields[binding.field];
  const auto &bounds = binding.fieldBounds;
  if (glm::any(glm::lessThan(position, bounds.lower))
      || glm::any(glm::greaterThan(position, bounds.upper))
      || glm::any(glm::lessThan(position, field.roi.lower))
      || glm::any(glm::greaterThan(position, field.roi.upper)))
    return true;
  VolumeSamplingState state;
#define SAMPLE_CASE(ENTRY, MEMBER, INIT)                                       \
  case SbtCallableEntryPoints::ENTRY:                                          \
    INIT(state.MEMBER, &field);                                                \
    value = sampleValue(state.MEMBER, field, position);                        \
    break
  switch (field.samplerCallableIndex) {
    SAMPLE_CASE(SpatialFieldSamplerRegular,
        structuredRegular,
        initStructuredRegularSampler);
    SAMPLE_CASE(SpatialFieldSamplerRectilinear,
        structuredRectilinear,
        initStructuredRectilinearSampler);
    SAMPLE_CASE(SpatialFieldSamplerNvdbFp4, nvdbFp4, initNvdbSampler);
    SAMPLE_CASE(SpatialFieldSamplerNvdbFp8, nvdbFp8, initNvdbSampler);
    SAMPLE_CASE(SpatialFieldSamplerNvdbFp16, nvdbFp16, initNvdbSampler);
    SAMPLE_CASE(SpatialFieldSamplerNvdbFpN, nvdbFpN, initNvdbSampler);
    SAMPLE_CASE(SpatialFieldSamplerNvdbFloat, nvdbFloat, initNvdbSampler);
    SAMPLE_CASE(SpatialFieldSamplerNvdbRectilinearFp4,
        nvdbRectilinearFp4,
        initNvdbRectilinearSampler);
    SAMPLE_CASE(SpatialFieldSamplerNvdbRectilinearFp8,
        nvdbRectilinearFp8,
        initNvdbRectilinearSampler);
    SAMPLE_CASE(SpatialFieldSamplerNvdbRectilinearFp16,
        nvdbRectilinearFp16,
        initNvdbRectilinearSampler);
    SAMPLE_CASE(SpatialFieldSamplerNvdbRectilinearFpN,
        nvdbRectilinearFpN,
        initNvdbRectilinearSampler);
    SAMPLE_CASE(SpatialFieldSamplerNvdbRectilinearFloat,
        nvdbRectilinearFloat,
        initNvdbRectilinearSampler);
  case SbtCallableEntryPoints::SpatialFieldSamplerCustom:
    initSamplerState(state, field);
    value = sampleValue(state, field, position);
    break;
  default:
    recordRenderError(fd.renderError,
        RenderErrorCategory::UNSUPPORTED_MEDIUM,
        hit.volume->id,
        hit.instance->id);
    return false;
  }
#undef SAMPLE_CASE
  supported = true;
  if (!isfinite(value)) {
    recordRenderError(fd.renderError,
        RenderErrorCategory::INVALID_FIELD_VALUE,
        hit.volume->id,
        hit.instance->id);
    return false;
  }
  return true;
}
} // namespace visrtx
