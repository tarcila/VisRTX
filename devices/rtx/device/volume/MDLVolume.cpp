// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#include "MDLVolume.h"
#include <cmath>
#include <typeinfo>
#include "spatial_field/SpatialField.h"

namespace visrtx {
MDLVolume::MDLVolume(DeviceGlobalState *d)
    : Volume(d), m_material(this), m_field(this)
{}
void MDLVolume::commitParameters()
{
  Volume::commitParameters();
  m_error.clear();
  auto fail = [&](const char *message) {
    if (m_error.empty())
      m_error = message;
  };
  m_material = getParamObject<Material>("material");
  if (!m_material || typeid(*m_material) != typeid(MDL)
      || m_material->deviceState() != deviceState())
    fail("mdl Volume requires a same-device mdl Material");
  m_field = getParamObject<SpatialField>("value");
  if (hasParam("value") && !hasParam("value", ANARI_SPATIAL_FIELD))
    fail("mdl Volume value must be ANARI_SPATIAL_FIELD");
  if (m_field && m_field->deviceState() != deviceState())
    fail("mdl Volume requires a same-device valid scalar field");
  m_fieldName = getParamString("fieldName", "");
  if (hasParam("fieldName") && !hasParam("fieldName", ANARI_STRING))
    fail("mdl Volume fieldName must be ANARI_STRING");
  if (m_field && m_fieldName.empty())
    fail("mdl Volume value requires nonempty fieldName");
  m_hasExtent = hasParam("extent");
  if (m_hasExtent) {
    if (!getParam("extent", ANARI_FLOAT32_BOX3, &m_extent))
      fail("mdl Volume extent must be ANARI_FLOAT32_BOX3");
    const vec3 sides = m_extent.upper - m_extent.lower;
    for (int c = 0; c < 3; ++c)
      if (!std::isfinite(m_extent.lower[c]) || !std::isfinite(m_extent.upper[c])
          || !std::isfinite(sides[c]) || sides[c] <= 0.f)
        fail(
            "mdl Volume extent must have finite strictly positive object-space sides");
  } else if (!m_field)
    fail("mdl Volume requires extent or value");
  if (!getParam("extinctionMajorant", ANARI_FLOAT32, &m_majorant)
      || !std::isfinite(m_majorant) || m_majorant < 0.f)
    fail(
        "mdl Volume requires a finite nonnegative ANARI_FLOAT32 extinctionMajorant");
  // Automatic marching scale is lazy: quality never needs or derives it.
  m_step = -1.f;
  m_hasStep = hasParam("stepSize");
  if (m_hasStep
      && (!getParam("stepSize", ANARI_FLOAT32, &m_step)
          || !std::isfinite(m_step) || m_step <= 0.f))
    fail("mdl Volume stepSize must be finite positive ANARI_FLOAT32");
  deviceState()->objectUpdates.lastVolumeBLASChange = helium::newTimeStamp();
}
void MDLVolume::finalize()
{
  if (const auto *error = renderError())
    reportMessage(ANARI_SEVERITY_ERROR, "%s", error);
  upload();
}
bool MDLVolume::isMDL() const
{
  return true;
}

bool MDLVolume::isValid() const
{
  return renderError() == nullptr;
}
const char *MDLVolume::renderError() const
{
  if (!m_error.empty())
    return m_error.c_str();
  const auto *material = dynamic_cast<const MDL *>(m_material.get());
  if (!material)
    return "invalid mdl Volume material";
  if (!material->volumeError().empty())
    return material->volumeError().c_str();
  if (m_field
      && (!m_field->isValid() || m_field->deviceState() != deviceState()))
    return "mdl Volume requires a same-device valid scalar field";
  if (m_field) {
    const auto bounds = m_field->bounds();
    const auto sides = bounds.upper - bounds.lower;
    for (int c = 0; c < 3; ++c)
      if (!std::isfinite(bounds.lower[c]) || !std::isfinite(bounds.upper[c])
          || !std::isfinite(sides[c]) || sides[c] <= 0.f)
        return "invalid mdl Volume field bounds";
  }
  return nullptr;
}
VolumeGPUData MDLVolume::gpuData() const
{
  auto data = Volume::gpuData();
  data.type = VolumeType::MDL;
  data.bounds = m_hasExtent || !m_field ? m_extent : m_field->bounds();
  data.clipBounds = data.bounds;
  // Intentionally prefer the field's scale even with an extent override.
  // Marching validates the derived scale only when it is actually used.
  data.stepSize = m_hasStep ? m_step : (m_field ? m_field->stepSize() : -1.f);
  data.data.mdl.field = m_field ? m_field->index() : -1;
  data.data.mdl.fieldBounds = m_field ? m_field->bounds() : box3{};
  const auto *material = dynamic_cast<const MDL *>(m_material.get());
  data.data.mdl.fieldNameId =
      m_field && material ? material->volumeFieldId(m_fieldName) : ~0u;
  data.data.mdl.material = m_material ? m_material->index() : -1;
  data.data.mdl.extinctionMajorant = m_majorant;
  return data;
}
} // namespace visrtx
