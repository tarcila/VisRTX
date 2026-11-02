// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "PrincipledVolume.h"
#include "PreviewColdPreparation.h"
#include "PreviewEligibility.h"
#include "QualityColdPreparation.h"
#include "utility/ScopedReferenceRounding.h"

#include <algorithm>
#include <stdexcept>

namespace visrtx {
namespace {
namespace ref = optical_reference;

void copyPayload(DeviceBuffer &buffer, const void *data, size_t bytes)
{
  buffer.reserve(bytes);
  const auto error =
      cudaMemcpy(buffer.ptr(), data, bytes, cudaMemcpyHostToDevice);
  if (error != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(error));
}

const char *preparationReason(ref::PreparationError error)
{
  switch (error) {
  case ref::PreparationError::FieldUnavailable:
    return "field is invalid or lacks a matching typed snapshot (custom fields require compiled opt-in, conservative ranges and immutable resources)";
  case ref::PreparationError::TypeMismatch:
    return "field scalar/RGB arity does not match the property";
  case ref::PreparationError::InvalidFieldMetadata:
    return "field domain or finalized revision metadata is unavailable";
  case ref::PreparationError::RevisionMismatch:
    return "field sampling and range revisions do not agree; prepare again after finalization";
  case ref::PreparationError::RangeUnavailable:
    return "extinction-driving field has no raw range certificate";
  case ref::PreparationError::InvalidRangeMetadata:
    return "raw/effective range metadata is empty, unordered or unrepresentable";
  case ref::PreparationError::OpticalRangeUnavailable:
    return "the effective range cannot establish a usable optical bound; a tighter range may be needed";
  case ref::PreparationError::InvalidMediumDomain:
    return "bounds must have finite coordinates and positive extents";
  default:
    return "preparation failed";
  }
}
} // namespace

PrincipledVolume::PrincipledVolume(DeviceGlobalState *d)
    : Volume(d, VolumeType::PRINCIPLED)
{}

bool PrincipledVolume::checkType(
    const std::string &name, ANARIDataType type, bool required)
{
  if (!hasParam(name)) {
    if (!required)
      return true;
    reportMessage(ANARI_SEVERITY_WARNING,
        "principledVolume requires parameter '%s'",
        name.c_str());
  } else if (hasParam(name, type))
    return true;
  else
    reportMessage(ANARI_SEVERITY_WARNING,
        "principledVolume parameter '%s' requires type %s",
        name.c_str(),
        anari::toString(type));
  m_parametersValid = false;
  return false;
}

template <typename T>
void PrincipledVolume::readBinding(
    const char *name, FieldBindingGPUData<T> &binding)
{
  constexpr ANARIDataType type =
      std::is_same_v<T, float> ? ANARI_FLOAT32 : ANARI_FLOAT32_VEC3;
  if (checkType(name, type))
    getParam(name, type, &binding.scale);
  const std::string outside = std::string(name) + ".outside";
  if (checkType(outside, type))
    getParam(outside, type, &binding.outside);
  const std::string fieldName = std::string(name) + ".field";
  if (!checkType(fieldName, ANARI_SPATIAL_FIELD))
    return;
  auto *object = getParamObject<helium::BaseObject>(fieldName);
  if (!object)
    return; // absent or explicitly null optional binding
  if (object->type() != ANARI_SPATIAL_FIELD) {
    m_parametersValid = false;
    reportMessage(ANARI_SEVERITY_WARNING,
        "principledVolume parameter '%s' does not reference a spatial field",
        fieldName.c_str());
    return;
  }
  auto *field = static_cast<SpatialField *>(object);
  const auto it = std::find(m_fields.begin(), m_fields.end(), field);
  if (it == m_fields.end()) {
    binding.field = int32_t(m_fields.size());
    m_fields.push_back(field);
  } else
    binding.field = int32_t(it - m_fields.begin());
}

void PrincipledVolume::updateObservers()
{
  // Preserve unchanged owners. Rebuilding a temporary duplicate set would make
  // destruction remove *all* equal SDK observer entries, losing subscriptions.
  for (auto it = m_observers.begin(); it != m_observers.end();) {
    if (std::find(m_fields.begin(), m_fields.end(), it->first)
        == m_fields.end())
      it = m_observers.erase(it);
    else
      ++it;
  }
  for (auto *field : m_fields)
    if (m_observers.find(field) == m_observers.end())
      m_observers.emplace(field,
          std::make_unique<helium::ChangeObserverPtr<SpatialField>>(
              this, field));
}

void PrincipledVolume::commitParameters()
{
  Volume::commitParameters();
  m_valid = false;
  m_parametersValid = true;
  m_bindings = {};
  m_fields.clear();
  if (checkType("bounds", ANARI_FLOAT32_BOX3, true)) {
    getParam("bounds", ANARI_FLOAT32_BOX3, &m_bindings.bounds);
    if (!ref::validDomainMetadata(m_bindings.bounds)
        || !ref::nonemptyDomain(m_bindings.bounds)) {
      m_parametersValid = false;
      reportMessage(ANARI_SEVERITY_WARNING,
          "principledVolume 'bounds' must have finite coordinates and strictly positive extents");
    }
  }
  checkType("id", ANARI_UINT32);
  checkType("visible", ANARI_BOOL);
  checkType("name", ANARI_STRING);
  for (const char *name :
      {"value", "valueRange", "color", "opacity", "unitDistance"}) {
    if (hasParam(name)) {
      m_parametersValid = false;
      reportMessage(ANARI_SEVERITY_WARNING,
          "principledVolume does not accept transfer-function parameter '%s'",
          name);
    }
  }
  readBinding("density", m_bindings.density);
  readBinding("scatteringColor", m_bindings.scatteringColor);
  readBinding("transmissionColor", m_bindings.transmissionColor);
  readBinding("transmissionDistance", m_bindings.transmissionDistance);
  readBinding("anisotropy", m_bindings.anisotropy);
  readBinding("secondaryAnisotropy", m_bindings.secondaryAnisotropy);
  readBinding("secondaryWeight", m_bindings.secondaryWeight);
  readBinding("emission", m_bindings.emission);
  readBinding("temperature", m_bindings.temperature);
  readBinding("blackbodyIntensity", m_bindings.blackbodyIntensity);
  readBinding("blackbodyTint", m_bindings.blackbodyTint);
  updateObservers();
}

void PrincipledVolume::reportPreparation(
    const ref::PreparationResult &result) const
{
  reportMessage(ANARI_SEVERITY_WARNING,
      "principledVolume property '%s' (local field %d): %s",
      result.property ? result.property : "bindings",
      result.field,
      preparationReason(result.error));
}

void PrincipledVolume::finalize()
{
  // Clear even on observer-only failure: no stale successful GPU payload may
  // survive an invalid field/recommit. Mutations follow the supported device
  // prepare/update boundary, not a new asynchronous ownership mechanism.
  m_valid = false;
  m_prepared = {};
  m_committed = {};
  m_previewTexturePrepared = false;
  m_payload.reset();
  m_fieldViews.reset();
  upload();
  if (!m_parametersValid)
    return;

  std::vector<const StoredFieldReference *> references;
  references.reserve(m_fields.size());
  for (auto *field : m_fields)
    references.push_back(field->isValid() ? field->referenceField() : nullptr);
  const StoredFieldReferenceProvider provider{
      references.data(), references.size()};
  ref::PreparedVolumeBindings candidate;
  const auto result =
      ref::prepareVolumeBindings(m_bindings, provider, candidate);
  if (result.error != ref::PreparationError::None) {
    reportPreparation(result);
    return;
  }
  std::vector<StoredFieldReferenceGPUData> views;
  views.reserve(references.size());
  for (const auto *reference : references)
    views.push_back(reference->gpuData());
  if (!views.empty())
    copyPayload(m_fieldViews,
        views.data(),
        views.size() * sizeof(StoredFieldReferenceGPUData));

  PrincipledVolumeGPUData data;
  data.bindings = candidate.bindings;
  data.opticalBounds = candidate.bounds;
  data.fields = m_fieldViews.ptrAs<StoredFieldReferenceGPUData>();
  data.fieldCount = uint32_t(views.size());
  {
    const ScopedReferenceRounding rounding;
    data.previewFloatEligible = preview_preparation::eligible(
        candidate.bindings, candidate.bounds, provider);
  }
  const auto &densityBinding = data.bindings.density;
  if (densityBinding.field >= 0)
    data.densityRevision = references[densityBinding.field]->descriptor().key;
  if (data.previewFloatEligible && densityBinding.field >= 0
      && densityBinding.scale != 0.f
      && data.bindings.transmissionColor.field < 0
      && data.bindings.transmissionDistance.field < 0
      && ref::domainContains(densityBinding.domain, data.bindings.bounds)) {
    ref::RawRangeCertificate<float> raw;
    const auto &field = views[densityBinding.field];
    const auto dims = field.indexMax - field.indexMin + ivec3(1);
    data.previewShadowCacheEligible = field.linear
        && std::max(dims.x, std::max(dims.y, dims.z)) >= 32
        && provider.range(densityBinding.field, raw)
        && raw.range.lower < raw.range.upper;
  }
  if (data.bindings.transmissionColor.field == -1) {
    const ScopedReferenceRounding rounding;
    for (int c = 0; c < 3; ++c)
      data.constantTransmissionLog[c] =
          ref::negativeLog(data.bindings.transmissionColor.scale[c]);
    data.hasConstantTransmissionLog = true;
    if (data.previewFloatEligible
        && data.bindings.transmissionDistance.field < 0) {
      for (int c = 0; c < 3; ++c)
        data.previewExtinctionPerDensity[c] =
            float(data.constantTransmissionLog[c])
            / data.bindings.transmissionDistance.scale;
      data.hasPreviewExtinction = true;
    }
  }
  if (data.previewFloatEligible
      && data.bindings.blackbodyIntensity.scale != 0.f) {
    const auto &cold = previewColdTable();
    copyPayload(m_previewCold, cold.data(), sizeof(cold));
    data.previewCold = m_previewCold.ptrAs<PreviewColdKnot>();
    const auto &quality = qualityColdTable();
    copyPayload(m_qualityCold, quality.data(), sizeof(quality));
    data.qualityCold = m_qualityCold.ptrAs<PreviewColdKnot>();
  }
  copyPayload(m_payload, &data, sizeof(data));
  const auto error = cudaStreamSynchronize(nullptr);
  if (error != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(error));
  m_referenceStep = std::numeric_limits<double>::max();
  for (int c = 0; c < 3; ++c)
    m_referenceStep = std::min(m_referenceStep,
        .5
            * (ref::promote(m_bindings.bounds.upper[c])
                - ref::promote(m_bindings.bounds.lower[c])));
  for (auto *field : m_fields) {
    const double hint = ref::promote(field->stepSize());
    if (hint > 0. && std::isfinite(hint))
      m_referenceStep = std::min(m_referenceStep, hint);
  }
  m_prepared = candidate;
  m_committed = data;
  m_valid = true;
  upload();
}

void PrincipledVolume::markFinalized()
{
  Volume::markFinalized();
  // Domain and validity can change independently of any bound field; force the
  // existing Group/World membership and AABB caches to observe both
  // transitions.
  deviceState()->objectUpdates.lastVolumeBLASChange = helium::newTimeStamp();
}

bool PrincipledVolume::isValid() const
{
  return m_valid;
}

void *PrincipledVolume::deviceData() const
{
  return m_valid ? m_payload.ptr() : nullptr;
}

ReferenceVolumeGPUData PrincipledVolume::referenceVolume()
{
  if (!m_valid)
    return {};
  const auto &b = m_committed.bindings;
  if (!m_previewTexturePrepared && m_committed.previewFloatEligible
      && b.density.field >= 0 && b.transmissionColor.field < 0
      && b.transmissionDistance.field < 0) {
    const auto *field = m_fields[b.density.field]->referenceField();
    if (field) {
      const auto view = field->previewGpuData();
      const auto error = cudaMemcpy(
          m_fieldViews.ptrAs<StoredFieldReferenceGPUData>() + b.density.field,
          &view,
          sizeof(view),
          cudaMemcpyHostToDevice);
      if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));
    }
    m_previewTexturePrepared = true;
  }
  return {VolumeType::PRINCIPLED,
      m_committed.bindings.bounds,
      m_payload.ptrAs<PrincipledVolumeGPUData>(),
      nullptr,
      m_referenceStep,
      index(),
      m_committed.previewFloatEligible,
      b.emission.scale != vec3(0.f)
          || (b.blackbodyIntensity.scale != 0.f && b.temperature.scale != 0.f
              && b.blackbodyTint.scale != vec3(0.f))};
}

bool PrincipledVolume::getProperty(const std::string_view &name,
    ANARIDataType type,
    void *ptr,
    uint64_t size,
    uint32_t flags)
{
  // BaseDevice owns WAIT-triggered flushing. NO_WAIT must not flush here.
  if (name == "bounds" && type == ANARI_FLOAT32_BOX3 && m_valid
      && size >= sizeof(box3)) {
    std::memcpy(ptr, &m_committed.bindings.bounds, sizeof(box3));
    return true;
  }
  return Volume::getProperty(name, type, ptr, size, flags);
}

VolumeGPUData PrincipledVolume::gpuData() const
{
  auto result = Volume::gpuData();
  result.type = VolumeType::PRINCIPLED;
  result.data.principled =
      m_valid ? m_payload.ptrAs<PrincipledVolumeGPUData>() : nullptr;
  result.bounds =
      m_valid ? m_committed.bindings.bounds : box3(vec3(0.f), vec3(0.f));
  result.stepSize = 0.f; // Legacy integration is not permitted for this type.
  return result;
}

} // namespace visrtx
