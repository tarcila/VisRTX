// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#include <cmath>
#include <cstring>
#include "FixtureProviders.h"
#include "array/Array1D.h"
#include "spatial_field/CustomField.h"
#include "spatial_field/SpatialFieldRegistry.h"

namespace visrtx {
namespace typed_custom_test {
template <typename Provider>
struct Field : CustomField
{
  Field(DeviceGlobalState *, bool optIn = true);
  ~Field() override;
  void commitParameters() override;
  void finalize() override;
  bool isValid() const override;
  box3 bounds() const override;
  float stepSize() const override;
  SpatialFieldGPUData gpuData() const override;
  std::unique_ptr<StoredFieldReference> createReferenceField(
      optical_reference::FieldRevision) const override;
  Payload payload;
  vec3 parameterValue{1.f};
  helium::ChangeObserverPtr<Array1D> valueArray;
  bool optIn, valid = false, hasRange = true, omitResources = false,
              unknown = false, badArity = false;
  uint32_t version = 1;
  std::shared_ptr<DeviceBuffer> resources;
};
template <typename P>
Field<P>::Field(DeviceGlobalState *s, bool enabled)
    : CustomField(s), valueArray(this), optIn(enabled)
{}
template <typename P>
Field<P>::~Field()
{
  m_uniformGrid.cleanup();
}
template <typename P>
void Field<P>::commitParameters()
{
  CustomField::commitParameters();
  if constexpr (std::is_same_v<typename P::Value, float>)
    payload.value = vec3(getParam<float>("value", 1.f));
  else
    payload.value = getParam<vec3>("value", vec3(.2f, .4f, .6f));
  parameterValue = payload.value;
  valueArray = getParamObject<Array1D>("valueArray");
  payload.slope = getParam<float>("slope", 0.f);
  const auto roi = getParam<box3>("roi", box3(vec3(0), vec3(1)));
  payload.domain = {glm::max(roi.lower, vec3(0)), glm::min(roi.upper, vec3(1))};
  version = getParam<uint32_t>("contractVersion", 1u);
  hasRange = getParam<bool>("hasRange", true);
  omitResources = getParam<bool>("omitResources", false);
  unknown = getParam<bool>("unknownProvider", false);
  badArity = getParam<bool>("badArity", false);
}
template <typename P>
void Field<P>::finalize()
{
  valid = std::isfinite(payload.slope) && payload.slope >= 0.f
      && payload.slope <= 1.f;
  for (int c = 0; c < 3; ++c)
    valid &= std::isfinite(parameterValue[c])
        && std::isfinite(payload.domain.lower[c])
        && std::isfinite(payload.domain.upper[c]);
  payload.value = parameterValue;
  if (valueArray) {
    valid &= P::RESOURCE_LEASE
        && valueArray->elementType() == ANARI_FLOAT32_VEC3
        && valueArray->size() == 1;
    // The single represented value constructs both the sampler generation and
    // its required raw range, not a separate content-validity scan.
    if (valid)
      payload.value = valueArray->beginAs<vec3>()[0];
  }
  payload.resource = nullptr;
  if constexpr (P::RESOURCE_LEASE) {
    auto generation = std::make_shared<DeviceBuffer>();
    generation->upload(&payload.value);
    resources = std::move(generation);
    payload.resource = resources->template ptrAs<vec3>();
  }
  m_uniformGrid.cleanup();
  if (valid && optical_reference::nonemptyDomain(payload.domain)) {
    m_uniformGrid.init(ivec3(2), payload.domain);
    // Legacy range computation remains separate from typed range certification.
    CustomField::finalize();
  } else
    upload();
}
template <typename P>
bool Field<P>::isValid() const
{
  return valid;
}
template <typename P>
box3 Field<P>::bounds() const
{
  return payload.domain;
}
template <typename P>
float Field<P>::stepSize() const
{
  return .125f;
}
template <typename P>
SpatialFieldGPUData Field<P>::gpuData() const
{
  SpatialFieldGPUData data;
  data.samplerCallableIndex = SbtCallableEntryPoints::SpatialFieldSamplerCustom;
  data.valueType = std::is_same_v<typename P::Value, float>
      ? FieldValueType::Scalar
      : FieldValueType::RGB;
  if (badArity)
    data.valueType = data.valueType == FieldValueType::Scalar
        ? FieldValueType::RGB
        : FieldValueType::Scalar;
  data.roi = payload.domain;
  data.grid = m_uniformGrid.gpuData();
  data.data.custom.subType = optIn ? P::ID : LEGACY_ID;
  static_assert(sizeof(Payload) <= sizeof(data.data.custom.fieldData));
  std::memcpy(data.data.custom.fieldData, &payload, sizeof(payload));
  return data;
}
template <typename P>
std::unique_ptr<StoredFieldReference> Field<P>::createReferenceField(
    optical_reference::FieldRevision key) const
{
  if (!optIn)
    return {};
  using T = typename P::Value;
  auto requested = typedCustomDescriptor<P>();
  requested.version = version;
  if (unknown)
    requested.provider = 0x7fffffffu;
  optical_reference::RawRange<T> range;
  range.empty = !optical_reference::nonemptyDomain(payload.domain);
  for (int c = 0; c < optical_reference::channels<T>(); ++c) {
    float lo = payload.value[c], hi = lo;
    if (payload.slope != 0.f) {
      const float magnitude = std::nextafter(
          float(std::abs(double(lo)) * (1. + double(payload.slope)) * 1.00001),
          INFINITY);
      lo = lo < 0.f ? -magnitude : 0.f;
      hi = payload.value[c] > 0.f ? magnitude : 0.f;
    }
    optical_reference::setComponent(range.lower, c, lo);
    optical_reference::setComponent(range.upper, c, hi);
  }
  TypedCustomError error;
  auto snapshot = StoredFieldReference::custom<T>(requested,
      payload,
      hasRange ? &range : nullptr,
      payload.domain,
      key,
      deviceState()->stream,
      omitResources ? nullptr : resources,
      &error);
  if (!snapshot)
    reportMessage(ANARI_SEVERITY_WARNING,
        "typed fixture: %s",
        typedCustomErrorString(error));
  return snapshot;
}
// Source-built host factories; no runtime extension or subtype advertisement.
struct RegisterFixtures
{
  RegisterFixtures();
};
RegisterFixtures::RegisterFixtures()
{
  registerCustomField("testTypedScalar",
      [](DeviceGlobalState *d) { return new Field<Scalar>(d); });
  registerCustomField("testTypedRGBOptix",
      [](DeviceGlobalState *d) { return new Field<RgbOptix>(d); });
  registerCustomField("testTypedRGBCuda",
      [](DeviceGlobalState *d) { return new Field<RgbCuda>(d); });
  registerCustomField("testTypedRGBOwned",
      [](DeviceGlobalState *d) { return new Field<RgbOwned>(d); });
  registerCustomField("testTypedRGBOwnedCuda",
      [](DeviceGlobalState *d) { return new Field<RgbOwnedCuda>(d); });
  registerCustomField("testTypedScalarOptix",
      [](DeviceGlobalState *d) { return new Field<ScalarOptix>(d); });
  registerCustomField("testLegacyScalar",
      [](DeviceGlobalState *d) { return new Field<Scalar>(d, false); });
}
static RegisterFixtures registration;
} // namespace typed_custom_test
} // namespace visrtx
