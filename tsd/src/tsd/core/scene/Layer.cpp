// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/core/scene/Layer.hpp"
#include "tsd/core/scene/Scene.hpp"

namespace tsd::core {

LayerNodeData::LayerNodeData(const char *n) : m_name(n)
{
  m_srt[0] = math::float3(1.f, 1.f, 1.f);
  m_srt[1] = math::float3(0.f, 0.f, 0.f);
  m_srt[2] = math::float3(0.f, 0.f, 0.f);
}

LayerNodeData::LayerNodeData(Object *o, const char *n) : LayerNodeData(n)
{
  setAsObject(o);
}

LayerNodeData::LayerNodeData(anari::DataType type, size_t index, const char *n)
    : LayerNodeData(n)
{
  setAsObject(type, index);
}

LayerNodeData::LayerNodeData(const math::mat4 &m, const char *n)
    : LayerNodeData(n)
{
  setAsTransform(m);
}

LayerNodeData::LayerNodeData(const math::mat3 &m, const char *n)
    : LayerNodeData(n)
{
  setAsTransform(m);
}

LayerNodeData::LayerNodeData(Array *a, const char *n) : LayerNodeData(n)
{
  setAsTransformArray(a);
}

bool LayerNodeData::hasDefault() const
{
  return m_defaultValue;
}

bool LayerNodeData::isDefaultValue() const
{
  return m_value == m_defaultValue;
}

void LayerNodeData::setToDefaultValue()
{
  if (hasDefault()) {
    m_value = m_defaultValue;
    if (isTransform())
      setAsTransform(getTransform()); // ensure srt matrix is up to date
  }
}

void LayerNodeData::setCurrentValueAsDefault()
{
  if (isTransform())
    m_defaultValue = m_value;
}

anari::DataType LayerNodeData::type() const
{
  return m_value.type();
}

bool LayerNodeData::isObject() const
{
  return anari::isObject(m_value.type());
}

bool LayerNodeData::isTransform() const
{
  return m_value.type() == ANARI_FLOAT32_MAT4;
}

bool LayerNodeData::isEmpty() const
{
  return m_value;
}

bool LayerNodeData::isEnabled() const
{
  return m_enabled;
}

void LayerNodeData::setAsObject(Object *o)
{
  m_value.reset();
  if (o)
    setAsObject(o->type(), o->index());
}

void LayerNodeData::setAsObject(anari::DataType type, size_t index)
{
  m_defaultValue.reset();
  m_value = Any(type, index);
}

void LayerNodeData::setAsTransform(const math::mat4 &m)
{
  m_value = m;
  if (!hasDefault())
    m_defaultValue = m_value;

  auto &sc = m_srt[0];
  auto &azelrot = m_srt[1];
  auto &tl = m_srt[2];
  math::mat4 rot;
  math::decomposeMatrix(m, sc, rot, tl);
  azelrot = math::degrees(math::matrixToAzElRoll(rot));
}

void LayerNodeData::setAsTransform(
    const math::mat4 &m, const math::mat4 &defaultM)
{
  m_defaultValue = defaultM;
  setAsTransform(m);
}

void LayerNodeData::setAsTransform(const math::mat3 &srt)
{
  m_srt = srt;
  auto &sc = srt[0];
  auto &azelrot = srt[1];
  auto &tl = srt[2];

  auto rot = math::IDENTITY_MAT4;
  rot = math::mul(rot,
      math::rotation_matrix(math::rotation_quat(
          math::float3(0.f, 1.f, 0.f), math::radians(azelrot.x))));
  rot = math::mul(rot,
      math::rotation_matrix(math::rotation_quat(
          math::float3(1.f, 0.f, 0.f), math::radians(azelrot.y))));
  rot = math::mul(rot,
      math::rotation_matrix(math::rotation_quat(
          math::float3(0.f, 0.f, 1.f), math::radians(azelrot.z))));

  m_value = math::mul(
      math::translation_matrix(tl), math::mul(rot, math::scaling_matrix(sc)));
  if (!hasDefault())
    m_defaultValue = m_value;
}

void LayerNodeData::setAsTransformArray(Array *a)
{
  setAsObject(a);
}

void LayerNodeData::setEnabled(bool e)
{
  m_enabled = e;
}

Object *LayerNodeData::getObject(Scene *scene) const
{
  return scene->getObject(m_value);
}

size_t LayerNodeData::getObjectIndex() const
{
  return m_value.getAsObjectIndex();
}

math::mat4 LayerNodeData::getTransform() const
{
  return isTransform() ? m_value.getAs<math::mat4>() : math::IDENTITY_MAT4;
}

math::mat3 LayerNodeData::getTransformSRT() const
{
  return isTransform() ? m_srt
                       : math::mat3{math::float3(1.f, 1.f, 1.f),
                             math::float3(0.f, 0.f, 0.f),
                             math::float3(0.f, 0.f, 0.f)};
}

Array *LayerNodeData::getTransformArray(Scene *scene) const
{
  auto *obj = getObject(scene);
  if (obj && obj->type() == ANARI_ARRAY1D) {
    auto *a = (Array *)obj;
    if (a->elementType() == ANARI_FLOAT32_MAT4)
      return a;
  }
  return nullptr;
}

std::string &LayerNodeData::name()
{
  return m_name;
}

Any LayerNodeData::getValueRaw() const
{
  return m_value;
}

void LayerNodeData::setValueRaw(const Any &v)
{
  m_value = v;
  setCurrentValueAsDefault();
}

const InstanceParameterMap &LayerNodeData::getInstanceParameters() const
{
  return m_instanceParameters;
}

void LayerNodeData::setInstanceParameter(const std::string &name, Any v)
{
  m_instanceParameters.set(name, v);
}

void LayerNodeData::clearInstanceParameters()
{
  m_instanceParameters.clear();
}

} // namespace tsd::core
