/*
 * Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "Geometry.h"
// specific types
#include "Cone.h"
#include "Curve.h"
#include "Cylinder.h"
#include "Isosurface.h"
#include "Quad.h"
#include "Sphere.h"
#include "Triangle.h"

#ifdef VISRTX_USE_NEURAL
#include "Neural.h"
#endif
#include "SDF.h"
#include "UnknownGeometry.h"
// std
#include <cstring>
#include <string_view>
// helium
#include <helium/helium_math.h>

#include "utility/AnariTypeHelpers.h"

namespace visrtx {

// Helper functions ///////////////////////////////////////////////////////////

static void populateAttributeData(Array1D *array, AttributeData &attr)
{
  attr.type = ANARI_UNKNOWN;
  attr.numChannels = 0;
  attr.data = nullptr;

  if (!array)
    return;

  auto type = array->elementType();

  if (!isColor(type))
    return;

  attr.type = type;
  attr.numChannels = numANARIChannels(attr.type);
  attr.data = array->dataGPU();
}

GeometryArrayPtr::GeometryArrayPtr(Geometry *owner, Array1D *array)
    : m_owner(owner), m_array(array)
{
  if (m_array)
    m_owner->addArrayObserver(m_array.ptr);
}

GeometryArrayPtr::~GeometryArrayPtr()
{
  if (m_array)
    m_owner->removeArrayObserver(m_array.ptr);
}

GeometryArrayPtr::GeometryArrayPtr(GeometryArrayPtr &&other) noexcept
    : m_owner(other.m_owner), m_array(other.m_array)
{
  other.m_array = nullptr;
}

GeometryArrayPtr &GeometryArrayPtr::operator=(Array1D *array)
{
  if (array == m_array.ptr)
    return *this;
  if (m_array)
    m_owner->removeArrayObserver(m_array.ptr);
  m_array = array;
  if (m_array)
    m_owner->addArrayObserver(m_array.ptr);
  return *this;
}

GeometryArrayPtr::operator bool() const
{
  return bool(m_array);
}

Array1D *GeometryArrayPtr::operator->() const
{
  return m_array.ptr;
}

Array1D *GeometryArrayPtr::get() const
{
  return m_array.ptr;
}

void Geometry::addArrayObserver(Array1D *array)
{
  if (m_arrayObserverCounts[array]++ == 0)
    array->addChangeObserver(this);
}

void Geometry::removeArrayObserver(Array1D *array)
{
  auto it = m_arrayObserverCounts.find(array);
  if (--it->second == 0) {
    array->removeChangeObserver(this);
    m_arrayObserverCounts.erase(it);
  }
}

GeometryAttributes::GeometryAttributes(Geometry *owner)
    : attribute0(owner),
      attribute1(owner),
      attribute2(owner),
      attribute3(owner),
      color(owner)
{}

// Geometry definitions ///////////////////////////////////////////////////////

Geometry::Geometry(DeviceGlobalState *s)
    : RegisteredObject<GeometryGPUData>(ANARI_GEOMETRY, s)
{
  setRegistry(s->registry.geometries);
}

Geometry::~Geometry()
{
  // Derived bindings have already detached. Clear the base bindings while
  // the count map is still alive; their later destructors then do no work.
  m_primitiveAttributes.attribute0 = nullptr;
  m_primitiveAttributes.attribute1 = nullptr;
  m_primitiveAttributes.attribute2 = nullptr;
  m_primitiveAttributes.attribute3 = nullptr;
  m_primitiveAttributes.color = nullptr;
  m_primitiveId = nullptr;
}

Geometry *Geometry::createInstance(
    std::string_view subtype, DeviceGlobalState *d)
{
  if (subtype == "triangle")
    return new Triangle(d);
  else if (subtype == "quad")
    return new Quad(d);
  else if (subtype == "sphere")
    return new Sphere(d);
  else if (subtype == "cylinder")
    return new Cylinder(d);
  else if (subtype == "cone")
    return new Cone(d);
  else if (subtype == "curve")
    return new Curve(d);
  else if (subtype == "sdf")
    return new SDF(d);
  else if (subtype == "isosurface")
    return new Isosurface(d);
#ifdef VISRTX_USE_NEURAL
  else if (subtype == "neural")
    return new Neural(d);
#endif
  else
    return new UnknownGeometry(subtype, d);
}

void Geometry::commitParameters()
{
  commitAttributes("primitive.", m_primitiveAttributes);

  auto getUniformAttribute =
      [&](const std::string &pName) -> std::optional<vec4> {
    vec4 v(0.f, 0.f, 0.f, 1.f);
    if (getParam(pName, ANARI_FLOAT32_VEC4, &v))
      return v;
    else
      return {};
  };

  m_uniformAttributes.attribute0 = getUniformAttribute("attribute0");
  m_uniformAttributes.attribute1 = getUniformAttribute("attribute1");
  m_uniformAttributes.attribute2 = getUniformAttribute("attribute2");
  m_uniformAttributes.attribute3 = getUniformAttribute("attribute3");
  m_uniformAttributes.color = getUniformAttribute("color");
  m_primitiveId = getParamObject<Array1D>("primitive.id");
}

void Geometry::markFinalized()
{
  markChanged(ChangeKind::SLOT);
  markChanged(ChangeKind::BLAS);
  Object::markFinalized();
}

bool Geometry::isAreaSamplingSupported() const
{
  return false;
}

void Geometry::ensureAreaData() {}

float Geometry::totalArea() const
{
  return 0.f;
}

GeometryGPUData Geometry::gpuData() const
{
  GeometryGPUData retval{};

  const vec4 defaultAttr(0.f, 0.f, 0.f, 1.f);
  retval.attrUniform[0] = m_uniformAttributes.attribute0.value_or(defaultAttr);
  retval.attrUniform[1] = m_uniformAttributes.attribute1.value_or(defaultAttr);
  retval.attrUniform[2] = m_uniformAttributes.attribute2.value_or(defaultAttr);
  retval.attrUniform[3] = m_uniformAttributes.attribute3.value_or(defaultAttr);
  retval.attrUniform[4] = m_uniformAttributes.color.value_or(defaultAttr);
  populateAttributeDataSet(m_primitiveAttributes, retval.attr);
  retval.primitiveId =
      (const uint32_t *)(m_primitiveId ? m_primitiveId->dataGPU() : nullptr);
  retval.epsilonScale = m_epsilonScale;

  return retval;
}

void Geometry::commitAttributes(const char *_prefix, GeometryAttributes &attrs)
{
  std::string prefix = _prefix;
  attrs.attribute0 = getParamObject<Array1D>(prefix + "attribute0");
  attrs.attribute1 = getParamObject<Array1D>(prefix + "attribute1");
  attrs.attribute2 = getParamObject<Array1D>(prefix + "attribute2");
  attrs.attribute3 = getParamObject<Array1D>(prefix + "attribute3");
  attrs.color = getParamObject<Array1D>(prefix + "color");
}

void Geometry::populateAttributeDataSet(
    const GeometryAttributes &hostAttrs, AttributeDataSet &gpuAttrs) const
{
  populateAttributeData(hostAttrs.attribute0.get(), gpuAttrs[0]);
  populateAttributeData(hostAttrs.attribute1.get(), gpuAttrs[1]);
  populateAttributeData(hostAttrs.attribute2.get(), gpuAttrs[2]);
  populateAttributeData(hostAttrs.attribute3.get(), gpuAttrs[3]);
  populateAttributeData(hostAttrs.color.get(), gpuAttrs[4]);
}

} // namespace visrtx

VISRTX_ANARI_TYPEFOR_DEFINITION(visrtx::Geometry *);
