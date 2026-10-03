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

#pragma once

#include "RegisteredObject.h"
#include "array/Array1D.h"
#include "gpu/gpu_objects.h"
// std
#include <optional>
#include <unordered_map>

namespace visrtx {

struct UniformAttributes
{
  helium::IntrusivePtr<Array1D> attribute0Array;
  helium::IntrusivePtr<Array1D> attribute1Array;
  helium::IntrusivePtr<Array1D> attribute2Array;
  helium::IntrusivePtr<Array1D> attribute3Array;
  helium::IntrusivePtr<Array1D> colorArray;

  std::optional<vec4> attribute0;
  std::optional<vec4> attribute1;
  std::optional<vec4> attribute2;
  std::optional<vec4> attribute3;
  std::optional<vec4> color;
};

struct Geometry;

// Helium observer registration is set-like, so aliased array bindings need
// one registration per owner, held until the last binding is removed.
struct GeometryArrayPtr
{
  GeometryArrayPtr(Geometry *owner, Array1D *array = nullptr);
  ~GeometryArrayPtr();
  GeometryArrayPtr(const GeometryArrayPtr &) = delete;
  GeometryArrayPtr &operator=(const GeometryArrayPtr &) = delete;
  GeometryArrayPtr(GeometryArrayPtr &&other) noexcept;
  GeometryArrayPtr &operator=(GeometryArrayPtr &&other) = delete;
  GeometryArrayPtr &operator=(Array1D *array);
  operator bool() const;
  Array1D *operator->() const;
  Array1D *get() const;

 private:
  Geometry *m_owner;
  helium::IntrusivePtr<Array1D> m_array;
};

struct GeometryAttributes
{
  GeometryAttributes(Geometry *owner);

  GeometryArrayPtr attribute0;
  GeometryArrayPtr attribute1;
  GeometryArrayPtr attribute2;
  GeometryArrayPtr attribute3;
  GeometryArrayPtr color;
};

struct Geometry : public RegisteredObject<GeometryGPUData>
{
 private:
  friend struct GeometryArrayPtr;
  void addArrayObserver(Array1D *array);
  void removeArrayObserver(Array1D *array);
  std::unordered_map<Array1D *, size_t> m_arrayObserverCounts;

 public:
  Geometry(DeviceGlobalState *d);

  static Geometry *createInstance(
      std::string_view subtype, DeviceGlobalState *d);

  void commitParameters() override;
  void markFinalized() override;

  virtual void populateBuildInput(OptixBuildInput &) const = 0;
  virtual int optixGeometryType() const = 0;

  // Geometry Light area sampling. Only types that can be sampled by area as an
  // Emissive Surface override these (Stage 1: Triangle). ensureAreaData()
  // builds the per-primitive area CDF; totalArea() is the object-space emitting
  // area.
  virtual bool isAreaSamplingSupported() const;
  virtual void ensureAreaData();
  virtual float totalArea() const;

 protected:
  GeometryGPUData gpuData() const override = 0;

  void commitAttributes(const char *prefix, GeometryAttributes &attrs);
  void populateAttributeDataSet(
      const GeometryAttributes &hostAttrs, AttributeDataSet &gpuAttrs) const;

  GeometryAttributes m_primitiveAttributes{this};
  UniformAttributes m_uniformAttributes;
  GeometryArrayPtr m_primitiveId{this};
  // Set by analytic-primitive subtypes in finalize(); see
  // GeometryGPUData::epsilonScale.
  float m_epsilonScale{0.f};
};

} // namespace visrtx

VISRTX_ANARI_TYPEFOR_SPECIALIZATION(visrtx::Geometry *, ANARI_GEOMETRY);
