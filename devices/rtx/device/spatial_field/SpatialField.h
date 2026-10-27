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
#include "StoredFieldReference.h"
#include "space_skipping/UniformGrid.h"

namespace visrtx {

struct SpatialField : public RegisteredObject<SpatialFieldGPUData>
{
  SpatialField(DeviceGlobalState *d);
  ~SpatialField() = default;

  virtual box3 bounds() const = 0;

  virtual float stepSize() const = 0;

  // Query after successful finalization. Existing custom fields default scalar.
  FieldValueType valueType() const;

  bool getProperty(const std::string_view &name,
      ANARIDataType type,
      void *ptr,
      uint64_t size,
      uint32_t flags) override;

  void markFinalized() override;

  // Lazy reference snapshot: range and sampler are built together from the
  // finalized inputs. No renderer is migrated by exposing this internal seam.
  const StoredFieldReference *referenceField();

  static SpatialField *createInstance(
      std::string_view subtype, DeviceGlobalState *d);

  UniformGrid m_uniformGrid;

 protected:
  virtual std::unique_ptr<StoredFieldReference> createReferenceField(
      optical_reference::FieldRevision key) const;

 private:
  uint64_t m_referenceIdentity{0};
  uint64_t m_referenceRevision{0};
  bool m_referenceAttempted{false};
  std::unique_ptr<StoredFieldReference> m_reference;
};

// Inlined definitions ////////////////////////////////////////////////////////

inline const StoredFieldReference *SpatialField::referenceField()
{
  if (!isValid())
    return nullptr;
  if (!m_referenceAttempted) {
    auto snapshot = createReferenceField({m_referenceIdentity,
        m_referenceRevision,
        STORED_FIELD_REFERENCE_CONTRACT});
    m_reference = std::move(snapshot);
    m_referenceAttempted = true;
  }
  return m_reference.get();
}

} // namespace visrtx

VISRTX_ANARI_TYPEFOR_SPECIALIZATION(
    visrtx::SpatialField *, ANARI_SPATIAL_FIELD);
