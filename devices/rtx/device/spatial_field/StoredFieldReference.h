// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "ReferenceFieldPreparation.h"
#include "utility/DeviceBuffer.h"

#include <anari/anari.h>
#include <nanovdb/NanoVDB.h>

#include <memory>

namespace visrtx {

// Bump when the canonical decoding, sampling layout or mapping contract
// changes.
inline constexpr uint64_t STORED_FIELD_REFERENCE_CONTRACT = 1;
static_assert(
    sizeof(vec3) == 3 * sizeof(float), "packed float3 storage required");

struct ReferenceAxes
{
  const float *values[3]{}; // host-visible committed axis coordinates
  size_t count[3]{};
};

// Non-owning packed view, retained by its StoredFieldReference owner. This is
// separate from SpatialFieldGPUData: legacy callable/texture semantics remain.
struct StoredFieldReferenceGPUData
{
  const void *data{nullptr};
  const float *axisLUT[3]{};
  vec3 origin{0.f};
  vec3 spacing{1.f};
  vec3 axisLower{0.f};
  vec3 axisUpper{1.f};
  ivec3 indexMin{0};
  ivec3 indexMax{0};
  nanovdb::GridType gridType{nanovdb::GridType::Unknown};
  bool sparse{false};
  bool rectilinear{false};
  bool cellCentered{false};
  bool linear{true};
};

// One field-owned finalized generation. Dense values are canonical float/vec3;
// sparse grids remain sparse and use controlled leaf decoding at sample/range.
// Building is lazy and may perform an acceleration range reduction, not a
// separate content-validity/NaN scan. Valid represented values are a
// precondition.
class StoredFieldReference
{
 public:
  StoredFieldReference() = default;
  ~StoredFieldReference() = default;
  StoredFieldReference(const StoredFieldReference &) = delete;
  StoredFieldReference &operator=(const StoredFieldReference &) = delete;
  StoredFieldReference(StoredFieldReference &&) = delete;
  StoredFieldReference &operator=(StoredFieldReference &&) = delete;

  static std::unique_ptr<StoredFieldReference> dense(const void *data,
      ANARIDataType type,
      uvec3 dims,
      vec3 origin,
      vec3 spacing,
      bool cellCentered,
      bool linear,
      box3 domain,
      optical_reference::FieldRevision key,
      const ReferenceAxes *axes = nullptr);
  static std::unique_ptr<StoredFieldReference> sparse(const void *hostGrid,
      size_t bytes,
      bool cellCentered,
      bool linear,
      box3 domain,
      optical_reference::FieldRevision key,
      const ReferenceAxes *axes = nullptr);

  const optical_reference::FieldSampleDescriptor &descriptor() const;
  StoredFieldReferenceGPUData gpuData() const;
  template <typename T>
  bool range(optical_reference::RawRangeCertificate<T> &out) const;

 private:
  bool buildAxes(const ReferenceAxes &axes);
  DeviceBuffer m_values;
  DeviceBuffer m_axes[3];
  StoredFieldReferenceGPUData m_gpu;
  optical_reference::FieldSampleDescriptor m_descriptor;
  optical_reference::RawRange<float> m_scalarRange;
  optical_reference::RawRange<vec3> m_rgbRange;
};

// Preparation adapter for a registry-indexed table of retained field snapshots.
// Missing/unsupported entries are null, never silently substituted as zero
// data.
struct StoredFieldReferenceProvider
{
  const StoredFieldReference *const *fields;
  size_t count;
  bool describe(
      int32_t index, optical_reference::FieldSampleDescriptor &out) const;
  template <typename T>
  bool range(
      int32_t index, optical_reference::RawRangeCertificate<T> &out) const;
};

// Inlined definitions ////////////////////////////////////////////////////////

inline const optical_reference::FieldSampleDescriptor &
StoredFieldReference::descriptor() const
{
  return m_descriptor;
}

inline StoredFieldReferenceGPUData StoredFieldReference::gpuData() const
{
  return m_gpu;
}

template <typename T>
inline bool StoredFieldReference::range(
    optical_reference::RawRangeCertificate<T> &out) const
{
  if (!fieldBindingAccepts<T>(m_descriptor.valueType))
    return false;
  out.key = m_descriptor.key;
  if constexpr (std::is_same_v<T, float>)
    out.range = m_scalarRange;
  else
    out.range = m_rgbRange;
  return true;
}

inline bool StoredFieldReferenceProvider::describe(
    int32_t index, optical_reference::FieldSampleDescriptor &out) const
{
  if (index < 0 || size_t(index) >= count || !fields[index])
    return false;
  out = fields[index]->descriptor();
  return true;
}

template <typename T>
inline bool StoredFieldReferenceProvider::range(
    int32_t index, optical_reference::RawRangeCertificate<T> &out) const
{
  return index >= 0 && size_t(index) < count && fields[index]
      && fields[index]->range(out);
}

} // namespace visrtx
