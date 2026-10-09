// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <anari/anari_cpp.hpp>

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace visrtx::fogtest {

/* Owns one public ANARI reference; the Device must outlive this scope. */
class ObjectOwner
{
 public:
  ObjectOwner(anari::Device device, anari::Object object) noexcept;
  ~ObjectOwner();
  ObjectOwner(const ObjectOwner &) = delete;
  ObjectOwner &operator=(const ObjectOwner &) = delete;
  ObjectOwner(ObjectOwner &&) = delete;
  ObjectOwner &operator=(ObjectOwner &&) = delete;

  // Transfer the reference to a caller or Scene only after setup succeeds.
  void release() noexcept;

 private:
  anari::Device m_device;
  anari::Object m_object;
};

/* Keeps a frame mapping alive through validation and allocating a copy. */
template <typename T>
class FrameMapping
{
 public:
  FrameMapping(anari::Device device, anari::Frame frame, const char *channel);
  ~FrameMapping();
  FrameMapping(const FrameMapping &) = delete;
  FrameMapping &operator=(const FrameMapping &) = delete;
  FrameMapping(FrameMapping &&) = delete;
  FrameMapping &operator=(FrameMapping &&) = delete;

  const T *checkedData(
      unsigned width, unsigned height, ANARIDataType type) const;

 private:
  anari::Device m_device;
  anari::Frame m_frame;
  const char *m_channel;
  anari::MappedFrameData<T> m_mapping;
};

/* Array edits must unmap even when fixture setup throws. */
template <typename T>
class ArrayMapping
{
 public:
  ArrayMapping(anari::Device device, anari::Array array);
  ~ArrayMapping();
  ArrayMapping(const ArrayMapping &) = delete;
  ArrayMapping &operator=(const ArrayMapping &) = delete;
  ArrayMapping(ArrayMapping &&) = delete;
  ArrayMapping &operator=(ArrayMapping &&) = delete;

  T *data() const;

 private:
  anari::Device m_device;
  anari::Array m_array;
  T *m_data;
};

// Inlined definitions ////////////////////////////////////////////////////////

inline ObjectOwner::ObjectOwner(
    anari::Device device, anari::Object object) noexcept
    : m_device(device), m_object(object)
{}

inline ObjectOwner::~ObjectOwner()
{
  if (m_object)
    anari::release(m_device, m_object);
}

inline void ObjectOwner::release() noexcept
{
  m_object = nullptr;
}

template <typename T>
FrameMapping<T>::FrameMapping(
    anari::Device device, anari::Frame frame, const char *channel)
    : m_device(device),
      m_frame(frame),
      m_channel(channel),
      m_mapping(anari::map<T>(device, frame, channel))
{}

template <typename T>
FrameMapping<T>::~FrameMapping()
{
  anari::unmap(m_device, m_frame, m_channel);
}

template <typename T>
const T *FrameMapping<T>::checkedData(
    unsigned width, unsigned height, ANARIDataType type) const
{
  if (!m_mapping.data || m_mapping.width != width || m_mapping.height != height
      || m_mapping.pixelType != type)
    throw std::runtime_error(
        std::string("Invalid mapped channel ") + m_channel);
  return m_mapping.data;
}

template <typename T>
ArrayMapping<T>::ArrayMapping(anari::Device device, anari::Array array)
    : m_device(device), m_array(array), m_data(anari::map<T>(device, array))
{}

template <typename T>
ArrayMapping<T>::~ArrayMapping()
{
  anari::unmap(m_device, m_array);
}

template <typename T>
T *ArrayMapping<T>::data() const
{
  if (!m_data)
    throw std::runtime_error("Invalid mapped array");
  return m_data;
}

template <typename T>
std::vector<T> readChannel(anari::Device device,
    anari::Frame frame,
    const char *name,
    unsigned width,
    unsigned height,
    ANARIDataType type = anari::ANARITypeFor<T>::value)
{
  const FrameMapping<T> mapping(device, frame, name);
  const auto *data = mapping.checkedData(width, height, type);
  return {data, data + std::size_t(width) * height};
}

// A missing advertised subtype is a release failure, never a skipped render.
inline int requireRendererFogSupport(anari::Device device)
{
  int failures = 0;
  for (const char *subtype : {"fast", "interactive", "default", "quality"}) {
    const auto *extensions =
        static_cast<const char *const *>(anariGetObjectInfo(
            device, ANARI_RENDERER, subtype, "extension", ANARI_STRING_LIST));
    bool found = false;
    if (extensions)
      for (auto e = extensions; *e; ++e)
        found |= std::strcmp(*e, "ANARI_VISRTX_RENDERER_FOG") == 0;
    if (!found) {
      fprintf(stderr, "required fog capability missing: %s\n", subtype);
      ++failures;
    }
  }
  return failures;
}

} // namespace visrtx::fogtest
