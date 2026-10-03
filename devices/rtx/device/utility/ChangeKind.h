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

// helium
#include <helium/utility/TimeStamp.h>
// std
#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <utility>
#include <vector>

namespace visrtx {

// Typed Change Kinds of ADR 0009. An object stamps the kinds it emits when it
// finalizes; consumers compare those stamps with the last ones they consumed.
enum class ChangeKind : uint8_t
{
  SLOT = 0,
  BLAS,
  TLAS,
  LIGHT_SET,
  PIPELINE,
  IMAGE,
  COUNT
};

constexpr size_t NUM_CHANGE_KINDS = size_t(ChangeKind::COUNT);

const char *toString(ChangeKind k);

// One monotonic "last change of kind K" stamp per Change Kind.
struct ChangeStamps
{
  void mark(ChangeKind k, helium::TimeStamp t);
  helium::TimeStamp last(ChangeKind k) const;

 private:
  std::array<helium::TimeStamp, NUM_CHANGE_KINDS> m_stamps{};
};

// A consumer's "last consumed K" record, per dependency and per kind. Stamps
// are per consumer (not cleared bits), so a dependency shared by several
// consumers is seen by each of them.
template <typename DEP>
struct ConsumedStamps
{
  // Returns true (and records 'stamp') if 'stamp' is newer than what this
  // consumer last consumed from 'dep' for kind 'k'.
  bool consume(const DEP *dep, ChangeKind k, helium::TimeStamp stamp);
  helium::TimeStamp lastConsumed(const DEP *dep, ChangeKind k) const;
  void forget(const DEP *dep);

 private:
  std::map<const DEP *, ChangeStamps> m_consumed;
};

// Queue of pending (child -> observer) propagations. Notifications enqueue;
// the commit-buffer flush drains. Duplicate edges collapse into one entry.
template <typename T>
struct PropagationQueue
{
  using Edge = std::pair<T *, T *>; // (child, observer)

  // Returns true if the edge was newly queued.
  bool push(T *child, T *observer);
  // Moves out every queued edge; edges pushed while draining go into the
  // next batch.
  std::vector<Edge> take();
  bool empty() const;

 private:
  mutable std::mutex m_mutex;
  std::vector<Edge> m_edges;
};

// Inlined definitions ////////////////////////////////////////////////////////

inline const char *toString(ChangeKind k)
{
  switch (k) {
  case ChangeKind::SLOT:
    return "Slot";
  case ChangeKind::BLAS:
    return "BLAS";
  case ChangeKind::TLAS:
    return "TLAS";
  case ChangeKind::LIGHT_SET:
    return "LightSet";
  case ChangeKind::PIPELINE:
    return "Pipeline";
  case ChangeKind::IMAGE:
    return "Image";
  default:
    return "unknown";
  }
}

inline void ChangeStamps::mark(ChangeKind k, helium::TimeStamp t)
{
  auto &s = m_stamps[size_t(k)];
  if (t > s)
    s = t;
}

inline helium::TimeStamp ChangeStamps::last(ChangeKind k) const
{
  return m_stamps[size_t(k)];
}

template <typename DEP>
inline bool ConsumedStamps<DEP>::consume(
    const DEP *dep, ChangeKind k, helium::TimeStamp stamp)
{
  auto &rec = m_consumed[dep];
  if (stamp <= rec.last(k))
    return false;
  rec.mark(k, stamp);
  return true;
}

template <typename DEP>
inline helium::TimeStamp ConsumedStamps<DEP>::lastConsumed(
    const DEP *dep, ChangeKind k) const
{
  auto it = m_consumed.find(dep);
  return it == m_consumed.end() ? helium::TimeStamp(0) : it->second.last(k);
}

template <typename DEP>
inline void ConsumedStamps<DEP>::forget(const DEP *dep)
{
  m_consumed.erase(dep);
}

template <typename T>
inline bool PropagationQueue<T>::push(T *child, T *observer)
{
  std::lock_guard<std::mutex> lock(m_mutex);
  const Edge e{child, observer};
  for (const auto &q : m_edges) {
    if (q == e)
      return false;
  }
  m_edges.push_back(e);
  return true;
}

template <typename T>
inline std::vector<typename PropagationQueue<T>::Edge>
PropagationQueue<T>::take()
{
  std::lock_guard<std::mutex> lock(m_mutex);
  std::vector<Edge> out;
  out.swap(m_edges);
  return out;
}

template <typename T>
inline bool PropagationQueue<T>::empty() const
{
  std::lock_guard<std::mutex> lock(m_mutex);
  return m_edges.empty();
}

} // namespace visrtx
