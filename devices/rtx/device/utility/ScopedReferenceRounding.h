// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cfenv>
#include <stdexcept>

namespace visrtx {

// Host-only reference construction. Restore the application's thread-local
// rounding mode on success, rejection and exceptions. Gradual underflow remains
// part of the documented host arithmetic environment.
struct ScopedReferenceRounding
{
  ScopedReferenceRounding();
  ~ScopedReferenceRounding();
  ScopedReferenceRounding(const ScopedReferenceRounding &) = delete;
  ScopedReferenceRounding &operator=(const ScopedReferenceRounding &) = delete;

 private:
  int m_previous;
};

inline ScopedReferenceRounding::ScopedReferenceRounding()
    : m_previous(std::fegetround())
{
  if (m_previous == -1 || std::fesetround(FE_TONEAREST) != 0)
    throw std::runtime_error("cannot establish reference rounding mode");
}

inline ScopedReferenceRounding::~ScopedReferenceRounding()
{
  std::fesetround(m_previous);
}

} // namespace visrtx
