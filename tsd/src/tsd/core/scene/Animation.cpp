// Copyright 2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/core/scene/Animation.hpp"

namespace tsd::core {

std::string &Animation::name()
{
  return m_name;
}

const std::string &Animation::name() const
{
  return m_name;
}

const std::string &Animation::info() const
{
  return m_info;
}

} // namespace tsd::core
