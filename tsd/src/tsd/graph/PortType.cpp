// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/graph/PortType.hpp"

namespace tsd::graph {

PortType PortTypeRegistry::registerType(const char *name)
{
  tsd::core::Token t(name);
  m_known.insert(static_cast<const void *>(t.value()));
  return PortType{t};
}

bool PortTypeRegistry::isRegistered(tsd::core::Token name) const
{
  return m_known.count(static_cast<const void *>(name.value())) > 0;
}

} // namespace tsd::graph
