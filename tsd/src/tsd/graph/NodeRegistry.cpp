// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/graph/NodeRegistry.hpp"

namespace tsd::graph {

void NodeRegistry::registerType(tsd::core::Token name, NodeFactory factory)
{
  m_entries.push_back(Entry{name, std::move(factory)});
}

std::unique_ptr<Node> NodeRegistry::create(tsd::core::Token name) const
{
  for (const auto &e : m_entries) {
    if (e.name == name)
      return e.factory();
  }
  return nullptr;
}

bool NodeRegistry::isRegistered(tsd::core::Token name) const
{
  for (const auto &e : m_entries)
    if (e.name == name)
      return true;
  return false;
}

NodeRegistry &GlobalNodeRegistry()
{
  static NodeRegistry registry;
  return registry;
}

} // namespace tsd::graph
