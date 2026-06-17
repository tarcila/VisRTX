// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/graph/Node.hpp"
// std
#include <functional>
#include <memory>
#include <vector>

namespace tsd::graph {

using NodeFactory = std::function<std::unique_ptr<Node>()>;

// Maps a node-type Token to a factory. Built-in node types self-register into
// a process-global registry via the GlobalNodeRegistry() accessor.
struct NodeRegistry
{
  void registerType(tsd::core::Token name, NodeFactory factory);
  std::unique_ptr<Node> create(tsd::core::Token name) const;
  bool isRegistered(tsd::core::Token name) const;

 private:
  struct Entry
  {
    tsd::core::Token name;
    NodeFactory factory;
  };
  std::vector<Entry> m_entries;
};

NodeRegistry &GlobalNodeRegistry();

// RAII registrar: a static instance self-registers a node type at static-init.
struct NodeRegistrar
{
  NodeRegistrar(tsd::core::Token name, NodeFactory factory)
  {
    GlobalNodeRegistry().registerType(name, std::move(factory));
  }
};

} // namespace tsd::graph

// Place in a node type's .cpp to self-register it:
//   TSD_GRAPH_REGISTER_NODE("MyNode", MyNodeClass)
#define TSD_GRAPH_REGISTER_NODE(NAME, TYPE)                                    \
  namespace {                                                                  \
  const ::tsd::graph::NodeRegistrar s_registrar_##TYPE(                        \
      ::tsd::core::Token(NAME), [] { return std::make_unique<TYPE>(); });      \
  }
